#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import copy
import hashlib
import pickle
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, Callable
from typing import Union

import ray
import torch
import vmas
import wandb
from ray.rllib import RolloutWorker, BaseEnv, Policy, VectorEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import PolicyID
from ray.tune import register_env
from torch import nn, Tensor
from vmas import make_env
from vmas.simulator.environment import Environment

from evaluate.distance_metrics import *
from evaluate.evaluate_model import TorchDiagGaussian
from models.fcnet import MyFullyConnectedNetwork
from models.gppo import GPPO
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)
from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer

from models.lip_multiagent_mlp import LipNormedMultiAgentMLP
from scenarios.simplified_het_mass import SimplifiedHetMass
from vmas_beta.vmas import VmasEnv
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
import matplotlib.pyplot as plt


class PathUtils:
    scratch_dir = (
        Path("/content/mlmi-thesis")
        if platform.system() == "Darwin" # native Mac OS X
        else Path("/content/mlmi-thesis/") # colab
    )
    gppo_dir = Path(__file__).parent.resolve()
    result_dir = gppo_dir / "results"
    rollout_storage = result_dir / "rollout_storage"


class InjectMode(Enum):
    ACTION_NOISE = 1
    OBS_NOISE = 2
    SWITCH_AGENTS = 3

    def is_noise(self):
        if self is InjectMode.OBS_NOISE or self is InjectMode.ACTION_NOISE:
            return True
        return False

    def is_obs(self):
        if self is InjectMode.OBS_NOISE or self is InjectMode.SWITCH_AGENTS:
            return True
        return False

    def is_action(self):
        if self is InjectMode.ACTION_NOISE or self is InjectMode.SWITCH_AGENTS:
            return True
        return False


class TrainingUtils:
    @staticmethod
    def init_ray(scenario_name: str, local_mode: bool = False):
        if not ray.is_initialized():
            ray.init(
                _temp_dir=str(PathUtils.scratch_dir / "ray"),
                local_mode=local_mode,
            )
            print("Ray init!")
        register_env(scenario_name, lambda config: TrainingUtils.env_creator(config))
        ModelCatalog.register_custom_model("GPPO", GPPO)
        ModelCatalog.register_custom_model(
            "MyFullyConnectedNetwork", MyFullyConnectedNetwork
        )
        ModelCatalog.register_custom_action_dist(
            "hom_multi_action", TorchHomogeneousMultiActionDistribution
        )

    @staticmethod
    def env_creator(config: Dict):
        env = make_env(
            scenario=config["scenario_name"],
            num_envs=config["num_envs"],
            device=config["device"],
            continuous_actions=config["continuous_actions"],
            wrapper=vmas.Wrapper.RLLIB,
            max_steps=config["max_steps"],
            # Scenario specific
            **config["scenario_config"],
        )
        return env

    class EvaluationCallbacks(DefaultCallbacks):
        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            episode: Episode,
            **kwargs,
        ):
            info = episode.last_info_for()
            for a_key in info.keys():
                for b_key in info[a_key]:
                    try:
                        episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                    except KeyError:
                        episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[str, Policy],
            episode: Episode,
            **kwargs,
        ):
            info = episode.last_info_for()
            for a_key in info.keys():
                for b_key in info[a_key]:
                    metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                    episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()

    class RenderingCallbacks(DefaultCallbacks):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.frames = []

        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Episode,
            **kwargs,
        ) -> None:
            self.frames.append(
                base_env.vector_env.try_render_at(
                    mode="rgb_array", agent_index_focus=None
                )
            )

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
        ) -> None:
            vid = np.transpose(self.frames, (0, 3, 1, 2))
            episode.media["rendering"] = wandb.Video(
                vid, fps=1 / base_env.vector_env.env.world.dt, format="mp4"
            )
            self.frames = []

    class HeterogeneityMeasureCallbacks(DefaultCallbacks):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.policy = None
            self.all_obs = []
            self.all_act = []

        def reset(self):
            self.all_obs = []
            self.all_act = []

        def on_episode_step(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            episode: Episode,
            **kwargs,
        ) -> None:
            obs = episode.last_raw_obs_for()
            act = episode.last_action_for()
            info = episode.last_info_for()
            reward = episode.last_reward_for()
            for i, agent_obs in enumerate(obs):
                obs[i] = torch.tensor(obs[i]).unsqueeze(0)
            # self.all_act.append(
            #     torch.chunk(torch.tensor(act), base_env.vector_env.env.n_agents)
            # )
            self.all_obs.append(obs)

        def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Episode,
            **kwargs,
        ) -> None:
            self.env: Environment = base_env.vector_env.env
            self.n_agents = self.env.n_agents
            self.input_lens = [
                2 * self.env.get_agent_action_size(agent) for agent in self.env.agents
            ]

            self.policy = policies["default_policy"]
            self.model = self.policy.model
            self.model_state_dict = self.model.state_dict()

            self.temp_model_i = copy.deepcopy(self.model)
            self.temp_model_j = copy.deepcopy(self.model)
            self.temp_model_i.eval()
            self.temp_model_j.eval()

            dists = torch.full(
                (
                    len(self.all_obs),
                    int((self.n_agents * (self.n_agents - 1)) / 2),
                    self.n_agents,
                    self.env.get_agent_action_size(self.env.agents[0]),
                ),
                -1.0,
                dtype=torch.float,
            )
            # num_obs,
            # number of unique pairs,
            # number of spots within an observation where I can evaluate the agents,
            # number of actions per agent

            all_measures = {
                "wasserstein": dists,
                "kl": dists.clone(),
                "kl_sym": dists.clone(),
                "hellinger": dists.clone(),
                "bhattacharyya": dists.clone(),
                "balch": dists.clone(),
            }

            # self.all_act = self.all_act[1:] + self.all_act[:1]
            pair_index = 0
            for i in range(self.n_agents):
                for j in range(self.n_agents):
                    if j <= i:
                        continue
                    # Line run for all pairs
                    for agent_index in range(self.n_agents):
                        self.temp_model_i.load_state_dict(self.model_state_dict)
                        self.temp_model_j.load_state_dict(self.model_state_dict)
                        try:
                            mdl = self.model.gnn
                            tmp_model_i = self.temp_model_i.gnn
                            tmp_model_j = self.temp_model_j.gnn
                        except AttributeError:
                            mdl = self.model
                            tmp_model_i = self.temp_model_i
                            tmp_model_j = self.temp_model_j
                        for tmp_layer_i, tmp_layer_j, layer in zip(
                            tmp_model_i.children(),
                            tmp_model_j.children(),
                            mdl.children(),
                        ):
                            assert isinstance(layer, nn.ModuleList)
                            if len(list(layer.children())) > 1:
                                assert len(list(layer.children())) == self.n_agents
                                self.load_agent_x_in_pos_y(
                                    tmp_layer_i, layer, x=i, y=agent_index
                                )
                                self.load_agent_x_in_pos_y(
                                    tmp_layer_j, layer, x=j, y=agent_index
                                )

                        for obs_index, obs in enumerate(self.all_obs):
                            return_dict = self.compute_distance(
                                temp_model_i=self.temp_model_i,
                                temp_model_j=self.temp_model_j,
                                obs=obs,
                                agent_index=agent_index,
                                i=i,
                                j=j,
                                act=None,
                                check_act=False,  # not obs_index == dists.shape[0] - 1,
                            )
                            for key, value in all_measures.items():
                                assert (
                                    all_measures[key][
                                        obs_index, pair_index, agent_index
                                    ].shape
                                    == return_dict[key].shape
                                )
                                all_measures[key][
                                    obs_index, pair_index, agent_index
                                ] = return_dict[key]
                    pair_index += 1

            all_measures_agent_matrix = self.get_distance_matrix(all_measures)
            self.upload_per_agent_contribution(all_measures_agent_matrix, episode)
            self.compute_hierarchical_social_entropy(all_measures_agent_matrix, episode)
            for key, value in all_measures.items():
                assert not (value < 0).any(), f"{key}_{value}"
                episode.custom_metrics[f"mine/{key}"] = value.mean().item()

            self.reset()

        def get_distance_matrix(
            self, all_measures: Dict[str, Tensor]
        ) -> Dict[str, Tensor]:
            all_measures_agent_matrix = {}
            for key, dists in all_measures.items():
                per_agent_distances = torch.full(
                    (self.n_agents, self.n_agents),
                    -1.0,
                    dtype=torch.float32,
                )
                per_agent_distances.diagonal()[:] = 0

                pair_index = 0
                for i in range(self.n_agents):
                    for j in range(self.n_agents):
                        if j <= i:
                            continue
                        pair_distance = dists[:, pair_index].mean()
                        per_agent_distances[i][j] = pair_distance
                        per_agent_distances[j][i] = pair_distance
                        pair_index += 1
                assert not (per_agent_distances < 0).any()
                all_measures_agent_matrix[key] = per_agent_distances
            return all_measures_agent_matrix

        def upload_per_agent_contribution(self, all_measures_agent_matrix, episode):
            for key, agent_matrix in all_measures_agent_matrix.items():
                for i in range(self.n_agents):
                    episode.custom_metrics[f"{key}/agent_{i}"] = agent_matrix[
                        i
                    ].sum().item() / (self.n_agents - 1)
                    for j in range(self.n_agents):
                        if j <= i:
                            continue
                        episode.custom_metrics[f"{key}/agent_{i}{j}"] = agent_matrix[
                            i, j
                        ].item()

        def compute_hierarchical_social_entropy(
            self, all_measures_agent_matrix, episode
        ):
            for metric_name, agent_matrix in all_measures_agent_matrix.items():
                distances = []
                for i in range(self.n_agents):
                    for j in range(self.n_agents):
                        if j <= i:
                            continue
                        distances.append(({i, j}, agent_matrix[i, j].item()))
                distances.sort(key=lambda e: e[1])
                intervals = []
                saved = 0
                for i in range(len(distances)):
                    intervals.append(distances[i][1] - saved)
                    saved = distances[i][1]

                hierarchical_social_ent = 0.0
                hs = [0.0] + [dist[1] for dist in distances[:-1]]

                for interval, h in zip(intervals, hs):
                    hierarchical_social_ent += interval * self.compute_social_entropy(
                        h, agent_matrix
                    )
                assert hierarchical_social_ent >= 0
                episode.custom_metrics[f"hse/{metric_name}"] = hierarchical_social_ent

        def compute_social_entropy(self, h, agent_matrix):
            clusters = self.cluster(h, agent_matrix)
            total_elements = np.array([len(cluster) for cluster in clusters]).sum()
            ps = [len(cluster) / total_elements for cluster in clusters]
            social_entropy = -np.array([p * np.log2(p) for p in ps]).sum()
            return social_entropy

        def cluster(self, h, agent_matrix):
            # Diametric clustering
            clusters = [{i} for i in range(self.n_agents)]
            for i, cluster in enumerate(clusters):
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    can_add = True
                    for k in cluster:
                        if agent_matrix[k, j].item() > h:
                            can_add = False
                            break
                    if can_add:
                        cluster.add(j)

            # Remove duplicate clusters
            clusters = [set(item) for item in set(frozenset(item) for item in clusters)]

            # Remove subsets (should not be used)
            final_clusters = copy.deepcopy(clusters)
            for i, c1 in enumerate(clusters):
                for j, c2 in enumerate(clusters):
                    if i != j and c1.issuperset(c2) and c2 in final_clusters:
                        final_clusters.remove(c2)
            assert final_clusters == clusters, "Superset check should be useless"
            return final_clusters

        def load_agent_x_in_pos_y(self, temp_model, model, x, y):
            temp_model[y].load_state_dict(model[x].state_dict())
            return temp_model

        def compute_distance(
            self,
            temp_model_i,
            temp_model_j,
            obs,
            agent_index,
            i,
            j,
            act,
            check_act,
        ):

            input_dict = {"obs": obs}

            logits_i = temp_model_i(input_dict)[0].detach()
            logits_j = temp_model_j(input_dict)[0].detach()

            split_inputs_i = torch.split(logits_i, self.input_lens, dim=1)
            split_inputs_j = torch.split(logits_j, self.input_lens, dim=1)

            distr_i = TorchDiagGaussian(
                split_inputs_i[agent_index], self.env.agents[agent_index].u_range
            )
            distr_j = TorchDiagGaussian(
                split_inputs_j[agent_index], self.env.agents[agent_index].u_range
            )

            mean_i = distr_i.dist.mean
            mean_j = distr_j.dist.mean

            # Check
            i_is_loaded_in_its_pos = agent_index == i
            j_is_loaded_in_its_pos = agent_index == j
            assert i != j
            if check_act:
                act = act[agent_index]
                if i_is_loaded_in_its_pos:
                    assert (act == mean_i).all()
                elif j_is_loaded_in_its_pos:
                    assert (act == mean_j).all()

            var_i = distr_i.dist.variance
            var_j = distr_i.dist.variance

            return_value = {}
            for name, distance in zip(
                ["wasserstein", "kl", "kl_sym", "hellinger", "bhattacharyya", "balch"],
                [
                    wasserstein_distance,
                    kl_divergence,
                    kl_symmetric,
                    hellinger_distance,
                    bhattacharyya_distance,
                    balch,
                ],
            ):
                distances = []
                for k in range(self.env.get_agent_action_size(self.env.agents[0])):
                    distances.append(
                        torch.tensor(
                            distance(
                                mean_i[..., k].numpy(),
                                var_i[..., k].unsqueeze(-1).numpy(),
                                mean_j[..., k].numpy(),
                                var_j[..., k].unsqueeze(-1).numpy(),
                            )
                        )
                    )
                    assert (
                        distances[k] >= 0
                    ).all(), f"{name}, [{distances[k]} with mean_i {mean_i[..., k]} var_i {var_i[...,k]}, mean_j {mean_j[..., k]} var_j {var_j[...,k]}"
                return_value[name] = torch.stack(distances)

            return return_value


class EvaluationUtils:

    # Resilience injection utils
    @staticmethod
    def __inject_noise_in_action(
        agent_actions: tuple,
        agent_indices: Set[int],
        noise_delta: float,
        env: VectorEnv,
    ) -> Tuple:
        assert len(agent_indices) <= len(agent_actions)
        agent_actions_new = list(agent_actions)
        for agent_index in agent_indices:
            noise = np.random.uniform(
                -noise_delta, noise_delta, size=agent_actions_new[agent_index].shape
            )
            agent_actions_new[agent_index] += noise
            agent_actions_new[agent_index] = np.clip(
                agent_actions_new[agent_index],
                -env.env.agents[agent_index].u_range,
                env.env.agents[agent_index].u_range,
            )
        return tuple(agent_actions_new)

    @staticmethod
    def __inject_noise_in_observation(
        observations: tuple,
        agent_indices: Set[int],
        noise_delta: float,
    ) -> Tuple:
        assert len(agent_indices) <= len(observations)
        observations_new = list(observations)
        for agent_index in agent_indices:
            noise = np.random.uniform(
                -noise_delta, noise_delta, size=observations_new[agent_index].shape
            )
            observations_new[agent_index] += noise
        return tuple(observations_new)

    @staticmethod
    def __switch_agents(
        angents_io: tuple,
        agent_indices: Set[int],
    ) -> Tuple:
        assert len(agent_indices) <= len(angents_io)
        assert len(agent_indices) == 2
        agent_indices = list(agent_indices)
        agents_io_new = list(angents_io)

        agents_io_new[agent_indices[0]] = angents_io[agent_indices[1]]
        agents_io_new[agent_indices[1]] = angents_io[agent_indices[0]]

        return tuple(agents_io_new)

    @staticmethod
    def get_inject_function(
        inject_mode: InjectMode,
        noise_delta: float,
        agents_to_inject: Set,
        env: VectorEnv,
    ):
        def inject_function(x):
            if inject_mode is InjectMode.ACTION_NOISE:
                return EvaluationUtils.__inject_noise_in_action(
                    x, agent_indices=agents_to_inject, noise_delta=noise_delta, env=env
                )
            elif inject_mode is InjectMode.OBS_NOISE:
                return EvaluationUtils.__inject_noise_in_observation(
                    x, noise_delta=noise_delta, agent_indices=agents_to_inject
                )
            elif inject_mode is InjectMode.SWITCH_AGENTS:
                assert noise_delta == 0
                return EvaluationUtils.__switch_agents(
                    x, agent_indices=agents_to_inject
                )
            else:
                assert False

        return inject_function

    @staticmethod
    def get_checkpoint_config(checkpoint_path: Union[str, Path]):
        params_path = Path(checkpoint_path).parent / "params.pkl"
        with open(params_path, "rb") as f:
            config = pickle.load(f)
        return config

    @staticmethod
    def get_config_trainer_and_env_from_checkpoint(
        checkpoint_path: Union[str, Path],
        for_evaluation: bool = True,
        config_update_fn: Callable[[Dict], Dict] = None,
    ):
        config = EvaluationUtils.get_checkpoint_config(checkpoint_path)
        scenario_name = config["env"]
        TrainingUtils.init_ray(scenario_name=scenario_name)

        if for_evaluation:

            # Env
            env_config = config["env_config"]
            env_config.update({"num_envs": 1})

            # Scenario
            # env_config["scenario_config"].update({"mass_position": 0.75})

            # Eval
            eval_config = config["evaluation_config"]
            eval_config.update({"callbacks": None})

            config_update = {
                "in_evaluation": True,
                "num_workers": 0,
                "num_gpus": 0,
                "num_envs_per_worker": 1,
                "callbacks": None,
                "env_config": env_config,
                "evaluation_config": eval_config
                # "explore": False,
            }
            config.update(config_update)

        if config_update_fn is not None:
            config = config_update_fn(config)

        print(f"\nConfig: {config}")

        trainer = MultiPPOTrainer(env=scenario_name, config=config)
        trainer.restore(str(checkpoint_path))
        trainer.start_config = config
        env = TrainingUtils.env_creator(config["env_config"])
        env.seed(config["seed"])

        return config, trainer, env

    @staticmethod
    def rollout_episodes(
        n_episodes: int,
        render: bool,
        get_obs: bool,
        get_actions: bool,
        trainer: MultiPPOTrainer,
        env: VectorEnv,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
        action_callback=None,
        use_pickle: bool = True,
    ):
        assert (trainer is None) != (action_callback is None)

        if trainer is not None:
            print(
                f"\nLoaded: {EvaluationUtils.get_model_name(trainer.config)[0]}, {EvaluationUtils.get_model_name(trainer.config)[2]}"
            )
        if inject:
            print(
                f"Injected: {EvaluationUtils.get_inject_name(inject_mode=inject_mode, agents_to_inject=agents_to_inject, noise_delta=noise_delta)[0]}"
            )
            inject_function = EvaluationUtils.get_inject_function(
                inject_mode,
                agents_to_inject=agents_to_inject,
                noise_delta=noise_delta,
                env=env,
            )

        best_gif = None
        rewards = []
        observations = []
        actions = []
        if use_pickle and trainer:
            (
                rewards,
                best_gif,
                observations,
                actions,
            ) = EvaluationUtils.__get_pickled_rollout(
                render,
                get_obs,
                get_actions,
                trainer,
                inject,
                inject_mode,
                agents_to_inject,
                noise_delta,
            )
            (rewards, observations, actions) = EvaluationUtils.__crop_rollout(
                rewards, observations, actions, get_obs, get_actions, n_episodes
            )
            print(f"Loaded from pickle {len(rewards)} episodes!")

        best_reward = max(rewards, default=float("-inf"))

        for j in range(len(rewards), n_episodes):
            env.seed(j)
            frame_list = []
            observations_this_episode = []
            actions_this_episode = []
            reward_sum = 0
            observation = env.vector_reset()[0]
            i = 0
            done = False
            if render:
                frame_list.append(
                    env.try_render_at(mode="rgb_array", visualize_when_rgb=True)
                )
            while not done:
                i += 1
                if inject and inject_mode.is_obs():
                    observation = inject_function(observation)
                if get_obs:
                    observations_this_episode.append(observation)

                if trainer is not None:
                    action = trainer.compute_single_action(observation)
                else:
                    action = action_callback(observation)

                if inject and inject_mode.is_action():
                    action = inject_function(action)
                if get_actions:
                    actions_this_episode.append(action)
                obss, rews, ds, infos = env.vector_step([action])
                observation = obss[0]
                reward = rews[0]
                done = ds[0]
                info = infos[0]
                reward_sum += reward
                if render:
                    frame_list.append(
                        env.try_render_at(mode="rgb_array", visualize_when_rgb=True)
                    )
            print(f"Episode: {j + 1}, total reward: {reward_sum}")
            rewards.append(reward_sum)
            if reward_sum > best_reward and render:
                best_reward = reward_sum
                best_gif = frame_list.copy()
            if get_obs:
                observations.append(observations_this_episode)
            if get_actions:
                actions.append(actions_this_episode)
        print(
            f"Max reward: {np.max(rewards)}\nReward mean: {np.mean(rewards)}\nMin reward: {np.min(rewards)}"
        )

        if use_pickle and trainer:
            EvaluationUtils.__store_pickled_rollout(
                rewards,
                best_gif,
                observations,
                actions,
                trainer,
                inject,
                inject_mode,
                agents_to_inject,
                noise_delta,
            )

        assert len(rewards) == n_episodes
        if get_obs:
            assert len(observations) == n_episodes
        if get_actions:
            assert len(actions) == n_episodes
        if render:
            assert best_gif

        return (
            rewards,
            best_gif,
            observations,
            actions,
        )

    @staticmethod
    def __crop_rollout(
        rewards,
        observations,
        actions,
        get_obs: bool,
        get_actions: bool,
        n_episodes: int,
    ):
        min_len = min(len(rewards), n_episodes)
        if get_actions:
            min_len = min(len(actions), min_len)
        if get_obs:
            min_len = min(len(observations), min_len)
        return (
            rewards[:min_len],
            observations[:min_len] if get_obs else observations,
            actions[:min_len] if get_actions else actions,
        )

    @staticmethod
    def __store_pickled_rollout(
        rewards,
        best_gif,
        observations,
        actions,
        trainer: MultiPPOTrainer,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
    ):
        (
            _,
            model_name,
            _,
            env_name,
        ) = EvaluationUtils.get_model_name(trainer.config)

        _, inject_name = EvaluationUtils.get_inject_name(
            agents_to_inject=agents_to_inject,
            noise_delta=noise_delta,
            inject_mode=inject_mode,
        )
        hash = hashlib.sha256()
        hash.update(bytes(str(trainer.start_config), "UTF-8"))

        name = (
            f"{model_name}_{env_name}"
            + ("_" + inject_name if inject else "")
            + f"_{hash.hexdigest()}"
        )

        reward_file = PathUtils.rollout_storage / f"rew_{name}.pkl"
        best_gif_file = PathUtils.rollout_storage / f"gif_{name}.pkl"
        observations_file = PathUtils.rollout_storage / f"obs_{name}.pkl"
        actions_file = PathUtils.rollout_storage / f"acts_{name}.pkl"

        (
            rewards_loaded,
            best_gif_loaded,
            observations_loaded,
            actions_loaded,
        ) = EvaluationUtils.__get_pickled_rollout(
            best_gif is not None,
            len(observations) > 0,
            len(actions) > 0,
            trainer,
            inject,
            inject_mode,
            agents_to_inject,
            noise_delta,
        )
        if len(rewards) > len(rewards_loaded):
            pickle.dump(rewards, open(reward_file, "wb"))
        if (
            best_gif is not None
            and (best_gif_loaded is None or len(rewards_loaded) < len(rewards))
            and False
        ):
            pickle.dump(best_gif, open(best_gif_file, "wb"))
        if len(observations) > len(observations_loaded):
            pickle.dump(observations, open(observations_file, "wb"))
        if len(actions) > len(actions_loaded):
            pickle.dump(actions, open(actions_file, "wb"))

    @staticmethod
    def __get_pickled_rollout(
        render: bool,
        get_obs: bool,
        get_actions: bool,
        trainer: MultiPPOTrainer,
        inject: bool,
        inject_mode: InjectMode,
        agents_to_inject: Set,
        noise_delta: float,
    ):

        (
            _,
            model_name,
            _,
            env_name,
        ) = EvaluationUtils.get_model_name(trainer.config)

        _, inject_name = EvaluationUtils.get_inject_name(
            agents_to_inject=agents_to_inject,
            noise_delta=noise_delta,
            inject_mode=inject_mode,
        )

        hash = hashlib.sha256()
        hash.update(bytes(str(trainer.start_config), "UTF-8"))
        name = (
            f"{model_name}_{env_name}"
            + ("_" + inject_name if inject else "")
            + f"_{hash.hexdigest()}"
        )

        reward_file = PathUtils.rollout_storage / f"rew_{name}.pkl"
        best_gif_file = PathUtils.rollout_storage / f"gif_{name}.pkl"
        observations_file = PathUtils.rollout_storage / f"obs_{name}.pkl"
        actions_file = PathUtils.rollout_storage / f"acts_{name}.pkl"

        best_gif = None
        rewards = []
        observations = []
        actions = []

        if reward_file.is_file():
            rewards = pickle.load(open(reward_file, "rb"))
            if render and best_gif_file.is_file():
                best_gif = pickle.load(open(best_gif_file, "rb"))
            if get_obs and observations_file.is_file():
                observations = pickle.load(open(observations_file, "rb"))
            if get_actions and actions_file.is_file():
                actions = pickle.load(open(actions_file, "rb"))

        return rewards, best_gif, observations, actions

    @staticmethod
    def get_model_name(config):

        # Model
        is_hetero = config["model"]["custom_model_config"]["heterogeneous"]
        is_gippo = config["model"]["custom_model_config"]["share_observations"]

        # Env
        env_config = config["env_config"]
        scenario_name = env_config["scenario_name"]

        model_title = f"{'Het' if is_hetero else ''}{'GPPO' if is_gippo else 'IPPO'}"
        model_name = model_title.lower().replace(" ", "_")

        env_title = scenario_name
        env_name = scenario_name.lower().replace(" ", "_")

        return model_title, model_name, env_title, env_name

    @staticmethod
    def get_inject_name(
        agents_to_inject: Set, inject_mode: InjectMode, noise_delta: float
    ):
        if agents_to_inject is not None and len(agents_to_inject) > 0:
            noise_title = (
                f"Agents injected: {agents_to_inject}, Inject mode: {inject_mode.name}"
                + (
                    " ($\\pm{}$ uniform noise)".format(noise_delta)
                    if inject_mode.is_noise()
                    else ""
                )
            )

            noise_name = (
                f"agents_injected_{agents_to_inject}_inject_mode_{inject_mode.name}"
                + (
                    "_{}_delta_noise".format(noise_delta)
                    if inject_mode.is_noise()
                    else ""
                )
            )

            return noise_title, noise_name
        return "", ""

class PlotUtils:
    @staticmethod
    def plot_function_arrows(
        SAVE_PATH: str,
        seed: int, 
        config: Dict,
        model_config: Dict,
        env_config: Dict,
        device: str,
    ):
        # if torch.has_cuda: # have to do something to make sure the devices are working properly
        #     USING_GPU = True
        with torch.no_grad():
            NUM_EVAL_LOCS = 13
            v0 = torch.from_numpy(np.linspace(-0.4, 0.4, NUM_EVAL_LOCS)).float().to(device)
            v1 = torch.from_numpy(np.linspace(-0.4, 0.4, NUM_EVAL_LOCS)).float().to(device)
    
            # create a grid of x, y values for evaluation
            X, Y = torch.meshgrid(v0, v1, indexing='xy') # use xy indexing to match numpy meshgrid functionality

            grid_inputs = torch.stack([X, Y], dim=-1)

            if env_config["scenario_name"] == "simplified_het_mass": 
                scen = SimplifiedHetMass()
            else:
                scen = env_config["scenario_name"]

            # create a VMAS env just so that we have the observation size and action size
            env = VmasEnv(
                scenario=scen,
                num_envs=config["vmas_envs"], # maybe need to use this to set the batch dimension to match the eval locations
                continuous_actions=True,
                max_steps=config["max_steps"],
                device=config["vmas_device"],
                seed=seed,
                # Scenario kwargs
                **env_config,
            )

            print(env.num_envs)
            print(env.max_steps)
            print(env.action_spec)

            # # specify the form of the actor_net
            actor_net = nn.Sequential(
                LipNormedMultiAgentMLP(
                    n_agent_inputs=env.observation_spec["observation"].shape[-1],
                    # two times the output because we want mu and sigma for the distribution
                    n_agent_outputs=2 * env.action_spec.shape[-1], 
                    n_agents=env.n_agents,
                    centralised=False, # policy for MAPPO is not centralized
                    share_params=model_config["shared_parameters"], # parameters are shared for homogeneous
                    device=config["training_device"],
                    depth=model_config["mlp_depth"], # changed to 3
                    num_cells=model_config["mlp_hidden_params"], # changed to 64 for het_mass
                    activation_class=model_config["MLP_activation"], # original: Tanh
                    lip_constrained=model_config["constrain_lipschitz"],
                    sigma=model_config["lip_sigma"],
                    groupsort_n_groups=model_config["groupsort_n_groups"],
                ),
                NormalParamExtractor(),
            )
            # nn.Module used to map the input to the output parameter space
            policy_module = TensorDictModule(
                actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
            )

            # run = wandb.init()
            # # must have weights and biases logged in in order to use 
            # # entity/project-name/model_path
            # artifact = run.use_artifact('alexshaw-mlmi-thesis/torchrl_het_mass/model-het_mass_MAPPO_31dcecef_23_06_01-16_13_23:v0/het_mass_MAPPO_31dcecef_23_06_01-16_13_23_model.pth', type='model')
            # artifact_dir = artifact.download()

            policy_module.load_state_dict(torch.load(SAVE_PATH)) # use SAVE_PATH for local
            policy_module.to(device) # sends it to the GPU if used
            policy_module.eval()

            # want the last dimension of the grid_input tensor (which should have the two input velocities) and want to also pass in 
            # position
            # so it should be ([p0 v0 p1 v1], [p1 v1 p0 v0]) as the last dimension of the tensor
            # create a new tensor for each of the 13 x 13 locations
            # save this output into a numpy array of size 13 x 13 x 2 (first dim is U second dim is V in the arrow) for the mean/loc
            # save the second part of the output into a 13 x 13 x 1 where the value is the scale/std deviation

            # batch_dim, env.n_agents, env.observation_spec["observation"].shape[-1] # shape of the input tensor
            batched_inputs = torch.flatten(grid_inputs, start_dim=0, end_dim=1) # 169 x 2 
            batched_inputs = torch.unsqueeze(batched_inputs, dim=-1) # 169 x 2 x 1
            zeros = torch.zeros_like(batched_inputs).float() # 169 x 2 x 1
            zeros = zeros.expand(-1, -1, 3).clone() # cloning in case this would cause problems, 169 x 2 x 3

            batched_inputs = torch.cat([zeros, batched_inputs], dim=-1) # 169 x 2 x 4 [0, 0, 0, vela], [0, 0, 0, velb]

            # flipped_clone required for SimplifiedHetMass because we have both agent's observations coming in 
            # flipped_clone = torch.flip(torch.clone(batched_inputs), dims=[range(batched_inputs.dims() - 1)]) # 169 x 2 x 2
            # batched_inputs = torch.cat([batched_inputs, flipped_clone], dim=-1) # 169 x 2 x 4
            assert batched_inputs.shape[-1] == env.observation_spec["observation"].shape[-1]
            assert batched_inputs.shape[-2] == env.n_agents

            locs, scales = policy_module(batched_inputs)  # dist params x batch_dim x n_agents x n_agent-outputs

            locs = locs.reshape(NUM_EVAL_LOCS, NUM_EVAL_LOCS, *locs.shape[1:]).cpu()
            scales = scales.reshape(NUM_EVAL_LOCS, NUM_EVAL_LOCS, *scales.shape[1:]).cpu()
            Y=Y.cpu()
            X=X.cpu()

            fig, ax = plt.subplots()
            ax.quiver(X, Y, locs[...,0,0], locs[...,0,1], color='g')  # plot arrows
            ax.quiver(X, Y, locs[...,1,0], locs[...,1,1], color='b')  # plot arrows

            ax.set_xlabel('Agent 0: v0')
            ax.set_ylabel('Agent 1: v1')
            ax.set_title('Policy')

            # plt.savefig('trial_policy.png')
            # plt.show()

            # wandb.finish()

            return fig
        # wandb.log({"policy": fig})

    @staticmethod
    def plot_lip_vs_heterogeneity():
        return 

    @staticmethod
    def plot_lip_vs_performance():
        return 