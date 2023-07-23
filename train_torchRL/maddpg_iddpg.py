# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from os import path

# import hydra
import torch

from dotmap import DotMap

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    AdditiveGaussianWrapper,
    ProbabilisticActor,
    TanhDelta,
    ValueOperator,
)
# from torchrl.modules.models.multiagent import MultiAgentMLP
from models.lip_multiagent_mlp import LipNormedMultiAgentMLP
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from logging_utils_new import init_logging, log_evaluation, log_training

from scenarios.simplified_het_mass import SimplifiedHetMass
from scenarios.simple_give_way import SimpleGiveWay
from scenarios.rel_give_way import RelGiveWay
from scenarios.balance import MyBalance
from scenarios.joint_passage import JointPassage

def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

def return_scenario(name):
    if name == "simplified_het_mass": 
        return SimplifiedHetMass()
    elif name == "simple_give_way":
        return SimpleGiveWay()
    elif name == "rel_give_way":
        return RelGiveWay()
    elif name == "balance":
        return MyBalance()
    elif name == "joint_passage":
        return JointPassage()
    
    return name

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, exp_name, best_mean_reward=float('-inf')
    ):
        self.best_mean_reward = best_mean_reward
        assert exp_name is not None
        self.exp_name = exp_name
        
    def __call__(
        self, current_reward,
        iteration, policy_module, value_module, optim
    ):
        if current_reward > self.best_mean_reward:
            self.best_mean_reward = current_reward
            print(f"\nBest eval reward: {self.best_mean_reward}")
            print(f"\nSaving best model for epoch: {iteration}\n")

            SAVE_PATH = path.join('saved_models', self.exp_name + '_best_model.pth')
            torch.save({
                'iter': iteration,
                'policy_model_state_dict': policy_module.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'critic_state_dict': value_module.state_dict(),
                }, SAVE_PATH)

# don't use hydra because have own config system
# @hydra.main(version_base="1.1", config_path=".", config_name="maddpg_iddpg")
def train(cfg: DotMap):  # noqa: F821
    # Seeding
    torch.manual_seed(cfg.seed)
    print(f"device: {cfg.env.device}")

    # Sampling
    cfg.env.vmas_envs = cfg.collector.frames_per_batch // cfg.env.max_steps
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    # Create env and env_test
    env = VmasEnv(
        scenario=return_scenario(cfg.env.scenario_name),
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )

    env_test = VmasEnv(
        scenario=return_scenario(cfg.env.scenario_name),
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        # Scenario kwargs
        **cfg.env.scenario,
    )

    # Policy
    actor_net = LipNormedMultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.action_spec.shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=cfg.model.mlp_depth,
        num_cells=cfg.model.mlp_hidden_params,
        activation_class=cfg.model.MLP_activation,
        lip_constrained=cfg.model.constrain_lipschitz,
        sigma=cfg.model.lip_sigma,
        groupsort_n_groups=cfg.model.groupsort_n_groups,
        norm_type=cfg.model.norm_type,
    )
    policy_module = TensorDictModule(
        actor_net, in_keys=[("agents", "observation")], out_keys=[("agents", "param")]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "param")],
        out_keys=[env.action_key],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env.unbatched_action_spec[("agents", "action")].space.minimum,
            "max": env.unbatched_action_spec[("agents", "action")].space.maximum,
        },
        return_log_prob=False,
    )

    policy_explore = AdditiveGaussianWrapper(
        policy,
        annealing_num_steps=int(cfg.collector.total_frames * (1 / 2)),
        action_key=env.action_key,
    )

    # Critic
    module = LipNormedMultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1]
        + env.action_spec.shape[-1],  # Q critic takes action and value
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.model.centralised_critic,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=cfg.model.mlp_depth,
        num_cells=cfg.model.mlp_hidden_params,
        activation_class=cfg.model.MLP_activation,
        lip_constrained=cfg.model.constrain_lipschitz,
        sigma=cfg.model.lip_sigma,
        groupsort_n_groups=cfg.model.groupsort_n_groups,
        norm_type=cfg.model.norm_type,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation"), env.action_key],
        out_keys=[("agents", "state_action_value")],
    )

    collector = SyncDataCollector(
        env,
        policy_explore,
        device=cfg.env.device,
        storing_device=cfg.train.device,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.train.minibatch_size,
    )

    loss_module = DDPGLoss(
        actor_network=policy, value_network=value_module, delay_value=True
    )
    loss_module.set_keys(
        state_action_value=("agents", "state_action_value"),
        reward=env.reward_key,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=cfg.loss.gamma)
    target_net_updater = SoftUpdate(loss_module, eps=1 - cfg.loss.tau)

    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    # Logging
    if cfg.logger.backend:
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "DDPG"
        )
        logger, exp_name = init_logging(cfg, model_name)
        save_best_model = SaveBestModel(exp_name)

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        print(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start

        tensordict_data.set(
            ("next", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get(("next", env.reward_key)).shape),
        )  # We need to expand the done to match the reward shape

        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for _ in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]

                loss_value.backward()

                if not cfg.model.constrain_lipschitz:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg.train.max_grad_norm
                    )
                    training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()
                # optim.zero_grad(set_to_none=True)
                target_net_updater.step()

        policy_explore.step(frames=current_frames)  # Update exploration annealing
        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and i % cfg.eval.evaluation_interval == 0
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad() and set_exploration_type(ExplorationType.MEAN):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                reward_mean = log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

                save_best_model(reward_mean, i, policy_module, value_module, optim)

        if cfg.logger.backend == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()


if __name__ == "__main__":
    train()