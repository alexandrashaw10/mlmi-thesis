# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch

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

from models.lip_multiagent_mlp import LipNormedMultiAgentMLP
from models.multiagent_mlp import MultiAgentMLP
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from logging_utils import log_evaluation, log_training

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
    
    return env_config["scenario_name"]

def trainMADDPG_IDDPG(seed, config, model_config, env_config, log=True):
    # Seeding
    torch.manual_seed(seed)

    # Create env and env_test
    env = VmasEnv(
        scenario=return_scenario(env_config["scenario_name"]),
        num_envs=config["vmas_envs"],
        continuous_actions=True,
        max_steps=config["max_steps"],
        device=config["vmas_device"],
        seed=seed,
        # Scenario kwargs
        **env_config,
    )
    # env = TransformedEnv(
    #     env,
    #     RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    # )

    env_test = VmasEnv(
        scenario=return_scenario(env_config["scenario_name"]),
        num_envs=config["evaluation_episodes"],
        continuous_actions=True,
        max_steps=config["max_steps"],
        device=config["vmas_device"],
        seed=seed,
        # Scenario kwargs
        **env_config,
    )

    # Policy
    actor_net = LipNormedMultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1],
        n_agent_outputs=env.action_spec.shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=model_config["shared_parameters"],
        device=config["training_device"],
        depth=model_config["mlp_depth"],
        num_cells=model_config["mlp_hidden_params"],
        activation_class=model_config["MLP_activation"],
        lip_constrained=model_config["constrain_lipschitz"],
        sigma=model_config["lip_sigma"],
        groupsort_n_groups=model_config["groupsort_n_groups"],
        norm_type=model_config["norm_type"],
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=[("observation")], out_keys=[("agents", "param")]
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec["action"],
        in_keys=[("agents", "param")],
        out_keys=[env.action_key],
        distribution_class=TanhDelta,
        distribution_kwargs={
            "min": env.unbatched_action_spec[("action")].space.minimum,
            "max": env.unbatched_action_spec[("action")].space.maximum,
        },
        return_log_prob=False,
    )

    policy_explore = AdditiveGaussianWrapper(
        policy,
        annealing_num_steps=int(config["total_frames"] * (1 / 2)),
        action_key=env.action_key,
    )

    # Critic
    module = LipNormedMultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1]
        + env.action_spec.shape[-1],  # Q critic takes action and value
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=model_config["centralised_critic"],
        share_params=model_config["shared_parameters"],
        device=config["training_device"],
        depth=model_config["mlp_depth"],
        num_cells=model_config["mlp_hidden_params"],
        activation_class=model_config["MLP_activation"],
        lip_constrained=model_config["constrain_critic"],
        sigma=model_config["lip_sigma"],
        groupsort_n_groups=model_config["groupsort_n_groups"],
        norm_type=model_config["norm_type"],
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("observation"), env.action_key],
        out_keys=[("state_action_value")],
    )

    collector = SyncDataCollector(
        env,
        policy_explore,
        device=config["vmas_device"],
        storing_device=config["training_device"],
        frames_per_batch=config["frames_per_batch"],
        total_frames=config["total_frames"],
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(config["memory_size"], device=config["training_device"]),
        sampler=SamplerWithoutReplacement(),
        batch_size=config["minibatch_size"],
    )

    loss_module = DDPGLoss(
        actor_network=policy, value_network=value_module, delay_value=True
    )
    loss_module.set_keys(
        state_action_value=("agents", "state_action_value"),
        reward=env.reward_key,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=config["gamma"])
    target_net_updater = SoftUpdate(loss_module, eps=1 - config["tau"])

    optim = torch.optim.Adam(loss_module.parameters(), config["lr"])

    # Logging
    if log:
        config.update({"model": model_config, "env": env_config})
        model_name = (
            ("Het" if not model_config["shared_parameters"] else "")
            + ("MA" if model_config["centralised_critic"] else "I")
            + "DDPG"
        )
        exp_name = generate_exp_name(env_config["scenario_name"], model_name)
        logger = WandbLogger(
            exp_name=exp_name,
            project=f"torchrl_{env_config['scenario_name']}",
            group=model_name,
            save_code=True,
            config=config,

        )
        wandb.run.log_code(".")

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
        for _ in range(config["num_epochs"]):
            for _ in range(config["frames_per_batch"] // config["minibatch_size"]):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = loss_vals["loss_actor"] + loss_vals["loss_value"]

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config["max_grad_norm"]
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()
                target_net_updater.step()

        policy_explore.step(frames=current_frames)  # Update exploration annealing
        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if log:
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
            config["evaluation_episodes"] > 0
            and i % config["evaluation_interval"] == 0
            and log
        ):
            evaluation_start = time.time()
            with torch.no_grad() and set_exploration_type(ExplorationType.MEAN):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=config["max_steps"],
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )

                evaluation_time = time.time() - evaluation_start

                log_evaluation(logger, rollouts, env_test, evaluation_time, step=i)

        if log == "wandb":
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()


if __name__ == "__main__":
    train()