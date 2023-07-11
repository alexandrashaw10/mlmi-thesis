import time

import torch
import wandb

from models.lip_multiagent_mlp import LipNormedMultiAgentMLP
from models.multiagent_mlp import MultiAgentMLP

from tensordict.nn import TensorDictModule
from modules.tensordict_module import exploration
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from vmas_beta.vmas import VmasEnv # torchrl.envs.libs.vmas
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers import generate_exp_name
from torchrl.record.loggers.wandb import WandbLogger
from monotonenorm import GroupSort
from logging_utils import log_evaluation, log_training, compute_neural_diversity
from scenarios.simplified_het_mass import SimplifiedHetMass
from scenarios.simple_give_way import SimpleGiveWay
from scenarios.rel_give_way import RelGiveWay
from scenarios.balance import MyBalance
import os
from os import path

from utils import PlotUtils

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


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

def trainMAPPO_IPPO(seed, config, model_config, env_config, log=True):
    # Create env and env_test
    if env_config["scenario_name"] == "simplified_het_mass": 
        scen = SimplifiedHetMass()
    elif env_config["scenario_name"] == "simple_give_way":
        scen = SimpleGiveWay()
    elif env_config["scenario_name"] == "rel_give_way":
        scen = RelGiveWay()
    elif env_config["scenario_name"] == "balance":
        scen = MyBalance()
    else:
        scen = env_config["scenario_name"]
    env = VmasEnv(
        scenario=scen,
        num_envs=config["vmas_envs"],
        continuous_actions=True,
        max_steps=config["max_steps"],
        device=config["vmas_device"],
        seed=seed,
        # Scenario kwargs
        **env_config,
    )
    if env_config["scenario_name"] == "simplified_het_mass": 
        scen = SimplifiedHetMass()
    elif env_config["scenario_name"] == "simple_give_way":
        scen = SimpleGiveWay()
    elif env_config["scenario_name"] == "rel_give_way":
        scen = RelGiveWay()
    elif env_config["scenario_name"] == "balance":
        scen = MyBalance()
    else:
        scen = env_config["scenario_name"]
    env_test = VmasEnv(
        scenario=scen,
        num_envs=config["evaluation_episodes"], # it must run a new episode to evaluate each time
        continuous_actions=True,
        max_steps=config["max_steps"],
        device=config["vmas_device"],
        seed=seed,
        # Scenario kwargs
        **env_config,
    )
    # why was this here?
    # env_config.update({"n_agents": env.n_agents, "scenario_name": env_config["scenario_name"]})

    # Policy
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
            norm_type=model_config["norm_type"],
        ),
        NormalParamExtractor(),
    )
    # nn.Module used to map the input to the output parameter space
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    # module used by torchRL which acts according to a stochastic policy
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_input_spec["action"], # specifies metadata with the kind of values you can expect
        in_keys=["loc", "scale"], # read these keywords from the input TensorDict to build the distribution
        # TanhNormal means transform the normal distribution using TanH so that the actions are between [-1,1]
        distribution_class=TanhNormal, # uses location scaling by default to make sure that youre not too far from 0
        distribution_kwargs={
            # moves the distribution to [min, max] instead of [-1,1]
            "min": env.unbatched_input_spec["action"].space.minimum,
            "max": env.unbatched_input_spec["action"].space.maximum,
            # tanh_loc=True (default)
        },
        return_log_prob=True, # puts the log probability of the sample in the tensordict
    )

    # Critic
    critic_module = LipNormedMultiAgentMLP(
        n_agent_inputs=env.observation_spec["observation"].shape[-1],
        n_agent_outputs=1, # why is the agent output only 1 ? for the critic? should be centralized critic for MAPPO
        # but why does this mean that the n_agent_outputs = 1 ?
        n_agents=env.n_agents,
        centralised=model_config["centralised_critic"], # True for MAPPO, False for IPPO
        share_params=model_config["shared_parameters"],
        device=config["training_device"],
        depth=model_config["mlp_depth"], # changed to 3
        num_cells=model_config["mlp_hidden_params"], # changed to 64 for het_mass
        activation_class=model_config["MLP_activation"],
        lip_constrained=model_config["constrain_critic"],
        sigma=model_config["lip_sigma"],
        groupsort_n_groups=model_config["groupsort_n_groups"],
        norm_type=model_config["norm_type"],
    )
    value_module = ValueOperator(
        module=critic_module, # didn't need a TensorDictModule here, probably because we don't have out_keys ?
        in_keys=["observation"],
    )

	# what does this line do?
    value_module(policy(env.reset().to(config["training_device"])))

	# sets the device for the policy and collects the data
    collector = SyncDataCollector(
        env,
        policy,
        device=config["vmas_device"],
        # makes sure that the device for the output tensordict has enough storage, 
		# may be different then where the policy and env are executed
        storing_device=config["training_device"],
        frames_per_batch=config["frames_per_batch"], # number of elements in a batch
        total_frames=config["total_frames"], # how much it will collect while running
    )

	# Set up storage for the replay buffer
    replay_buffer = ReplayBuffer(
        # where to store it/how large is the buffer
        storage=LazyTensorStorage(config["memory_size"], device=config["training_device"]),
        # sampler to be used
        sampler=SamplerWithoutReplacement(),
        # defines the batch size to be used
        batch_size=config["minibatch_size"],
        collate_fn=lambda x: x,  # Make it not clone when sampling
    )

    # Loss
    loss_module = ClipPPOLoss( # clipped importance weighted loss
        actor=policy,
        critic=value_module,
        advantage_key="advantage", # dict keyword in input where advantage is written
        clip_epsilon=config["clip_epsilon"], # weight clipping threshold in clipped PPO loss eq
        entropy_coef=config["entropy_eps"], # critic loss multiplier w total loss
        normalize_advantage=False, # don't normalize the advantage function
    )

	# set up the parameters for the value function
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=config["gamma"], lmbda=config["lmbda"]
    )
    optim = torch.optim.Adam(loss_module.parameters(), config["lr"])

    # Logging
    if log:
        config.update({"model": model_config, "env": env_config})
        model_name = (
            ("Het" if not model_config["shared_parameters"] else "")
            + ("MA" if model_config["centralised_critic"] else "I")
            + "PPO"
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


	# where does the actual sampling happen to fill the replay buffer ?
    if not model_config["constrain_lipschitz"]:
        print('Training without lipschitz constraint and with gradient clipping')

    total_time = 0
    total_frames = 0
    sampling_start = time.time()
    for i, tensordict_data in enumerate(collector):
        # get data from the collector
        print(f"\nIteration {i}")

        sampling_time = time.time() - sampling_start
        print(f"Sampling took: {sampling_time}")

		# disables local gradients, so it doesn't call backwards()
        with torch.no_grad(): # no operation should build the computation graph
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_params.detach(),
                target_params=loss_module.target_critic_params,
            )
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
                training_tds.append(loss_vals.detach()) # removes it from the computational graph

                loss_value = ( # copy the loss values
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                if not model_config["constrain_lipschitz"]:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), config["max_grad_norm"]
                    )
                    training_tds[-1]["grad_norm"] = total_norm.mean()

                optim.step()
                optim.zero_grad() # sets the gradients of all optimized tensors to zero

        collector.update_policy_weights_() # handles policy of data collector and trained policy on diff devices

        training_time = time.time() - training_start
        print(f"Training took: {training_time}")

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if log:
            reward_mean = log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
            )

            # save_best_model(reward_mean, i, policy_module, value_module, optim)

        if (
            config["evaluation_episodes"] > 0
            and i % config["evaluation_interval"] == 0
            and log
        ):
            evaluation_start = time.time()
            with torch.no_grad():
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
                print(f"Evaluation took: {evaluation_time}")

                reward_mean = log_evaluation(
                    logger,
                    rollouts,
                    env_test,
                    evaluation_time,
                    policy_module,
                    (not model_config["shared_parameters"]),
                )

                save_best_model(reward_mean, i, policy_module, value_module, optim)

        if log:
            logger.experiment.log({}, commit=True)
        sampling_start = time.time()


    if log:
        SAVE_DIR = './saved_models'
        # if not path.isdir(SAVE_DIR):
        #     os.makedirs(SAVE_DIR)

        # POLICY_SAVE_PATH = path.join(SAVE_DIR, exp_name + '_model.pth')
        # CRITIC_SAVE_PATH = path.join(SAVE_DIR, exp_name + '_critic.pth')
        
        # torch.save(policy_module.state_dict(), POLICY_SAVE_PATH)
        # torch.save(critic_module.state_dict(), CRITIC_SAVE_PATH)

        # artifact = wandb.Artifact(name=f"model-{exp_name}", type='model')
        # artifact.add_file(SAVE_PATH)
        # logger.experiment.log_artifact(artifact)

        # plt = PlotUtils.plot_function_arrows(SAVE_PATH, seed, config, model_config, env_config, config['vmas_device'])
        # wandb.log({"plot": wandb.Image(plt)})
        SAVE_PATH = path.join(SAVE_DIR, exp_name + '_model.pth')
        torch.save({
            'policy_model_state_dict': policy_module.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'critic_state_dict': value_module.state_dict(),
            }, SAVE_PATH)

    wandb.finish()


if __name__ == "__main__":
    for seed in [0]:
        # Device
        training_device = "cpu" if not torch.has_cuda else "cuda:0"
        vmas_device = training_device

        # Seeding
        seed = seed # should this be a diff name ?
        torch.manual_seed(seed)

        # Log
        write_logs = False

        # Sampling
        frames_per_batch = 60_000  # Frames sampled each sampling iteration
        max_steps = 100 # defines the horizon, None=infinite. max number of steps in each vectorized env before it returns Done
        vmas_envs = frames_per_batch // max_steps
        n_iters = 500  # Number of sampling/training iterations, "stop"
        total_frames = frames_per_batch * n_iters
        memory_size = frames_per_batch

        env_config = {
            # Scenario
            "scenario_name": "het_mass",
            "n_agents": 2,
        }

        config = {
            # RL
            "gamma": 0.9,
            "seed": seed,
            # PPO
            "lmbda": 0.9,
            "entropy_eps": 0,
            "clip_epsilon": 0.2,
            # Sampling,
            "frames_per_batch": frames_per_batch,
            "max_steps": max_steps,
            "vmas_envs": vmas_envs,
            "n_iters": n_iters,
            "total_frames": total_frames,
            "memory_size": memory_size,
            "vmas_device": vmas_device,
            # Training
            "num_epochs": 15,  # optimization steps per batch of data collected
            "minibatch_size": 4096,  # size of minibatches used in each epoch
            "lr": 5e-5, # what algorithm is using this learning rate ?
            "max_grad_norm": 40.0,
            "training_device": training_device,
            # Evaluation
            "evaluation_interval": 20, # what is this interval
            "evaluation_episodes": 200, # number of episodes to use during evaluation
        }

        model_config = {
            "shared_parameters": True, # True = homogeneous, False = Heterogeneous
            "centralised_critic": True,  # MAPPO if True, IPPO if False
            "MLP_activation": nn.Tanh, # can try with GroupSort activation instead
            "GroupSort_n_groups": 8,
            "constrain_lipschitz": True,
            "lip_sigma": 1.0,
            "mlp_hidden_params":256,
            "mlp_depth":3,
        }

        trainMAPPO_IPPO(seed, config, model_config, env_config, write_logs)
