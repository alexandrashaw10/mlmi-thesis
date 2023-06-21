import time

import torch
import wandb

from models.lip_multiagent_mlp import LipNormedMultiAgentMLP

from tensordict.nn import TensorDictModule
from modules.tensordict_module import exploration
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from vmas_beta.vmas import VmasEnv # torchrl.envs.libs.vmas is the true package, but using a debug version
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers import generate_exp_name
from torchrl.record.loggers.wandb import WandbLogger
from logging_utils import log_evaluation, log_training
from monotonenorm import GroupSort
import argparse

# train this run using MAPPO and HetMAPPO
from train_torchRL.mappo_ippo import trainMAPPO_IPPO

parser = argparse.ArgumentParser(description = 'Running Simple Give Way Test')

# RL
parser.add_argument('--gamma', type=float, default=0.9) 
parser.add_argument('--seed', nargs='+', type=int, default=0) # for list of seeds
# PPO
parser.add_argument('--lmbda', type=float, default=0.9)
parser.add_argument('--entropy_eps', type=int, default=0)
parser.add_argument('--clip_epsilon', type=float, default=0.2)
# Sampling
parser.add_argument('--frames_per_batch', type=int, default=60_000)
parser.add_argument('--max_steps', type=int, default=300)
parser.add_argument('--n_iters', type=int, default=500)
parser.add_argument('--vmas_device', type=str, default="cuda:0")
# Training
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--minibatch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--max_grad_norm', type=float, default=40.0)
parser.add_argument('--training_device', type=str, default="cuda:0")
# Evaluation
parser.add_argument('--evaluation_interval', type=int, default=20)
parser.add_argument('--evaluation_episodes', type=int, default=600)

# Model
parser.add_argument('--shared_parameters', type=bool, default=False) # True = homogeneous, False = Heterogeneous
parser.add_argument('--centralised_critic', type=bool, default=True) # MAPPO if True, IPPO if False
parser.add_argument('--MLP_activation') # TanH may not be suitable for this model, as we might need GroupSort, doesn't accept the type of nn.Module
parser.add_argument('--constrain_lipschitz', type=bool, default=False) # constrain the lipschitz constraint so that we can test if it runs
parser.add_argument('--lip_sigma', type=float, nargs='*', default=1.0)
parser.add_argument('--groupsort_n_groups', type=int, default=8)
parser.add_argument('--mlp_hidden_params', type=int, default=256)
parser.add_argument('--mlp_depth', type=int, default=3)
parser.add_argument('--constrain_critic', type=bool, default=False)

# run parameters
# will run each constant for the number of seeds that are provided
# parser.add_argument('--num_constants', type=int, default=5) # don't go over 
# parser.add_argument('--num_seeds', type=int, default=3)
parser.add_argument('--log', type=bool, default=True)
args = parser.parse_args()

# for some reason the argparse things are not coming through
print("Lipschitz constraint from argparse", args.constrain_lipschitz)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

args.vmas_device = device
args.training_device = device
print(args)

activation = nn.Tanh
if args.MLP_activation == "GroupSort":
    activation = GroupSort

frames_per_batch = 60_000
max_steps = 100

config = {
    # RL
    "gamma": 0.9, #args.gamma,
    "seed": 0, #args.seed,
    # PPO
    "lmbda": 0.9, # args.lmbda,
    "entropy_eps": 0,#args.entropy_eps,
    "clip_epsilon": 0.2,#args.clip_epsilon,
    # Sampling
    "frames_per_batch": 60_000,#args.frames_per_batch,
    "max_steps": args.max_steps,
    "vmas_envs": frames_per_batch // max_steps, # args.frames_per_batch // args.max_steps,
    "n_iters": args.n_iters, # args.n_iters,
    "total_frames": frames_per_batch * args.n_iters, #args.frames_per_batch * args.n_iters,
    "memory_size": frames_per_batch, # args.frames_per_batch,
    "vmas_device": device, #args.vmas_device,
    # Training
    "num_epochs": 25, #args.num_epochs, # optimization steps per batch of data collected
    "minibatch_size": 4096, #args.minibatch_size, # size of minibatches used in each epoch
    "lr": 5e-5, #args.lr,
    "max_grad_norm": 40.0,# args.max_grad_norm,
    "training_device": device, #args.vmas_device,
    # Evaluation
    "evaluation_interval": 20, # args.evaluation_interval,
    "evaluation_episodes": 200, # args.evaluation_episodes, # number of episodes to use during evaluation
}

model_config = {
    "shared_parameters": args.shared_parameters, # True = homogeneous, False = Heterogeneous
    "centralised_critic": True, # args.centralised_critic, # MAPPO if True, IPPO if False
    "MLP_activation": activation, # TanH may not be suitable for this model, as we might need GroupSort
    "constrain_lipschitz": args.constrain_lipschitz, #args.constrain_lipschitz,  # constrain the lipschitz constraint so that we can test if it runs
    "lip_sigma": 1.0, #args.lip_sigma, # will be overwritten by the constraints
    "mlp_hidden_params": 256, #args.mlp_hidden_params,
    "groupsort_n_groups": 8, #args.groupsort_n_groups,
    "mlp_depth": 3, #args.mlp_depth,
    "constrain_critic": False,
}

env_config = {
    # Scenario
    "scenario_name": "simple_give_way",
    "n_agents": 2,
}

print(config)
print(model_config)
print(env_config)

for seed in args.seed:
    torch.manual_seed(seed)
    # update config with new seed
    config.update({"seed": seed})
    if model_config['constrain_lipschitz']:
        for lip_constant in args.lip_sigma:
            # update config with new lipschitz constraint
            model_config.update({"lip_sigma": lip_constant})
            trainMAPPO_IPPO(seed, config, model_config, env_config, log=True)

    else:
        trainMAPPO_IPPO(seed, config, model_config, env_config, log=True)


