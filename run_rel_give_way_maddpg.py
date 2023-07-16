import time

import torch
import wandb

from dotmap import DotMap

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
from train_torchRL.maddpg_iddpg import train

parser = argparse.ArgumentParser(description = 'Running Rel Give Way Test')

# RL
parser.add_argument('--gamma', type=float, default=0.9) 
parser.add_argument('--seed', nargs='+', type=int, default=0) # for list of seeds
# DDPG
parser.add_argument('--tau', type=float, default=0.005) 

# Sampling
parser.add_argument('--frames_per_batch', type=int, default=60_000)
parser.add_argument('--max_steps', type=int, default=300)
parser.add_argument('--n_iters', type=int, default=400)
parser.add_argument('--device', type=str, default="cuda:0")
# Training
parser.add_argument('--num_epochs', type=int, default=45)
parser.add_argument('--minibatch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--max_grad_norm', type=float, default=40.0)
parser.add_argument('--training_device', type=str, default="cuda:0")
# Evaluation
parser.add_argument('--evaluation_interval', type=int, default=20)
parser.add_argument('--evaluation_episodes', type=int, default=200)

# Model
parser.add_argument('--centralised_critic', type=bool, default=False) # MAPPO if True, IPPO if False, use False if using full information
parser.add_argument('--MLP_activation') # TanH may not be suitable for this model, as we might need GroupSort, doesn't accept the type of nn.Module
parser.add_argument('--constrain_lipschitz', type=bool, default=False) # constrain the lipschitz constraint so that we can test if it runs
parser.add_argument('--lip_sigma', type=float, nargs='*', default=float('inf'))
parser.add_argument('--groupsort_n_groups', type=int, default=8)
parser.add_argument('--mlp_hidden_params', type=int, default=256)
parser.add_argument('--mlp_depth', type=int, default=3)
parser.add_argument('--constrain_critic', type=bool, default=False)
parser.add_argument('--norm_type', type=str, default='1')

# Env
parser.add_argument('--agent_radius', type=float, default=0.12)

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

args.device = device
print(args)

activation = nn.Tanh
if args.MLP_activation == "GroupSort":
    activation = GroupSort

config = {
    "seed": args.seed,
    "loss":{
        "gamma": args.gamma,
        "tau": args.tau,
    },
    "collector": {
         "frames_per_batch": args.frames_per_batch,
         "n_iters": args.n_iters,
    },
    "env": {
        "max_steps": args.max_steps,
        "scenario_name": "rel_give_way",
        "device": device,
        "n_agents": 2,
        "scenario": {
            "agent_radius": args.agent_radius,
        }
    },
    "model": {
        "shared_parameters": False,
        "centralised_critic": args.centralised_critic, # args.centralised_critic, # MADDPG if True, IDDPG if False (if full information use False)
        "constrain_lipschitz": args.constrain_lipschitz, #args.constrain_lipschitz,  # constrain the lipschitz constraint so that we can test if it runs
        "lip_sigma": 1.0, #args.lip_sigma, # will be overwritten by the constraints
        "mlp_hidden_params": args.mlp_hidden_params,
        "groupsort_n_groups": args.groupsort_n_groups,
        "mlp_depth": args.mlp_depth,
        "constrain_critic": False,
        "norm_type": args.norm_type,
        "MLP_activation": activation,
    },
    "train": {
        "device": device,
        "minibatch_size": args.minibatch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "max_grad_norm": args.max_grad_norm,
    },
    "eval": {
        "evaluation_interval": args.evaluation_interval, # args.evaluation_interval,
        "evaluation_episodes": args.evaluation_episodes, # args.evaluation_episodes, # number of episodes to use during evaluation
    },
    "logger": {
        "backend": "wandb",
    }
}

print(config)

for seed in args.seed:
    torch.manual_seed(seed)
    # update config with new seed
    config.update({"seed": seed})
    if config["model"]['constrain_lipschitz']:
        for lip_constant in args.lip_sigma:
            # update config with new lipschitz constraint
            config["model"].update({"lip_sigma": lip_constant})
            train(DotMap(config))

    else:
        train(DotMap(config))


