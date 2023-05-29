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

parser = argparse.ArgumentParser(description = 'Running Het Mass Test')

# RL
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=seed)
# PPO
parser.add_argument('--lmbda', type=float, default=0.9)
parser.add_argument('--entropy_eps', type=int, default=0)
parser.add_argument('--clip_epsilon', type=float, default=0.2)
# Sampling
parser.add_argument('--frames_per_batch', type=int, default=60_000)
parser.add_argument('--max_steps', type=int, default=100)
parser.add_argument('--n_iters', type=int, default=100)
parser.add_argument('--vmas_device', type=str, default="cpu")
# Training
parser.add_argument('--num_epochs', type=int, default=25)
parser.add_argument('--minibatch_size', type=int, default=4096)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--max_grad_norm', type=float, default=40.0)
parser.add_argument('--training_device', type=str, default="cpu")
# Evaluation
parser.add_argument('--evaluation_interval', type=int, default=20)
parser.add_argument('--evaluation_episodes', type=int, default=200)

# Model
parser.add_argument('--shared_parameters', type=bool, default=True) # True = homogeneous, False = Heterogeneous
parser.add_argument('--centralised_critic', type=bool, default=True) # MAPPO if True, IPPO if False
parser.add_argument('--MLP_activation', type=nn.Module, default=nn.Tanh) # TanH may not be suitable for this model, as we might need GroupSort
parser.add_argument('--constrain_lipschitz', type=bool, default=True) # constrain the lipschitz constraint so that we can test if it runs
parser.add_argument('--lip_sigma', type=float, default=1.0)
parser.add_argument('--mlp_hidden_params', type=int, default=256)
parser.add_argument('--mlp_depth', type=int, default=3)

args = parser.parse_args()

config = {
    # RL
    "gamma": args.gamma,
    "seed": args.seed,
    # PPO
    "lmbda": args.lmbda,
    "entropy_eps": args.entropy_eps,
    "clip_epsilon": args.clip_epsilon,
    # Sampling
    "frames_per_batch": args.frames_per_batch,
    "max_steps": args.max_steps,
    "vmas_envs": args.frames_per_batch // args.max_steps,
    "n_iters": args.n_iters,
    "total_frames": args.frames_per_batch * args.n_iters,
    "memory_size": args.frames_per_batch,
    "vmas_device": args.vmas_device,
    # Training
    "num_epochs": args.num_epochs, # optimization steps per batch of data collected
    "minibatch_size": args.minibatch_size, # size of minibatches used in each epoch
    "lr": args.lr,
    "max_grad_norm": args.max_grad_norm,
    "training_device": args.training_device,
    # Evaluation
    "evaluation_interval": args.evaluation_interval,
    "evaluation_episodes": args.evaluation_episodes, # number of episodes to use during evaluation
}

model_config = {
    "shared_parameters": args.shared_parameters, # True = homogeneous, False = Heterogeneous
    "centralised_critic": args.centralised_critic, # MAPPO if True, IPPO if False
    "MLP_activation": args.MLP_activation, # TanH may not be suitable for this model, as we might need GroupSort
    "constrain_lipschitz": args.constrain_lipschitz,  # constrain the lipschitz constraint so that we can test if it runs
    "lip_sigma": args.lip_sigma,
    "mlp_hidden_params": args.mlp_hidden_params,
    "mlp_depth": args.mlp_depth,
}

env_config = {
    # Scenario
    "scenario_name": "het_mass",
    "n_agents": 2,
}