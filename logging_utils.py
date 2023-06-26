import numpy as np
import torch
import wandb
import copy
import vmas
import hashlib
import pickle
import platform
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, Callable
from typing import Union
from torch import nn, Tensor
from vmas_beta.vmas import VmasEnv
from vmas.simulator.environment import Environment

from tensordict import TensorDictBase
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from models.lip_multiagent_mlp import LipNormedMultiAgentMLP
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
import matplotlib.pyplot as plt
from monotonenorm import GroupSort
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

from torchrl.record.loggers.wandb import WandbLogger

from evaluate.distance_metrics import *
from evaluate.evaluate_model import TorchDiagGaussian
from models.fcnet import MyFullyConnectedNetwork
from models.gppo import GPPO
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)
from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer


def compute_neural_diversity(
    rollouts: TensorDictBase,
    policy_module: TensorDictModule,
):
  dist = np.zeros(len(rollouts) * 2)

  for i, td in enumerate(rollouts):
    # td["observation"] includes max_steps (3) x agents (2) x observation (8)
    orig_input = td["observation"]
    flipped_input = torch.flip(orig_input,[-2]) # create reverse copy of the input (assumes just 2 agents)

    # print(input.shape) # max_steps (3) x agents (2) x observation (8)
    orig_loc, orig_scale = policy_module(orig_input)
    # orig_loc = orig_loc[...,0]
    # orig_scale = orig_scale[...,0]
    # orig_loc = orig_loc.unsqueeze(-1)
    # orig_scale = orig_scale.unsqueeze(-1)

    compare_loc, compare_scale = policy_module(flipped_input)
    # align the 
    compare_loc = torch.flip(compare_loc,[-2])
    compare_scale = torch.flip(compare_scale,[-2])

    # compare_loc = compare_loc[...,0]
    # compare_scale = compare_scale[...,0]
    # compare_loc = compare_loc.unsqueeze(-1)
    # compare_scale = compare_scale.unsqueeze(-1)

    # compute the distance between the means
    # m_diff = torch.sub(orig_loc, compare_loc)
    # #print(m_diff)
    # m_diff_norm = torch.linalg.vector_norm(m_diff, dim=-1)

    # orig_cov = torch.square(orig_scale)
    # compare_cov = torch.square(compare_scale)

    # diff = torch.sub(orig_scale, compare_scale)
    # frob = torch.linalg.matrix_norm(diff.unsqueeze(-1), ord='fro', dim=(-2,-1))
    
    # distance = torch.mean(torch.add(m_diff_norm, frob))
    # dist[i]=distance

    dist_1 = wasserstein_distance(orig_loc[...,0,:], orig_scale[...,0,:], 
                                  compare_loc[...,0,:], compare_scale[...,0,:])
    dist_2 = wasserstein_distance(orig_loc[...,1,:], orig_scale[...,1,:],
                                  compare_loc[...,1,:], compare_scale[...,1,:])
    dist[2 * i] = dist_1
    dist[2 * i + 1] = dist_2
  
  return np.mean(dist), np.std(dist), np.min(dist), np.max(dist)


def log_training(
    logger: WandbLogger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int,
    total_frames: int,
):
    logger.experiment.log(
        {
            f"train/learner/{key}": value.mean().item()
            for key, value in training_td.items()
        },
        commit=False,
    )
    if "info" in sampling_td.keys():
        logger.experiment.log(
            {
                f"train/info/{key}": value.mean().item()
                for key, value in sampling_td["info"].items()
            },
            commit=False,
        )

    mean_reward = sampling_td["next", "reward"].mean().item()

    logger.experiment.log(
        {
            "train/reward/reward_min": sampling_td["next", "reward"]
            .mean(-2)  # Agents
            .min()
            .item(),
            "train/reward/reward_mean": mean_reward,
            "train/reward/reward_max": sampling_td["next", "reward"]
            .mean(-2)  # Agents
            .max()
            .item(),
            "train/reward/reward_std_dev": sampling_td["next", "reward"].std().item(),
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        },
        commit=False,
    )

    return sampling_td["next", "reward"].mean().item()



def log_evaluation(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    evaluation_time: float,
    policy_module: TensorDictModule,
    log_diversity: bool,
):
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r["next"]["done"].sum(
            tuple(range(r.batch_dims, r["next", "done"].ndim)),
            dtype=torch.bool,
        )
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]
    vid = np.transpose(env_test.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2))
    logger.experiment.log(
        {
            "eval/video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4"),
        },
        commit=False,
    ),

    reward_mean = sum([td["next", "reward"].sum(0).mean() for td in rollouts]) / len(rollouts)

    #snd_mean, snd_std, snd_min, snd_max = compute_neural_diversity(rollouts, policy_module)
    # if log_diversity:
    #     logger.experiment.log(
    #         {
    #             "eval/episode_reward_min": min(
    #                 [td["next", "reward"].sum(0).mean() for td in rollouts]
    #             ),
    #             "eval/episode_reward_max": max(
    #                 [td["next", "reward"].sum(0).mean() for td in rollouts]
    #             ),
    #             "eval/episode_reward_mean": reward_mean,
    #             "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
    #             / len(rollouts),
    #             "eval/evaluation_time": evaluation_time,
    #             # "eval/snd_mean": snd_mean,
    #             # "eval/snd_std": snd_std,
    #             # "eval/snd_min": snd_min,
    #             # "eval/snd_max": snd_max,
    #         },
    #         commit=False,
    #     )
    # else:
    logger.experiment.log(
        {
            "eval/episode_reward_min": min(
                [td["next", "reward"].sum(0).mean() for td in rollouts]
            ),
            "eval/episode_reward_max": max(
                [td["next", "reward"].sum(0).mean() for td in rollouts]
            ),
            "eval/episode_reward_mean": reward_mean,
            "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
            / len(rollouts),
            "eval/evaluation_time": evaluation_time
        },
        commit=False,
    )

    return reward_mean