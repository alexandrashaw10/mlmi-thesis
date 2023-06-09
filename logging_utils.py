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


# def get_distance_matrix(
#     self, all_measures: Dict[str, Tensor]
# ) -> Dict[str, Tensor]:
#     all_measures_agent_matrix = {}
#     for key, dists in all_measures.items():
#         per_agent_distances = torch.full(
#             (self.n_agents, self.n_agents),
#             -1.0,
#             dtype=torch.float32,
#         )
#         per_agent_distances.diagonal()[:] = 0

#         pair_index = 0
#         for i in range(self.n_agents):
#             for j in range(self.n_agents):
#                 if j <= i:
#                     continue
#                 pair_distance = dists[:, pair_index].mean()
#                 per_agent_distances[i][j] = pair_distance
#                 per_agent_distances[j][i] = pair_distance
#                 pair_index += 1
#         assert not (per_agent_distances < 0).any()
#         all_measures_agent_matrix[key] = per_agent_distances
#     return all_measures_agent_matrix

# def upload_per_agent_contribution(self, all_measures_agent_matrix, episode):
#     for key, agent_matrix in all_measures_agent_matrix.items():
#         for i in range(self.n_agents):
#             episode.custom_metrics[f"{key}/agent_{i}"] = agent_matrix[
#                 i
#             ].sum().item() / (self.n_agents - 1)
#             for j in range(self.n_agents):
#                 if j <= i:
#                     continue
#                 episode.custom_metrics[f"{key}/agent_{i}{j}"] = agent_matrix[
#                     i, j
#                 ].item()

# def compute_hierarchical_social_entropy(
#     self, all_measures_agent_matrix, episode
# ):
#     for metric_name, agent_matrix in all_measures_agent_matrix.items():
#         distances = []
#         for i in range(self.n_agents):
#             for j in range(self.n_agents):
#                 if j <= i:
#                     continue
#                 distances.append(({i, j}, agent_matrix[i, j].item()))
#         distances.sort(key=lambda e: e[1])
#         intervals = []
#         saved = 0
#         for i in range(len(distances)):
#             intervals.append(distances[i][1] - saved)
#             saved = distances[i][1]

#         hierarchical_social_ent = 0.0
#         hs = [0.0] + [dist[1] for dist in distances[:-1]]

#         for interval, h in zip(intervals, hs):
#             hierarchical_social_ent += interval * self.compute_social_entropy(
#                 h, agent_matrix
#             )
#         assert hierarchical_social_ent >= 0
#         episode.custom_metrics[f"hse/{metric_name}"] = hierarchical_social_ent

# def compute_social_entropy(self, h, agent_matrix):
#     clusters = self.cluster(h, agent_matrix)
#     total_elements = np.array([len(cluster) for cluster in clusters]).sum()
#     ps = [len(cluster) / total_elements for cluster in clusters]
#     social_entropy = -np.array([p * np.log2(p) for p in ps]).sum()
#     return social_entropy

# def cluster(self, h, agent_matrix):
#     # Diametric clustering
#     clusters = [{i} for i in range(self.n_agents)]
#     for i, cluster in enumerate(clusters):
#         for j in range(self.n_agents):
#             if i == j:
#                 continue
#             can_add = True
#             for k in cluster:
#                 if agent_matrix[k, j].item() > h:
#                     can_add = False
#                     break
#             if can_add:
#                 cluster.add(j)

#     # Remove duplicate clusters
#     clusters = [set(item) for item in set(frozenset(item) for item in clusters)]

#     # Remove subsets (should not be used)
#     final_clusters = copy.deepcopy(clusters)
#     for i, c1 in enumerate(clusters):
#         for j, c2 in enumerate(clusters):
#             if i != j and c1.issuperset(c2) and c2 in final_clusters:
#                 final_clusters.remove(c2)
#     assert final_clusters == clusters, "Superset check should be useless"
#     return final_clusters

# # why do we need to do all this stuff with loading in the different models to each? 
# # why can't we just compare the location and scales directly?
# # e.g by calling the policy on the different observations
# def load_agent_x_in_pos_y(self, temp_model, model, x, y):
#     temp_model[y].load_state_dict(model[x].state_dict())
#     return temp_model

# def compute_distance(
#     self,
#     temp_model_i,
#     temp_model_j,
#     obs,
#     agent_index,
#     i,
#     j,
#     act,
#     check_act,
# ):

#     input_dict = {"obs": obs}

#     logits_i = temp_model_i(input_dict)[0].detach()
#     logits_j = temp_model_j(input_dict)[0].detach()

#     split_inputs_i = torch.split(logits_i, self.input_lens, dim=1)
#     split_inputs_j = torch.split(logits_j, self.input_lens, dim=1)

#     distr_i = TorchDiagGaussian(
#         split_inputs_i[agent_index], self.env.agents[agent_index].u_range
#     )
#     distr_j = TorchDiagGaussian(
#         split_inputs_j[agent_index], self.env.agents[agent_index].u_range
#     )

#     mean_i = distr_i.dist.mean
#     mean_j = distr_j.dist.mean

#     # Check
#     i_is_loaded_in_its_pos = agent_index == i
#     j_is_loaded_in_its_pos = agent_index == j
#     assert i != j
#     if check_act:
#         act = act[agent_index]
#         if i_is_loaded_in_its_pos:
#             assert (act == mean_i).all()
#         elif j_is_loaded_in_its_pos:
#             assert (act == mean_j).all()

#     var_i = distr_i.dist.variance
#     var_j = distr_i.dist.variance

#     return_value = {}
#     for name, distance in zip(
#         ["wasserstein", "kl", "kl_sym", "hellinger", "bhattacharyya", "balch"],
#         [
#             wasserstein_distance,
#             kl_divergence,
#             kl_symmetric,
#             hellinger_distance,
#             bhattacharyya_distance,
#             balch,
#         ],
#     ):
#         distances = []
#         for k in range(self.env.get_agent_action_size(self.env.agents[0])):
#             distances.append(
#                 torch.tensor(
#                     distance(
#                         mean_i[..., k].numpy(),
#                         var_i[..., k].unsqueeze(-1).numpy(),
#                         mean_j[..., k].numpy(),
#                         var_j[..., k].unsqueeze(-1).numpy(),
#                     )
#                 )
#             )
#             assert (
#                 distances[k] >= 0
#             ).all(), f"{name}, [{distances[k]} with mean_i {mean_i[..., k]} var_i {var_i[...,k]}, mean_j {mean_j[..., k]} var_j {var_j[...,k]}"
#         return_value[name] = torch.stack(distances)

#     return return_value

# # take in rollouts from the TensorDictBase and a policy_module and compute the Wasserstein distance between 
# # all of the output of the policy_modules for all of the observations
# def compute_neural_diversity(
#     env: VmasEnv,
#     rollouts: TensorDictBase,
#     policy_module: TensorDictModule,
#     model_state_dict: Dict,
# ) -> int:
#     n_agents = env.n_agents
#     input_lens = [
#         2 * env.get_agent_action_size(agent) for agent in env.agents
#     ]

#     temp_model_i = copy.deepcopy(policy_module)
#     temp_model_j = copy.deepcopy(policy_module)
#     temp_model_i.eval()
#     temp_model_j.eval()

#     distance_metrics = [] # going to hold a list of the distance matrices, where each key corresponds to a matrix of measures
#     for td in rollouts:
#     #   print(td)
#       # get the loc and scale for the actions for this td for each step
#       # shape of loc : # of steps x # of agents x # of dimensions of the space  
#       # where it says episode . logmetrics, that is where I would return this
#       # into a list because I just want those metrics
#       # take the metrics into a list and then take the mean across all steps

#       # metric is defined on a per agent basis at first, then summed at the end
#       # since we only have two agents, there is no reason to sum them
#       # can log this one number per run and add it to the df where the plot of
#       # reward is included

#     # create a distribution holder with -1 in each index
#         dists = torch.full(
#             (
#                 rollouts.shape[0], # num observations
#                 int((n_agents * (n_agents - 1)) / 2), # number of unique pairs
#                 n_agents, # number of spots within an observation where I can evaluate the agents
#                 env.get_agent_action_size(env.agents[0]), # number of actions per agent
#             ),
#             -1.0, # fill value
#             dtype=torch.float,
#         )

#         all_measures = {
#             "wasserstein": dists,
#             "kl": dists.clone(),
#             "kl_sym": dists.clone(),
#             "hellinger": dists.clone(),
#             "bhattacharyya": dists.clone(),
#             "balch": dists.clone(),
#         }

#         # self.all_act = self.all_act[1:] + self.all_act[:1]
#         pair_index = 0
#         for i in range(n_agents):
#             for j in range(i + 1, n_agents):
#                 if j <= i:
#                     continue
#                 # Line run for all pairs
#                 for agent_index in range(n_agents):
#                     # what is the point of the deepcopy if we just load in the state dict later?
#                     temp_model_i.load_state_dict(model_state_dict)
#                     temp_model_j.load_state_dict(model_state_dict)
#                     try: # not using GNNs
#                         mdl = policy_module.gnn
#                         tmp_model_i = temp_model_i.gnn
#                         tmp_model_j = temp_model_j.gnn
#                     except AttributeError:
#                         mdl = policy_module
#                         tmp_model_i = temp_model_i
#                         tmp_model_j = temp_model_j
#                     for temp_layer_i, temp_layer_j, layer in zip(
#                         tmp_model_i.children(),
#                         tmp_model_j.children(),
#                         mdl.children(),
#                     ):
#                         assert isinstance(layer, nn.ModuleList)
#                         if len(list(layer.children())) > 1:
#                             assert len(list(layer.children())) == n_agents
#                             load_agent_x_in_pos_y(
#                                 temp_layer_i, layer, x=i, y=agent_index
#                             )
#                             load_agent_x_in_pos_y(
#                                 temp_layer_j, layer, x=j, y=agent_index
#                             )
#                     print(td["observation"])
#                     return 0
#                     # for obs_index, obs in enumerate(self.all_obs):
#                     #     return_dict = self.compute_distance(
#                     #         temp_model_i=self.temp_model_i,
#                     #         temp_model_j=self.temp_model_j,
#                     #         obs=obs,
#                     #         agent_index=agent_index,
#                     #         i=i,
#                     #         j=j,
#                     #         act=None,
#                     #         check_act=False,  # not obs_index == dists.shape[0] - 1,
#                     #     )
#                     #     for key, value in all_measures.items():
#                     #         assert (
#                     #             all_measures[key][
#                     #                 obs_index, pair_index, agent_index
#                     #             ].shape
#                     #             == return_dict[key].shape
#                     #         )
#                     #         all_measures[key][
#                     #             obs_index, pair_index, agent_index
#                     #         ] = return_dict[key]
#                 #pair_index += 1

#         # all_measures_agent_matrix = self.get_distance_matrix(all_measures)
#         # self.upload_per_agent_contribution(all_measures_agent_matrix, episode)
#         # self.compute_hierarchical_social_entropy(all_measures_agent_matrix, episode)
#         # for key, value in all_measures.items():
#         #     assert not (value < 0).any(), f"{key}_{value}"
#         #     episode.custom_metrics[f"mine/{key}"] = value.mean().item()

#         # self.reset() don't need to reset because not using class, and the observations come from the rollout dict


# def log_test(
#     #logger: WandbLogger,
#     rollouts: TensorDictBase,
#     env_test: VmasEnv,
#     #evaluation_time: float,
#     policy_module: TensorDictModule,
#     state_dict: Dict,
# ):
#     rollouts = list(rollouts.unbind(0))
#     # for k, r in enumerate(rollouts):
#     #     next_done = r["next"]["done"].sum(
#     #         tuple(range(r.batch_dims, r["next", "done"].ndim)),
#     #         dtype=torch.bool,
#     #     )
#     #     done_index = next_done.nonzero(as_tuple=True)[0][
#     #         0
#     #     ]  # First done index for this traj
#     #     rollouts[k] = r[: done_index + 1]
#     vid = np.transpose(env_test.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2))
#     # logger.experiment.log(
#     #     {
#     #         "test/video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4"),
#     #     },
#     #     commit=False,
#     # ),

#     print(compute_neural_diversity(env_test, rollouts, policy_module, state_dict))
#     #logger.experiment.log(
#         # {
#         #     "test/episode_reward_min": min(
#         #         [td["next", "reward"].sum(0).mean() for td in rollouts]
#         #     ),
#         #     "test/episode_reward_max": max(
#         #         [td["next", "reward"].sum(0).mean() for td in rollouts]
#         #     ),
#         #     "test/episode_reward_mean": sum(
#         #         [td["next", "reward"].sum(0).mean() for td in rollouts]
#         #     )
#         #     / len(rollouts),
#         #     "test/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
#         #     / len(rollouts),
#         #     "test/evaluation_time": evaluation_time,
#         #     "test/system_neural_diversity": compute_neural_diversity(env_test, rollouts, policy_module),
#         # },
#         # commit=False,
#     # )



# torch.set_grad_enabled(False)

# def rendering_callback(env, td):
#     env.frames.append(env.render(mode="rgb_array", agent_index_focus=None))

# def test_heterogeneity(
#     SAVE_PATH: str,
#     seed: int, 
#     config: Dict,
#     model_config: Dict,
#     env_config: Dict,
#     device: str
#   ):
#   with torch.no_grad():
#       # create a VMAS env just so that we have the observation size and action size
#       test_env = VmasEnv(
#           scenario=env_config["scenario_name"],
#           num_envs=config["vmas_envs"], # maybe need to use this to set the batch dimension to match the eval locations
#           continuous_actions=True,
#           max_steps=config["max_steps"],
#           device=device,
#           seed=seed,
#           # Scenario kwargs
#           **env_config,
#       )

#       print(test_env.num_envs)
#       print(test_env.max_steps)
#       print(test_env.action_spec)

#       def return_activation():
#         if "GroupSort" in model_config["MLP_activation"]:
#           return GroupSort
#         else:
#           return nn.Tanh

#       # specify the form of the actor_net
#       actor_net = nn.Sequential(
#           LipNormedMultiAgentMLP(
#               n_agent_inputs=test_env.observation_spec["observation"].shape[-1],
#               # two times the output because we want mu and sigma for the distribution
#               n_agent_outputs=2 * test_env.action_spec.shape[-1], 
#               n_agents=test_env.n_agents,
#               centralised=False, # policy for MAPPO is not centralized
#               share_params=model_config["shared_parameters"], # parameters are shared for homogeneous
#               device=device,
#               depth=model_config["mlp_depth"], # changed to 3
#               num_cells=model_config["mlp_hidden_params"], # changed to 64 for het_mass
#               activation_class=return_activation(), # original: Tanh
#               lip_constrained=model_config["constrain_lipschitz"],
#               sigma=model_config["lip_sigma"],
#               groupsort_n_groups=model_config["groupsort_n_groups"],
#           ),
#           NormalParamExtractor(),
#       )
#       # nn.Module used to map the input to the output parameter space
#       policy_module = TensorDictModule(
#           actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
#       )

#       state_dict = torch.load(SAVE_PATH, map_location=torch.device('cpu'))
#       policy_module.load_state_dict(state_dict) # use SAVE_PATH for local
#       policy_module.eval()

#       ####
#       ## SND Over the rollouts ######
#       ####  

#       ## I think can just run a few rollouts in a loop and save the statistics?
#       ## why is there so much variation in the mean of all of the policies in each step?
#       ## save these rollouts into the mean reward
#       ## plot the policy and the graph from monday with these rollouts
#       policy = ProbabilisticActor(
#         module=policy_module,
#         spec=test_env.unbatched_input_spec["action"], # specifies metadata with the kind of values you can expect
#         in_keys=["loc", "scale"], # read these keywords from the input TensorDict to build the distribution
#         # TanhNormal means transform the normal distribution using TanH so that the actions are between [-1,1]
#         distribution_class=TanhNormal, # uses location scaling by default to make sure that youre not too far from 0
#         distribution_kwargs={
#             # moves the distribution to [min, max] instead of [-1,1]
#             "min": test_env.unbatched_input_spec["action"].space.minimum,
#             "max": test_env.unbatched_input_spec["action"].space.maximum,
#             # tanh_loc=True (default)
#         },
#         return_log_prob=True, # puts the log probability of the sample in the tensordict
#     )
     
#       test_env.frames = []
#       rollouts = test_env.rollout( # how many rollouts will this do?
#         max_steps=3,
#         policy=policy,
#         callback=rendering_callback, 
#         auto_cast_to_device=True,
#         break_when_any_done=False,
#         # We are running vectorized evaluation we do not want it to stop when just one env is done
#       )

#       print(rollouts.shape)

#       log_test(
#           #logger,
#           rollouts,
#           test_env,
#           #evaluation_time,
#           policy_module,
#           state_dict,
#       )
#       # logger.experiment.log({}, commit=True)

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

    logger.experiment.log(
        {
            "train/reward/reward_min": sampling_td["next", "reward"]
            .mean(-2)  # Agents
            .min()
            .item(),
            "train/reward/reward_mean": sampling_td["next", "reward"].mean().item(),
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




def log_evaluation(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    evaluation_time: float,
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

    logger.experiment.log(
        {
            "eval/episode_reward_min": min(
                [td["next", "reward"].sum(0).mean() for td in rollouts]
            ),
            "eval/episode_reward_max": max(
                [td["next", "reward"].sum(0).mean() for td in rollouts]
            ),
            "eval/episode_reward_mean": sum(
                [td["next", "reward"].sum(0).mean() for td in rollouts]
            )
            / len(rollouts),
            "eval/episode_len_mean": sum([td.batch_size[0] for td in rollouts])
            / len(rollouts),
            "eval/evaluation_time": evaluation_time,
        },
        commit=False,
    )