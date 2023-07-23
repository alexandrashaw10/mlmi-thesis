#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Dict

import numpy as np
from itertools import chain
import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y

# every class called "Scenario" to make it easy to load
class LeftRight(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.get("green_mass", 2)
        self.blue_mass = kwargs.get("blue_mass", 2)
        self.spawn_distance = kwargs.get("spawn_dist", 0.0)
        self.mass_noise = kwargs.get("mass_noise", 0)
        self.obs_noise = kwargs.get("obs_noise", 0.0)

        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1.0)
        self.final_reward = kwargs.get("final_reward", 0.01)

        # was originally 1
        self.plot_grid = False

        LIGHT_BLUE = (0.45, 0.45, 0.95)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        self.green_agent = Agent(
            name="agent 0",
            collide=False,
            color=Color.GREEN,
            render_action=True,
            mass=self.green_mass,
        )
        world.add_agent(self.green_agent)
        self.blue_agent = Agent(
            name="agent 1",
            collide=False,
            color=Color.BLUE,
            render_action=True,
            mass=self.blue_mass,
        )
        world.add_agent(self.blue_agent)

        green_goal = Landmark(
            name="green_goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )

        blue_goal = Landmark(
            name="blue_goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=LIGHT_BLUE,
        )
        world.add_landmark(blue_goal)
        world.add_landmark(green_goal)
        self.blue_goal = [blue_goal, green_goal]

        self.blue_agent.goal = blue_goal
        self.green_agent.goal = green_goal
        self.blue_goal = blue_goal
        self.green_goal = green_goal

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        self.goal_reached = self.pos_rew.clone().to(torch.bool)

        return world

    def reset_world_at(self, env_index: int = None):
        self.blue_agent.mass = self.blue_mass + np.random.uniform(
            -self.mass_noise, self.mass_noise
        )
        self.green_agent.mass = self.green_mass + np.random.uniform(
            -self.mass_noise, self.mass_noise
        )

        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.green_goal.set_pos(
            torch.cat([torch.full(
                (1, 1)
                if env_index is not None
                else (self.world.batch_dim, 1),
                fill_value=1,
                device=self.world.device,
                dtype=torch.float32,
            ), torch.zeros((1, self.world.dim_p - 1)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p - 1),
                device=self.world.device,
                dtype=torch.float32,)
            ], dim=-1),
            batch_index=env_index,
        )

        self.blue_goal.set_pos(
            torch.cat([torch.full(
                (1, 1)
                if env_index is not None
                else (self.world.batch_dim, 1),
                fill_value=-1,
                device=self.world.device,
                dtype=torch.float32,
            ), torch.zeros((1, self.world.dim_p - 1)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p - 1),
                device=self.world.device,
                dtype=torch.float32,)
            ], dim=-1),
            batch_index=env_index,
        )

        if env_index is not None:
          self.rew[env_index] = 0
          self.goal_reached[env_index] = False
          self.on_goal[env_index] = torch.zeros(2,
            device=self.world.device, 
            dtype=torch.float32
          ).to(torch.bool)
        else:
          self.rew = torch.zeros(
              self.world.batch_dim, device=self.world.device, dtype=torch.float32
          )
          self.goal_reached = torch.zeros(
              self.world.batch_dim, device=self.world.device
          ).to(torch.bool)

    def process_action(self, agent: Agent):
        agent.action.u[:, Y] = 0

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0] # why is the reward only for the first agent?

        if is_first:
            blue_agent = self.blue_agent
            green_agent = self.green_agent
            self.blue_distance = torch.linalg.vector_norm(
                blue_agent.state.pos - blue_agent.goal.state.pos,
                dim=1,
            )
            self.green_distance = torch.linalg.vector_norm(
                green_agent.state.pos - green_agent.goal.state.pos,
                dim=1,
            )
            self.blue_on_goal = self.blue_distance < blue_agent.goal.shape.radius
            self.green_on_goal = self.green_distance < green_agent.goal.shape.radius
            self.goal_reached = self.green_on_goal * self.blue_on_goal

            green_shaping = self.green_distance * self.pos_shaping_factor
            self.green_rew = green_shaping
            # self.green_rew = green_agent.shaping - green_shaping
            # green_agent.shaping = green_shaping

            blue_shaping = self.blue_distance * self.pos_shaping_factor
            self.blue_rew = blue_shaping
            # self.blue_rew = self.blue_agent.shaping - blue_shaping
            # blue_agent.shaping = blue_shaping

            self.pos_rew += self.blue_rew + self.green_rew

            self.final_rew[self.goal_reached] = self.final_reward

        return self.pos_rew + self.final_rew

    def observation(self, agent: Agent):
        # pass in the position and velocity of all of the other agents as well
        # so that its pos_curr_agent, vel_curr_agent, pos0, vel0, ...
        positions = [agent.state.pos]
        velocities = [agent.state.vel]
        positions.extend([a.state.pos - agent.state.pos for a in self.world.agents if a is not agent])
        velocities.extend([a.state.vel - agent.state.vel for a in self.world.agents if a is not agent])

        agents = list(chain.from_iterable(zip(positions, velocities)))

        if self.obs_noise > 0:
            for i, obs in enumerate(agents):
                noise = torch.zeros(
                    *obs.shape,
                    device=self.world.device,
                ).uniform_(
                    -self.obs_noise,
                    self.obs_noise,
                )
                agents[i] = obs + noise

        return torch.cat(agents, dim=-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "success": self.goal_reached,
        }

if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=True,
        control_two_agents=True,
    )