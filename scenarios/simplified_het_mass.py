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
class SimplifiedHetMass(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.green_mass = kwargs.get("green_mass", 4)
        self.blue_mass = kwargs.get("blue_mass", 2)
        self.mass_noise = kwargs.get("mass_noise", 0.5) # why would I want mass noise - keeping for now but want to try with and without, right now setting to 0
        self.obs_noise = kwargs.get("obs_noise", 0.0)
        # was originally 1
        self.plot_grid = True

        # Make world
        world = World(batch_dim, device)
        # Add agents
        self.green_agent = Agent(
            name="agent 0",
            collide=False,
            color=Color.GREEN,
            render_action=True,
            mass=self.green_mass,
            f_range=1,
        )
        world.add_agent(self.green_agent)
        self.blue_agent = Agent(
            name="agent 1",
            collide=False,
            color=Color.BLUE,
            render_action=True,
            mass=self.blue_mass,
            f_range=1,
        )
        world.add_agent(self.blue_agent)

        self.max_speed = torch.zeros(batch_dim, device=device)
        self.energy_expenditure = self.max_speed.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        # Temp - right now not always the agents have different masses when they restart ? theoretically each could have mass 3.
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
                ).uniform_(-1, 1), # why do I want to uniform give them different positions ?
                batch_index=env_index,
            )

    def process_action(self, agent: Agent):
        agent.action.u[:, Y] = 0

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0] # why is the reward only for the first agent?

        if is_first:
            self.max_speed = torch.stack(
                [
                    torch.linalg.vector_norm(a.state.vel, dim=1)
                    for a in self.world.agents
                ],
                dim=1,
            ).max(dim=1)[0]

            self.energy_expenditure = (
                -torch.stack(
                    [
                        torch.linalg.vector_norm(a.action.u, dim=-1)
                        / math.sqrt(self.world.dim_p * (a.f_range**2))
                        for a in self.world.agents # convert the action into energy for each agent (?)
                    ],
                    dim=1,
                ).sum(-1) # sum the energies
                * 0.17 # multiply by a scaling factor which is not explained ?
            )

            # why is this here:
            # print(self.max_speed)
            # print(self.energy_expenditure)
            # self.energy_rew_1 = (self.world.agents[0].action.u[:, X] - 0).abs()
            # self.energy_rew_1 += (self.world.agents[0].action.u[:, Y] - 0).abs()
            #
            # self.energy_rew_2 = (self.world.agents[1].action.u[:, X] - 0).abs()
            # self.energy_rew_2 += (self.world.agents[1].action.u[:, Y] - 0).abs()

        return self.max_speed + self.energy_expenditure

    def observation(self, agent: Agent):
        # pass in the position and velocity of all of the other agents as well
        # so that its pos_curr_agent, vel_curr_agent, pos0, vel0, ...
        positions = [agent.state.pos]
        velocities = [agent.state.vel]
        positions.extend([a.state.pos for a in self.world.agents if a is not agent])
        velocities.extend([a.state.vel for a in self.world.agents if a is not agent])

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
            "max_speed": self.max_speed,
            "energy_expenditure": self.energy_expenditure,
        }


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)