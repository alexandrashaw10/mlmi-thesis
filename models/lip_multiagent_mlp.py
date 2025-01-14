from typing import Tuple

import time

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

#from torchrl.modules import MLP
from models.lip_mlp import LipNormedMLP

# MLP model which uses the newly create MLP
class LipNormedMultiAgentMLP(nn.Module):
    def __init__(
        self,
        n_agent_inputs,
        n_agent_outputs,
        n_agents,
        centralised,
        share_params,
        device,
        depth,
        num_cells,
        activation_class, # try with GroupSort and with tanh
        lip_constrained=False,
        sigma=1.0,
        groupsort_n_groups="8",
        norm_type='1',
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.centralised = centralised

        self.agent_networks = nn.ModuleList(
            [
                # switched from using nn.Modules MLP to own MLP implementation
                LipNormedMLP(
                    in_features=n_agent_inputs
                    if not centralised
                    else n_agent_inputs * n_agents,
                    out_features=n_agent_outputs,
                    depth=depth,
                    num_cells=num_cells,
                    activation_class=activation_class,
                    device=device,
                    lip_constrained=lip_constrained,
                    sigma=sigma,
                    always_norm = False, # need to add this to the config
                    groupsort_n_groups=groupsort_n_groups,
                    norm_type=norm_type
                )
                for _ in range(self.n_agents if not self.share_params else 1)
            ]
        )

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)
        inputs = inputs[0]

        if inputs.shape[-2:] != (self.n_agents, self.n_agent_inputs):
            raise ValueError(
                f"Multi-agent network expected input with last 2 dimensions {[self.n_agents, self.n_agent_inputs]},"
                f" but got {inputs.shape}"
            )

        # If the model is centralized, agents have full observability. concatenate them
        if self.centralised:
            inputs = inputs.view(
                *inputs.shape[:-2], self.n_agents * self.n_agent_inputs
            )

        # If parameters are not shared, each agent has its own network
        if not self.share_params:
            # all agents get all inputs
            if self.centralised:
                # have to stack all of the networks together if theyre centralized
                output = torch.stack(
                    [net(inputs) for i, net in enumerate(self.agent_networks)],
                    dim=-2,
                )
            else:
                # if decentralized critic, need to pick out just the right part of the inputs to pass through
                output = torch.stack(
                    [
                        # include all of the dimensions up to the second to last, pick i, 
						# then include the last dimension
                        net(inputs[..., i, :])
                        for i, net in enumerate(self.agent_networks)
                    ],
                    dim=-2,
                )
        # If parameters are shared, agents use the same network
        else:
            # there is only one network so just use the first index
            output = self.agent_networks[0](inputs)

            if self.centralised: 
                # why do we take off the last layer of the output?
                output = (
                    # output.view reshapes the output so that we take the shape exluding the last dimension and replace it with one per agent
                    # what is n_agent_outputs ? the num of output features per agent would make sense
					output.view(*output.shape[:-1], self.n_agent_outputs)
                    .unsqueeze(-2) # add another dimension of 1 second to last dimension
                    # adds a dimension of one before the output
					# increase the new dimension to include one per agent
                    .expand(*output.shape[:-1], self.n_agents, self.n_agent_outputs)
                )
            #### if not centralized what happens ? ####
		# verifies the output shape at the end
        if output.shape[-2:] != (self.n_agents, self.n_agent_outputs):
            raise ValueError(
                f"Multi-agent network expected output with last 2 dimensions {[self.n_agents, self.n_agent_outputs]},"
                f" but got {output.shape}"
            )

        return output