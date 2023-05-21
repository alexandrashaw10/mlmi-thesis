
import time

import torch
from torch import nn

#from torchrl.modules import MLP
from monotonenorm import direct_norm

class LipNormedMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 depth: int, num_cells: int, 
                 activation_class: nn.Module, device: torch.device | str | int | None = None,
                 lip_constrained: bool = False, sigma: float | None = None):
        super().__init__()

        # credit to monotonenorm package
        def lipschitz_norm(module):
            return direct_norm(
                module,  # the layer to constrain
                "one",  # |W|_1 constraint type
                max_norm=sigma ** (1 / depth),  # norm of the layer (LIP ** (1/nlayers))
            )

        self.layers = nn.ModuleList()

        # lipschitz constraints are added to the linear layer forward pre-hook 
        # so that it is normalized before the backward pass and the normalized gradients are passed through
        if lip_constrained:
            assert sigma is not None, "if lipschitz constraint active, sigma must be a float"

            self.layers.append(lipschitz_norm(nn.Linear(in_features, num_cells)))
            self.layers.append(activation_class())
            for d in range(1,depth-1):
                self.layers.append(lipschitz_norm(nn.Linear(num_cells, num_cells)))
                if activation_class is not None:
                    self.layers.append(activation_class())
            self.layers.append(lipschitz_norm(nn.Linear(num_cells, out_features)))

        else:
            self.layers.append(nn.Linear(in_features, num_cells))
            self.layers.append(activation_class())
            for d in range(1,depth-1):
                self.layers.append(nn.Linear(num_cells, num_cells))
                if activation_class is not None:
                    self.layers.append(activation_class())
            self.layers.append(nn.Linear(num_cells, out_features))

        self.device = device
        self.to(device)

    def forward(self, input_data):
        # does this need to include something with the normalization to work?
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    
