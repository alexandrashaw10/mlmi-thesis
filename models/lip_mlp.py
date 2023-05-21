
import time

import torch
from torch import nn

#from torchrl.modules import MLP
# from monotonenorm import direct_norm
from torch.nn.utils.parametrize import register_parametrization


class LipNormedMLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 depth: int, num_cells: int, 
                 activation_class: nn.Module, device: torch.device | str | int | None = None,
                 lip_constrained: bool = False, sigma: float | None = None, always_norm: bool = False):
        super().__init__()

        # credit to monotonenorm package
        def lipschitz_norm(layer: nn.Linear):
            '''
            layer - the nn.Linear layer to normalize
            always_norm - always normalize the weight matrix to the max_norm if True. Default is False so it can be < max_norm
            max_norm - the lipschitz constraint on the network
            '''
            #max_norm = sigma ** (1 / depth) # the norm of each layer

            class LipNormalize(nn.Module):
                def forward(self, W):
                    # norms = W.abs().sum(axis=0) # compute 1-norm of W
                    
                    # if not always_norm:
                    #     # 1 / max_norm is the lambda ^ (-1/depth) term in the paper
                    #     norms = torch.max(torch.ones_like(norms), norms / max_norm)
                    # else:
                    #     # otherwise just take the norm divided by the max_norm
                    #     norms = norms / max_norm

                    # norm_W = W / torch.max(norms, torch.ones_like(norms)*1e-10) # second term protects from divide by zero errors

                    # return norm_W
                    return W
            
            #register_parametrization(layer, "weight", LipNormalize())

            return layer

            # return direct_norm(
            #     module,  # the layer to constrain
            #     "one",  # |W|_1 constraint type
            #     max_norm=sigma ** (1 / depth),  # norm of the layer (LIP ** (1/nlayers))
            # )

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
    
