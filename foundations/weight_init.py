import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        return torch.round(torch.randn(fan_out, fan_in) * math.sqrt(2/(fan_in + fan_out)), decimals = 4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        return torch.round(torch.randn(fan_out, fan_in) * math.sqrt(2/(fan_in)), decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        a = []
        w = []
        d = [input_dim] + [hidden_dim] * num_layers

        for i in range(num_layers):
            wi = torch.randn(d[i+1], d[i])
            if init_type == 'xavier':
                wi *= math.sqrt(2/(d[i] + d[i+1]))
            elif init_type == 'kaiming':
                wi *= math.sqrt(2/d[i])
            w.append(wi)
        
        x = torch.randn(1, input_dim)

        for m in w:
            x = torch.nn.functional.relu(x @ m.T)
            a.append(round(x.std().item(), 2))

        return a
