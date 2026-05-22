import numpy as np
from typing import Tuple, List
import torch


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = torch.tensor(x)
        gamma = torch.tensor(gamma)
        beta = torch.tensor(beta)
        running_mean = torch.tensor(running_mean)
        running_var = torch.tensor(running_var)

        N = x.size(0)

        if N > 1:
            factor = 1/(N/(N-1))
        else:
            factor = 1
        
        print(N)

        if training: 
            if (N > 1):
                running_mean = (1-momentum) * running_mean + torch.mean(x, axis = 0) * momentum
                running_var =  (1-momentum) * running_var + torch.var(x, axis = 0) * factor * momentum
                mu  = torch.mean(x, axis = 0)
            else:
                mu  = torch.mean(x, axis = 0)
                running_mean = (1-momentum) * running_mean + torch.mean(x, axis = 0) * momentum
                running_var =  (1-momentum) * running_var + torch.zeros_like(running_var) * momentum


        mu  = torch.mean(x, axis = 0)
        if N > 1:
            var = torch.var (x, axis = 0) * factor
            var = torch.sqrt(var + eps)
        else:
            var = torch.zeros_like(running_var)
            var = torch.sqrt(var + eps)


    
        if training:
            x_hat = (x-mu)/var
        else:
            x_hat = (x-running_mean)/torch.sqrt(running_var + eps)
        
        y = gamma * x_hat + beta

        y = y.tolist()
        for i in range(len(y)):
            for j in range(len(y[i])):
                y[i][j] = round(y[i][j], 4)

        b = [round(x, 4) for x in running_mean.tolist()]
        c = [round(x, 4) for x in running_var.tolist()]

        return (y, b, c)
