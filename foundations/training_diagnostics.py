import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        torch.no_grad()
        ans = []

        for layer in model.children():
            x = layer(x)
            if isinstance(layer, nn.Linear):
                curr_el = {}
                curr_el['mean'] = round(x.mean().item(), 4)
                curr_el['std'] =  round(x.std().item(), 4)
                curr_el['dead_fraction'] = round((x <= 0).all(dim = 0).float().mean().item(), 4)
                ans.append(curr_el)

        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        ans = []
        
        y_pred = model(x)
        loss_function = nn.MSELoss()

        loss = loss_function(y_pred, y)
        loss.backward()

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                gradients = layer.weight.grad
                curr = {}
                curr['mean'] = round(gradients.mean().item(), 4)
                curr['std'] =  round(gradients.std().item(), 4)
                curr['norm'] = round(gradients.norm().item(), 4)
                ans.append(curr)

        return ans

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for d in activation_stats:
            if d['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for d in gradient_stats:
            if d['norm'] > 1000:
                return 'exploding_gradients'
        
        if (len(gradient_stats) >= 0):
            f = gradient_stats[len(gradient_stats) - 1]
            if f['norm'] < 1e-5:
                return 'vanishing_gradients'
        
        for d in activation_stats:
            if d['std'] < 0.1:
                return 'vanishing_gradients'
            
        for d in activation_stats:
            if d['std'] > 10.0:
                return 'exploding_gradients'


        return 'healthy'
