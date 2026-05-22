import numpy as np
from typing import List
import torch

class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        N = len(x)
        x = torch.tensor(x)
        gamma = torch.tensor(gamma)


        const = pow(torch.sum(torch.square(x))/N + eps, 0.5)
        
        x = x/const

        x = x * gamma

        return [round(e, 4) for e in x.tolist()]
