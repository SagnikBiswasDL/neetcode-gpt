import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.model = nn.Sequential(
            nn.Embedding(vocabulary_size, 16), 
            nn.Linear(16, 1), 
            nn.Sigmoid()
        )

        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        batch_size = x.size(0)
        sequence_length = x.size(1)

        x_1 = self.model[0](x)
        x_1 = x_1.mean(dim = 1)
        

        x_2 = self.model[1](x_1)

        x_3 = self.model[2](x_2)

        

        return torch.round(x_3, decimals = 4)
