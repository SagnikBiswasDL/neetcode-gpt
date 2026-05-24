import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim, bias = False),
            nn.Linear(embedding_dim, attention_dim, bias = False), 
            nn.Linear(embedding_dim, attention_dim, bias = False)
        )

        self.scale = pow(attention_dim, 0.5)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        K = self.model[0](embedded)
        Q = self.model[1](embedded)
        V = self.model[2](embedded)

        print(Q.shape)
        print(K.mT.shape)

        A = (Q @ K.mT) / self.scale
        
        mask = torch.tril(torch.ones(A.size(1), A.size(1)))

        A = torch.where(mask == 0, float('-inf'), A)

        A = F.softmax(A, dim = 2)

        return torch.round(A @ V, decimals = 4)
