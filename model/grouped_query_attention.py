import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        rat = self.num_heads / self.num_kv_heads

        heads = None

        for i in range(self.num_heads):
            Q_curr = Q[:,:, i*self.head_dim : (i+1) * self.head_dim]
            K_curr = K[:,:, int(i//rat) * self.head_dim : (int(i//rat) + 1) * self.head_dim]
            V_curr = V[:,:, int(i//rat) * self.head_dim : (int(i//rat) + 1) * self.head_dim]

            scores = Q_curr @ K_curr.transpose(1, 2) / (self.head_dim ** 0.5)
            mask = torch.tril(torch.ones(T, T))
            scores = scores.masked_fill(mask == 0, float('-inf'))

            curr_head = nn.functional.softmax(scores , dim = -1) @ V_curr
            if heads is None:
                heads = curr_head
            else:
                heads = torch.cat((heads, curr_head), dim = -1)
        
        heads = self.output_proj(heads)
        return torch.round(heads, decimals = 4)
