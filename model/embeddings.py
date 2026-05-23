import numpy as np
from numpy.typing import NDArray
import torch


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        tensor = torch.zeros((token_ids.size, embeddings.shape[1]), dtype = torch.float64)

        for i in range(token_ids.size):
            tensor[i] = torch.from_numpy(embeddings[token_ids[i]])
        
        return torch.round(tensor, decimals = 5).numpy()
