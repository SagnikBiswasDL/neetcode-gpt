import numpy as np
from numpy.typing import NDArray
import torch
import math

class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        pos_enc = torch.zeros((seq_len, d_model), dtype = torch.float64)

        for i in range(seq_len):
            for j in range(d_model):
                if j % 2 == 0:
                    pos_enc[i][j] = math.sin( i/( pow(10000, (j/d_model) ) )  )
                else:
                    pos_enc[i][j] = math.cos( i/( pow(10000, ( (j-1) /d_model) ) )  )

        return torch.round(pos_enc, decimals = 5).numpy()
