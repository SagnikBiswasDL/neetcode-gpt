import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        n = len(weights)
        curr_h = x.reshape(1, -1)

        for i in range(n):
            print(curr_h.shape, weights[i].shape, biases[i].shape)
            curr_h = np.maximum(0, curr_h @ weights[i] + biases[i].reshape(1, -1))
        
        curr_h = curr_h.flatten()

        return np.round(curr_h, 5)
