import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import torch


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_samples  = X.shape[0]
        n_features = X.shape[1]

        X = torch.from_numpy(X).reshape(n_samples, n_features)
        y = torch.from_numpy(y).reshape(n_samples, 1)

        w = torch.zeros((n_features, 1), dtype = torch.float64)
        b = 0

        for i in range(epochs):
            y_hat = X @ w + b
            delta = y_hat - y
            w -= lr * 2/n_samples * X.T @ (delta)
            b -= lr * 2/n_samples * torch.sum(delta)
            
        w = w.reshape(n_features)

        return (np.round(w.numpy(), 5), np.round(b.item(),5))
