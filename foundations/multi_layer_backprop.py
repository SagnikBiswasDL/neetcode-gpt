import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        o1 = x @ W1.T + b1
        # print("Before", o1.shape)
        o1 = o1.reshape(1, -1)
        o1 = np.maximum(0, o1)
        # print("After", o1.shape)

        o2 = o1 @ W2.T + b2
        o2 = o2.reshape(1, -1)

        # print(o2.shape, o1.shape)
        
        n = 1 if y_true.ndim == 1 else y_true.size
        L = np.mean(np.square(y_true - o2))

        L = np.round(L, 4)

        db2 = 2 * (o2 - y_true) / n
        # print(db2.shape, o1.shape)
        dW2 = db2.T @ o1
        print(db2 @ W2)
        z1 = (x @ W1.T + b1).reshape(1, -1)
        z1 = (z1 > 0).astype(int)
        db1 = np.multiply((db2 @ W2), z1)
        # print(db1.shape, x.shape)
        x = x.reshape(1, -1)
        dW1 = db1.T @ x

        ans = {}
        ans['loss'] = L

        db2 = np.round(db2, 4)
        db2 = db2.flatten()
        ans['db2'] = db2.tolist()

        dW2 = np.round(dW2, 4)
        ans['dW2'] = dW2.tolist()

        db1 = np.round(db1, 4)
        db1 = db1.flatten()
        ans['db1'] = db1.tolist()

        dW1 = np.round(dW1, 4)
        ans['dW1'] = dW1.tolist()

        return ans
