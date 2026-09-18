import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def forward_sigmoid(self, x, w, b):
        z = np.dot(x, w) + b
        return 1 / (1 + np.exp(-z))

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        y_hat = self.forward_sigmoid(x, w, b)

        # dL/dy_hat i.e. dL/dy_pred
        dL_dy = y_hat - y_true

        # dy_hat/dz
        dy_dz = y_hat * (1 - y_hat)

        # dL/dw = dL_dy * dy_dz * dy/dw; dy/dw = x
        dL_dw = dL_dy * dy_dz * x

        # dL/db = dL/dz * dz/db; dz/db = 1
        dL_db = dL_dy * dy_dz

        return np.round(dL_dw, 5), np.round(dL_db, 5)

