from .pinhole import Pinhole
import numpy as np


class DoubleSphere(Pinhole):
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        super().__init__(fx, fy, cx, cy)
        self.xi, self.alpha = xi, alpha

    def __str__(self):
        return "Double Sphere (ds)"

    def distort(self, X_normalized, Y_normalized):
        r_2 = X_normalized ** 2 + Y_normalized ** 2
        mz = (1 - (self.alpha ** 2) * r_2) / (self.alpha * np.sqrt(1 - (2 * self.alpha - 1) * r_2) + (1 - self.alpha))
        coefficient = (mz * self.xi + np.sqrt(mz ** 2 + (1 - self.xi ** 2) * r_2)) / (mz ** 2 + r_2)
        X_distorted = coefficient * X_normalized
        Y_distorted = coefficient * Y_normalized
        # 2d is the
        return X_distorted, Y_distorted
