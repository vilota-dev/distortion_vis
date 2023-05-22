from .pinhole import Pinhole
import numpy as np


class KB4(Pinhole):
    def __init__(self, fx, fy, cx, cy, k1, k2, k3, k4):
        super().__init__(fx, fy, cx, cy)
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4

    def __str__(self):
        return "Kannala Brandt 4 (kb4)"

    def distort(self, X_normalized, Y_normalized):
        r = np.sqrt(X_normalized ** 2 + Y_normalized ** 2)
        theta = np.arctan(r)
        theta_distorted = theta * (1 + self.k1 * theta ** 2 + self.k2 * theta ** 4 +
                                   self.k3 * theta ** 6 + self.k4 * theta ** 8)
        X_distorted = X_normalized * theta_distorted / r
        Y_distorted = Y_normalized * theta_distorted / r
        return X_distorted, Y_distorted
