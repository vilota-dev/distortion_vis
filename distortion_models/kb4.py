from .pinhole import Pinhole
import numpy as np
import torch


class KB4(Pinhole):
    def __init__(self, fx, fy, cx, cy, k1, k2, k3, k4):
        super().__init__(fx, fy, cx, cy)
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4

    def __str__(self):
        return "Kannala Brandt 4 (kb4)"

    def world2cam(self, points):
        x, y, z = points.T

        r = np.sqrt(x ** 2 + y ** 2)

        theta = np.arctan(r / z)
        theta_distorted = theta + self.k1 * theta ** 3 + self.k2 * theta ** 5 + \
                          self.k3 * theta ** 7 + self.k4 * theta ** 9

        u = self.fx * theta_distorted * x / r + self.cx
        v = self.fy * theta_distorted * y / r + self.cy

        return torch.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))