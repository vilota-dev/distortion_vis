from .pinhole import Pinhole
import numpy as np
import torch


class DoubleSphere(Pinhole):
    def __init__(self, fx, fy, cx, cy, xi, alpha, fov: float = 180):
        super().__init__(fx, fy, cx, cy)
        self.xi, self.alpha = xi, alpha
        # Below parameters are not used for now.
        self.fov = fov
        fov_rad = self.fov / 180 * np.pi
        self.fov_cos = np.cos(fov_rad / 2)

    def __str__(self):
        return "Double Sphere (ds)"

    def world2cam(self, points):
        x, y, z = points.T

        d1 = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        d2 = torch.sqrt(x ** 2 + y ** 2 + (self.xi * d1 + z) ** 2)

        denominator = self.alpha * d2 + (1 - self.alpha) * (self.xi * d1 + z)

        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy

        return torch.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))
