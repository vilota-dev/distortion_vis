from .pinhole import Pinhole
import torch
import numpy as np

class EUCM(Pinhole):
    def __init__(self, fx, fy, cx, cy, alpha, beta):
        super().__init__(fx, fy, cx, cy)
        self.alpha, self.beta = alpha, beta
        self.fov_eucm = 180

    def __str__(self):
        return "Extended Unified Camera Model (eucm)"

    def world2cam(self, points):
        x, y, z = points.T

        polar_angle = np.arctan2(np.sqrt(x**2 + y**2), z)
        valid = np.abs(polar_angle) < np.deg2rad(self.fov / 2)

        d = torch.sqrt(self.beta * (x ** 2 + y ** 2) + z ** 2)
        denominator = self.alpha * d + (1 - self.alpha) * z

        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy

        return torch.hstack((u.reshape(-1, 1), v.reshape(-1, 1))), valid

    def cam2world(self, points_2D):
        u, v = points_2D.T

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy

        r_2 = mx ** 2 + my ** 2

        mz = (1 - self.beta * (self.alpha ** 2) * (r_2 ** 2)) / \
                (self.alpha * torch.sqrt(1 - (2 * self.alpha - 1) * self.beta * r_2) +
                    (1 - self.alpha))

        coefficient = 1 / torch.sqrt(mx ** 2 + my ** 2 + mz ** 2)

        x = coefficient * mx
        y = coefficient * my
        z = coefficient * mz

        return torch.hstack((x.reshape(-1, 1),
                             y.reshape(-1, 1),
                             z.reshape(-1, 1)))