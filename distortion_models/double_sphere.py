from .pinhole import Pinhole
import numpy as np
import torch


class DoubleSphere(Pinhole):
    def __init__(self, fx, fy, cx, cy, xi, alpha, fov: float = 180):
        self.fx_original = fx
        self.fy_original = fy
        fx_adj = fx * (1 - alpha)
        fy_adj = fy * (1 - alpha)
        super().__init__(fx_adj, fy_adj, cx, cy)
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

        u = self.fx_original * x / denominator + self.cx
        v = self.fy_original * y / denominator + self.cy

        return torch.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))

    def cam2world(self, points_2D):
        u, v = points_2D.T

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r_2 = mx ** 2 + my ** 2
        mz = (1 - (self.alpha ** 2) * r_2) / \
             (self.alpha * torch.sqrt(1 - (2 * self.alpha - 1) * r_2) +
              (1 - self.alpha))

        coefficient = (mz * self.xi +
                       torch.sqrt(mz ** 2 + (1 - self.xi ** 2) * r_2)) / \
                      (mz ** 2 + r_2)

        x = coefficient * mx
        y = coefficient * my
        z = coefficient * mz - self.xi

        return torch.hstack((x.reshape(-1, 1),
                             y.reshape(-1, 1),
                             z.reshape(-1, 1)))