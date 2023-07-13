from .pinhole import Pinhole
import numpy as np
import numpy as np


class DoubleSphere:
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.xi, self.alpha = xi, alpha
        self.fov_ds = 220

        # fix the ds model scaling on focal length, to match the equivalent pinhole model
        fx_adj = fx / (1 + xi)
        fy_adj = fy / (1 + xi)

    def __str__(self):
        return "ds"

    def project(self, points):
        x, y, z = points.T

        polar_angle = np.arctan2(np.sqrt(x**2 + y**2), z)
        valid = np.abs(polar_angle) < np.deg2rad(self.fov_ds / 2)

        d1 = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        d2 = np.sqrt(x ** 2 + y ** 2 + (self.xi * d1 + z) ** 2)

        denominator = self.alpha * d2 + (1 - self.alpha) * (self.xi * d1 + z)

        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy

        return np.hstack((u.reshape(-1, 1), v.reshape(-1, 1))), valid

    def cam2world(self, points_2D):
        u, v = points_2D.T

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r_2 = mx ** 2 + my ** 2
        mz = (1 - (self.alpha ** 2) * r_2) / \
             (self.alpha * np.sqrt(1 - (2 * self.alpha - 1) * r_2) +
              (1 - self.alpha))

        coefficient = (mz * self.xi +
                       np.sqrt(mz ** 2 + (1 - self.xi ** 2) * r_2)) / \
                      (mz ** 2 + r_2)

        x = coefficient * mx
        y = coefficient * my
        z = coefficient * mz - self.xi

        return np.hstack((x.reshape(-1, 1),
                             y.reshape(-1, 1),
                             z.reshape(-1, 1)))