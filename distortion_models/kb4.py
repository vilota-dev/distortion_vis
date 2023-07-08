from .pinhole import Pinhole
import numpy as np


class KB4(Pinhole):
    def __init__(self, fx, fy, cx, cy, k1, k2, k3, k4):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.fov_kb4 = 180

    def __str__(self):
        return "Kannala Brandt 4 (kb4)"

    def project(self, points):
        x, y, z = points.T

        polar_angle = np.arctan2(np.sqrt(x**2 + y**2), z)
        valid = np.abs(polar_angle) < np.deg2rad(self.fov_kb4 / 2)

        r = np.sqrt(x ** 2 + y ** 2)

        theta = np.arctan2(r, z)
        theta_distorted = theta + self.k1 * theta ** 3 + self.k2 * theta ** 5 + \
                          self.k3 * theta ** 7 + self.k4 * theta ** 9

        u = self.fx * theta_distorted * x / r + self.cx
        v = self.fy * theta_distorted * y / r + self.cy

        return np.hstack((u.reshape(-1, 1), v.reshape(-1, 1))), valid

    def cam2world(self, points_2D):
        u, v = points_2D.T

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy

        ru = np.sqrt(mx ** 2 + my ** 2)

        theta = ru