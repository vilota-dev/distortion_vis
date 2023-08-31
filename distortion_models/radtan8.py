import numpy as np

class RadTan8:
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6, self.rpmax = k1, k2, p1, p2, k3, k4, k5, k6, rpmax
        self.fov = 150

    def __str__(self):
        return "radtan8"

    def project(self, points):
        x, y, z = points.T

        polar_angle = np.arctan2(np.sqrt(x**2 + y**2), z)
        valid = np.abs(polar_angle) < np.deg2rad(self.fov / 2)

        xp = x / z
        yp = y / z
        r2 = xp ** 2 + yp ** 2
        cdist = (1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))) / (
                1 + r2 * (self.k4 + r2 * (self.k5 + r2 * self.k6))
        )

        deltaX = 2 * self.p1 * xp * yp + self.p2 * (r2 + 2 * xp * xp)
        deltaY = 2 * self.p2 * xp * yp + self.p1 * (r2 + 2 * yp * yp)

        xpp = xp * cdist + deltaX
        ypp = yp * cdist + deltaY

        u = self.fx * xpp + self.cx
        v = self.fy * ypp + self.cy
        return np.hstack((u.reshape(-1, 1), v.reshape(-1, 1))), valid

    def cam2world(self, points_2D):
        raise NotImplementedError
