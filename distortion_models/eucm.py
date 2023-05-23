from .pinhole import Pinhole
import torch


class EUCM(Pinhole):
    def __init__(self, fx, fy, cx, cy, alpha, beta):
        super().__init__(fx, fy, cx, cy)
        self.alpha, self.beta = alpha, beta

    def __str__(self):
        return "Extended Unified Camera Model (eucm)"

    def world2cam(self, points):
        x, y, z = points.T

        d = torch.sqrt(self.beta * (x ** 2 + y ** 2) + z ** 2)
        denominator = self.alpha * d + (1 - self.alpha) * z

        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy

        return torch.hstack((u.reshape(-1, 1), v.reshape(-1, 1)))
