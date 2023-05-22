from .pinhole import Pinhole
import numpy as np


class RadTan8(Pinhole):
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax):
        super().__init__(fx, fy, cx, cy)
        self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6, self.rpmax = k1, k2, p1, p2, k3, k4, k5, k6, rpmax

    def __str__(self):
        return "Rational Tangential 8 (radtan8)"

    def distort(self, X_normalized, Y_normalized):
        pass
