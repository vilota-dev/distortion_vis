import numpy as np


class Pinhole:
    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def distort(self, X_normalized, Y_normalized):
        raise NotImplementedError


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


class DoubleSphere(Pinhole):
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        super().__init__(fx, fy, cx, cy)
        self.xi, self.alpha = xi, alpha

    def __str__(self):
        return "Double Sphere (ds)"

    def distort(self, X_normalized, Y_normalized):
        r_2 = X_normalized ** 2 + Y_normalized ** 2
        mz = (1 - (self.alpha ** 2) * r_2) / (self.alpha * np.sqrt(1 - (2 * self.alpha - 1) * r_2) + (1 - self.alpha))
        coefficient = (mz * self.xi + np.sqrt(mz ** 2 + (1 - self.xi ** 2) * r_2)) / (mz ** 2 + r_2)
        X_distorted = coefficient * X_normalized
        Y_distorted = coefficient * Y_normalized
        # 2d is the
        return X_distorted, Y_distorted


class RadTan8(Pinhole):
    def __init__(self, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax):
        super().__init__(fx, fy, cx, cy)
        self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6, self.rpmax = k1, k2, p1, p2, k3, k4, k5, k6, rpmax

    def __str__(self):
        return "Rational Tangential 8 (radtan8)"

    def distort(self, X_normalized, Y_normalized):
        pass


class EUCM(Pinhole):
    def __init__(self, fx, fy, cx, cy, alpha, beta):
        super().__init__(fx, fy, cx, cy)
        self.alpha, self.beta = alpha, beta

    def __str__(self):
        return "Extended Unified Camera Model (eucm)"

    def distort(self, X_normalized, Y_normalized):
        pass
