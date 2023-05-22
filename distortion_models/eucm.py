from .pinhole import Pinhole


class EUCM(Pinhole):
    def __init__(self, fx, fy, cx, cy, alpha, beta):
        super().__init__(fx, fy, cx, cy)
        self.alpha, self.beta = alpha, beta

    def __str__(self):
        return "Extended Unified Camera Model (eucm)"

    def distort(self, X_normalized, Y_normalized):
        pass
