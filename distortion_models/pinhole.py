class Pinhole:
    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def distort(self, X_normalized, Y_normalized):
        raise NotImplementedError
