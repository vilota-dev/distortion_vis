import numpy as np

class Pinhole:
    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.fov = 179

    def project(self, points_3D):
        """ Use FOV to reject invalid points """
        x = points_3D[:, 0]
        y = points_3D[:, 1]
        z = points_3D[:, 2]

        # Reject points that are behind the camera
        # Calculate polar angle for each of the points
        polar_angle = np.arctan2(np.sqrt(x**2 + y**2), z)
        valid = np.abs(polar_angle) < np.deg2rad(self.fov / 2)

        x_projected = self.fx * x / z + self.cx
        y_projected = self.fy * y / z + self.cy

        return x_projected, y_projected, valid

    def unproject(self, points_2D):
        """ Instead of arbitrarily generated a grid of 3D points, we can generate a 2D grid with the resolution given,
        and then unproject the 2D grid to 3D points. The actual way its supposed to be:
        1. Generate 2d grid, not 3d grid, then using the pinhole model, unproject to 3d space,
        2. Pass the unprojected points to the distortion model, then project back to 2d space
        3. Then plot the displacement vectors """
        raise NotImplementedError

    def world2cam(self, point3D):
        """ Must be implemented by the child class (distortion models) """
        raise NotImplementedError
