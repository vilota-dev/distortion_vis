import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

class DistortionVisualizer:
    """ Class for the matplotlib.pyplot quiver plot visualizer """
    def __init__(self, width, height, num_points, model):
        self.width = width
        self.height = height
        self.num_points = num_points
        self.model = model

    def _create_grid(self):
        """ Create a grid of points."""
        x = np.linspace(0, self.width, self.num_points)
        y = np.linspace(0, self.height, self.num_points)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def _create_3D_grid(self):
        x = np.linspace((-self.width / self.model.fx) / 2, (self.width / self.model.fx) / 2, self.num_points)
        y = np.linspace((-self.height / self.model.fy) / 2, (self.height / self.model.fy) / 2, self.num_points)
        z = 1  # Ignoring the depth for visualization purposes

        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z)

        points_3D = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        return points_3D

    def _filter_points(self, generated_points, projected_points):
        valid = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < self.width)
        valid = valid & (projected_points[:, 1] >= 0) & (projected_points[:, 1] < self.height)

        projected_points = projected_points[valid]
        generated_points = generated_points[valid]
        return generated_points, projected_points

    @staticmethod
    def _calculate_displacement_vectors(X, Y, X_distorted, Y_distorted):
        """ Calculate the displacement vectors between the distorted and undistorted points."""
        U = X_distorted - X
        V = Y_distorted - Y
        return U, V

    def _plot_quiver(self, X, Y, X_distorted, Y_distorted, U, V):
        """ Plot the displacement vectors. """
        plt.figure(figsize=(10, 5))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.002, color='r', alpha=0.5)
        plt.scatter(X, Y, marker='.', color='blue', label='Original Points')
        plt.scatter(X_distorted, Y_distorted, marker='.', color='red', label='Distorted Points')
        plt.legend()
        plt.title("Quiver Plot of Distortion Model: {}".format(self.model))
        plt.gca().set_aspect('equal', adjustable='box')
        st.pyplot(plt)

    def _plot_histogram(self, U, V):
        """ Plot the histogram of the displacement vectors. """
        # Grab the euclidean distance of the displacement vectors
        euclidean_distance = np.sqrt(U ** 2 + V ** 2)
        euclidean_distance = torch.round(euclidean_distance).to(torch.int)

        max_distance = torch.max(euclidean_distance).item()  # Get the maximum value

        if max_distance == 0:
            bins = torch.tensor([0, 1])  # Set bins with a single bar at zero
        else:
            bins = torch.arange(0, max_distance + 1, max_distance / 10)  # Set the bins with 0.5 pixel step

        plt.figure(figsize=(10, 5))
        plt.hist(euclidean_distance, bins=bins, color='r', alpha=0.5)
        plt.xticks(bins)
        plt.xlabel('Euclidean Distance (pixels)')
        plt.ylabel('Frequency')
        plt.title("Histogram of Displacement Vectors")
        st.pyplot(plt)

    def visualize_distortion(self):
        generated_points = self._create_3D_grid()

        # Use the distortion model's method to convert the 3D points to 2D points
        distorted_points = self.model.world2cam(torch.tensor(generated_points, dtype=torch.float))

        # Filter out the points that are outside the self.width and self.height (defined by resolution of camera)
        generated_points, distorted_points = self._filter_points(generated_points, distorted_points)

        # Pinhole model
        x_original, y_original = self.model.project(generated_points)

        # Calculate the displacement vectors between the distorted and undistorted points (from pinhole model)
        U, V = self._calculate_displacement_vectors(x_original, y_original, distorted_points[:, 0], distorted_points[:, 1])

        self._plot_quiver(x_original, y_original, distorted_points[:, 0], distorted_points[:, 1], U, V)

        self._plot_histogram(U, V)
