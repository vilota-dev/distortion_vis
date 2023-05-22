import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


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

    def _normalize(self, X, Y):
        """ Normalize points to the camera coordinate system. From 2D -> 3D """
        X_normalized = (X - self.model.cx) / self.model.fx
        Y_normalized = (Y - self.model.cy) / self.model.fy
        return X_normalized, Y_normalized

    def _denormalize(self, X_normalized, Y_normalized):
        """ Denormalize points to the image coordinate system. From 3D -> 2D
        """
        X = X_normalized * self.model.fx + self.model.cx
        Y = Y_normalized * self.model.fy + self.model.cy
        return X, Y

    def _calculate_displacement_vectors(self, X, Y, X_distorted, Y_distorted):
        """ Calculate the displacement vectors between the distorted and undistorted points."""
        U = X_distorted - X
        V = Y_distorted - Y
        return U, V

    def _plot_quiver(self, X, Y, X_distorted, Y_distorted, U, V):
        """ Plot the displacement vectors. """
        plt.figure(figsize=(10, 6))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.002, color='r', alpha=0.5)
        plt.scatter(X, Y, marker='.', color='b', label='Original Points')
        plt.scatter(X_distorted, Y_distorted, marker='.', color='r', label='Distorted Points')
        plt.legend()
        plt.title("Quiver Plot of Distortion Model: {}".format(self.model))
        plt.gca().set_aspect('equal', adjustable='box')
        st.pyplot(plt)

    def generate_distortion_quiver(self):
        X, Y = self._create_grid()
        X_normalized, Y_normalized = self._normalize(X, Y)
        X_distorted, Y_distorted = self.model.distort(X_normalized, Y_normalized)
        X_distorted, Y_distorted = self._denormalize(X_distorted, Y_distorted)
        U, V = self._calculate_displacement_vectors(X, Y, X_distorted, Y_distorted)
        self._plot_quiver(X, Y, X_distorted, Y_distorted, U, V)
