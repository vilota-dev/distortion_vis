import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from matplotlib.patches import Rectangle
import pandas as pd

"""
# Rotate 3D points 90 degrees
# Use scipy to rotate
points_3D_1 = np.dot(points_3D, np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]))
# Generate points for all 6 sides of the cube
points_3D_2 = np.dot(points_3D, np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
points_3D_3 = np.dot(points_3D, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
points_3D_4 = np.dot(points_3D, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))
points_3D_5 = np.dot(points_3D, np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0]]))
points_3D_6 = np.dot(points_3D, np.array([[0, 0, -1], [0, 0, -1], [-1, 0, 0]]))
"""

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
        z = np.linspace(1, 2, self.num_points)  # Generate a range of z values
        X, Y, Z = np.meshgrid(x, y, z)
        # Z = np.full_like(X, z)

        points_3D = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        return points_3D


    def create_3D_cube_surface(self):
        # x = np.linspace(0, self.width / self.model.fx, self.num_points)
        # y = np.linspace(0, self.height / self.model.fy, self.num_points)
        x = np.linspace((-self.width / self.model.fx) / 2, (self.width / self.model.fx) / 2, self.num_points)
        y = np.linspace((-self.height / self.model.fy) / 2, (self.height / self.model.fy) / 2, self.num_points)
        z = np.linspace(-1, 1, self.num_points)  # Generate a range of z values

        # Create empty list to store surface points
        surface_points = []

        # Generate points on the six faces of the cube
        for i in [x[0], x[-1]]:
            for j in y:
                for k in z:
                    surface_points.append([i, j, k])
        for i in x:
            for j in [y[0], y[-1]]:
                for k in z:
                    surface_points.append([i, j, k])
        for i in x:
            for j in y:
                for k in [z[0], z[-1]]:
                    surface_points.append([i, j, k])

        # Convert list to numpy array
        points_3D = np.array(surface_points)

        return points_3D

    def _create_fibonacci_sphere(self):
        samples = st.session_state['num_points']
        phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

        indices = np.arange(samples)
        y = 1 - (indices / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * indices  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points_3D = np.column_stack((x, y, z))
        return points_3D

    @staticmethod
    def _calculate_displacement_vectors(X, Y, X_distorted, Y_distorted):
        """ Calculate the displacement vectors between the distorted and undistorted points."""
        U = X_distorted - X
        V = Y_distorted - Y
        return U, V

    def _calc_angles(self, points_3D):
        """
        Given the set of generated points, calculate the azimuth and polar angles for the points.
        :return:
        """
        x, y, z = points_3D[:, 0], points_3D[:, 1], points_3D[:, 2]

        # careful of the axis convention, here we drop the y axis
        azimuth_rad = np.arctan2(x, z)
        azimuth_deg = np.rad2deg(azimuth_rad)

        polar_rad = np.arctan2(np.sqrt(x**2 + y**2), z)
        polar_deg = np.rad2deg(polar_rad)

        return azimuth_deg, polar_deg

    def _plot_quiver(self, X, Y, X_distorted, Y_distorted, U, V):
        """ Plot the displacement vectors. """
        plt.figure(figsize=(10, 5))

        plt.scatter(self.model.cx, self.model.cy, s=4, marker='o', color='green', label='Center Point')
        plt.plot([self.model.cx, self.model.cx], [0, self.model.cy], 'g--', linewidth=0.5)
        plt.plot([0, self.model.cx], [self.model.cy, self.model.cy], 'g--', linewidth=0.5)

        if not st.session_state['hide_displacement_vectors']:
            plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.002, color='r', alpha=0.5)
        if not st.session_state['hide_pinhole_points']:
            plt.scatter(X, Y, s=1, marker='.', color='blue', label='Original Points')
        if not st.session_state['hide_distorted_points']:
            plt.scatter(X_distorted, Y_distorted, s=1, marker='.', color='red', label='Distorted Points')

        # Draw a rectangle at 0 to self.width and 0 to self.height
        plt.gca().add_patch(Rectangle((0, 0), self.width, self.height, linewidth=1, edgecolor='black', facecolor='none'))

        plt.legend()
        plt.title("Quiver Plot of Distortion Model: {}".format(self.model))
        plt.gca().set_aspect('equal', adjustable='box')
        buffer = st.session_state['buffer']
        plt.axis([0 - self.width * buffer, self.width * (1 + buffer), 0 - self.width * buffer, self.height * (1 + buffer)])
        st.pyplot(plt)

    def _plot_histogram(self, x_original, y_original, U, V):
        current_scale = st.session_state['buffer']
        current_width = self.width * (1 + 2 * current_scale)
        current_height = self.height * (1 + 2 * current_scale)

        visible = (x_original > 0) & (x_original < current_width)
        visible &= (y_original > 0) & (y_original < current_height)

        euclidean_distance = np.sqrt(U ** 2 + V ** 2)
        # st.write("There are {} valid points".format(valid_points.__sizeof__()))
        euclidean_distance = euclidean_distance[visible]
        # st.write(euclidean_distance)

        bins = np.arange(0, torch.max(euclidean_distance), 0.1)

        plt.figure(figsize=(10, 5))
        plt.hist(euclidean_distance, bins=bins, color='red', alpha=0.5)
        plt.title('Histogram of Displacement Vectors')
        plt.xlabel('Euclidean Distance (pixels)')
        plt.ylabel('Frequency')

        st.pyplot(plt)

    def _plot_statistics(self, azimuth_deg, polar_deg, x_original, y_original, U, V):
        current_scale = st.session_state['buffer']
        current_width = self.width * (1 + 2 * current_scale)
        current_height = self.height * (1 + 2 * current_scale)

        visible = (x_original > 0) & (x_original < current_width)
        visible &= (y_original > 0) & (y_original < current_height)

        offset = np.sqrt(U ** 2 + V ** 2)
        offset = offset[visible]
        azimuth_deg = azimuth_deg[visible]
        polar_deg = polar_deg[visible]

        azi_data = pd.DataFrame({'azimuth': azimuth_deg, 'offset': offset})
        polar_data = pd.DataFrame({'polar': polar_deg, 'offset': offset})
        azi_data_grouped = azi_data.groupby('azimuth').mean().reset_index()
        polar_data_grouped = polar_data.groupby('polar').mean().reset_index()


        plt.figure(figsize=(10, 5))
        plt.plot(azi_data_grouped['azimuth'], azi_data_grouped['offset'], '-o', color='red', alpha=0.5)
        plt.plot(polar_data_grouped['polar'], polar_data_grouped['offset'], '-o', color='blue', alpha=0.5)
        plt.title('Azimuth vs Offset')
        plt.xlabel('Azimuth Angle (degrees)')
        plt.ylabel('Offset (pixels)')

        st.pyplot(plt)

    def _plot_3D(self, points_3D):
        fig = go.Figure(data=[go.Scatter3d(
            x=points_3D[:, 0],
            y=points_3D[:, 1],
            z=points_3D[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='blue',
            )
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X', range=[min(points_3D[:, 0]), max(points_3D[:, 0])]),
                yaxis=dict(title='Y', range=[min(points_3D[:, 1]), max(points_3D[:, 1])]),
                zaxis=dict(title='Z', range=[min(points_3D[:, 2]), max(points_3D[:, 2])]),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            title='Fibonacci Sphere'
        )

        st.plotly_chart(fig)

    def visualize_distortion(self):
        if st.session_state['3d_shape'] == "Fibonacci Sphere":
            generated_points = self._create_fibonacci_sphere()
        else:
            generated_points = self.create_3D_cube_surface()

        self._plot_3D(generated_points)

        azimuth, polar = self._calc_angles(generated_points)

        # Use the distortion model's method to convert the 3D points to 2D points
        distorted_points, valid_distorted = self.model.world2cam(torch.tensor(generated_points, dtype=torch.float))

        # Pinhole model
        x_original, y_original, valid_original = self.model.project(generated_points)

        valid_both = valid_distorted.numpy() & valid_original

        azimuth = azimuth[valid_both]
        polar = polar[valid_both]


        # Calculate the displacement vectors between the distorted and undistorted points (from pinhole model)
        U, V = self._calculate_displacement_vectors(x_original[valid_both], y_original[valid_both], distorted_points[valid_both, 0], distorted_points[valid_both, 1])

        self._plot_quiver(x_original[valid_original], y_original[valid_original], distorted_points[valid_distorted, 0], distorted_points[valid_distorted, 1], U, V)

        self._plot_histogram(x_original[valid_both], y_original[valid_both], U, V)
        self._plot_statistics(azimuth, polar, x_original[valid_both], y_original[valid_both], U, V)
