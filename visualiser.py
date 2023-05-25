import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import plotly.graph_objects as go

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

    def _create_fibonacci_sphere(self):
        samples = st.session_state['sphere_points']
        phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

        indices = np.arange(samples)
        y = 1 - (indices / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * indices  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points_3D = np.column_stack((x, y, z))
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
        if not st.session_state['hide_displacement_vectors']:
            plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.002, color='r', alpha=0.5)
        if not st.session_state['hide_pinhole_points']:
            plt.scatter(X, Y, marker='.', color='blue', label='Original Points')
        if not st.session_state['hide_distorted_points']:
            plt.scatter(X_distorted, Y_distorted, marker='.', color='red', label='Distorted Points')
        plt.legend()
        plt.title("Quiver Plot of Distortion Model: {}".format(self.model))
        plt.gca().set_aspect('equal', adjustable='box')
        st.pyplot(plt)

    def _plot_quiver_plotly(self, X, Y, X_distorted, Y_distorted, U, V):
        """ Plot the displacement vectors. """
        fig = go.Figure()

        # Add the original points as scatter plot
        fig.add_trace(go.Scatter(
            x=X,
            y=Y,
            mode='markers',
            marker=dict(
                size=5,
                color='blue'
            ),
            name='Original Points'
        ))

        # Add the distorted points as scatter plot
        fig.add_trace(go.Scatter(
            x=X_distorted,
            y=Y_distorted,
            mode='markers',
            marker=dict(
                size=5,
                color='red'
            ),
            name='Distorted Points'
        ))

        # Set layout properties
        fig.update_layout(
            title="Quiver Plot of Distortion Model: {}".format(self.model),
            xaxis=dict(title='X', range=[min(X) - 1, max(X) + 1]),
            yaxis=dict(title='Y', range=[min(Y) - 1, max(Y) + 1]),
            width=960,  # Adjust the width of the plot as desired
            height=600,  # Adjust the height of the plot as desired
            showlegend=True
        )

        st.plotly_chart(fig, theme='streamlit')

    def _plot_histogram(self, U, V):
        euclidean_distance = np.sqrt(U ** 2 + V ** 2)

        max_distance = torch.max(euclidean_distance).item()  # Get the maximum value

        if max_distance == 0:
            bins = torch.tensor([0, 1])  # Set bins with a single bar at zero
        else:
            bins = torch.arange(0, max_distance + 1, max_distance / 10)  # Set the bins with 0.5 pixel step

        # Create the histogram trace
        hist_trace = go.Histogram(x=euclidean_distance.numpy(), nbinsx=len(bins), marker=dict(color='red', opacity=0.5))

        # Create the layout
        layout = go.Layout(
            title='Histogram of Displacement Vectors',
            xaxis=dict(title='Euclidean Distance (pixels)', tickvals=bins),
            yaxis=dict(title='Frequency'),
            bargap=0.2,
            bargroupgap=0.1
        )

        # Create the figure
        fig = go.Figure(data=[hist_trace], layout=layout)

        # Display the figure using Streamlit
        st.plotly_chart(fig)

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
                xaxis=dict(title='X', range=[2 * min(points_3D[:, 0]), 2 * max(points_3D[:, 0])]),
                yaxis=dict(title='Y', range=[2 * min(points_3D[:, 1]), 2 * max(points_3D[:, 1])]),
                zaxis=dict(title='Z', range=[2 * min(points_3D[:, 2]), 2 * max(points_3D[:, 2])]),
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
            generated_points = self._create_3D_grid()

        self._plot_3D(generated_points)

        # Use the distortion model's method to convert the 3D points to 2D points
        distorted_points = self.model.world2cam(torch.tensor(generated_points, dtype=torch.float))

        # Filter out the points that are outside the self.width and self.height (defined by resolution of camera)
        # generated_points, distorted_points = self._filter_points(generated_points, distorted_points)

        # Pinhole model
        x_original, y_original, valid_original = self.model.project(generated_points)

        # Calculate the displacement vectors between the distorted and undistorted points (from pinhole model)
        U, V = self._calculate_displacement_vectors(x_original[valid_original], y_original[valid_original], distorted_points[valid_original, 0], distorted_points[valid_original, 1])

        self._plot_quiver_plotly(x_original[valid_original], y_original[valid_original], distorted_points[:, 0], distorted_points[:, 1], U, V)

        self._plot_histogram(U, V)
