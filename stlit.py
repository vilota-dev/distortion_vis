import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_distortion_quiver(distortion_model, params):
    # Given distortion model parameters
    if distortion_model == "rantan8":
        # Your implementation for rantan8 distortion model parameters here
        pass
    elif distortion_model == "eucm":
        # Your implementation for eucm distortion model parameters here
        pass
    elif distortion_model == "kb4":
        fx, fy, cx, cy, k1, k2, k3, k4 = params
    elif distortion_model == "ds":
        # Your implementation for ds distortion model parameters here
        pass
    else:
        raise ValueError("Unsupported distortion model.")

    width, height = 1920, 1200

    # Create a grid of points
    num_points = 20
    x = np.linspace(0, width, num_points)
    y = np.linspace(0, height, num_points)
    X, Y = np.meshgrid(x, y)

    # Normalize the points to the camera coordinate system
    x_normalized = (X - cx) / fx
    y_normalized = (Y - cy) / fy

    # Calculate the distorted points
    if distortion_model == "kb4":
        r = np.sqrt(x_normalized**2 + y_normalized**2)
        theta = np.arctan(r)
        theta_distorted = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
        x_distorted = x_normalized * theta_distorted / r
        y_distorted = y_normalized * theta_distorted / r
    else:
        # Your implementation for other distortion models here
        pass

    # Convert back to pixel coordinates
    X_distorted = x_distorted * fx + cx
    Y_distorted = y_distorted * fy + cy

    # Calculate displacement vectors
    U = X_distorted - X
    V = Y_distorted - Y

    # Plot the quiver plot
    plt.figure(figsize=(10, 6))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.002, color='r', alpha=0.5)
    plt.scatter(X, Y, marker='.', color='b', label='Original Points')
    plt.scatter(X_distorted, Y_distorted, marker='.', color='r', label='Distorted Points')
    plt.legend()
    plt.title("Quiver Plot of Distortion ({})".format(distortion_model))
    plt.gca().set_aspect('equal', adjustable='box')
    st.pyplot(plt)

def main():
    # Set page title
    st.title("Distortion Model Visualiser")

    # Sidebar
    st.sidebar.title('Input Parameters')

    # Distortion model selection
    distortion_model = st.sidebar.selectbox('Distortion Model', ['rantan8', 'eucm', 'kb4', 'ds'])

    # Parameter input
    if distortion_model == "kb4":
        fx = st.sidebar.number_input('fx', value=622.0)
        fy = st.sidebar.number_input('fy', value=622.0)
        cx = st.sidebar.number_input('cx', value=965.0)
        cy = st.sidebar.number_input('cy', value=631.0)
        k1 = st.sidebar.number_input('k1', value=-0.256)
        k2 = st.sidebar.number_input('k2', value=-0.0015)
        k3 = st.sidebar.number_input('k3', value=0.0007)
        k4 = st.sidebar.number_input('k4', value=-0.0002)
        params = [fx, fy, cx, cy, k1, k2, k3, k4]
    else:
        st.sidebar.write("Parameter input not implemented for the selected distortion model.")

    # Generate the distortion quiver plot
    generate_distortion_quiver(distortion_model, params)

if __name__ == '__main__':
    main()

