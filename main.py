import streamlit as st
from distortion_models import *
import visualiser
import json

from widgets import *
from helper_functions import *


def main():
    # ---------------------- Session State ---------------------- #
    full_name = {
        'kb4': 'Kannala Brandt 4 (kb4)',
        'ds': 'Double Sphere (ds)',
        'pinhole-radtan8': 'Radial Tangential 8 (radtan8)',
        'eucm': "Extended Unified Camera Model (eucm)"
    }
    if "data" not in st.session_state:
        st.session_state['data'] = None

    # ------------------ Sidebar Placeholders ------------------ #
    file_uploader_placeholder = st.sidebar.empty()  # Upload a JSON config
    st.sidebar.divider()
    camera_selector_placeholder = st.sidebar.empty()  # e.g: cam_a, cam_b, cam_c
    resolution_placeholder = st.sidebar.empty()  # e.g: 1280x800
    st.sidebar.slider("Focal length", 0.0, 1.0, 0.2, key='buffer')

    # Selectbox for 3d sphere or 3d cube
    st.sidebar.selectbox("3D shape", ['Fibonacci Sphere', '3D Cube'], key='3d_shape')
    if st.session_state['3d_shape'] == 'Fibonacci Sphere':
        num_points = st.sidebar.slider("Point Density", 1000, 10000, 1000, key='num_points')
    else:
        num_points = st.sidebar.slider("Point Density", 20, 100, 10, key='num_points')  # Not retrieved from JSON so no need to save state

    # ------------------ Main page Placeholders ------------------ #
    title_placeholder = st.empty()

    # ------------------ FILE UPLOADING SECTION ------------------ #
    # Upload distortion model (.json format)
    uploaded_file = file_uploader_placeholder.file_uploader('Upload a distortion model', type=['json'])
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        st.session_state['data'] = data

    # ------------------ MAIN SECTION ------------------ #
    title_placeholder.title("Distortion Model Visualizer")

    if st.session_state['data'] is not None:
        selected_camera = camera_selector_placeholder.selectbox('Camera selected', get_camera_names(), key='selected_camera')
        width, height = draw_resolution_config(resolution_placeholder)

        st.subheader(full_name[get_selected_model()])
        pinhole_fx, pinhole_fy, pinhole_cx, pinhole_cy = draw_ideal_pinhole_config()
        pinhole_model = Pinhole(pinhole_fx, pinhole_fy, pinhole_cx, pinhole_cy)
        fx, fy, cx, cy = draw_pinhole_config()

        selected_model = get_selected_model()
        if selected_model == "kb4":
            k1, k2, k3, k4 = draw_kb4_config()
            model = KB4(fx, fy, cx, cy, k1, k2, k3, k4)

        elif selected_model == 'ds':
            xi, alpha = draw_ds_config()
            model = DoubleSphere(fx, fy, cx, cy, xi, alpha)

        elif selected_model == "pinhole-radtan8":
            k1, k2, k3, k4, k5, k6, p1, p2, rpmax = draw_radtan8_config()
            model = RadTan8(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax)

        elif selected_model == 'eucm':
            alpha, beta = draw_eucm_config()
            model = EUCM(fx, fy, cx, cy, alpha, beta)

        else:
            raise ValueError("Unsupported distortion model")

        c1, c2, c3 = st.columns(3)
        c1.checkbox("Hide Ideal pinhole points (blue)", key='hide_pinhole_points')
        c2.checkbox("Hide Distorted points (red)", key='hide_distorted_points')
        c3.checkbox("Hide displacement vectors (red arrows)", key='hide_displacement_vectors')

        viz = visualiser.DistortionVisualizer(width, height, num_points=num_points, model=model, pinhole_model=pinhole_model)

        viz.visualize_distortion()


if __name__ == '__main__':
    main()
