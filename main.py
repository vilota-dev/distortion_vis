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
        'radtan8': 'Radial Tangential 8 (radtan8)',
        'eucm': "Extended Unified Camera Model (eucm)"
    }
    if "data" not in st.session_state:
        st.session_state['data'] = None

    # ------------------ Sidebar Placeholders ------------------ #
    file_uploader_placeholder = st.sidebar.empty()  # Upload a JSON config
    st.sidebar.divider()
    camera_selector_placeholder = st.sidebar.empty()  # e.g: cam_a, cam_b, cam_c
    resolution_placeholder = st.sidebar.empty()  # e.g: 1280x800
    num_points = st.sidebar.slider("Number of points", 10, 50, 20)  # Not retrieved from JSON so no need to save state

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
        fx, fy, cx, cy = draw_pinhole_config()

        selected_model = get_selected_model()
        if selected_model == "kb4":
            k1, k2, k3, k4 = draw_kb4_config()
            model = KB4(fx, fy, cx, cy, k1, k2, k3, k4)

        elif selected_model == 'ds':
            xi, alpha = draw_ds_config()
            model = DoubleSphere(fx, fy, cx, cy, xi, alpha)

        elif selected_model == "radtan8":
            k1, k2, k3, k4, k5, k6, p1, p2, rpmax = draw_radtan8_config()
            model = RadTan8(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax)

        elif selected_model == 'eucm':
            alpha, beta = draw_eucm_config()
            model = EUCM(fx, fy, cx, cy, alpha, beta)

        else:
            raise ValueError("Unsupported distortion model")

        viz = visualiser.DistortionVisualizer(width, height, num_points=num_points, model=model)

        viz.visualize_distortion()
        st.write(st.session_state['data'])


if __name__ == '__main__':
    main()
