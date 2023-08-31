import streamlit as st
from distortion_models import *
import visualiser
import distortion_fit
import json

from widgets import *
from helper_functions import *


def main():
    # ---------------------- Session State ---------------------- #
    full_name = {
        'pinhole': 'Pinhole (pinhole)',
        'kb4': 'Kannala Brandt 4 (kb4)',
        'ds': 'Double Sphere (ds)',
        'pinhole-radtan8': 'Radial Tangential 8 (radtan8)',
        'eucm': "Extended Unified Camera Model (eucm)"
    }
    if "data" not in st.session_state:
        st.session_state['data'] = None

    # ------------------ Sidebar Placeholders ------------------ #
    file_form = st.sidebar.empty()
    # file_uploader_placeholder_a = st.sidebar.empty()  # Upload a JSON config
    # file_uploader_placeholder_b = st.sidebar.empty()
    st.sidebar.divider()
    camera_selector_placeholder_a = st.sidebar.empty()  # e.g: cam_a, cam_b, cam_c
    camera_selector_placeholder_b = st.sidebar.empty()
    resolution_placeholder = st.sidebar.empty()  # e.g: 1280x800
    st.sidebar.slider("Buffer Size", 0.0, 1.0, 0.2, key='buffer')

    # Selectbox for 3d sphere or 3d cube
    st.sidebar.selectbox("3D shape", ['Fibonacci Sphere', '3D Cube'], key='3d_shape')
    if st.session_state['3d_shape'] == 'Fibonacci Sphere':
        num_points = st.sidebar.slider("Point Density", 1000, 10000, 1000, key='num_points')
    else:
        num_points = st.sidebar.slider("Point Density", 20, 100, 10, key='num_points')  # Not retrieved from JSON so no need to save state

    # FOV Controls
    # st.sidebar.divider()
    # st.sidebar.subheader("FOV Controls")
    


    # ------------------ Main page Placeholders ------------------ #
    title_placeholder = st.empty()

    # ------------------ FILE UPLOADING SECTION ------------------ #
    # Upload distortion model (.json format)
    with file_form.form(key="calib_files_form"):
        uploaded_file_a = st.file_uploader('Upload a distortion model A', type=['json'])
        uploaded_file_b = st.file_uploader('Upload a distortion model B', type=['json'])

        calib_files_submitted = st.form_submit_button("Compare Distortion Models")

    # update state when uploading is done
    if calib_files_submitted:
        st.session_state['data_a'] = json.load(uploaded_file_a)
        st.session_state['data_b'] = json.load(uploaded_file_b)

    if 'data_a' in st.session_state:
        st.session_state['data_a']['selected_camera'] = camera_selector_placeholder_a.selectbox('Camera selected A', get_camera_names(st.session_state['data_a']))
    if 'data_b' in st.session_state:
        st.session_state['data_b']['selected_camera'] = camera_selector_placeholder_b.selectbox('Camera selected B', get_camera_names(st.session_state['data_b']))

    # ------------------ MAIN SECTION ------------------ #
    title_placeholder.title("Distortion Model Visualizer")

    if 'data_a' in st.session_state and 'data_b' in st.session_state:
        
        # only take the first camera
        width, height = draw_resolution_config(st.session_state['data_a'], resolution_placeholder)

        if get_selected_model(st.session_state['data_a']) not in full_name:
            st.error(f"{get_selected_model(st.session_state['data_a'])} is not supported currently.")

        if get_selected_model(st.session_state['data_b']) not in full_name:
            st.error(f"{get_selected_model(st.session_state['data_b'])} is not supported currently.")

        st.subheader(f"{full_name[get_selected_model(st.session_state['data_a'])]} and {full_name[get_selected_model(st.session_state['data_b'])]}")
    #     # pinhole_fx, pinhole_fy, pinhole_cx, pinhole_cy = draw_ideal_pinhole_config()
    #     # pinhole_model = Pinhole(pinhole_fx, pinhole_fy, pinhole_cx, pinhole_cy)

        models = [None, None]
        data_names = ['data_a', 'data_b']

        with st.form(key="models_form"):

            print("runed!")
            # cols = st.rows(2)
            for i in range(2):
                # with cols[i]:
                with st.container():
                    st.subheader(data_names[i])
                    data = st.session_state[data_names[i]]
                    selected_model = get_selected_model(data)

                    #     selected_model = get_selected_model()
                    #     model = None
                    #     reference_model = None
                    if selected_model == "pinhole":
                        fx, fy, cx, cy = draw_pinhole_config(data, i)
                        model = Pinhole(fx, fy, cx, cy)
                    

                        
                    elif selected_model == "kb4":
                        fx, fy, cx, cy, k1, k2, k3, k4 = draw_kb4_config(data, i)
                        model = KB4(fx, fy, cx, cy, k1, k2, k3, k4)

                    elif selected_model == 'ds':
                        fx, fy, cx, cy, xi, alpha = draw_ds_config(data, i)
                        model = DoubleSphere(fx, fy, cx, cy, xi, alpha)

                    elif selected_model == "pinhole-radtan8":
                        k1, k2, k3, k4, k5, k6, p1, p2, rpmax = draw_radtan8_config(data, i)
                        model = RadTan8(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax)

                    elif selected_model == 'eucm':
                        fx, fy, cx, cy, alpha, beta = draw_eucm_config(data, i)
                        model = EUCM(fx, fy, cx, cy, alpha, beta)

                    else:
                        raise ValueError("Unsupported distortion model")
                    
                    st.slider("FOV (degrees)", 0, 360, 90, key='fov' + str(i))
                    
                    models[i] = model

            st.slider("FOV (degrees) Visualised", 0, 360, 90, key='fov')

            models_form_submitted = st.form_submit_button("Update Distortion Params")

        # detailed comparison analysis

        st.info(models[0] is None)

        if models_form_submitted:
            # st.info("submitted")
        
    #     if model is not None and reference_model is not None:

            c1, c2, c3 = st.columns(3)
            c1.checkbox("Hide Model A (blue)", key='hide_model_a')
            c2.checkbox("Hide Model B (red)", key='hide_model_b')
            c3.checkbox("Hide displacement vectors (red arrows)", key='hide_displacement_vectors')

            models[0].fov = st.session_state['fov0']
            models[1].fov = st.session_state['fov1']

            viz = visualiser.DistortionVisualizer(width, height, num_points=num_points, model_a=models[0], model_b=models[1])

            viz.visualize_distortion()
            
            st.slider("test", min_value=-1, max_value=1, value=0, key="slider_test")

        
        # parameter play

        # fit = distortion_fit.draw_r_theta_curve(models[0])
        


if __name__ == '__main__':
    main()
