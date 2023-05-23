import streamlit as st
from distortion_models import *
import visualiser
import json


def main():
    st.title("Distortion Model Visualizer")

    # ------------------ FILE UPLOADING SECTION ------------------ #
    # Upload distortion model (.json format)
    uploaded_file = st.sidebar.file_uploader('Upload a distortion model', type=['json'])
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        # Grab the list of camera models in value0, cam_names
        cam_names = data['value0']['cam_names']
        intrinsics = data['value0']['intrinsics']

        # Create a tab for each camera name
        for (i, cam_name) in enumerate(cam_names):
            with st.sidebar.expander(cam_name):
                # Access the corresponding parameters in the "intrinsics" part of the JSON
                data = intrinsics[i]
                st.write(data)
    st.sidebar.divider()

    model_list = [
        'Kannala-Brandt Model (kb4)',
        'Double Sphere (ds)',
        'Radial Tangential (radtan8)',
        'Enhanced Unified Camera Model (eucm)'
    ]

    # Resolution parameters
    st.sidebar.subheader('Resolution')
    c1, c2 = st.sidebar.columns(2)
    width = c1.number_input('width', step=1, format="%d", value=1280)
    height = c2.number_input('height', step=1, format="%d", value=800)


    # Pinhole Camera common parameters
    # Since the parameters are default for all camera models, the points can look iffy for some models
    # To get proper output, use ur own values
    with st.expander('Parameters'):
        selected_model = st.selectbox('Distortion Model', model_list)
        num_points = st.sidebar.slider("Number of points", 10, 50, 20)
        col1, col2, col3, col4 = st.columns(4)
        fx = col1.number_input('fx', step=0.1, format="%.3f", value=409.56013372715798)
        fy = col2.number_input('fy', step=0.1, format="%.3f", value=410.48431621672327)
        cx = col3.number_input('cx', step=0.1, format="%.3f", value=654.3040038316136)
        cy = col4.number_input('cy', step=0.1, format="%.3f", value=411.0859377592162)


        # Parameter input
        if selected_model == "Kannala-Brandt Model (kb4)":
            k1 = col1.number_input('k1', step=1e-8, format="%.8f", value=-0.256)
            k2 = col2.number_input('k2', step=1e-8, format="%.8f", value=-0.0015)
            k3 = col3.number_input('k3', step=1e-8, format="%.8f", value=0.0007)
            k4 = col4.number_input('k4', step=1e-8, format="%.8f", value=-0.0002)

            model = KB4(fx, fy, cx, cy, k1, k2, k3, k4)

        elif selected_model == 'Double Sphere (ds)':
            second_col1, second_col2 = st.columns(2)
            xi = second_col1.number_input('xi', step=0.01, format="%.8f", value=-0.1183471725196422)
            alpha = second_col2.number_input('alpha', step=0.01, format="%.8f", value=0.10426563424702325)

            model = DoubleSphere(fx, fy, cx, cy, xi, alpha)

        elif selected_model == "Radial Tangential (radtan8)":
            k1 = col1.number_input('k1', step=1e-8, format="%.8f", value=-0.34343072695540086)
            k2 = col2.number_input('k2', step=1e-8, format="%.8f", value=-0.035587492995016487)
            k3 = col3.number_input('k3', step=1e-8, format="%.8f", value=0.008503833816478223)
            k4 = col4.number_input('k4', step=1e-8, format="%.8f", value=-0.3437950776046646)
            k5 = col1.number_input('k5', step=1e-8, format="%.8f", value=-0.03480285769096108)
            k6 = col2.number_input('k6', step=1e-8, format="%.8f", value=0.008240062843272636)
            p1 = col3.number_input('p1', step=1e-8, format="%.8f", value=0.0023416182385782777)
            p2 = col4.number_input('p2', step=1e-8, format="%.8f", value=-0.0034607048670843899)
            rpmax = col1.number_input('rpmax', step=1e-6, format="%.6f", value=0.0)

            model = RadTan8(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax)

        elif selected_model == 'Enhanced Unified Camera Model (eucm)':
            second_col1, second_col2 = st.columns(2)
            alpha = second_col1.number_input('alpha', min_value=0.0, max_value=1.0, step=1e-8, format="%.8f", value=0.00007992209678892431)
            beta = second_col2.number_input('beta', step=1e-8, format="%.8f", value=20.180947456852793)

            model = EUCM(fx, fy, cx, cy, alpha, beta)
        else:
            raise ValueError("Unsupported distortion model")



    viz = visualiser.DistortionVisualizer(width, height, num_points=num_points, model=model)

    viz.visualize_distortion()


if __name__ == '__main__':
    main()
