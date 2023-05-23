import streamlit as st
from distortion_models import *
import visualiser


def main():
    # Only put stuff relevant to streamlit style here
    st.title("Distortion Model Visualiser")

    st.sidebar.title('Input Parameters')

    uploaded_file = st.sidebar.file_uploader('Upload a distortion model', type=['json', 'yaml', 'yml'])
    if uploaded_file is not None:
        st.sidebar.write('File uploaded successfully')
        st.sidebar.write(uploaded_file)

    model_list = [
        'Kannala-Brandt Model (kb4)',
        'Double Sphere (ds)',
        'Radial Tangential (radtan8)',
        'Enhanced Unified Camera Model (eucm)'
    ]

    # Distortion model selection
    selected_model = st.sidebar.selectbox('Distortion Model', model_list)
    st.sidebar.divider()

    st.subheader(selected_model)

    num_points = st.slider("Number of points", 10, 50, 20)

    # Resolution parameters
    st.sidebar.subheader('Resolution')
    width = st.sidebar.number_input('width', step=1, format="%d", value=1280)
    height = st.sidebar.number_input('height', step=1, format="%d", value=800)
    st.sidebar.divider()

    # Pinhole Camera common parameters
    # Since the parameters are default for all camera models, the points can look iffy for some models
    # To get proper output, use ur own values
    fx = st.sidebar.number_input('fx', step=0.1, format="%.3f", value=409.56013372715798)
    fy = st.sidebar.number_input('fy', step=0.1, format="%.3f", value=410.48431621672327)
    cx = st.sidebar.number_input('cx', step=0.1, format="%.3f", value=654.3040038316136)
    cy = st.sidebar.number_input('cy', step=0.1, format="%.3f", value=411.0859377592162)
    st.sidebar.divider()

    # Parameter input
    if selected_model == "Kannala-Brandt Model (kb4)":
        k1 = st.sidebar.number_input('k1', step=1e-8, format="%.8f", value=-0.256)
        k2 = st.sidebar.number_input('k2', step=1e-8, format="%.8f", value=-0.0015)
        k3 = st.sidebar.number_input('k3', step=1e-8, format="%.8f", value=0.0007)
        k4 = st.sidebar.number_input('k4', step=1e-8, format="%.8f", value=-0.0002)

        model = KB4(fx, fy, cx, cy, k1, k2, k3, k4)

    elif selected_model == 'Double Sphere (ds)':
        xi = st.sidebar.number_input('xi', step=0.01, format="%.8f", value=-0.1183471725196422)
        alpha = st.sidebar.number_input('alpha', step=0.01, format="%.8f", value=0.10426563424702325)

        model = DoubleSphere(fx, fy, cx, cy, xi, alpha)

    elif selected_model == "Radial Tangential (radtan8)":
        k1 = st.sidebar.number_input('k1', step=1e-8, format="%.8f", value=-0.34343072695540086)
        k2 = st.sidebar.number_input('k2', step=1e-8, format="%.8f", value=-0.035587492995016487)
        p1 = st.sidebar.number_input('p1', step=1e-8, format="%.8f", value=0.0023416182385782777)
        p2 = st.sidebar.number_input('p2', step=1e-8, format="%.8f", value=-0.0034607048670843899)
        k3 = st.sidebar.number_input('k3', step=1e-8, format="%.8f", value=0.008503833816478223)
        k4 = st.sidebar.number_input('k4', step=1e-8, format="%.8f", value=-0.3437950776046646)
        k5 = st.sidebar.number_input('k5', step=1e-8, format="%.8f", value=-0.03480285769096108)
        k6 = st.sidebar.number_input('k6', step=1e-8, format="%.8f", value=0.008240062843272636)
        rpmax = st.sidebar.number_input('rpmax', step=1e-6, format="%.6f", value=0.0)

        model = RadTan8(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, rpmax)

    elif selected_model == 'Enhanced Unified Camera Model (eucm)':
        alpha = st.sidebar.number_input('alpha', min_value=0.0, max_value=1.0, step=1e-8, format="%.8f", value=0.00007992209678892431)
        beta = st.sidebar.number_input('beta', step=1e-8, format="%.8f", value=20.180947456852793)

        model = EUCM(fx, fy, cx, cy, alpha, beta)
    else:
        raise ValueError("Unsupported distortion model")

    viz = visualiser.DistortionVisualizer(width, height, num_points=num_points, model=model)

    viz.visualize_distortion()


if __name__ == '__main__':
    main()
