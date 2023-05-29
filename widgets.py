import streamlit as st
from helper_functions import *


def draw_resolution_config(placeholder):
    res = st.session_state['data']['value0']['resolution'][get_selected_camera_idx()]
    c1, c2 = placeholder.columns(2)
    width = c1.number_input('width', step=1, format="%d", value=res[0])
    height = c2.number_input('height', step=1, format="%d", value=res[1])

    return width, height

def draw_ideal_pinhole_config():
    st.subheader("Ideal Pinhole Intrinsics")
    temp = st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['intrinsics']
    col1, col2, col3, col4 = st.columns(4)
    fx = col1.number_input('pinhole_fx', step=0.1, format="%.3f", value=temp['fx'])
    fy = col2.number_input('pinhole_fy', step=0.1, format="%.3f", value=temp['fy'])
    cx = col3.number_input('pinhole_cx', step=0.1, format="%.3f", value=temp['cx'])
    cy = col4.number_input('pinhole_cy', step=0.1, format="%.3f", value=temp['cy'])

    return fx, fy, cx, cy


def draw_pinhole_config():
    temp = st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['intrinsics']
    col1, col2, col3, col4 = st.columns(4)
    fx = col1.number_input('fx', step=0.1, format="%.3f", value=temp['fx'])
    fy = col2.number_input('fy', step=0.1, format="%.3f", value=temp['fy'])
    cx = col3.number_input('cx', step=0.1, format="%.3f", value=temp['cx'])
    cy = col4.number_input('cy', step=0.1, format="%.3f", value=temp['cy'])

    return fx, fy, cx, cy


def draw_kb4_config():
    temp = st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['intrinsics']
    c1, c2, c3, c4 = st.columns(4)
    k1 = c1.number_input('k1', step=1e-8, format="%.8f", value=temp['k1'])
    k2 = c2.number_input('k2', step=1e-8, format="%.8f", value=temp['k2'])
    k3 = c3.number_input('k3', step=1e-8, format="%.8f", value=temp['k3'])
    k4 = c4.number_input('k4', step=1e-8, format="%.8f", value=temp['k4'])

    return k1, k2, k3, k4


def draw_ds_config():
    temp = st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['intrinsics']
    c1, c2 = st.columns(2)
    xi = c1.number_input('xi', step=0.01, format="%.8f", value=temp['xi'])
    alpha = c2.number_input('alpha', step=0.01, format="%.8f", value=temp['alpha'])

    return xi, alpha


def draw_radtan8_config():
    temp = st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['intrinsics']
    c1, c2, c3, c4 = st.columns(4)
    k1 = c1.number_input('k1', step=1e-8, format="%.8f", value=temp['k1'])
    k2 = c2.number_input('k2', step=1e-8, format="%.8f", value=temp['k2'])
    k3 = c3.number_input('k3', step=1e-8, format="%.8f", value=temp['k3'])
    k4 = c4.number_input('k4', step=1e-8, format="%.8f", value=temp['k4'])
    k5 = c1.number_input('k5', step=1e-8, format="%.8f", value=temp['k5'])
    k6 = c2.number_input('k6', step=1e-8, format="%.8f", value=temp['k6'])
    p1 = c3.number_input('p1', step=1e-8, format="%.8f", value=temp['p1'])
    p2 = c4.number_input('p2', step=1e-8, format="%.8f", value=temp['p2'])
    rpmax = c1.number_input('rpmax', step=1e-6, format="%.6f", value=temp['rpmax'])

    return k1, k2, k3, k4, k5, k6, p1, p2, rpmax


def draw_eucm_config():
    temp = st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['intrinsics']

    c1, c2 = st.columns(2)
    alpha = c1.number_input('alpha', min_value=0.0, max_value=1.0, step=1e-8, format="%.8f", value=temp['alpha'])
    beta = c2.number_input('beta', step=1e-8, format="%.8f", value=temp['beta'])

    return alpha, beta
