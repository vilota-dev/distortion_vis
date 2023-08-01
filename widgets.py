import streamlit as st
from helper_functions import *


def draw_resolution_config(data, placeholder):
    res = data['value0']['resolution'][get_selected_camera_idx(data)]
    c1, c2 = placeholder.columns(2)
    width = c1.number_input('width', step=1, format="%d", value=res[0])
    height = c2.number_input('height', step=1, format="%d", value=res[1])

    return width, height

# def draw_ideal_pinhole_config(data):
#     # st.subheader("Ideal Pinhole Intrinsics")
#     temp = data['value0']['intrinsics'][get_selected_camera_idx(data)]['intrinsics']
#     col1, col2, col3, col4 = st.columns(4)
#     fx = col1.number_input('pinhole_fx', step=0.1, format="%.3f", value=temp['fx'])
#     fy = col2.number_input('pinhole_fy', step=0.1, format="%.3f", value=temp['fy'])
#     cx = col3.number_input('pinhole_cx', step=0.1, format="%.3f", value=temp['cx'])
#     cy = col4.number_input('pinhole_cy', step=0.1, format="%.3f", value=temp['cy'])

#     return fx, fy, cx, cy


def draw_pinhole_config(data, i):
    temp = data['value0']['intrinsics'][get_selected_camera_idx(data)]['intrinsics']
    col1, col2, col3, col4 = st.columns(4)
    fx = col1.number_input('fx' + str(i), step=0.1, format="%.3f", value=temp['fx'])
    fy = col2.number_input('fy' + str(i), step=0.1, format="%.3f", value=temp['fy'])
    cx = col3.number_input('cx' + str(i), step=0.1, format="%.3f", value=temp['cx'])
    cy = col4.number_input('cy' + str(i), step=0.1, format="%.3f", value=temp['cy'])

    return fx, fy, cx, cy


def draw_kb4_config(data, i):
    temp = data['value0']['intrinsics'][get_selected_camera_idx(data)]['intrinsics']

    fx, fy, cx, cy = draw_pinhole_config(data, i)

    c1, c2, c3, c4 = st.columns(4)
    k1 = c1.number_input('k1' + str(i), step=1e-8, format="%.8f", value=temp['k1'])
    k2 = c2.number_input('k2' + str(i), step=1e-8, format="%.8f", value=temp['k2'])
    k3 = c3.number_input('k3' + str(i), step=1e-8, format="%.8f", value=temp['k3'])
    k4 = c4.number_input('k4' + str(i), step=1e-8, format="%.8f", value=temp['k4'])

    return fx, fy, cx, cy, k1, k2, k3, k4


def draw_ds_config(data, i):
    temp = data['value0']['intrinsics'][get_selected_camera_idx(data)]['intrinsics']

    fx, fy, cx, cy = draw_pinhole_config(data, i)

    c1, c2 = st.columns(2)
    xi = c1.number_input('xi' + str(i), step=0.01, format="%.8f", value=temp['xi'])
    alpha = c2.number_input('alpha' + str(i), step=0.01, format="%.8f", value=temp['alpha'])

    return fx, fy, cx, cy, xi, alpha


def draw_radtan8_config(data, i):
    temp = data['value0']['intrinsics'][get_selected_camera_idx(data)]['intrinsics']
    c1, c2, c3, c4 = st.columns(4)
    k1 = c1.number_input('k1' + str(i), step=1e-8, format="%.8f", value=temp['k1'])
    k2 = c2.number_input('k2' + str(i), step=1e-8, format="%.8f", value=temp['k2'])
    k3 = c3.number_input('k3' + str(i), step=1e-8, format="%.8f", value=temp['k3'])
    k4 = c4.number_input('k4' + str(i), step=1e-8, format="%.8f", value=temp['k4'])
    k5 = c1.number_input('k5' + str(i), step=1e-8, format="%.8f", value=temp['k5'])
    k6 = c2.number_input('k6' + str(i), step=1e-8, format="%.8f", value=temp['k6'])
    p1 = c3.number_input('p1' + str(i), step=1e-8, format="%.8f", value=temp['p1'])
    p2 = c4.number_input('p2' + str(i), step=1e-8, format="%.8f", value=temp['p2'])
    rpmax = c1.number_input('rpmax' + str(i), step=1e-6, format="%.6f", value=temp['rpmax'])

    return k1, k2, k3, k4, k5, k6, p1, p2, rpmax


def draw_eucm_config(data, i):
    temp = data['value0']['intrinsics'][get_selected_camera_idx(data)]['intrinsics']

    fx, fy, cx, cy = draw_pinhole_config(data, i)

    c1, c2 = st.columns(2)
    alpha = c1.number_input('alpha' + str(i), min_value=0.0, max_value=1.0, step=1e-8, format="%.8f", value=temp['alpha'])
    beta = c2.number_input('beta' + str(i), step=1e-8, format="%.8f", value=temp['beta'])

    return fx, fy, cx, cy, alpha, beta
