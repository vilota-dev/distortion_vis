import streamlit as st


def get_camera_names():
    return st.session_state['data']['value0']['cam_names']


def get_selected_model():
    """ Returns the selected model """
    return st.session_state['data']['value0']['intrinsics'][get_selected_camera_idx()]['camera_type']


def get_selected_camera_idx():
    """ Returns the index of the selected camera """
    # We have a string containing the selected camera, we want to get the index of that camera using .index method
    return get_camera_names().index(st.session_state['selected_camera'])
