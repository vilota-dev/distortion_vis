import streamlit as st


def get_camera_names(data):
    return data['value0']['cam_names']


def get_selected_model(data):
    """ Returns the selected model """
    return data['value0']['intrinsics'][get_selected_camera_idx(data)]['camera_type']


def get_selected_camera_idx(data):
    """ Returns the index of the selected camera """
    # We have a string containing the selected camera, we want to get the index of that camera using .index method
    return get_camera_names(data).index(data['selected_camera'])
