import io
import os
import streamlit as st
from time import sleep
import math
from PIL import Image
from pathlib import Path
import time
import numpy as np
from layouts import configure_sidebar, init_prediction_visualization_layout
from pbs import is_pbs_available
import matplotlib.pyplot as plt
import imageio
from datetime import datetime

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status

import imageio


def create_gif_from_saved_figures(figures_dict):
    # Create a BytesIO buffer to hold the GIF in memory
    buf = io.BytesIO()
    frames = []

    # Sort the dictionary by key to ensure the frames are in chronological order
    sorted_keys = sorted(figures_dict.keys())

    # Loop through the sorted keys and retrieve the corresponding figure
    for key in sorted_keys:
        fig = figures_dict[key]  # Retrieve the figure from the dictionary

        # Save the figure to a buffer
        buf_tracked = io.BytesIO()
        fig.savefig(buf_tracked, format='png', bbox_inches='tight', pad_inches=0)
        buf_tracked.seek(0)
        img = Image.open(buf_tracked)
        frames.append(np.array(img))  # Append frame

    # Create the GIF with the frames
    imageio.mimsave(buf, frames, format='GIF', fps=5, loop=0)  # Adjust duration for frame rate
    buf.seek(0)  # Go to the start of the buffer to send it to st.image

    return buf


def create_figure_dict_from_array(gt_array):
    # Initialize an empty dictionary to store the figures
    figure_dict = {}

    # Loop through the array and create a figure for each element
    for i in range(gt_array.shape[0]):  # Assuming gt_array has shape (n_frames, height, width)
        # Create a figure for the current frame
        fig, ax = plt.subplots()
        ax.imshow(gt_array[i], cmap='jet')  # Use 'jet' colormap for visualization
        ax.axis('off')  # Turn off axis for better visual
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove borders

        # Save the figure in the dictionary with a key based on the index or time
        figure_dict[f"frame_{i}"] = fig  # Use the index as the key, or modify as needed

    return figure_dict


def update_prediction_visualization(gt_array, pred_array):
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60 = init_prediction_visualization_layout()

    gt_array = gt_array[:, 0, :, :]
    gt0_dict = create_figure_dict_from_array(gt_array)

    gt0_gif = create_gif_from_saved_figures(gt0_dict)

    # Store the GIF in session state for tab1
    st.session_state.tab1_gif = gt0_gif.getvalue()

    # Display the GIF using Streamlit
    gt_current.image(gt0_gif, caption="Prediction Animation", use_container_width=True)
    pred_current.empty()
    gt_plus_30.image(gt0_gif, caption="Prediction Animation", use_container_width=True)
    pred_plus_30.image(gt0_gif, caption="Prediction Animation", use_container_width=True)
    gt_plus_60.image(gt0_gif, caption="Prediction Animation", use_container_width=True)
    pred_plus_60.image(gt0_gif, caption="Prediction Animation", use_container_width=True)


def submit_prediction_job(sidebar_args):
    error = None
    with st.status(f':hammer_and_wrench: **Running prediction...**', expanded=True) as status:
        prediction_placeholder = st.empty()
        with prediction_placeholder:
            with prediction_placeholder.container():
                pbs_job_id, out_dir = submit_inference(sidebar_args)
                # if pbs_job_id is None:
                #     error = "Error submitting prediction job"
                #     status.update(label="❌ Prediction failed!", state="error", expanded=True)
                #     return error
                # while get_job_status(pbs_job_id)=="R":
                #     sleep(1)
                #     status.update(label="🔄 Prediction in progress...", state="running", expanded=True)
        status.update(label="✅ Prediction completed!", state="complete", expanded=False)
    return error, out_dir


def get_prediction_results(out_dir):
    gt_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:10]
    gt_array = np.array(gt_array)
    gt_array[gt_array < 0] = 0
    gt_array[gt_array > 50] = 50
    gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    pred_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:10]
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    pred_array[pred_array > 50] = 50
    pred_array = (pred_array - np.min(pred_array)) / (np.max(pred_array) - np.min(pred_array))
    print("*** LOADED DATA ***")
    return gt_array, pred_array


def display_prediction_results(out_dir):
    gt_array, pred_array = get_prediction_results(out_dir)
    update_prediction_visualization(gt_array, pred_array)


def main_page(sidebar_args) -> None:
    # Only run prediction if not already done
    if 'prediction_result' not in st.session_state:
        submitted = sidebar_args[-1]
        if submitted:
            error, out_dir = submit_prediction_job(sidebar_args)
            if not error:
                gt_array, pred_array = get_prediction_results(out_dir)
                # Store results in session state
                st.session_state.prediction_result = {
                    'gt_array': gt_array,
                    'pred_array': pred_array
                }
                update_prediction_visualization(gt_array, pred_array)
            else:
                st.error(error)
    else:
        # If prediction results already exist, reuse them
        gt_array = st.session_state.prediction_result['gt_array']
        pred_array = st.session_state.prediction_result['pred_array']
        update_prediction_visualization(gt_array, pred_array)


def show_prediction_page():
    st.title("Select Date and Time for Prediction")

    selected_date = st.date_input("Select Date", min_value=datetime(2020, 1, 1), max_value=datetime.today())
    selected_time = st.time_input("Select Time", datetime.now().time())

    if st.button("Submit"):
        st.session_state.selected_date = selected_date
        st.session_state.selected_time = selected_time

        st.write(f"Selected Date: {selected_date}")
        st.write(f"Selected Time: {selected_time}")

        sample_image = np.random.rand(100, 100)  # Create a dummy image
        buf = io.BytesIO()
        Image.fromarray((sample_image * 255).astype(np.uint8)).save(buf, format="PNG")

        # Store the generated GIF buffer for tab2
        st.session_state.tab2_gif = buf.getvalue()

        # Display the GIF
        st.image(st.session_state.tab2_gif, caption="Prediction Image", use_column_width=True)
    else:
        # Display the stored GIF buffer for tab2 if it exists
        if "tab2_gif" in st.session_state:
            st.image(st.session_state.tab2_gif, caption="Prediction Image", use_column_width=True)


def show_home_page():
    st.title("Weather Prediction")
    st.write("This is the home page with prediction results and other details.")

    # Display the stored GIF buffer for tab1
    if "tab1_gif" in st.session_state:
        st.image(st.session_state.tab1_gif, caption="Prediction Animation", use_column_width=True)
    else:
        st.write("No GIFs available yet.")


# Sidebar (remains as is)
sidebar_args = configure_sidebar()

# Create tabs using st.tabs
tab1, tab2, tab3 = st.tabs(["Home", "Prediction by Date & Time", "Tab 3"])

with tab1:
    main_page(sidebar_args)

with tab2:
    show_prediction_page()

with tab3:
    st.title("Tab 3")
    st.write("This is another tab. You can put your content here.")
