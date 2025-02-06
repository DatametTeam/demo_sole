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

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")
# st.image(Image.open(Path(__file__).parent / "static/faradai_logo.png"))

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status

import imageio


def create_gif_in_memory(gt_array, pred_array, index):
    # Create a BytesIO buffer to hold the GIF in memory
    buf = io.BytesIO()
    frames = []

    for i in range(gt_array.shape[0]):  # You can adjust this for how many frames to include
        gt_frame = gt_array[i, 0, :, :]
        # pred_frame = pred_array[i, 0, :, :]

        # Create an image frame by stacking ground truth and prediction
        fig, ax = plt.subplots()
        ax.imshow(gt_frame, cmap='viridis', alpha=1)
        # ax.imshow(pred_frame, cmap='inferno', alpha=0.5)
        ax.axis('off')  # Turn off axis for better visual
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove borders
        buf.seek(0)

        # Convert the buffer to a PIL image and append it to frames
        buf_tracked = io.BytesIO()  # new buffer to store each frame
        plt.savefig(buf_tracked, format='png', bbox_inches='tight', pad_inches=0)
        buf_tracked.seek(0)
        img = Image.open(buf_tracked)
        frames.append(np.array(img))  # Append frame
        plt.close(fig)  # Close the figure to avoid memory issues

    # Ensure the frames are in the correct format
    frames = [np.array(frame) for frame in frames]
    if frames[0].ndim == 3 and frames[0].shape[2] == 4:
        # If the images have an alpha channel, convert to RGB
        frames = [frame[:, :, :3] for frame in frames]

    # Save the frames as a GIF directly into the buffer
    imageio.mimsave(buf, frames, format='GIF', duration=0.5, loop=0)  # Adjust duration for frame rate
    buf.seek(0)  # Go to the start of the buffer to send it to st.image
    return buf


def update_prediction_visualization(gt_array, pred_array):
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60 = init_prediction_visualization_layout()
    i = 0
    gif_path = create_gif_in_memory(gt_array, pred_array, i)

    while i < gt_array.shape[0]:
        # Create and save the GIF

        # Display the GIF using Streamlit
        gt_current.image(gif_path, caption="Prediction Animation", use_container_width=True)

        # Clear previous prediction visuals
        pred_current.empty()
        gt_plus_30.empty()
        pred_plus_30.empty()
        gt_plus_60.empty()
        pred_plus_60.empty()

        # time.sleep(1)  # Delay to simulate animation, can be adjusted
        i += 1



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
    gt_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:5]
    gt_array = np.array(gt_array)
    gt_array[gt_array < 0] = 0
    gt_array[gt_array > 50] = 50
    gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    pred_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:5]
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
    submitted = sidebar_args[-1]
    if submitted:
        error, out_dir = submit_prediction_job(sidebar_args)
        if not error:
            display_prediction_results(out_dir)
        else:
            st.error(error)


def main():
    sidebar_args = configure_sidebar()
    main_page(sidebar_args)


if __name__ == "__main__":
    main()
