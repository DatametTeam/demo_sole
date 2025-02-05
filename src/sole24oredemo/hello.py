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

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")
# st.image(Image.open(Path(__file__).parent / "static/faradai_logo.png"))

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status

def update_prediction_visualization(gt_array, pred_array):
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60 = init_prediction_visualization_layout()
    i = 0
    while i < gt_array.shape[0]:
        # Create a figure and plot the image using the colormap
        fig, ax = plt.subplots()
        cax = ax.imshow(gt_array[i, 0, :, :], cmap='viridis')

        # Remove axis if desired
        ax.axis('off')

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        # Open the image from the buffer with PIL
        img = Image.open(buf)

        # Display the image using Streamlit
        gt_current.image(img, caption="GT Frame t0", use_container_width=True)

        pred_current.empty()
        gt_plus_30.image(gt_array[i, 3, :, :], caption="GT Frame t+30", use_container_width=True)
        pred_plus_30.image(pred_array[i, 3, :, :], caption="Pred Frame t+30", use_container_width=True)
        gt_plus_60.image(gt_array[i, 6, :, :], caption="GT Frame t+60", use_container_width=True)
        pred_plus_60.image(pred_array[i, 6, :, :], caption="Pred Frame t+60", use_container_width=True)
        time.sleep(1)
        i += 1

def submit_prediction_job(sidebar_args):
    error = None
    with st.status(f':hammer_and_wrench: **Running prediction...**', expanded=True) as status:
        prediction_placeholder = st.empty() 
        with prediction_placeholder:
            with prediction_placeholder.container():
                pbs_job_id, out_dir = submit_inference(sidebar_args)
                if pbs_job_id is None:
                    error = "Error submitting prediction job"
                    status.update(label="❌ Prediction failed!", state="error", expanded=True)
                    return error
                while get_job_status(pbs_job_id)=="R":
                    sleep(1)
                    status.update(label="🔄 Prediction in progress...", state="running", expanded=True)
        status.update(label="✅ Prediction completed!", state="complete", expanded=False)
    return error, out_dir

def get_prediction_results(out_dir):
    gt_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:50]
    gt_array = np.array(gt_array)
    gt_array[gt_array < 0] = 0
    gt_array[gt_array > 200] = 200
    gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    pred_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:50]
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    pred_array[pred_array > 200] = 200
    pred_array = (pred_array - np.min(pred_array)) / (np.max(pred_array) - np.min(pred_array))

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
