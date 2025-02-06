import io
import os
import streamlit as st
from time import sleep
import math
from pathlib import Path
import time
import numpy as np
from layouts import configure_sidebar, init_prediction_visualization_layout
from pbs import is_pbs_available
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sole24oredemo.parallel_code import create_fig_dict_in_parallel
from sole24oredemo.utils import compute_figure

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")
# st.image(Image.open(Path(__file__).parent / "static/faradai_logo.png"))

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status


def update_prediction_visualization(gt_dict, pred_dict):
    """
    Visualizes ground truth and prediction images side by side using Streamlit.

    Parameters:
    - gt_dict: Dictionary containing GT figures with keys formatted as "%d%m%Y_%H%M".
    - pred_dict: Dictionary containing prediction figures with keys formatted as "%d%m%Y_%H%M",
                 and values as nested dictionaries with keys '+0min', '+30min', '+55min'.
    """
    # Initialize layout containers
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60 = \
        init_prediction_visualization_layout()

    # Sort GT keys (dates)
    sorted_gt_keys = sorted(gt_dict.keys(), key=lambda x: datetime.strptime(x, "%d%m%Y_%H%M"))

    # Iterate over sorted GT keys
    for key in sorted_gt_keys:
        # Retrieve GT figures
        gt_current_fig = gt_dict.get(key)
        gt_plus_30_key = (datetime.strptime(key, "%d%m%Y_%H%M") + timedelta(minutes=30)).strftime("%d%m%Y_%H%M")
        gt_plus_60_key = (datetime.strptime(key, "%d%m%Y_%H%M") + timedelta(minutes=60)).strftime("%d%m%Y_%H%M")
        gt_plus_30_fig = gt_dict.get(gt_plus_30_key)
        gt_plus_60_fig = gt_dict.get(gt_plus_60_key)

        # Retrieve Prediction figures
        pred_data = pred_dict.get(key, {})
        # pred_current_fig = pred_data.get('+0min')
        pred_plus_30_fig = pred_data.get('+30min')
        pred_plus_60_fig = pred_data.get('+60min')

        # Display GT figures
        if gt_current_fig is not None:
            gt_current.pyplot(gt_current_fig)

        if gt_plus_30_fig is not None:
            gt_plus_30.pyplot(gt_plus_30_fig)

        if gt_plus_60_fig is not None:
            gt_plus_60.pyplot(gt_plus_60_fig)

        # # Display Prediction figures
        # if pred_current_fig is not None:
        #     pred_current.pyplot(pred_current_fig)

        if pred_plus_30_fig is not None:
            pred_plus_30.pyplot(pred_plus_30_fig)

        if pred_plus_60_fig is not None:
            pred_plus_60.pyplot(pred_plus_60_fig)

        # Pause for visualization
        time.sleep(0.2)


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
                # while get_job_status(pbs_job_id)=="R":
                while pbs_job_id == 0:
                    sleep(2)
                    status.update(label="Prediction in progress...", state="running", expanded=True)
                    pbs_job_id = 1
                status.update(label="Prediction completed!", state="complete", expanded=False)
                status.update(label="Computing images", state="running", expanded=True)
                gt_dict_figs, pred_dict_figs = create_fig_dict_in_parallel()
        status.update(label="Done!", state="complete", expanded=False)

    return error, gt_dict_figs, pred_dict_figs


def get_prediction_results(out_dir):
    # TODO: da fixare
    out_dir = Path("/archive/SSD/home/guidim/demo_sole/data/output/ConvLSTM/20250205/20250205/generations")

    gt_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:10]
    gt_array = np.array(gt_array)
    gt_array[gt_array < 0] = 0
    gt_array[gt_array > 200] = 200
    # gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    pred_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:10]
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    pred_array[pred_array > 200] = 200
    # pred_array = (pred_array - np.min(pred_array)) / (np.max(pred_array) - np.min(pred_array))

    return gt_array, pred_array


def display_prediction_results(gt_images, pred_images):
    # gt_array, pred_array = get_prediction_results(out_dir)
    update_prediction_visualization(gt_images, pred_images)


def main_page(sidebar_args) -> None:
    submitted = sidebar_args[-1]
    if submitted:
        error, gt_dict_figs, pred_dict_figs = submit_prediction_job(sidebar_args)
        if not error:
            display_prediction_results(gt_dict_figs, pred_dict_figs)
        else:
            st.error(error)


def main():
    sidebar_args = configure_sidebar()
    main_page(sidebar_args)


if __name__ == "__main__":
    main()
