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

from sole24oredemo.utils import compute_figure

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")
# st.image(Image.open(Path(__file__).parent / "static/faradai_logo.png"))

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status


def update_prediction_visualization(gt_array, pred_array):
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60 = \
        init_prediction_visualization_layout()
    i = 0

    # Define the hardcoded date
    base_date = datetime.strptime('31-01-2025', "%d-%m-%Y")

    # Add labels for rows
    st.markdown("### GROUNDTRUTH", unsafe_allow_html=True)
    gt_row_container = st.container()
    st.markdown("### PREDICTIONS", unsafe_allow_html=True)
    pred_row_container = st.container()

    while i < gt_array.shape[0]:
        # Calculate the current datetime based on the iteration
        start_time = datetime.strptime('00:00', "%H:%M") + timedelta(minutes=i * 5)
        current_datetime = base_date + timedelta(hours=start_time.hour, minutes=start_time.minute)

        # Format date and time for captions
        datetime_caption = current_datetime.strftime("%d-%m-%Y %H:%M")

        with gt_row_container:
            # Create a figure and plot the image using the colormap

            fig1 = compute_figure(gt_array[i, 0, :, :])
            # Display the image using Streamlit
            gt_current.pyplot(fig1)

            fig2 = compute_figure(gt_array[i, 6, :, :])
            # Display the image using Streamlit
            gt_plus_30.pyplot(fig2)

            fig3 = compute_figure(gt_array[i, -1, :, :])
            # Display the image using Streamlit
            gt_plus_60.pyplot(fig3)

            # plus_30_date = current_datetime + timedelta(minutes=30)
            # gt_plus_30.image(gt_array[i, 3, :, :], caption=plus_30_date.strftime("%d-%m-%Y %H:%M"),
            #                  use_container_width=True)
            #
            # plus_60_date = current_datetime + timedelta(minutes=60)
            # gt_plus_60.image(gt_array[i, 6, :, :], caption=plus_60_date.strftime("%d-%m-%Y %H:%M"),
            #                  use_container_width=True)

        with pred_row_container:
            fig4 = compute_figure(gt_array[i, 0, :, :], basemap)
            # Display the image using Streamlit
            pred_plus_30.pyplot(fig4)

            fig5 = compute_figure(gt_array[i, 0, :, :], basemap)
            # Display the image using Streamlit
            pred_plus_60.pyplot(fig5)

            # pred_current.empty()
            # pred_plus_30.image(pred_array[i, 3, :, :], caption=plus_30_date.strftime("%d-%m-%Y %H:%M"),
            #                    use_container_width=True)
            # pred_plus_60.image(pred_array[i, 6, :, :], caption=plus_60_date.strftime("%d-%m-%Y %H:%M"),
            #                    use_container_width=True)

        time.sleep(0.2)
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
                # while get_job_status(pbs_job_id)=="R":
                while pbs_job_id == 0:
                    sleep(2)
                    status.update(label="Prediction in progress...", state="running", expanded=True)
                    pbs_job_id = 1
        status.update(label="Prediction completed!", state="complete", expanded=False)
    return error, out_dir


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
