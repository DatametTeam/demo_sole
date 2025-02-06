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
from datetime import datetime, timedelta

from sole24oredemo.parallel_code import create_fig_dict_in_parallel
from sole24oredemo.utils import compute_figure
import imageio
from datetime import datetime, timedelta

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status


def create_gif_from_saved_figures(figures_dict, progress_placeholder):
    # Create a BytesIO buffer to hold the GIF in memory
    buf = io.BytesIO()
    frames = []

    # Sort the dictionary by key to ensure the frames are in chronological order
    sorted_keys = sorted(figures_dict.keys())

    progress_bar = st.progress(0)

    # Loop through the sorted keys and retrieve the corresponding figure
    for i, key in enumerate(sorted_keys):
        fig = figures_dict[key]  # Retrieve the figure from the dictionary

        progress = (i + 1) / len(sorted_keys)
        progress_bar.progress(progress)
        progress_placeholder.write(f"Processing frame {i + 1}/{len(sorted_keys)}")

        # Save the figure to a buffer
        buf_tracked = io.BytesIO()
        fig.savefig(buf_tracked, format='png', bbox_inches='tight', pad_inches=0)
        buf_tracked.seek(0)
        img = Image.open(buf_tracked)
        frames.append(np.array(img))  # Append frame

    # Create the GIF with the frames
    imageio.mimsave(buf, frames, format='GIF', fps=5, loop=0)  # Adjust duration for frame rate
    buf.seek(0)  # Go to the start of the buffer to send it to st.image
    progress_placeholder.empty()

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

def update_prediction_visualization(gt0_gif):
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60 = init_prediction_visualization_layout()
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
                progress_bar = st.progress(0)
                your_array = np.zeros(100)
                status_placeholder = st.empty()
                for i in range(len(your_array)):
                    # Your processing logic here
                    progress = (i + 1) / len(your_array)
                    progress_bar.progress(progress)
                    status_placeholder.write(f"Processing item {i + 1}/{len(your_array)}")
                    time.sleep(0.01)
                # status1.update(label="Processing complete!")

        status.update(label="✅ Prediction completed!", state="complete", expanded=False)
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
    print("*** LOADED DATA ***")

    return gt_array, pred_array


def main_page(sidebar_args) -> None:
    # Only run prediction if not already done
    if 'prediction_result' not in st.session_state:
        submitted = sidebar_args[-1]
        if submitted:
            error, out_dir = submit_prediction_job(sidebar_args)
            if not error:
                with st.status(f':hammer_and_wrench: **Loading results...**', expanded=True) as status:

                    prediction_placeholder = st.empty()
                    progress_placeholder = st.empty()  # Add this line for progress bar

                    with prediction_placeholder:
                        status.update(label="🔄 Loading results...", state="running", expanded=True)

                        gt_array, pred_array = get_prediction_results(out_dir)

                        status.update(label="🔄 Creating GIFs...", state="running", expanded=True)

                        gt_array = gt_array[:, 0, :, :]
                        gt0_dict = create_figure_dict_from_array(gt_array)
                        gt0_gif = create_gif_from_saved_figures(gt0_dict, progress_placeholder)

                        # Store results in session state
                        st.session_state.prediction_result = {
                            'gt0_gif': gt0_gif,
                        }
                        st.session_state.tab1_gif = gt0_gif.getvalue()

                        status.update(label="✅ Done!", state="complete", expanded=True)
                        update_prediction_visualization(gt0_gif)
            else:
                st.error(error)
    else:
        # If prediction results already exist, reuse them
        gt0_gif = st.session_state.prediction_result['gt0_gif']
        update_prediction_visualization(gt0_gif)


def get_closest_5_minute_time():
    now = datetime.now()
    # Calculate the number of minutes past the closest earlier 5-minute mark
    minutes = now.minute - (now.minute % 5)
    return now.replace(minute=minutes, second=0, microsecond=0).time()


def show_prediction_page():
    st.title("Select Date and Time for Prediction")

    # Date and time selection
    selected_date = st.date_input(
        "Select Date", min_value=datetime(2020, 1, 1).date(), max_value=datetime.today().date()
    )
    selected_time = st.time_input("Select Time", get_closest_5_minute_time())

    if st.button("Submit"):
        # Combine selected date and time
        selected_datetime = datetime.combine(selected_date, selected_time)
        selected_key = selected_datetime.strftime("%Y-%m-%d_%H:%M")

        # Prepare ground truth frames
        groundtruth_frames = []
        for i in range(12):
            frame_time = selected_datetime - timedelta(minutes=5 * (12 - i - 1))
            frame_key = frame_time.strftime("%Y-%m-%d_%H:%M")
            if frame_key in groundtruth_dict:
                groundtruth_frames.append(groundtruth_dict[frame_key])
            else:
                groundtruth_frames.append(None)  # Placeholder for missing frames

        # Prepare target frames
        target_frames = []
        for i in range(12):
            frame_time = selected_datetime + timedelta(minutes=5 * (i + 1))
            frame_key = frame_time.strftime("%Y-%m-%d_%H:%M")
            if frame_key in groundtruth_dict:
                target_frames.append(groundtruth_dict[frame_key])
            else:
                target_frames.append(None)  # Placeholder for missing frames

        # Prepare prediction frames
        pred_frames = []
        if selected_key in pred_dict:
            for offset, pred_frame in pred_dict[selected_key].items():
                pred_frames.append(pred_frame)
        while len(pred_frames) < 12:
            pred_frames.append(None)  # Fill missing frames

        # Create 3 columns
        cols = st.columns(3)

        # Groundtruth column
        with cols[0]:
            st.markdown("<h5 style='font-size:14px;'>Groundtruth</h5>", unsafe_allow_html=True)
            for frame in groundtruth_frames:
                if frame is not None:
                    st.image((frame * 255).astype(np.uint8), use_container_width=True)
                else:
                    st.text("Missing Frame")

        # Target column
        with cols[1]:
            st.markdown("<h5 style='font-size:14px;'>Target</h5>", unsafe_allow_html=True)
            for frame in target_frames:
                if frame is not None:
                    st.image((frame * 255).astype(np.uint8), use_container_width=True)
                else:
                    st.text("Missing Frame")

        # Prediction column
        with cols[2]:
            st.markdown("<h5 style='font-size:14px;'>Prediction</h5>", unsafe_allow_html=True)
            for frame in pred_frames:
                if frame is not None:
                    st.image((frame * 255).astype(np.uint8), use_container_width=True)
                else:
                    st.text("Missing Frame")


def show_home_page():
    st.title("Weather Prediction")
    st.write("This is the home page with prediction results and other details.")

    # Display the stored GIF buffer for tab1
    if "tab1_gif" in st.session_state:
        st.image(st.session_state.tab1_gif, caption="Prediction Animation", use_column_width=True)
    else:
        st.write("No GIFs available yet.")



def main():
    sidebar_args = configure_sidebar()

    # Create tabs using st.tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Prediction by Date & Time", "Tab 3"])

    with tab1:
        main_page(sidebar_args)

    with tab2:
        show_prediction_page()


# Dummy dictionaries for ground truth and predictions
groundtruth_dict = {
    "2025-02-06_14:00": np.random.rand(1400, 1200),
    "2025-02-06_14:05": np.random.rand(1400, 1200),
    "2025-02-06_14:10": np.random.rand(1400, 1200),
    "2025-02-06_14:15": np.random.rand(1400, 1200),
    "2025-02-06_14:20": np.random.rand(1400, 1200),
    "2025-02-06_14:25": np.random.rand(1400, 1200),
    "2025-02-06_14:30": np.random.rand(1400, 1200),
    "2025-02-06_14:35": np.random.rand(1400, 1200),
    "2025-02-06_14:40": np.random.rand(1400, 1200),
    "2025-02-06_14:45": np.random.rand(1400, 1200),
    "2025-02-06_14:50": np.random.rand(1400, 1200),
    "2025-02-06_14:55": np.random.rand(1400, 1200),
    "2025-02-06_15:00": np.random.rand(1400, 1200),
    "2025-02-06_15:05": np.random.rand(1400, 1200),
    "2025-02-06_15:10": np.random.rand(1400, 1200),
    "2025-02-06_15:15": np.random.rand(1400, 1200),
    "2025-02-06_15:20": np.random.rand(1400, 1200),
    "2025-02-06_15:25": np.random.rand(1400, 1200),
    "2025-02-06_15:30": np.random.rand(1400, 1200),
    "2025-02-06_15:35": np.random.rand(1400, 1200),
    "2025-02-06_15:40": np.random.rand(1400, 1200),
    "2025-02-06_15:45": np.random.rand(1400, 1200),
    "2025-02-06_15:50": np.random.rand(1400, 1200),
    "2025-02-06_15:55": np.random.rand(1400, 1200),
    "2025-02-06_16:00": np.random.rand(1400, 1200),
}
pred_dict = {
    "2025-02-06_15:00": {
        "+5mins": np.random.rand(1400, 1200),
        "+10mins": np.random.rand(1400, 1200),
        "+15mins": np.random.rand(1400, 1200),
        "+20mins": np.random.rand(1400, 1200),
        "+25mins": np.random.rand(1400, 1200),
        "+30mins": np.random.rand(1400, 1200),
        "+35mins": np.random.rand(1400, 1200),
        "+40mins": np.random.rand(1400, 1200),
        "+45mins": np.random.rand(1400, 1200),
        "+50mins": np.random.rand(1400, 1200),
        "+55mins": np.random.rand(1400, 1200),
        "+60mins": np.random.rand(1400, 1200),
    },
    # Add more date-time entries for predictions
}

if __name__ == "__main__":
    main()
