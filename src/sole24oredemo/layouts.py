import io

import streamlit as st
from datetime import datetime, time, timedelta

from PIL import Image
from sole24oredemo.graphics import generate_metrics_plot
from sole24oredemo.utils import compute_figure_gpd, create_colorbar_fig


def configure_sidebar(model_list):
    with st.sidebar:
        st.image("imgs/LDO_logo_transp.png", use_container_width=True)  # Replace with the path to your logo

        st.markdown("<h1 style='font-size: 32px; font-weight: bold;'>NOWCASTING</h1>", unsafe_allow_html=True)
        with st.form("weather_prediction_form"):
            # Date inputs
            start_date = st.date_input("Select a start date", value=datetime(2025, 1, 31).date(),
                                       format="DD/MM/YYYY")  # TODO: rimettere now
            # Time inputs
            start_time = st.time_input(
                "Select a start time",
                value=time(1, 0),
                step=timedelta(minutes=5)  # 5-minute intervals
            )
            end_date = st.date_input("Select an end date", value=datetime(2025, 1, 31).date(),
                                     format="DD/MM/YYYY")  # TODO: da rimettere now

            end_time = st.time_input(
                "Select an end time",
                value=time(1, 55),
                step=timedelta(minutes=5)  # 5-minute intervals
            )

            # Model selection
            model_name = st.selectbox("Select a model", model_list)

            # Form submission
            submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

        return {"start_date": start_date, "end_date": end_date, "start_time": start_time, "end_time": end_time,
                "model_name": model_name, "submitted": submitted}


def init_prediction_visualization_layout():
    with st.container():
        row1 = st.columns([3, 0.3, 3, 3, 0.4], vertical_alignment='center')
        row2 = st.columns([3, 0.3, 3, 3, 0.4], vertical_alignment='center')
        row3 = st.columns([3, 0.3, 3, 3, 0.4], vertical_alignment='center')

        with row1[0]:
            st.markdown("<h3 style='text-align: center;'>Current Time</h3>", unsafe_allow_html=True)
        with row1[2]:
            st.markdown("<h3 style='text-align: center;'>Groundtruths</h3>", unsafe_allow_html=True)
        with row1[3]:
            st.markdown("<h3 style='text-align: center;'>Predictions</h3>", unsafe_allow_html=True)

        with row2[0]:
            gt_current = st.empty()

        with row3[0]:
            pred_current = st.empty()

        with row2[1]:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                    <div style="transform: rotate(-90deg); font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                        +30min
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with row3[1]:
            st.markdown(
                """
                <div style="position: relative; height: 100%; width: 100%;">
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(
                    270deg);
                    font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                        +60min
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with row2[2]:
            gt_plus_30 = st.empty()
        with row3[2]:
            gt_plus_60 = st.empty()

        with row2[3]:
            pred_plus_30 = st.empty()
        with row3[3]:
            pred_plus_60 = st.empty()

        with row2[4]:
            colorbar30 = st.empty()
        with row3[4]:
            colorbar60 = st.empty()

    return gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60, colorbar30, colorbar60


def precompute_images(frame_dict):
    """
    Precomputes images from frame data and stores them in a list.
    """
    precomputed_images = []
    total_frames = len(frame_dict)

    # Initialize a progress bar
    progress = st.progress(0)
    progress_text = st.empty()

    for idx, (timestamp, frame) in enumerate(frame_dict.items()):
        if frame is not None:
            fig = compute_figure_gpd(frame, timestamp)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            precomputed_images.append((timestamp, Image.open(buf)))
        progress.progress((idx + 1) / total_frames)
        progress_text.text(f"Processing image {idx + 1}/{total_frames}")

    progress.empty()

    return precomputed_images


def init_second_tab_layout(groundtruth_images, target_frames, pred_frames):
    # Define the layout with 5 columns (1 for the label and 4 for images)
    groundtruth_rows = st.columns([0.2] + [1] * 4 + [0.2], vertical_alignment='center')

    # First column spanning all 3 rows (label column)
    with groundtruth_rows[0]:
        st.markdown(
            """
            <div style="position: relative; height: 100%; width: 100%;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(270deg);
                font-weight: bold; font-size: 1.5em; white-space: nowrap;">
                    Groundtruths
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with groundtruth_rows[-1]:
        st.image(create_colorbar_fig())

    # Loop through the 4 columns for the images
    for row_idx in range(3):  # Loop through rows 1 to 3
        row_offset = row_idx * 4  # 4 frames per row
        for col_idx in range(1, 5):  # Columns 1 to 4 (skip index 0)
            with groundtruth_rows[col_idx]:
                timestamp_idx = col_idx - 1 + row_offset
                if timestamp_idx < len(groundtruth_images):
                    timestamp, image = groundtruth_images[timestamp_idx]
                    st.image(image, caption=timestamp, use_container_width=True)

    # Additional layout for TARGET and PREDICTION columns
    target_pred_rows = []

    for i in range(13):
        target_pred_rows.append(st.columns([0.5, 0.2, 1.5, 1.5, 0.2, 0.5], vertical_alignment='center'))

    # Titles for TARGET and PREDICTION
    with target_pred_rows[0][2]:
        st.markdown(
            """<div style="text-align: center; font-weight: bold; font-size: 2em;">Target</div>""",
            unsafe_allow_html=True,
        )

    with target_pred_rows[0][3]:
        st.markdown(
            """<div style="text-align: center; font-weight: bold; font-size: 2em;">Prediction</div>""",
            unsafe_allow_html=True,
        )

    # Fill TARGET and PREDICTION frames row by row
    for row_idx in range(1, 13):  # Skip the title row
        # Left empty column with +Xmins labels
        with target_pred_rows[row_idx][1]:
            st.markdown(
                f"""<div style="position: relative; height: 100%; width: 100%;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%) rotate(270deg);
                font-size: 1em; font-weight: bold;">+{row_idx * 5}mins</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # TARGET frames
        with target_pred_rows[row_idx][2]:
            if row_idx - 1 < len(target_frames):
                timestamp = list(target_frames.keys())[row_idx - 1]
                frame = target_frames.get(timestamp, None)
                if frame is not None:
                    fig = compute_figure_gpd(frame, timestamp)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    image = Image.open(buf)
                    st.image(image, use_container_width=True)

        # PREDICTION frames
        with target_pred_rows[row_idx][3]:
            if row_idx - 1 < len(pred_frames):
                timestamp = list(pred_frames.keys())[row_idx - 1]
                frame = pred_frames.get(timestamp, None)
                if frame is not None:
                    fig = compute_figure_gpd(frame, 'PRED @ ' + timestamp)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    image = Image.open(buf)
                    st.image(image, use_container_width=True)

        with target_pred_rows[row_idx][-2]:
            st.image(create_colorbar_fig(top_adj=0.85, bot_adj=0.07))

    return


def show_metrics_page(config):
    model_list = config.get("models", [])
    # Select time
    selected_date = st.date_input("Select date", value=datetime(2025, 1, 31).date(),
                                  format="DD/MM/YYYY")  # TODO: rimettere now

    # Select date
    selected_time = st.time_input(
        "Select time",
        value=time(1, 0),
        step=timedelta(minutes=5)  # 5-minute intervals
    )

    # Initialize session state for selected models and plots
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = []
    if "plotted_metrics" not in st.session_state:
        st.session_state["plotted_metrics"] = []

    # Display checkboxes for all models
    st.subheader("Select Models to Display Data")

    selected_models = []
    num_columns = 5  # Adjust the number of models per row here

    # Create rows of checkboxes
    for i in range(0, len(model_list), num_columns):
        cols = st.columns(num_columns)
        for col, model in zip(cols, model_list[i:i + num_columns]):
            with col:
                if st.checkbox(model, value=model in st.session_state["selected_models"]):
                    selected_models.append(model)

    st.session_state["selected_models"] = selected_models

    # Button to generate the plot
    if st.button("Generate Plot"):
        if selected_models:

            empty_space = st.empty()
            with empty_space.container():
                with st.status(f':hammer_and_wrench: **Loading results...**', expanded=True) as status:
                    plotted_metrics = generate_metrics_plot(selected_date, selected_time, selected_models, config)
                    # status.update(label=f"Done!", state="complete", expanded=True)
                    st.session_state["plotted_metrics"] = plotted_metrics
            empty_space.empty()

        else:
            st.warning("Please select at least one model.")

    # Display the formula and plots if they exist in session state
    if st.session_state["plotted_metrics"]:
        with st.status(f'Done!', state='complete', expanded=True) as status:
            columns = st.columns([0.5, 0.05, 0.3])
            with columns[2]:
                st.markdown(r"""
                    ### CSI Formula
                    The Critical Success Index (CSI) is calculated as:
    
                    ### $$CSI = \frac{TP}{TP + FP + FN}$$
    
                    Where:
                    - **TP** is the True Positives
                    - **FP** is the False Positives
                    - **FN** is the False Negatives
                """, unsafe_allow_html=True)

            with columns[0]:
                for i, plot_buffer in enumerate(st.session_state["plotted_metrics"]):
                    st.image(plot_buffer)