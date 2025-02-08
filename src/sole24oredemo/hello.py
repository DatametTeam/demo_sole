import io
import os
import streamlit as st
from PIL import Image
from pathlib import Path
import time
import numpy as np
from layouts import configure_sidebar, init_prediction_visualization_layout, init_second_tab_layout
from pbs import is_pbs_available

from sole24oredemo.parallel_code import create_fig_dict_in_parallel
from sole24oredemo.utils import check_if_gif_present, load_gif_as_bytesio, create_colorbar_fig, \
    get_closest_5_minute_time, read_groundtruth_and_target_data
import imageio
from datetime import datetime, time, timedelta
from multiprocessing import Manager, Process
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")

if is_pbs_available():
    from pbs import submit_inference, get_job_status
else:
    from mock import inference_mock as submit_inference, get_job_status


def create_single_gif_for_parallel(queue, start_pos, figures_dict, window_size, sorted_keys, process_idx,
                                   save_on_disk=True, fps_gif=3):
    """
    Create a single GIF and send progress updates through a queue.
    """
    buf = io.BytesIO()
    frames = []
    print(f"{start_pos} / {start_pos + window_size}")
    window_keys = sorted_keys[start_pos:start_pos + window_size]
    print(window_keys)

    for i, key in enumerate(window_keys):
        fig = figures_dict[key]
        buf_tracked = io.BytesIO()
        fig.savefig(buf_tracked, format='png', bbox_inches='tight', pad_inches=0)
        buf_tracked.seek(0)
        img = Image.open(buf_tracked)
        frames.append(np.array(img))

        # Send progress update through queue
        progress = (i + 1) / len(window_keys)
        queue.put(('progress', process_idx, progress))

    imageio.mimsave(buf, frames, format='GIF', fps=fps_gif, loop=0)
    buf.seek(0)
    # Send completed GIF through queue
    queue.put(('complete', process_idx, buf.getvalue()))

    if save_on_disk:
        save_path = "/archive/SSD/home/guidim/demo_sole/data/output/gifs/gt"
        file_name = f"{window_keys[0]}_{window_keys[-1]}"
        save_path = os.path.join(save_path, file_name + '.gif')
        imageio.mimsave(save_path, frames, format='GIF', fps=fps_gif, loop=0)
        print(f"GIF save @ path {save_path}")


def create_sliding_window_gifs(figures_dict, progress_placeholder, start_positions=[0, 6, 12], save_on_disk=True,
                               fps_gif=3):
    """
    Create multiple GIFs in parallel with progress tracking.
    """
    window_size = len(figures_dict) - 12
    sorted_keys = sorted(figures_dict.keys())

    # Create progress bars and text containers
    progress_bars = []
    progress_texts = []

    with st.container():
        for i in range(len(start_positions)):
            progress_bars.append(st.progress(0))
            progress_texts.append(st.empty())

    # Setup multiprocessing queue and processes
    queue = Manager().Queue()
    processes = []
    for idx, start_pos in enumerate(start_positions):
        p = Process(
            target=create_single_gif_for_parallel,
            args=(queue, start_pos, figures_dict, window_size, sorted_keys, idx, save_on_disk, fps_gif)
        )
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()

    # Initialize storage for completed GIFs
    completed_gifs = [None] * len(start_positions)
    completed_count = 0

    # Monitor queue for updates
    while completed_count < len(start_positions):
        try:
            msg_type, idx, data = queue.get(timeout=1)

            if msg_type == 'progress':
                total_steps = window_size
                current_step = int(data * total_steps)
                progress_bars[idx].progress(data)
                progress_texts[idx].text(f"GIF {idx + 1}: {current_step}/{total_steps} complete")

            elif msg_type == 'complete':
                completed_gifs[idx] = io.BytesIO(data)
                completed_count += 1
                progress_texts[idx].text(f"GIF {idx + 1}: Complete!")

        except:
            # Check if all processes are still alive
            if not any(p.is_alive() for p in processes):
                break

    # Clean up processes
    for p in processes:
        p.join()

    # Clear progress indicators
    for bar in progress_bars:
        bar.empty()
    for text in progress_texts:
        text.empty()

    return completed_gifs


def create_sliding_window_gifs_for_predictions(prediction_dict, progress_placeholder, save_on_disk=True, fps_gif=3):
    """
    Create GIFs for the predictions dictionary. Each GIF corresponds to:
    - +30 mins: Figures from the "+30mins" key in the nested dictionary.
    - +60 mins: Figures from the "+60mins" key in the nested dictionary.

    Args:
        prediction_dict: Dictionary where each key is a timestamp, and the value is a nested dictionary
                         with "+30mins" and "+60mins" keys containing figures.
        progress_placeholder: Streamlit placeholder for progress updates.

    Returns:
        Tuple of BytesIO buffers containing the two GIFs (+30 mins, +60 mins).
    """

    def create_single_gif(queue, figures, gif_type, process_idx, start_key, end_key, fps_gif=3, save_on_disk=True):
        buf = io.BytesIO()
        frames = []

        for idx, fig in enumerate(figures):
            try:
                buf_tracked = io.BytesIO()
                fig.savefig(buf_tracked, format='png', bbox_inches='tight', pad_inches=0)
                buf_tracked.seek(0)
                img = Image.open(buf_tracked)
                frames.append(np.array(img))

                # Update progress
                progress = (idx + 1) / len(figures)
                queue.put(('progress', process_idx, progress))

            except Exception as e:
                print(f"Error processing figure for {gif_type}, index {idx}: {e}")
                continue

        if not frames:
            raise ValueError(f"No frames generated for {gif_type}.")

        # Create the GIF
        imageio.mimsave(buf, frames, format='GIF', fps=fps_gif, loop=0)
        buf.seek(0)
        queue.put(('complete', process_idx, buf.getvalue()))

        if save_on_disk:
            save_path = "/archive/SSD/home/guidim/demo_sole/data/output/gifs/pred"
            file_name = f"{start_key}_{end_key}_{gif_type}"
            save_path = os.path.join(save_path, file_name + '.gif')
            imageio.mimsave(save_path, frames, format='GIF', fps=fps_gif, loop=0)
            print(f"GIF save @ path {save_path}")

    sorted_keys = sorted(prediction_dict.keys())

    # Select all except the last 12 keys
    keys_except_last_12 = sorted_keys[:-12]

    # Filter the dictionary
    prediction_dict = {key: prediction_dict[key] for key in keys_except_last_12}

    # Extract figures for +30 mins and +60 mins
    figures_30 = [nested_dict["+30min"] for nested_dict in prediction_dict.values() if "+30min" in nested_dict]
    figures_60 = [nested_dict["+60min"] for nested_dict in prediction_dict.values() if "+60min" in nested_dict]
    start_key = keys_except_last_12[0]
    end_key = keys_except_last_12[-1]

    if not figures_30 or not figures_60:
        raise ValueError("Insufficient figures for +30 min or +60 min.")

    # Create progress bars and placeholders
    progress_bars = []
    progress_texts = []

    with st.container():
        for _ in range(2):
            progress_bars.append(st.progress(0))
            progress_texts.append(st.empty())

    # Setup multiprocessing queue and processes
    queue = Manager().Queue()
    processes = []

    for idx, (figures, gif_type) in enumerate([(figures_30, "+30 mins"), (figures_60, "+60 mins")]):
        p = Process(target=create_single_gif,
                    args=(queue, figures, gif_type, idx, start_key, end_key, fps_gif, save_on_disk))
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()

    # Initialize storage for completed GIFs
    completed_gifs = [None] * len(processes)
    completed_count = 0

    # Monitor queue for updates
    while completed_count < len(processes):
        try:
            msg_type, idx, data = queue.get(timeout=1)

            if msg_type == 'progress':
                progress_bars[idx].progress(data)
                total_steps = len(keys_except_last_12)
                current_step = int(data * total_steps)
                progress_bars[idx].progress(data)
                progress_texts[idx].text(f"GIF {idx + 1}: {current_step}/{total_steps} complete")

            elif msg_type == 'complete':
                completed_gifs[idx] = io.BytesIO(data)
                completed_count += 1
                progress_texts[idx].text(f"GIF {idx + 1}: Complete!")

        except:
            # Check if all processes are still alive
            if not any(p.is_alive() for p in processes):
                break

    # Clean up processes
    for p in processes:
        p.join()

    # Clear progress indicators
    for bar in progress_bars:
        bar.empty()
    for text in progress_texts:
        text.empty()

    return completed_gifs


def update_prediction_visualization(gt0_gif, gt6_gif, gt12_gif, pred_gif_6, pred_gif_12):
    gt_current, pred_current, gt_plus_30, pred_plus_30, gt_plus_60, pred_plus_60, colorbar30, colorbar60 = \
        init_prediction_visualization_layout()
    # Display the GIF using Streamlit
    gt_current.image(gt0_gif, caption="Current data", use_container_width=True)
    pred_current.image(gt0_gif, caption="Current data", use_container_width=True)
    gt_plus_30.image(gt6_gif, caption="Data +30 minutes", use_container_width=True)
    pred_plus_30.image(pred_gif_6, caption="Prediction +30 minutes", use_container_width=True)
    gt_plus_60.image(gt12_gif, caption="Data +60 minutes", use_container_width=True)
    pred_plus_60.image(pred_gif_12, caption="Prediction +60 minutes", use_container_width=True)
    colorbar30.image(create_colorbar_fig(top_adj=0.96, bot_adj=0.12))
    colorbar60.image(create_colorbar_fig(top_adj=0.96, bot_adj=0.12))


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

    gt_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:24]
    gt_array = np.array(gt_array)
    gt_array[gt_array < 0] = 0
    gt_array[gt_array > 200] = 200
    # gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    pred_array = np.load(out_dir / "data.npy", mmap_mode='r')[0:24]
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    pred_array[pred_array > 200] = 200
    # pred_array = (pred_array - np.min(pred_array)) / (np.max(pred_array) - np.min(pred_array))
    print("*** LOADED DATA ***")

    return gt_array, pred_array


def compute_prediction_results(sidebar_args):
    error, out_dir = submit_prediction_job(sidebar_args)
    if not error:
        with st.status(f':hammer_and_wrench: **Loading results...**', expanded=True) as status:

            prediction_placeholder = st.empty()
            progress_placeholder = st.empty()  # Add this line for progress bar

            with prediction_placeholder:
                status.update(label="🔄 Loading results...", state="running", expanded=True)

                gt_gif_ok, pred_gif_ok, gt_paths, pred_paths = check_if_gif_present(sidebar_args)
                if gt_gif_ok:
                    gt_gifs = load_gif_as_bytesio(gt_paths)

                gt_array, pred_array = get_prediction_results(out_dir)

                status.update(label="🔄 Creating dictionaries...", state="running", expanded=True)
                gt_dict, pred_dict = create_fig_dict_in_parallel(gt_array, pred_array)

                if not gt_gif_ok:
                    status.update(label="🔄 Creating GT GIFs...", state="running", expanded=True)
                    gt_gifs = create_sliding_window_gifs(gt_dict, progress_placeholder, fps_gif=3,
                                                         save_on_disk=True)

                status.update(label="🔄 Creating Pred GIFs...", state="running", expanded=True)
                pred_gifs = create_sliding_window_gifs_for_predictions(pred_dict, progress_placeholder,
                                                                       fps_gif=3, save_on_disk=True)

                gt0_gif = gt_gifs[0]  # Full sequence
                gt_gif_6 = gt_gifs[1]  # Starts from frame 6
                gt_gif_12 = gt_gifs[2]  # Starts from frame 12
                pred_gif_6 = pred_gifs[0]
                pred_gif_12 = pred_gifs[1]

                # Store results in session state
                st.session_state.prediction_result = {
                    'gt0_gif': gt0_gif,
                    'gt6_gif': gt_gif_6,
                    'gt12_gif': gt_gif_12,
                    'pred6_gif': pred_gif_6,
                    'pred12_gif': pred_gif_12,
                }
                st.session_state.tab1_gif = gt0_gif.getvalue()

                status.update(label=f"Done!", state="complete", expanded=True)
                update_prediction_visualization(gt0_gif, gt_gif_6, gt_gif_12, pred_gif_6, pred_gif_12)
                display_results(gt_gifs, pred_gifs)
    else:
        st.error(error)


def display_results(gt_gifs, pred_gifs):
    gt0_gif = gt_gifs[0]  # Full sequence
    gt_gif_6 = gt_gifs[1]  # Starts from frame 6
    gt_gif_12 = gt_gifs[2]  # Starts from frame 12
    pred_gif_6 = pred_gifs[0]
    pred_gif_12 = pred_gifs[1]

    # Store results in session state
    st.session_state.prediction_result = {
        'gt0_gif': gt0_gif,
        'gt6_gif': gt_gif_6,
        'gt12_gif': gt_gif_12,
        'pred6_gif': pred_gif_6,
        'pred12_gif': pred_gif_12,
    }
    st.session_state.tab1_gif = gt0_gif.getvalue()
    update_prediction_visualization(gt0_gif, gt_gif_6, gt_gif_12, pred_gif_6, pred_gif_12)


def main_page(sidebar_args) -> None:
    # Only run prediction if not already done
    if 'prediction_result' not in st.session_state or st.session_state.prediction_result == {}:
        if 'submitted' not in st.session_state:
            submitted = sidebar_args['submitted']
            if submitted:
                st.session_state.submitted = True
        if 'submitted' in st.session_state and st.session_state.submitted:

            gt_gif_ok, pred_gif_ok, gt_paths, pred_paths = check_if_gif_present(sidebar_args)

            if gt_gif_ok and pred_gif_ok:
                st.warning("Prediction data already present. Do you want to recompute?")
                col1, _, col2, _ = st.columns([1, 0.5, 1, 3])  # Ensure both buttons take half the page
                with col1:
                    compute_ok = False
                    if st.button("YES", use_container_width=True):
                        compute_ok = True
                if compute_ok:
                    compute_prediction_results(sidebar_args)

                with col2:
                    compute_nok = False
                    if st.button("NO", use_container_width=True):
                        compute_nok = True
                if compute_nok:
                    gt_gifs = load_gif_as_bytesio(gt_paths)
                    pred_gifs = load_gif_as_bytesio(pred_paths)
                    display_results(gt_gifs, pred_gifs)

            else:
                compute_prediction_results(sidebar_args)
            return
    else:
        # If prediction results already exist, reuse them
        gt0_gif = st.session_state.prediction_result['gt0_gif']
        gt_gif_6 = st.session_state.prediction_result['gt6_gif']
        gt_gif_12 = st.session_state.prediction_result['gt12_gif']
        pred_gif_6 = st.session_state.prediction_result['pred6_gif']
        pred_gif_12 = st.session_state.prediction_result['pred12_gif']
        update_prediction_visualization(gt0_gif, gt_gif_6, gt_gif_12, pred_gif_6, pred_gif_12)


def show_prediction_page():
    st.title("Select Date and Time for Prediction")

    # Date and time selection
    selected_date = st.date_input(
        "Select Date", min_value=datetime(2020, 1, 1).date(), max_value=datetime.today().date(), format="DD/MM/YYYY",
        value=datetime(2025, 2, 6).date())
    selected_time = st.time_input("Select Time", value=time(15, 00))  # get_closest_5_minute_time(), s

    if st.button("Submit"):
        # Combine selected date and time
        selected_datetime = datetime.combine(selected_date, selected_time)
        selected_key = selected_datetime.strftime("%d%m%Y_%H%M")

        groundtruth_dict, target_dict = read_groundtruth_and_target_data(selected_key)

        init_second_tab_layout(groundtruth_dict, target_dict, target_dict)


def show_home_page():
    st.title("Weather Prediction")
    st.write("This is the home page with prediction results and other details.")

    # Display the stored GIF buffer for tab1
    if "tab1_gif" in st.session_state:
        st.image(st.session_state.tab1_gif, caption="Prediction Animation", use_column_width=True)
    else:
        st.write("No GIFs available yet.")


def show_real_time_prediction():
    map = folium.Map(location=[45.0, 7.0], zoom_start=10)  # Specify a location and zoom level

    # Add a marker for demonstration purposes
    folium.Marker(
        location=[45.0, 7.0],
        popup="Demo Location",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(map)

    # Display the map in Streamlit
    st_map = st_folium(map, width=700, height=500)


def main():
    sidebar_args = configure_sidebar()
    if sidebar_args['submitted'] and 'prediction_result' in st.session_state:
        st.session_state.prediction_result = {}

    # Create tabs using st.tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Prediction by Date & Time", "Real Time Prediction"])

    with tab1:
        main_page(sidebar_args)

    with tab2:
        show_prediction_page()
    
    with tab3:
        show_real_time_prediction()


if __name__ == "__main__":
    print("***NEWRUN***")
    main()
