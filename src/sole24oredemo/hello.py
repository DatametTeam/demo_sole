import os
import threading
import h5py
import pyproj
import streamlit as st
from streamlit.components.v1 import html
from pathlib import Path
import time
import numpy as np
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx, add_script_run_ctx

from layouts import configure_sidebar, init_prediction_visualization_layout, init_second_tab_layout, \
    precompute_images, \
    show_metrics_page, display_map_layout
from pbs import is_pbs_available

from sole24oredemo.parallel_code import create_fig_dict_in_parallel, create_sliding_window_gifs, \
    create_sliding_window_gifs_for_predictions
from sole24oredemo.sou_py import dpg
from sole24oredemo.utils import check_if_gif_present, load_gif_as_bytesio, create_colorbar_fig, \
    get_closest_5_minute_time, read_groundtruth_and_target_data, lincol_2_yx, yx_2_latlon, cmap, norm, load_config, \
    get_latest_file, load_prediction_data, launch_thread_execution
from datetime import time as dt_time
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from threading import Event
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Weather prediction", page_icon=":flag-eu:", layout="wide")

# if is_pbs_available():
from pbs import submit_inference, get_job_status


# else:
#     from mock import inference_mock as submit_inference, get_job_status


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


def get_prediction_results(out_dir, sidebar_args, get_only_pred=False):
    # TODO: da fixare
    model_name = sidebar_args['model_name']
    pred_out_dir = Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/Test")
    model_out_dir = Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/{model_name}")
    gt_array = None

    if not get_only_pred:
        print("Loading GT data")
        gt_array = np.load(pred_out_dir / "predictions.npy", mmap_mode='r')[12:36]
        print("GT data loaded")
        gt_array = np.array(gt_array)
        gt_array[gt_array < 0] = 0
        # gt_array[gt_array > 200] = 200
        # gt_array = (gt_array - np.min(gt_array)) / (np.max(gt_array) - np.min(gt_array))

    print("Loading pred data")
    pred_array = np.load(model_out_dir / "predictions.npy", mmap_mode='r')[0:24]
    if model_name == 'Test':  # TODO: sistemare
        pred_array = np.load(model_out_dir / "predictions.npy", mmap_mode='r')[12:36]
    pred_array = np.array(pred_array)
    pred_array[pred_array < 0] = 0
    print("Loaded pred data")
    # pred_array[pred_array > 200] = 200
    # pred_array = (pred_array - np.min(pred_array)) / (np.max(pred_array) - np.min(pred_array))
    print("Loading radar mask")
    with h5py.File("src/mask/radar_mask.hdf", "r") as f:
        radar_mask = f["mask"][()]
    print("Radar mask loaded")

    pred_array = pred_array * radar_mask

    print("*** LOADED DATA ***")

    return gt_array, pred_array


def compute_prediction_results(sidebar_args):
    error, out_dir = submit_prediction_job(sidebar_args)
    if not error:
        with st.status(f':hammer_and_wrench: **Loading results...**', expanded=True) as status:

            prediction_placeholder = st.empty()

            with prediction_placeholder:
                status.update(label="🔄 Loading results...", state="running", expanded=True)

                gt_gif_ok, pred_gif_ok, gt_paths, pred_paths = check_if_gif_present(sidebar_args)
                if gt_gif_ok:
                    gt_gifs = load_gif_as_bytesio(gt_paths)

                gt_array, pred_array = get_prediction_results(out_dir, sidebar_args)

                status.update(label="🔄 Creating dictionaries...", state="running", expanded=True)
                gt_dict, pred_dict = create_fig_dict_in_parallel(gt_array, pred_array, sidebar_args)

                if not gt_gif_ok:
                    status.update(label="🔄 Creating GT GIFs...", state="running", expanded=True)
                    gt_gifs = create_sliding_window_gifs(gt_dict, sidebar_args, fps_gif=3,
                                                         save_on_disk=True)

                status.update(label="🔄 Creating Pred GIFs...", state="running", expanded=True)
                pred_gifs = create_sliding_window_gifs_for_predictions(pred_dict, sidebar_args,
                                                                       fps_gif=3, save_on_disk=True)

                status.update(label=f"Done!", state="complete", expanded=True)
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
    selected_time = st.time_input("Select Time", value=dt_time(15, 00))  # get_closest_5_minute_time(), s
    selected_model = st.selectbox("Select model", model_list)

    if st.button("Submit"):
        # Combine selected date and time
        selected_datetime = datetime.combine(selected_date, selected_time)
        prediction_start_datetime = selected_datetime - timedelta(hours=1)
        selected_key = prediction_start_datetime.strftime("%d%m%Y_%H%M")

        args = {'start_date': selected_datetime, 'start_time': prediction_start_datetime, 'model_name': selected_model,
                'submitted': True}

        submit_prediction_job(args)

        # Check if groundtruths are already in session state
        if selected_key not in st.session_state:
            groundtruth_dict, target_dict, pred_dict = read_groundtruth_and_target_data(selected_key, selected_model)

            # Precompute and cache images for groundtruths
            st.session_state[selected_key] = {
                "groundtruths": precompute_images(groundtruth_dict),
                "target_dict": target_dict,
                "pred_dict": pred_dict,
            }
        else:
            # If groundtruths exist, just update the target and prediction dictionaries for the new model
            _, target_dict, pred_dict = read_groundtruth_and_target_data(selected_key, selected_model)
            st.session_state[selected_key]["target_dict"] = target_dict
            st.session_state[selected_key]["pred_dict"] = pred_dict

        # Use cached groundtruths, targets, and predictions
        groundtruth_images = st.session_state[selected_key]["groundtruths"]
        target_dict = st.session_state[selected_key]["target_dict"]
        pred_dict = st.session_state[selected_key]["pred_dict"]

        # Initialize the second tab layout with precomputed images
        init_second_tab_layout(groundtruth_images, target_dict, pred_dict)


def show_home_page():
    st.title("Weather Prediction")
    st.write("This is the home page with prediction results and other details.")

    # Display the stored GIF buffer for tab1
    if "tab1_gif" in st.session_state:
        st.image(st.session_state.tab1_gif, caption="Prediction Animation", use_column_width=True)
    else:
        st.write("No GIFs available yet.")


def thread_for_position():
    thread_id = threading.get_ident()
    print(f"Worker thread (ID: {thread_id}) is starting...")
    while True:
        if "st_map" in st.session_state:
            st_map = st.session_state["st_map"]
            if 'center' in st_map.keys() and 'zoom' in st_map.keys():
                # print("THREAD - " + str(st_map['center']) + " -- " + str(st_map['zoom']))
                st.session_state["center"] = st_map['center']
                st.session_state["zoom"] = st_map['zoom']
            else:
                # print("THREAD - center / zoom not available..")
                pass
        else:
            # print("THREAD - st_map not available..")
            pass
        time.sleep(0.4)


def initial_state_management():
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'selected_time' not in st.session_state:
        st.session_state.selected_time = None
    if 'latest_file' not in st.session_state:
        st.session_state.latest_file = None
    if 'rgba_image' not in st.session_state:
        st.session_state.rgba_image = None
    if 'thread_started' not in st.session_state:
        st.session_state.thread_started = None
    if 'old_count' not in st.session_state:
        st.session_state.old_count = COUNT


def map_state_initialization():
    if "last_map_update" not in st.session_state:
        st.session_state["last_map_update"] = None
        st.session_state["last_map_update"] = datetime.now().second
        get_latest_file(SRI_FOLDER_DIR, None)

        if "new_update" not in st.session_state:
            st.session_state["new_update"] = None
        st.session_state["new_update"] = True


def create_only_map(rgba_img, prediction: bool = False):
    if st.session_state.selected_model and st.session_state.selected_time:
        if "new_prediction" in st.session_state and st.session_state["new_prediction"]:
            # 3 --> nuova predizione da caricare, si aggiorna il centro
            center = st.session_state["center"]
            zoom = st.session_state["zoom"]

            st.session_state["old_center"] = center
            st.session_state["old_zoom"] = zoom

            st.session_state["new_prediction"] = False
        elif "old_center" in st.session_state and "old_zoom" in st.session_state:
            center = st.session_state["old_center"]
            zoom = st.session_state["old_zoom"]
        elif "center" in st.session_state and "zoom" in st.session_state:
            # 1 --> direttamente sull'overlay
            center = st.session_state["center"]
            zoom = st.session_state["zoom"]

            # 2 --> salvataggio come valori precedenti
            st.session_state["old_center"] = center
            st.session_state["old_zoom"] = zoom
    else:
        center = {'lat': 42.5, 'lng': 12.5}
        zoom = 5

    map = folium.Map(location=[center['lat'], center['lng']],
                     zoom_start=zoom,
                     control_scale=False,  # Disable control scale
                     tiles='Esri.WorldGrayCanvas',  # Watercolor map style
                     name="WorldGray",
                     )
    folium.TileLayer(
        tiles='Esri.WorldImagery',  # Satellite imagery
        name="Satellite",
        control=True
    ).add_to(map)

    folium.TileLayer(
        tiles='OpenStreetMap.Mapnik',  # Satellite imagery
        name="OSM",
        control=True
    ).add_to(map)

    if prediction:
        # ricreazione totale della mappa + predizione
        folium.raster_layers.ImageOverlay(
            image=rgba_img,
            bounds=[[35.0623, 4.51987], [47.5730, 20.4801]],
            mercator_project=False,
            origin="lower",
            name="NWC_pred"
            # opacity=0.5
        ).add_to(map)

        data_min = 0  # Minimum value in your data
        data_max = 100  # Maximum value in your data

        data_values = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100]
        normalized_values = norm(data_values)

        colormap = cm.LinearColormap(
            colors=[cmap(n) for n in normalized_values],  # Generate 10 colors
            index=data_values,  # Map to actual data values
            vmin=data_min,
            vmax=data_max
        )

        colormap.caption = "Prediction Intensity (mm/h)"
        map.add_child(colormap)

        st.session_state["new_update"] = False

    folium.LayerControl().add_to(map)
    st_map = st_folium(map, width=800, height=600, use_container_width=True)
    st.session_state["st_map"] = st_map
    if "center" in st_map.keys():
        st.session_state["center"] = st_map["center"]
        st.session_state["zoom"] = st_map["zoom"]


def show_real_time_prediction():
    time_for_reloading_data = 55

    # Initial state management
    initial_state_management()

    model_options = model_list
    time_options = ["+5min", "+10min", "+15min", "+20min", "+25min",
                    "+30min", "+35min", "+40min", "+45min", "+50min",
                    "+55min", "+60min"]

    columns = st.columns([0.5, 0.5])
    with (columns[0]):
        internal_columns = st.columns([0.3, 0.1, 0.3])
        with internal_columns[0]:
            # Select model, bound to session state
            st.selectbox(
                "Select a model",
                options=model_options,
                key="selected_model"
            )

        with internal_columns[2]:
            # Select time, bound to session state
            st.selectbox(
                "Select a prediction time",
                options=time_options,
                key="selected_time",
            )

        # questa cosa è chiamata solo la prima volta che viene eseguito il codice
        map_state_initialization()

        # qua serve:
        # -- thread che controlla l'aggiornamento dei dati
        # -- thread che calcola le previsioni quando ci sono dati nuovi

        now = datetime.now().second

        print("----------------------------------------------------")
        diff = now - st.session_state["last_map_update"] if now >= st.session_state["last_map_update"] else (
                60 - st.session_state["last_map_update"] + now)
        print(now)
        print(st.session_state["last_map_update"])
        print("---> " + str(diff))
        print("----------------------------------------------------")

        # ogni tot secondi è possibile verificare se esiste un nuovo aggiornamento dei dati di input
        # in questa versione basta refreshare la pagina (interagire con la mappa)
        if diff >= time_for_reloading_data:
            print("OBTAINING NEW INPUT FILE VERSION..")
            st.session_state["last_map_update"] = datetime.now().second

            # posso lanciare questa roba su un thread e fare in modo che se l'applicazione refresha il thread continua
            obtain_input_ev = threading.Event()

            ctx = get_script_run_ctx()
            obtain_input_th = threading.Thread(target=get_latest_file, args=(SRI_FOLDER_DIR, obtain_input_ev))
            add_script_run_ctx(obtain_input_th, ctx)
            obtain_input_th.start()
            obtain_input_ev.wait()

            print("Input data received with event!")

            st.session_state["new_update"] = True

        st.markdown("<div style='text-align: center; font-size: 18px;'>"
                    f"<b>Current Date: {st.session_state.latest_file}</b>"
                    "</div>",
                    unsafe_allow_html=True)

        # devo simulare questa cosa randomica
        # praticamente ogni tot secondi devo fare in modo che un file nuovo venga messo nella cartella dei latest_files
        # in modo che la modifica sia rilevata
        latest_file = st.session_state["latest_"]
        if latest_file != st.session_state.latest_file:
            # calcolo della previsione
            print("LAUNCH PREDICTION..")
            with columns[1]:
                with st.status(f':hammer_and_wrench: **Launch prediction...**', expanded=True) as status:
                    launch_thread_execution(st, latest_file, columns)
                status.update(label="✅ Prediction completed!", state="complete", expanded=False)
                st.session_state.latest_file = latest_file
            st.session_state.selection = None
            st.session_state["new_prediction"] = True
        else:
            print(f"Current SRI == Latest file processed! {latest_file}. Skipped prediction")

            with columns[1]:
                st.write("")
                st.write("")
                st.status(label="✅ Using latest data available", state="complete", expanded=False)

        if st.session_state.selected_model and st.session_state.selected_time:
            with columns[1]:
                with st.status(f':hammer_and_wrench: **Loading prediction...**', expanded=True) as status:
                    print("LOAD PREDICTION DATA..")

                    # problema di caricamento lungo risolto con la cache e ttl
                    rgba_img = load_prediction_data(st, time_options, latest_file)

                status.update(label="✅ Loading completed!", state="complete", expanded=False)
            create_only_map(rgba_img, prediction=True)
        else:
            create_only_map(None)


def main(model_list):
    sidebar_args = configure_sidebar(model_list)
    if sidebar_args['submitted'] and 'prediction_result' in st.session_state:
        st.session_state.prediction_result = {}

    # Create tabs using st.tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Nowcasting", "Prediction by Date & Time", "Metrics", "Real Time Prediction"])

    with tab1:
        main_page(sidebar_args)

    with tab2:
        show_prediction_page()

    with tab3:
        show_metrics_page(config)

    with tab4:
        show_real_time_prediction()


# Initial auto-refresh interval (in seconds)
seconds_for_autorefresh = 10000
COUNT = st_autorefresh(interval=seconds_for_autorefresh * 1000)


# Function to monitor time and adjust the refresh interval
def monitor_time():
    print("Starting Monitor time thread")
    global seconds_for_autorefresh, COUNT
    while True:
        now = datetime.now()
        # Check if the current minute is a multiple of 5 and seconds are close to 0
        if now.minute % 5 == 0 and now.second < 5:
            print(f"Time is {now}! Restarting app to force new prediction")
            time.sleep(5)
            st.rerun()

        time.sleep(2)  # Check every second


# Initialize the thread only once using session state
if "autorefresh_thread_started" not in st.session_state:
    st.session_state["autorefresh_thread_started"] = False

if not st.session_state["autorefresh_thread_started"]:
    # thread = threading.Thread(target=monitor_time, daemon=True)
    # thread.start()
    st.session_state["autorefresh_thread_started"] = True

src_dir = Path(__file__).resolve().parent.parent
config = load_config(os.path.join(src_dir, "sole24oredemo/cfg/cfg.yaml"))
model_list = config.get("models", [])

# tampone locale, da non pushare!
root_dir = src_dir.parent
SRI_FOLDER_DIR = str(os.path.join(root_dir, "SRI_adj"))

if __name__ == "__main__":
    print(f"***NEWRUN @ {datetime.now()}***")
    # print(st.session_state)

    main(model_list)
