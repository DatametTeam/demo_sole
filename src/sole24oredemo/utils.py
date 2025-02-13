import io
import os
import threading
import time
from pathlib import Path

import h5py
import pyproj
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.basemap import Basemap
from numba import njit
import warnings
import geopandas as gpd
from datetime import datetime, timedelta
import yaml
import sou_py.dpg as dpg

ROOT_PATH = Path(__file__).parent.parent.absolute()

par = np.array([600., 1000., 650., -1000.])
lat_0 = 42.0
lon_0 = 12.5
map_ = pyproj.Proj({"proj": 'tmerc', "lat_0": lat_0, "lon_0": lon_0})
SHAPEFILE_FOLDER = ROOT_PATH / "shapefiles"

warnings.filterwarnings('ignore', category=UserWarning,
                        message='The input coordinates to pcolormesh are interpreted as cell centers.*')


def compute_figure_gpd(img1, timestamp):
    global x, y
    # gdf = gdf.to_crs(crs="EPSG:4326")
    fig, ax = plt.subplots(figsize=(10, 10))
    italy_shape.plot(ax=ax, edgecolor='black', color='white')
    mesh = ax.pcolormesh(x, y, img1, shading="auto", cmap=cmap, norm=norm, vmin=None if norm else vmin,
                         vmax=None if norm else vmax, snap=True, linewidths=0, )

    # Remove the axis
    plt.axis("off")

    # Set a white background
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    # Adjust the suptitle to be closer to the image
    plt.suptitle(timestamp, y=0.92, fontsize=14)  # Adjust `y` and `fontsize` as needed
    plt.close()
    return fig


def get_legend_data(filepath) -> dict:
    legend_data = {
        "Thresh": [],
        "rgb": [],
        "null_color": (0, 0, 0, 0),
        "void_color": (0, 0, 0, 0),
        "discrete": 0,
        "label": []
    }

    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        key = parts[0].lower()
        if key == 'thresh':
            legend_data["Thresh"].append(float(parts[2]))
        elif key == 'rgb':
            color = tuple(float(c) / 255.0 for c in parts[2:])
            legend_data["rgb"].append(color)
        elif key == 'null_color':
            legend_data["null_color"] = tuple(float(c) / 255.0 for c in parts[2:])
        elif key == 'void_color':
            legend_data["void_color"] = tuple(float(c) / 255.0 for c in parts[2:])
        elif key == 'discrete':
            legend_data["discrete"] = int(parts[2])
        elif key == 'label':
            legend_data["label"].append(" ".join((parts[2:])))

    return legend_data


def build_legend_file_path(parname):
    legend_file_path = ROOT_PATH / "legends" / parname / "legend.txt"

    return legend_file_path


def forward(x, thresh):
    """ Map the threshold values to a [0, 1] scale. """
    return np.interp(x, thresh, np.linspace(0, 1, len(thresh)))


def inverse(x, thresh):
    """ Map normalized values [0, 1] back to the original threshold values. """
    return np.interp(x, np.linspace(0, 1, len(thresh)), thresh)


class CustomNorm(mcolors.Normalize):
    """ Custom normalization to handle forward and inverse functions with thresholds. """

    def __init__(self, thresh, vmin=None, vmax=None):
        super().__init__(vmin, vmax)
        self.thresh = thresh

    def __call__(self, value, clip=None):
        return forward(value, self.thresh)

    def inverse(self, value):
        return inverse(value, self.thresh)


def create_colormap_from_legend(legend_data, parname, min_value, max_value):
    cmap_name = 'colormap_from_legend'

    if legend_data["discrete"] == 0:
        thresh = legend_data["Thresh"]
        extended_thresh = thresh
        rgb_colors = legend_data["rgb"]
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, rgb_colors, N=256)

        norm = CustomNorm(thresh, vmin=thresh[0], vmax=thresh[-1])
    else:
        extended_thresh = legend_data["Thresh"]
        rgb_colors = legend_data["rgb"]
        cmap = mcolors.ListedColormap(rgb_colors)
        norm = mcolors.BoundaryNorm(extended_thresh, cmap.N)

    return cmap, norm, extended_thresh


def configure_colorbar(parameter_name, min_val, max_val):
    legend_file_path = build_legend_file_path(parameter_name)
    if legend_file_path.exists():
        legend_data = get_legend_data(legend_file_path)
        cmap, norm, extended_thresh = create_colormap_from_legend(legend_data, parameter_name, min_value=min_val,
                                                                  max_value=max_val)
        if legend_data["discrete"] == 1:
            vmin = 0
        else:
            vmin = min(extended_thresh)
        vmax = max(extended_thresh)
        null_color = legend_data.get("null_color", (0, 0, 0, 0))
        void_color = legend_data.get("void_color", (0, 0, 0, 0))
        discrete = legend_data.get("discrete")
        ticks = extended_thresh
    else:
        cmap = 'jet'
        norm = None
        vmin = None
        vmax = None
        null_color = (0, 0, 0, 0)
        void_color = (0, 0, 0, 0)
        discrete = 0
        ticks = []
    return cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks


def lincol_2_yx(
        lin: np.ndarray,
        col: np.ndarray,
        params: list,
        dim=None,
        az_coords: tuple = (),
        el_coords=None,
        set_center: bool = False,
):
    """

    Args:
        lin:
        col:
        params: parametri del file NAVIGATION.txt
        az_coords: contenuto del file AZIMUTH.txt
        el_coords: contenuto del file ELEVATION.txt
        dim: parametro opzionale che serve solo nel caso della mosaicatura.

    Returns:

    """
    if dim is None:
        dim = []

    if len(params) >= 10:
        # Caso dati polari

        # if set_az:
        """
        TODO: qua manca una parte complicata del set_az, dovuta al fatto che gli azimuth
        non sempre sono ordinati, ma non partono necessariamente da 0 e non sempre hanno
        un passo regolare. Serve una lista chiamata az_coords che è un vettore la cui lunghezza
        coincide col numero di linee. Per ogni linea abbiamo l'angolo di azimuth corrispondente.
        Il pass in azimuth può avere delle oscillazioni, quindi serve sapere ogni singolo fascio
        dove sta puntando, che sta dentro az_coords. (nel caso di dati grezzi).
        Questa complicazione viene risolta nel momento in cui abbiamo i dati polari campionati.
        Viene riportato tutto l'array in una matriche che va a passi di 1° da 0 e 360, e si fa
        anche un campionamento in range, di solito di 1km. Quella grezza è del tutto variabile in
        base al radar. La risoluzione è variabile pure in elevazione.
        I vari array che corrispondo al ppi (giro a 360°) hanno una dimensione variabile, non sono
        tutte uguali. Quindi sono salvate in file diversi, l'albero dei dati grezzi è molto più
        profondo. Ogni singola matrice ha un numero di righe e di colonne diverso e una ris9oluzione
        che è anche essa diversa.
        Più i dati sono in elevazione, meno ci interessano, perchè si potrebbero andare a coprire
        parti che non ci interessano (oltre i 20km si scarta tutto).
        """

        x, y = lincol_2_radyx(
            lin, col, params, az_coords, el_coords, set_center=set_center
        )

    else:
        # Caso dati non polari
        xoff = params[0]
        xres = params[1]
        yoff = params[2]
        yres = params[3]

        if set_center:
            xoff -= 0.5
            yoff -= 0.5
        if xres == 0 or yres == 0:
            x = 0.0
            y = 0.0
            return

        x = (col - xoff) * xres
        y = (lin - yoff) * yres

        # TODO: caso particolare, poi nel caso lo facciamo. Da vedere se spsotare in un'altra fuznione esterna
        # if SET_Z:
        #     get_altitudes()

    if len(dim) != 2:
        return y, x

    # TODO: da controllare, è solo check sulle richieste fuori matrice
    ind = np.where((lin < 0) | lin >= dim[1])[0]

    if len(ind) > 0:
        notValid = ind

    ind = np.where((col < 0) | col >= dim[0])[0]
    if len(ind) > 0:
        if len(notValid) > 0:
            notValid = np.concatenate((notValid, ind))
        else:
            notValid = ind

    if len(notValid) <= 0:
        return

    y[notValid] = np.nan
    x[notValid] = np.nan

    return y, x


@njit(cache=True, parallel=True, fastmath=True)
def lincol_2_radyx(
        lin,
        col,
        par: dict,
        az_coords=np.array(()),
        el_coords: np.ndarray = None,
        set_az: bool = False,
        set_center: bool = False,
        lev=np.array(()),
        set_z: bool = False,
        radz=np.array(()),
):
    """
    Converts linear and column coordinates to radar coordinates in the azimuth and range dimensions.

    This function transforms linear and column indices (lin, col) into radar coordinates (rady, radx)
    based on specified parameters (par) and optional azimuth and elevation coordinates. It calculates
    the azimuth and range for each point and converts these into x and y coordinates in a radar-based
    coordinate system.

    Args:
        lin (np.ndarray or int): Linear indices or a single linear index.
        col (np.ndarray or int): Column indices or a single column index.
        par (dict): Parameters containing offsets and resolutions for the transformation.
        az_coords (np.ndarray, optional): Azimuth coordinates corresponding to the linear indices. Defaults to None.
        el_coords (np.ndarray, optional): Elevation coordinates. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - radx (np.ndarray or float): The x-coordinates in the radar coordinate system.
            - rady (np.ndarray or float): The y-coordinates in the radar coordinate system.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the provided 'par' dictionary and
        'az_coords'.
    """
    azres = 0
    polres = 0
    azoff = 0
    poloff = 0
    if az_coords is None:
        az_coords = np.array(())

    if len(par) >= 10:
        poloff = par[6]
        polres = par[7]
        azoff = par[8]
        azres = par[9]

    if polres == 0:
        radx = col
        rady = lin
        return

    az = azoff

    if len(az_coords) == 0:
        if azres != 0:
            az += lin * azres
    else:
        if len(lin) == 1:
            if 0 <= lin <= len(az_coords):
                az = az_coords[lin]
        else:
            az = az_coords[lin]

    if len(az) == 1 or set_az:
        azimuth = az
        if len(azimuth) == 1:
            if azimuth >= 360:
                azimuth -= 360

    ro = col * polres

    if set_center:
        if len(az_coords) == 0:
            az += azres / 2.0
        ro += polres / 2.0

    # Perchè gli angoli in polare partono da 0 a nord e vanno in senso orario
    # al contrario della trigonometria normale
    az = 450 - az  # rende antiorario

    # if n_elements(ro) eq 1 or keyword_set(set_range) eq 1 then range=ro

    if poloff > 0:
        ro += poloff

    az *= np.pi / 180

    # # Invertiamo così evitiamo istruzione precedente in cui togliamo 450
    # cosaz = np.sin(az)
    # sinaz = np.cos(az)

    cosaz = np.cos(az)
    sinaz = np.sin(az)

    radx = ro * cosaz
    rady = ro * sinaz

    if azres == 0 and len(lev) == 0:
        lev = lin
    if not set_z:
        return radx, rady
    if radz is not None:
        return radx, rady

    # dpg.map.get_altitudes()

    return radx, rady


# @njit(cache=True, parallel=True, fastmath=True)
def yx_2_latlon(y, x, map_proj):
    """
    Converts map projection coordinates to latitude and longitude.

    This function transforms x and y coordinates in a specified map projection back into geographic coordinates
    (latitude and longitude). The transformation is performed using the provided map projection object.

    Args:
        y (np.ndarray or float): The y-coordinate(s) in the map projection.
        x (np.ndarray or float): The x-coordinate(s) in the map projection.
        map_proj (pyproj.Proj): The map projection object used for the conversion.

    Returns:
        tuple: A tuple containing two elements:
            - lat (np.ndarray or float): The latitude(s) in degrees.
            - lon (np.ndarray or float): The longitude(s) in degrees.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the 'pyproj.Proj' function.
    """

    lon, lat = map_proj(longitude=x, latitude=y, inverse=True)

    # Prova per provare a velocizzare la conversione, al momento non sembra migliorare le perfomance
    # transformer = get_transformer(map_proj, 'latlong')
    # lon, lat = transformer.transform(x, y, direction="FORWARD")

    return lat, lon


def get_italian_region_shapefile() -> Path:
    italian_regions_folder_path = SHAPEFILE_FOLDER / "italian_regions"
    files_in_folder = list(italian_regions_folder_path.glob("*"))
    filename = files_in_folder[0].stem
    return italian_regions_folder_path / filename


def check_if_gif_present(sidebar_args):
    model = sidebar_args['model_name']
    gif_dir = f"/davinci-1/home/guidim/demo_sole/data/output/gifs/{model}"
    start_date = sidebar_args['start_date']
    start_time = sidebar_args['start_time']

    # Generate datetime objects for start, +30 mins, and +60 mins
    start_datetime = datetime.combine(start_date, start_time)
    datetime_plus_30 = start_datetime + timedelta(minutes=30)
    datetime_plus_60 = start_datetime + timedelta(minutes=60)

    # File names for groundtruths
    groundtruth_files = [
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
        f"{datetime_plus_30.strftime('%d%m%Y_%H%M')}_"
        f"{(datetime_plus_30 + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif",
        f"{datetime_plus_60.strftime('%d%m%Y_%H%M')}_"
        f"{(datetime_plus_60 + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}.gif"
    ]

    # File names for predictions
    prediction_files = [
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}_+30 mins.gif",
        f"{start_datetime.strftime('%d%m%Y_%H%M')}_"
        f"{(start_datetime + timedelta(minutes=55)).strftime('%d%m%Y_%H%M')}_+60 mins.gif"
    ]

    groundtruth_paths = [os.path.join(gif_dir, 'gt', file) for file in groundtruth_files]
    prediction_paths = [os.path.join(gif_dir, 'pred', file) for file in prediction_files]

    # Check presence of groundtruth files
    groundtruth_present = all(os.path.exists(path) for path in groundtruth_paths)

    # Check presence of prediction files
    prediction_present = all(os.path.exists(path) for path in prediction_paths)

    return groundtruth_present, prediction_present, groundtruth_paths, prediction_paths


def load_gif_as_bytesio(gif_paths):
    """
    Loads a GIF from a specified path into an io.BytesIO object.

    Args:
        gif_path (str): Path to the GIF file.

    Returns:
        list: In-memory binary representation of the GIF.
    """
    gifs = []
    for gif_path in gif_paths:
        with open(gif_path, "rb") as f:
            gif_data = f.read()
        gifs.append(io.BytesIO(gif_data))

    return gifs


def create_colorbar_fig(top_adj=None, bot_adj=None):
    # Create a figure
    fig, ax = plt.subplots(figsize=(2, 25))  # Adjust the figsize as needed
    fig.subplots_adjust(right=0.5, top=top_adj, bottom=bot_adj)

    # Create a colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
        ticks=ticks, )

    cbar.ax.tick_params(labelsize=25, length=10, width=3)  # Larger ticks and labels
    product_unit = "mm/h"
    cbar.ax.set_title(product_unit, fontsize=30, pad=50)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", )  # bbox_inches="tight")
    buf.seek(0)

    # Show the plot
    return buf


def get_closest_5_minute_time():
    now = datetime.now()
    # Calculate the number of minutes past the closest earlier 5-minute mark
    minutes = now.minute - (now.minute % 5)
    return now.replace(minute=minutes, second=0, microsecond=0).time()


def read_groundtruth_and_target_data(selected_key, selected_model):
    # Define output directory and load arrays
    # TODO: da sistemare
    out_dir = Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/{selected_model}")
    gt_array = np.load(Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/Test") / "predictions.npy",
                       mmap_mode='r')[0:12, 0]
    target_array = np.load(
        Path(f"/davinci-1/work/protezionecivile/sole24/pred_teo/Test") / "predictions.npy",
        mmap_mode='r')[12:24, 0]
    pred_array = np.load(out_dir / "predictions.npy", mmap_mode='r')[12]
    if selected_model == 'Test':
        pred_array = np.load(out_dir / "predictions.npy", mmap_mode='r')[12:24, 0]

    with h5py.File("/archive/SSD/home/guidim/demo_sole/src/mask/radar_mask.hdf", "r") as f:
        radar_mask = f["mask"][()]

    pred_array = pred_array * radar_mask
    target_array = target_array * radar_mask
    gt_array = gt_array * radar_mask

    # Clean and normalize arrays
    gt_array = np.clip(gt_array, 0, 200)
    pred_array = np.clip(pred_array, 0, 200)
    target_array = np.clip(target_array, 0, 200)

    # Convert selected_key to a datetime object
    selected_time = datetime.strptime(selected_key, "%d%m%Y_%H%M")

    # Create dictionaries for ground truth and predictions
    gt_dict = {}
    pred_dict = {}
    target_dict = {}

    # Fill ground truth dictionary
    for i in range(len(gt_array)):
        timestamp = (selected_time + timedelta(minutes=5 * i)).strftime("%d%m%Y_%H%M")
        gt_dict[timestamp] = gt_array[i]

    for i in range(len(target_array)):
        timestamp = (selected_time + timedelta(minutes=5 * (12 + i))).strftime("%d%m%Y_%H%M")
        target_dict[timestamp] = target_array[i]

    # Adjust start time for predictions (60 minutes later)
    # pred_start_time = selected_time + timedelta(minutes=5 * 12)
    for i in range(len(pred_array)):
        timestamp = (selected_time + timedelta(minutes=5 * (12 + i))).strftime("%d%m%Y_%H%M")
        pred_dict[timestamp] = pred_array[i]

    return gt_dict, target_dict, pred_dict


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_latest_file(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".hdf")]
    if not files:
        return None
    # Sort files based on the timestamp in their names
    files.sort(key=lambda x: datetime.strptime(x.split(".")[0], "%d-%m-%Y-%H-%M"), reverse=True)
    return files[0]  # Latest file


def load_prediction_data(st, time_options):
    if st.session_state.selected_model and st.session_state.selected_time:

        img1 = np.load(
            Path(
                f"/davinci-1/work/protezionecivile/sole24/pred_teo/{st.session_state.selected_model}") /
            "predictions.npy", mmap_mode='r')[0, time_options.index(st.session_state.selected_time)]
        # img1 = np.load(
        #     Path(
        #         f"/davinci-1/work/protezionecivile/sole24/pred_teo/Test") /
        #     "predictions.npy", mmap_mode='r')[0, 0]
        img1 = np.array(img1)
        img1[img1 < 0] = 0
        with h5py.File("src/mask/radar_mask.hdf", "r") as f:
            radar_mask = f["mask"][()]
        img1 = img1 * radar_mask

        sourceNode = dpg.tree.createTree("/davinci-1/home/guidim/demo_sole/data/output/nodes/sourceNode")
        destNode = dpg.tree.createTree("/davinci-1/home/guidim/demo_sole/data/output/nodes/destNode")
        img1 = dpg.warp.warp_map(sourceNode, destNode=destNode, source_data=img1)
        img1 = np.nan_to_num(img1, nan=0)

        img1[img1 < 0] = 0
        img1 = img1.astype(float)

        img_norm = norm(img1)
        rgba_img = cmap(img_norm)
        return rgba_img
    else:
        return None


def worker_thread(event):
    thread_id = threading.get_ident()  # Get the thread ID
    print(f"Worker thread (ID: {thread_id}) is doing some work...")
    time.sleep(10)  # Simulate some work being done
    print(f"Worker thread (ID: {thread_id}) has finished!")
    event.set()  # Signal that the worker thread is done


def launch_thread_execution(st, latest_file, columns):
    st.session_state.latest_file = latest_file
    print(f"New SRI file available! {latest_file}")
    with columns[1]:
        st.write("")
        st.write("")
        st.status(label="✅ Found new data!", state="complete", expanded=False)

        event = threading.Event()
        print(f"prima dell'if stato thread_started: {st.session_state.thread_started}")
        if st.session_state.thread_started is None:
            print("Starting thread")
            # Start the worker thread only if no thread is running
            thread = threading.Thread(target=worker_thread, args=(event,))
            st.session_state.thread_started = True
            thread.start()

        with st.status(f':hammer_and_wrench: **Running prediction...**', expanded=True) as status:
            status_placeholder = st.empty()
            while not event.is_set():
                time.sleep(0.1)  # Sleep for a short time to avoid blocking
                status_placeholder.text("Still waiting...")
        thread.join()
        status.update(label="✅ Prediction completed!", state="complete", expanded=False)


lat_0 = 42.0
lon_0 = 12.5
cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks = (
    configure_colorbar('R', min_val=None, max_val=None)
)
destlines = 1400
destcols = 1200
y = np.arange(destlines).reshape(-1, 1) * np.ones((1, destcols))
x = np.ones((destlines, 1)) * np.arange(destcols).reshape(1, -1).astype(int)
y, x = lincol_2_yx(lin=y, col=x, params=par, set_center=True)
lat, lon = yx_2_latlon(y, x, map_)
ll_lat = 35
ur_lat = 47
ll_lon = 6.5
ur_lon = 20

italy_shape = gpd.read_file("/davinci-1/home/guidim/demo_sole/src/shapefiles/italian_regions/gadm41_ITA_1.shp")
# Define the custom Transverse Mercator projection
custom_crs = {
    "proj": "tmerc",  # Transverse Mercator projection
    "lat_0": 42,  # Latitude of the origin
    "lon_0": 12.5,  # Longitude of the origin
    "k": 1,  # Scale factor
    "x_0": 0,  # False easting (no shift applied)
    "y_0": 0,  # False northing (no shift applied)
    "datum": "WGS84",  # Geodetic datum
    "units": "m",  # Units in meters
    "no_defs": True  # Do not use external defaults
}

# Reproject the GeoDataFrame to the custom CRS
italy_shape = italy_shape.to_crs(crs=custom_crs)
