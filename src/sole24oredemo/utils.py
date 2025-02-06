import io
import os
from pathlib import Path

import pyproj
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.basemap import Basemap
from numba import njit

ROOT_PATH = Path(__file__).parent.parent.absolute()

par = np.array([600., 1000., 650., -1000.])
lat_0 = 42.0
lon_0 = 12.5
map_ = pyproj.Proj({"proj": 'tmerc', "lat_0": lat_0, "lon_0": lon_0})
SHAPEFILE_FOLDER = ROOT_PATH / "shapefiles"


def compute_figure(img1, timestamp):
    fig = plt.figure()
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

    m = Basemap(
        projection=map_.name,
        llcrnrlat=ll_lat,
        urcrnrlat=ur_lat,
        llcrnrlon=ll_lon,
        urcrnrlon=ur_lon,
        lat_0=lat_0,
        lon_0=lon_0,
        resolution="c",
    )

    m.readshapefile(get_italian_region_shapefile(), "italy_regions")

    # Converto le coordinate geografiche in coordinate della mappa
    x, y = m(lon, lat)

    # Sovrappongo l'immagine radar alla mappa con maggiore trasparenza
    c = m.pcolormesh(
        x,
        y,
        img1,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        snap=True,
        linewidths=0,
    )
    c.set_edgecolor("face")
    plt.suptitle(timestamp)
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
