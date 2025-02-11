import os
from pathlib import Path
from sys import prefix
from typing import Tuple
import matplotlib.colors as mcolors

from matplotlib.ticker import MultipleLocator, MaxNLocator

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

from sou_py import dpg
from sou_py import dpb
from sou_py.data_utils import get_italian_region_shapefile
from sou_py.dpg.log import log_message

from sou_py.dpv.datamet_getters import (
    get_product_unit_max_min,
    get_azoff,
    get_italian_region_shapefile,
    check_quality_path,
    check_navigation,
)


def reformat_path(path):
    """
    The function takes a path string as input, splits it to extract relevant
    components, and conditionally appends a version number if the path includes an integer
    indicating a version. The reformatted path is returned as a string with components joined
    by a delimiter (" - ") for consistency.

    Args:
        path (str): The file path to be reformatted.

    Returns:
        str: The reformatted file path, with optional version numbering if applicable.

    """

    # adapting title for image 1
    img_split = path.split("+")
    img_path_split = img_split[0].split("\\")
    index = len(img_path_split)

    if isinstance(img_path_split[index - 1], int):
        img_final = img_path_split[:-1]

        # creating version number string for image 1
        version_number = " ("
        num = str(img_path_split[index - 1])
        final_version_number = version_number + num
        final_version_number = final_version_number + ")"

        return (" - ".join(img_final)) + final_version_number

    else:
        img_final = img_path_split
        return " - ".join(img_final)


def reformat_title_and_paths(path_1, path_2, title):
    """
    This method change the format for the title of the image and for the titles of the single plots.
    It inserts white dashes between the path folders and isolate the elevation number into brackets.
    It only consider thye elevation number and remove the second number at the end of the path if exists.
    This method relies on reformat_path() to rearrange the image path to add dashes between the path folders.

    Args:
        path_1: path of the first image.
        path_2: path of the second image.

    Returns:
        string_img_1: string modified of the first image.
        string_img_2: string  modified of the second image.
        suptitle_2: subtitle of the image

    """

    string_img_1 = reformat_path(path_1)
    string_img_2 = reformat_path(path_2)

    # creating title
    title_list = title.split("_")
    value = len(title_list)

    try:
        int(title_list[-1])
        value -= 1
    except:
        pass

    suptitle_2 = " ".join(title_list[:value])

    return string_img_1, string_img_2, suptitle_2


def get_calibration_parname(node) -> dict:
    """
    Extracts calibration parameters from a given node object.
    Processes calibration data from the node and converts it into a dictionary.

    Args:
        node: A node object containing calibration data.

    Returns:
        dict: A dictionary with calibration parameters extracted from the node.
    """
    calibrationData, _ = node.getValues()
    if calibrationData is None:
        return None
    if not isinstance(calibrationData, list):
        calibration_data = [calibrationData]

    calibration_dict = {}
    for attr in calibration_data:
        if isinstance(attr, dpg.attr__define.Attr):
            calibration_dict.update(attr.pointer)

    return calibration_dict


def configure_colorbar(parameter_name, min_val, max_val):
    """
    Configures a colorbar using parameter name and value ranges.
    Utilizes legend data to create a colormap and normalization values.

    Args:
        parameter_name (str): Name of the parameter for which the colorbar is configured.
        min_val (float): Minimum value of the parameter.
        max_val (float): Maximum value of the parameter.

    Returns:
        tuple: Contains colormap, normalization, min/max values, colors for null/void data,
               discrete flag, and ticks.
    """
    legend_file_path = build_legend_file_path(parameter_name)
    if legend_file_path.exists():
        legend_data = get_legend_data(legend_file_path)
        cmap, norm, extended_thresh = create_colormap_from_legend(
            legend_data, parameter_name, min_value=min_val, max_value=max_val
        )
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
        cmap = "jet"
        norm = None
        vmin = None
        vmax = None
        null_color = (0, 0, 0, 0)
        void_color = (0, 0, 0, 0)
        discrete = 0
        ticks = []
    return cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks


def create_colormap_from_legend(legend_data, parname, min_value, max_value):
    """
    Creates a colormap based on legend data for a given parameter.
    Handles both discrete and continuous colormaps.

    Args:
        legend_data (dict): Data from the legend file, including thresholds and colors.
        parname (str): Parameter name for which the colormap is created.
        min_value (float): Minimum value for normalization.
        max_value (float): Maximum value for normalization.

    Returns:
        tuple: Contains colormap, normalization object, and extended thresholds.
    """

    cmap_name = "colormap_from_legend"
    if legend_data["discrete"] == 0:
        thresh = legend_data["Thresh"]
        extended_thresh = thresh
        rgb_colors = legend_data["rgb"]
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, rgb_colors, N=256)

        def forward(x):
            """Map the threshold values to a [0, 1] scale."""
            return np.interp(x, thresh, np.linspace(0, 1, len(thresh)))

        def inverse(x):
            """Map normalized values [0, 1] back to the original threshold values."""
            return np.interp(x, np.linspace(0, 1, len(thresh)), thresh)

        norm = mcolors.FuncNorm((forward, inverse), vmin=thresh[0], vmax=thresh[-1])

    else:
        extended_thresh = legend_data["Thresh"]
        rgb_colors = legend_data["rgb"]
        cmap = mcolors.ListedColormap(rgb_colors)
        norm = mcolors.BoundaryNorm(extended_thresh, cmap.N)
    return cmap, norm, extended_thresh


def get_legend_data(filepath) -> dict:
    """
    Reads a legend file and extracts threshold, color, and other metadata.

    Args:
        filepath (str): Path to the legend file.

    Returns:
        dict: A dictionary containing legend information like thresholds, colors, and flags.
    """

    legend_data = {
        "Thresh": [],
        "rgb": [],
        "null_color": (0, 0, 0, 0),
        "void_color": (0, 0, 0, 0),
        "discrete": 0,
        "label": [],
    }

    with open(filepath, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        key = parts[0].lower()
        if key == "thresh":
            legend_data["Thresh"].append(float(parts[2]))
        elif key == "rgb":
            color = tuple(float(c) / 255.0 for c in parts[2:])
            legend_data["rgb"].append(color)
        elif key == "null_color":
            legend_data["null_color"] = tuple(float(c) / 255.0 for c in parts[2:])
        elif key == "void_color":
            legend_data["void_color"] = tuple(float(c) / 255.0 for c in parts[2:])
        elif key == "discrete":
            legend_data["discrete"] = int(parts[2])
        elif key == "label":
            legend_data["label"].append(" ".join((parts[2:])))

    return legend_data


def build_legend_file_path(parname):
    """
    Constructs the file path to the legend file for a given parameter.

    Args:
        parname (str): Name of the parameter.

    Returns:
        Path: File path to the corresponding legend file.
    """

    file_path = Path("datamet_data/data/legends")
    legend_file_path = file_path / parname / "legend.txt"

    return legend_file_path


def get_data_with_nodes(path: Path) -> Tuple[object, np.array]:
    """
    Retrieves data and creates tree nodes from a given file path.

    Args:
        path (Path): Path to the data file.

    Returns:
        Tuple: The node object and associated numeric data.
    """

    node = dpg.tree.createTree(str(path))
    return (node, get_data(node, numeric=True))


def get_data(
    node,
    name="",
    numeric=None,
    linear=False,
    schedule=None,
    date=False,
    n_hours=None,
    warp=False,
    mosaic=False,
    regular=False,
    maximize=False,
    silent=False,
    path="",
    interactive=False,
    site_name=None,
    aux=False,
    remove=False,
    noRemove=False,
    getMinMaxVal=False,
    main=False,
    time=None,
):
    """
    Retrieves data from the specified node with various optional processing and retrieval options.

    Args:
        node (Node): The node from which to retrieve the information.
        name (str, optional): The name of the data to retrieve. Defaults to ''.
        numeric (bool, optional): Flag to specify if the data should be numeric. Defaults to None.
        linear (bool, optional): Flag to indicate if the data should be linearized. Defaults to False.
        schedule (Any, optional): Schedule used for data retrieval. Defaults to None.
        date (bool, optional): Flag to specify if the date should be used in retrieval. Defaults to False.
        n_hours (int, optional): Number of hours to consider for data retrieval. Defaults to None.
        warp (bool, optional): Flag to indicate if the data should be warped. Defaults to False.
        mosaic (bool, optional): Flag to indicate if the data should be a mosaic. Defaults to False.
        regular (bool, optional): Flag to specify if regular processing should be applied. Defaults to False.
        maximize (bool, optional): Flag to indicate if the data should be maximized. Defaults to False.
        silent (bool, optional): Flag to suppress output messages. Defaults to False.
        path (str, optional): The path to the data tree. Defaults to ''.
        interactive (bool, optional): Flag to indicate if interactive mode is enabled. Defaults to False.
        site_name (str, optional): The site name to use for data retrieval. Defaults to None.
        aux (bool, optional): Flag to indicate if auxiliary data should be included. Defaults to False.
        remove (bool, optional): Flag to remove the data tree after processing. Defaults to False.
        noRemove (bool, optional): Flag to prevent the removal of the data tree. Defaults to False.
        getMinMaxVal (bool, optional): Flag to return the minimum and maximum values. Defaults to False.
        main (Node, optional): Main node. Defaults to False.
        time (Any, optional): Time information. Defaults to None.

    Returns:
        var (np.ndarray): The retrieved data array.
        Optional[Tuple[np.ndarray, float, float]]: If getMinMaxVal is True, returns a tuple of the data array,
                                            minimum value, and maximum value.
    """

    var = np.array([0])
    tree = None
    son = None
    if isinstance(node, dpg.node__define.Node):
        son = node
    if schedule is not None:
        if n_hours is None:
            n_hours = 1
        if date is None:
            date, time, _ = dpg.times.get_time(node)
        ret, prod_path = dpg.access.get_aux_products(
            node,
            schedule=schedule,
            site_name=site_name,
            interactive=interactive,
            last_date=date,
            last_time=time,
            n_hours=n_hours,
        )
        if ret <= 0:
            return np.zeros(1)
        path = prod_path[0]

    if path is not None and path != "":
        tree = dpg.tree.createTree(path)
        if site_name is not None:
            son = dpg.radar.find_site_node(tree, site_name)
        else:
            sons = tree.getSons()
            if len(sons) != 1:
                son = tree

    if name is not None and name != "":
        sons = dpg.tree.findAllDescendant(son, name)
        if sons is None:
            sons = dpg.tree.findAllDescendant(son.parent, name)
            if sons is None:
                return None
        son = sons[0]
        if isinstance(sons, list):
            son = sons[0]
        else:
            son = sons
    if not isinstance(son, dpg.node__define.Node):
        return var

    if numeric is None:
        numeric = True

    if warp or mosaic:
        if not silent:
            log_message(f"Using node {son.path}")
        if maximize:
            pointer, _, _ = dpb.dpb.get_pointer(son, aux)
            if pointer is not None:
                pointer = dpg.prcs.maximize_data(pointer, maximize)
                # TODO: da controllare che effettivamente il
                #  pointer che torna la maximize_data sia uguale a quello dentro il son (i.e. son.pointer e pointer
                #  devono essere uguali per riferimento)
        var = dpg.warp.warp_map(
            son,
            node,
            numeric=numeric,
            linear=linear,
            mosaic=mosaic,
            aux=aux,
            regular=regular,
            remove=remove,
        )

        dpg.tree.removeTree(tree)
        return var

    pointer, out_dict = dpg.array.get_array(son, aux=aux, silent=silent)

    if pointer is None:
        dpg.tree.removeTree(tree)
        return var

    var = pointer
    if numeric or linear:
        # get_array_values deve restituire values e scale
        values, calib, out_dict = dpg.calibration.get_array_values(son)
        scale = out_dict["scale"]
        # convertData deve restituire var
        var = dpg.calibration.convertData(var, values, linear=linear, scale=scale)
        if getMinMaxVal:
            if len(values) > 0:
                minVal = np.nanmin(values[np.isfinite(values)])
                maxVal = np.nanmax(values[np.isfinite(values)])

    if not silent:
        pass
        # TODO: log

    if main is not None:
        if isinstance(main, dpg.node__define.Node):
            noRemove = True
            main = son

    if not (noRemove):
        dpg.tree.removeTree(tree)

    if getMinMaxVal:
        return var, minVal, maxVal

    return var


def plot_comparison(
    img1: np.ndarray,
    img2: np.ndarray,
    img1_title: str = "",
    img2_title: str = "",
    suptitle: str = "",
    img1_unit: str = "General",
    img2_unit: str = "General",
    vmin: int = None,
    vmax: int = None,
):
    """
    Plots a side-by-side comparison of two 2D arrays (images) with optional color scaling, titles, and units.
    The function automatically adjusts colorbars based on the provided or calculated vmin and vmax values.

    Args:
        img1 (np.ndarray): First image (2D array) to be displayed.
        img2 (np.ndarray): Second image (2D array) to be displayed.
        img1_title (str): Title for the first image (default is an empty string).
        img2_title (str): Title for the second image (default is an empty string).
        suptitle (str): Overall title for the comparison (default is an empty string).
        img1_unit (str): Unit label for the first image's colorbar (default is "General").
        img2_unit (str): Unit label for the second image's colorbar (default is "General").
        vmin (int, optional): Minimum value for color scaling (applied to both images). Defaults to None.
        vmax (int, optional): Maximum value for color scaling (applied to both images). Defaults to None.

    Returns:
        None
    """

    if vmin is None:
        vmin_img1 = np.nanmin(img1[img1 != -np.inf])
        vmin_img2 = np.nanmin(img2[img2 != -np.inf])
    else:
        vmin_img1 = vmin
        vmin_img2 = vmin

    if vmax is None:
        vmax_img1 = np.nanmax(img1[img1 != np.inf])
        vmax_img2 = np.nanmax(img2[img2 != np.inf])
    else:
        vmax_img1 = vmax
        vmax_img2 = vmax

    col, row = _get_rows_cols(img1)

    fig, (ax1, ax2) = plt.subplots(
        row,
        col,
        figsize=(15, 8),
        gridspec_kw={"wspace": 0.2, "hspace": 0.2},
        clear=True,
    )
    plt.suptitle(suptitle, fontsize=20)

    # Plot the first image on the left
    im1 = ax1.imshow(img1, cmap="jet", vmin=vmin_img1, vmax=vmax_img1)
    ax1.set_title(img1_title, fontsize=7)

    # Plot the second image on the right
    im2 = ax2.imshow(img2, cmap="jet", vmin=vmin_img2, vmax=vmax_img2)
    ax2.set_title(img2_title, fontsize=7)

    if (vmin_img1 == vmin_img2) and (vmax_img1 == vmax_img2):
        # If vmin and vmax are the same for both images, create a single colorbar
        cbar = fig.colorbar(
            im2, ax=[ax1, ax2], orientation="vertical", fraction=0.02, pad=0.04
        )
        cbar.ax.set_title(img1_unit, fontsize=7)

    else:
        # If vmin and vmax are different, create separate colorbars for each image
        cbar1 = fig.colorbar(
            im1, ax=ax1, orientation="vertical", fraction=0.02, pad=0.04
        )
        cbar2 = fig.colorbar(
            im2, ax=ax2, orientation="vertical", fraction=0.02, pad=0.04
        )
        cbar1.ax.set_title(img1_unit, fontsize=7)
        cbar2.ax.set_title(img2_unit, fontsize=7)
    # plt.show()


def _plot_comparison_polar(
    img1,
    img2,
    img1_title="",
    img2_title="",
    suptitle="",
    product_unit="General",
    vmin=0,
    vmax=0,
    save_path=None,
    test_results=None,
    azoff=0,
):
    """
    Creates a polar plot comparison of two 2D arrays (images).
    This is particularly useful for data with polar coordinate representations, such as radar imagery.

    Args:
        img1 (np.ndarray): First polar image (2D array) to be displayed.
        img2 (np.ndarray): Second polar image (2D array) to be displayed.
        img1_title (str): Title for the first image (default is an empty string).
        img2_title (str): Title for the second image (default is an empty string).
        suptitle (str): Overall title for the comparison (default is an empty string).
        product_unit (str): Unit label for the colorbar (default is "General").
        vmin (int): Minimum value for color scaling (default is 0).
        vmax (int): Maximum value for color scaling (default is 0).
        save_path (str, optional): File path to save the plot (default is None).
        test_results (object, optional): Additional data for plotting or testing (default is None).
        azoff (int): Azimuth offset for rotating the polar plot (default is 0).

    Returns:
        None
    """

    azimuth = np.linspace(0, 2 * np.pi, 360)
    range_ = np.linspace(0, 1, img1.shape[-1])  # Adjust range values as necessary

    # Adjust the azimuth values by the azimuth offset
    adjusted_azimuth = azimuth + np.radians(azoff)

    # Create meshgrid
    A, R = np.meshgrid(adjusted_azimuth, range_)

    # Transpose the data for correct orientation in the polar plot
    data1 = img1.T
    data2 = img2.T

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(
        1, 2, subplot_kw=dict(projection="polar"), figsize=(15, 8), clear=True
    )
    plt.suptitle(suptitle, fontsize=20)

    # Adjust to have 0 degrees at the top (north) for both plots
    for ax in (ax1, ax2):
        ax.set_theta_direction(-1)  # Clockwise direction
        ax.set_theta_offset((np.pi / 2))  # Ensure 0 degrees is always at the top

        # Remove the radial ticks (radius numbers)
        ax.set_yticklabels([])

        # Set custom angle labels
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        ax.set_thetagrids(angles, labels)

    cmap = plt.get_cmap("turbo_r")
    new_cmap = _truncate_colormap(cmap, 0.0, 0.6)

    # Plot the first data set
    c1 = ax1.pcolormesh(
        A, R, data1, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax
    )
    ax1.set_title(f"{img1_title}", fontsize=7)

    # Plot the second data set
    c2 = ax2.pcolormesh(
        A, R, data2, shading="auto", cmap=new_cmap, vmin=vmin, vmax=vmax
    )
    ax2.set_title(f"{img2_title}", fontsize=7)

    # Create an axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(c1, cax=cbar_ax)
    cbar.ax.set_title(product_unit, fontsize=7)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])  # Adjust rect to leave space for colorbar

    plt.close()


def plot_comparison_polar_on_map(
    img1,
    img2,
    par_img1,
    par_img2,
    map_img1,
    map_img2,
    img1_title="",
    img2_title="",
    suptitle="",
    img1_unit="General",
    img2_unit="General",
    vmin_img1=None,
    vmax_img1=None,
    vmin_img2=None,
    vmax_img2=None,
):
    """
    Plots two radar images with polar data overlaid on a geographic map, allowing for side-by-side comparison.
    The function uses a geographic projection to align radar data with real-world coordinates.

    Args:
        img1 (np.ndarray): First radar image data (2D array).
        img2 (np.ndarray): Second radar image data (2D array).
        par_img1 (dict): Parameters associated with the first image.
        par_img2 (dict): Parameters associated with the second image.
        map_img1 (object): Map projection object for the first image.
        map_img2 (object): Map projection object for the second image.
        img1_title (str): Title for the first radar image (default is an empty string).
        img2_title (str): Title for the second radar image (default is an empty string).
        suptitle (str): Overall title for the comparison (default is an empty string).
        img1_unit (str): Unit label for the first image's colorbar (default is "General").
        img2_unit (str): Unit label for the second image's colorbar (default is "General").
        vmin_img1 (float, optional): Minimum value for the first image's color scaling. Defaults to None.
        vmax_img1 (float, optional): Maximum value for the first image's color scaling. Defaults to None.
        vmin_img2 (float, optional): Minimum value for the second image's color scaling. Defaults to None.
        vmax_img2 (float, optional): Maximum value for the second image's color scaling. Defaults to None.

    Returns:
        None
    """

    if vmin_img1 is None:
        vmin_img1 = np.nanmin(img1[img1 != -np.inf])

    if vmin_img2 is None:
        vmin_img2 = np.nanmin(img2[img2 != -np.inf])

    if vmax_img1 is None:
        vmax_img1 = np.nanmax(img1[img1 != np.inf])

    if vmax_img2 is None:
        vmax_img2 = np.nanmax(img2[img2 != np.inf])

    dpi = 150
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 8), dpi=dpi)

    # Adjust the space between plots
    plt.subplots_adjust(wspace=0.1)

    # Main plotting loop
    for ax, img, title, map, par, vmin, vmax, product_unit in zip(
        [ax1, ax2],
        [img1, img2],
        [img1_title, img2_title],
        [map_img1, map_img2],
        [par_img1, par_img2],
        [vmin_img1, vmin_img2],
        [vmax_img1, vmax_img2],
        [img1_unit, img2_unit],
    ):

        projection = map.mapProj
        destlines = img.shape[-2]
        destcols = img.shape[-1]
        y = np.arange(destlines).reshape(-1, 1) * np.ones((1, destcols))
        x = np.ones((destlines, 1)) * np.arange(destcols).reshape(1, -1).astype(int)

        y, x = dpg.map.lincol_2_yx(lin=y, col=x, params=par, set_center=True)
        lat, lon = dpg.map.yx_2_latlon(y, x, map)

        if projection.name == "tmerc":
            ll_lat = 35
            ur_lat = 47
            ll_lon = 6.5
            ur_lon = 20
        else:
            buffer = 0.5  # Buffer per creare una zona di interesse più ampia attorno al sito radar
            ll_lat = lat.min() - buffer
            ur_lat = lat.max() + buffer
            ll_lon = lon.min() - buffer
            ur_lon = lon.max() + buffer

        m = Basemap(
            projection=projection.name,
            llcrnrlat=ll_lat,
            urcrnrlat=ur_lat,
            llcrnrlon=ll_lon,
            urcrnrlon=ur_lon,
            lat_0=map.p0lat,
            lon_0=map.p0lon,
            resolution="i",
            ax=ax,
        )

        m.readshapefile(get_italian_region_shapefile(), "italy_regions")

        x, y = m(lon, lat)

        # Sovrappongo l'immagine radar alla mappa con maggiore trasparenza
        c = m.pcolormesh(
            x, y, img, shading="auto", cmap="jet", vmin=vmin, vmax=vmax, snap=True
        )
        c.set_edgecolor("face")

        ax.set_title(title, fontsize=8)
        cbar = fig.colorbar(c, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.ax.set_title(product_unit, fontsize=7)
        cbar.ax.tick_params(labelsize=5)

    plt.suptitle(suptitle, fontsize=20, y=0.99)

    # plt.close()
    plt.show()


def _plot_differences_and_hist(
    prd_image: np.ndarray,
    gt_image: np.ndarray,
    img1_title: str,
    img2_title: str,
    suptitle: str,
    save_path: str,
    test_results: dict,
):
    # ristrutturazione algoritmo:
    # 1) individuazione della riga in cui è presente il maggior numero di differenze
    # 2) selezionare la matrice composta della riga interessata ed altre due righe:
    #                                                                               - una sopra e una sotto
    #                                                                               - due sopra
    #                                                                               - due sotto
    # 3) relizzazione di 3 matrici circostanti a una delle 3 aree in cui si conta il maggior numero di differenze
    # 4) composizione del layout
    difference_range_error = 80

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2, 1]}, clear=True
    )
    plt.suptitle(suptitle, fontsize=20)

    prd_image = np.nan_to_num(prd_image, neginf=0)
    gt_image = np.nan_to_num(gt_image, neginf=0)
    differences = np.abs(prd_image - gt_image)

    # 1:
    count = 0
    prec = count
    max_row = -1
    for i, row in enumerate(differences):
        count = np.sum(row > difference_range_error)
        if count > prec:
            prec = count
            max_row = i

    # plot delle differenze tra le immagini nel subplot sinistro
    left_im = ax1.imshow(differences, cmap="jet", vmin=1e-12)
    ax1.set_title("Differences between images")

    # aggiunta la barra dei colori al subplot sinistro
    cbar_left = fig.colorbar(left_im, ax=ax1, fraction=0.05, pad=0.04)
    cbar_left.set_label("Difference")

    # plot dell'istogramma delle differenze
    differences_flat = differences.flatten()
    ax2.hist(differences_flat, bins=30, color="blue", alpha=0.7)
    ax2.set_title("Histogram of differences")

    plt.close()


def _plot_differences_zoom(
    prd_image: np.ndarray,
    gt_image: np.ndarray,
    img1_title: str,
    img2_title: str,
    suptitle: str,
    save_path: str,
    test_results: dict,
):
    difference_range_error = 80

    prd_image = np.nan_to_num(prd_image, neginf=0)
    gt_image = np.nan_to_num(gt_image, neginf=0)
    differences = np.abs(prd_image - gt_image)

    count = 0
    prec = count
    max_row = -1
    for i, row in enumerate(differences):
        count = np.sum(row > difference_range_error)
        if count > prec:
            prec = count
            max_row = i

    # plot ingrandito per le ultime righe se necessario
    if max_row > -1:
        # 2)
        if _test_max_row(max_row, differences) == 0:
            three_rows_diff = differences[0:3, :]  # selezione delle prime tre righe
            interested_row_index = max_row
        elif _test_max_row(max_row, differences) == 1:
            three_rows_diff = differences[-3:, :]  # selezione delle ultime tre righe
            interested_row_index = 2
        else:
            three_rows_diff = differences[
                max_row - 1 : max_row + 1, :
            ]  # selezione della riga individuata più una sopra ed una sotto
            interested_row_index = 1

        # 3)
        adjacent_elements = 0
        flg = False
        # inizializzazione variabili dizionario
        zones = {}
        zone_values = []
        zone_index = 0
        start_i = 0
        for i in range(
            len(three_rows_diff[interested_row_index])
        ):  # Per ogni colonna nella riga in cui si è rilevato il maggior numero di valori di differenza sballati
            if (
                (three_rows_diff[interested_row_index][i] >= difference_range_error)
                or (
                    (i > 0)
                    and (
                        three_rows_diff[interested_row_index][i]
                        < difference_range_error
                    )
                    and (
                        three_rows_diff[interested_row_index][i - 1]
                        >= difference_range_error
                    )
                )
                or (
                    (i > 0)
                    and (
                        three_rows_diff[interested_row_index][i]
                        < difference_range_error
                    )
                    and (
                        three_rows_diff[interested_row_index][i - 2]
                        >= difference_range_error
                    )
                )
            ):
                if flg == False:
                    start_i = i
                    flg = True
                adjacent_elements += 1
            # devo memorizzare le coordinate della zona in cui sto registrando questi elementi adiacenti
            else:
                if adjacent_elements > 0:
                    adjacent_elements -= 2
                    zone_values.append(start_i)
                    zone_values.append(adjacent_elements)
                    zones["zone_" + str(zone_index)] = zone_values
                    zone_index += 1

                    flg = False
                    start_i = 0
                    adjacent_elements = 0
                    zone_values = []

        fig, ax = plt.subplots(1, 1, figsize=(20, 15), clear=True)
        plt.suptitle(suptitle, fontsize=20)

        up_im = ax.imshow(three_rows_diff, cmap="jet", vmin=1e-12, aspect=60)
        ax.set_title("Differences " + suptitle)
        plt.tight_layout()

        cbar_left = fig.colorbar(
            up_im, ax=ax, fraction=0.05, pad=0.04, aspect=10, shrink=0.5
        )
        cbar_left.set_label("Difference")

        # seconda riga --> altra immagine
        fig, axs = plt.subplots(3, 1, figsize=(10, 13))

        # a questo punto devo ottenere una stima dei 3 valori più grandi che ci sono in zones come numero di elementi
        # adiacenti
        sorted_zones = sorted(zones.items(), key=lambda item: item[1][1], reverse=True)
        top_three_zones = sorted_zones[:3]

        for i, _ in enumerate(top_three_zones):
            target_start_column = top_three_zones[i][1][0]
            target_final_column = target_start_column + top_three_zones[i][1][1]
            interest_matrix = three_rows_diff[
                :, target_start_column:target_final_column
            ]

            im = axs[i].imshow(interest_matrix, cmap="jet", aspect="auto")
            axs[i].set_title("Differences " + suptitle)

            if _test_max_row(max_row, differences) == 0:
                axs[i].set_xlabel(
                    "rows: (0 - 3) columns ("
                    + str(target_start_column)
                    + " - "
                    + str(target_final_column)
                    + ")"
                )
            elif _test_max_row(max_row, differences) == 1:
                axs[i].set_xlabel(
                    "rows: ("
                    + str(max_row - 2)
                    + " - "
                    + str(max_row)
                    + ") columns ("
                    + str(target_start_column)
                    + " - "
                    + str(target_final_column)
                    + ")"
                )
            else:
                axs[i].set_xlabel(
                    "rows: ("
                    + str(max_row - 1)
                    + " - "
                    + str(max_row + 1)
                    + ") columns ("
                    + str(target_start_column)
                    + " - "
                    + str(target_final_column)
                    + ")"
                )

            rg = list(range(target_start_column, target_final_column))
            axs[i].xaxis.set_ticks(np.arange(len(rg)))
            axs[i].xaxis.set_ticks([tick - 0.5 for tick in axs[i].get_xticks()])
            axs[i].xaxis.set_ticklabels(rg)
            axs[i].yaxis.set_ticklabels([])

            # Adjust tick label size and padding
            axs[i].tick_params(
                axis="x", labelsize=7, pad=2
            )  # Adjust labelsize and pad as needed
            axs[i].tick_params(axis="y", labelsize=7, pad=2)

            axs[i].tick_params(axis="x", rotation=45)

            # Add a colorbar for each subplot
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            fig.colorbar(im, cax=cax).set_label("Difference")

        # Adjust layout for better spacing
        fig.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1, hspace=0.4)

        plt.close()


def _get_rows_cols(img: np.ndarray) -> tuple:
    if img.shape[0] > img.shape[1]:
        return 2, 1
    else:
        return 1, 2


def _test_max_row(max_row, diff_matrix):
    if max_row == 0:
        return 0
    elif max_row == diff_matrix.shape[0] - 1:
        return 1
    else:
        return 2


# TODO: delete this mess!
def _render_text(fig, text_items: list) -> None:
    # Calculate total text width considering average character width
    total_text_width = (
        sum(len(text) for text, color in text_items) * 0.006
    )  # Adjust this scaling factor if needed

    # Calculate the starting x position to center the text
    x_center = 0.5  # Center position
    start_x_pos = x_center - total_text_width / 2  # Adjust starting x position

    # Fixed y position for the text
    y_pos = 0.92

    # Add each text item to the figure
    for text, color in text_items:
        text_length = len(text) * 0.012  # Adjust this scaling factor if needed
        fig.text(start_x_pos, y_pos, text, ha="left", color=color, fontsize=12)
        start_x_pos += text_length * 0.5  # Increment x position based on text length


def _truncate_colormap(
    cmap, minval=0.0, maxval=1.0, n=100
) -> colors.LinearSegmentedColormap:
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def find_elevation_by_el_coords(data, target_el_coords):
    """
    Finds the elevation key in a nested data dictionary that corresponds to the
    specified elevation coordinates (`target_el_coords`).

    Parameters:
        data: A dictionary containing elevation levels and corresponding metadata.
        target_el_coords: The target elevation coordinate to search for.

    Returns:
        elev_key: The key of the matching elevation level, or None if not found.
    """
    for elev_key, sub_dict in data.items():
        if "navigation_dict_phi" in sub_dict:
            nav_dict_phi = sub_dict["navigation_dict_phi"]
            if (
                "el_coords" in nav_dict_phi
                and nav_dict_phi["el_coords"] == target_el_coords
            ):
                return elev_key
    return None


def find_elevation_by_el_coords_uz(data, target_el_coords):
    """
    Finds the elevation key in a nested data dictionary that corresponds to the
    specified elevation coordinates (`target_el_coords`) within the "navigation_dict_uz" sub-dictionary.

    Parameters:
        data: A dictionary containing elevation levels and corresponding metadata.
        target_el_coords: The target elevation coordinate to search for.

    Returns:
        elev_key: The key of the matching elevation level, or None if not found.
    """
    for elev_key, sub_dict in data.items():
        if "navigation_dict_uz" in sub_dict:
            nav_dict_phi = sub_dict["navigation_dict_uz"]
            if (
                "el_coords" in nav_dict_phi
                and nav_dict_phi["el_coords"] == target_el_coords
            ):
                return elev_key
    return None


def plot_phi_and_kdp(data, azimuth=0, elevation=0.5, save_path=None, plot_data=False):
    """
    Plots the corrected differential phase (`phi`) and specific differential phase (`kdp`)
    as a function of range for a specific azimuth and elevation.

    Parameters:
        data: A nested dictionary containing radar data and navigation information.
        azimuth: The azimuth angle (degrees) for which data is to be plotted. Default is 0.
        elevation: The elevation angle (degrees) to locate the elevation data. Default is 0.5.
        save_path: File path to save the plot. If None, the plot is not saved. Default is None.
        plot_data: If True, displays the plot interactively. Default is False.

    Outputs:
        Creates a multi-panel plot showing `phi` and `kdp` for the selected azimuth and elevation.
    """
    dict_elevation = find_elevation_by_el_coords(data, elevation)
    nav_dict = data[dict_elevation]["navigation_dict_phi"]
    row_index = dpg.map.get_az_index(
        azimuth, nav_dict["azimut_res"], nav_dict["azimut_off"], nav_dict["az_coords"]
    )
    # Extract the keys with the "_corr" suffix for plotting
    phi_keys = [
        key
        for key in data[dict_elevation]
        if key.startswith("phi_") and key != "phi_raw"
    ]
    kdp_keys = [key for key in data[dict_elevation] if key.startswith("kdp_")]

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(len(phi_keys), 1, figsize=(10, len(phi_keys) * 3))

    # Ensure axes is a list even if there's only one subplot
    if len(phi_keys) == 1:
        axes = [axes]

    # Loop through each pair and plot
    for i, (phi_key, kdp_key) in enumerate(zip(phi_keys, kdp_keys)):
        ax1 = axes[i]
        ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        # Plot phi values on the left y-axis
        (line1,) = ax1.plot(
            data[dict_elevation][phi_key][row_index], label=phi_key, color="b", zorder=3
        )
        ax1.set_ylabel(r"$\Phi_{dp}$ (deg)", color="b", fontsize=7)
        ax1.tick_params(axis="y", labelcolor="b", labelsize=7)

        # Plot kdp values on the right y-axis
        (line2,) = ax2.plot(
            data[dict_elevation][kdp_key][row_index], label=kdp_key, color="r", zorder=4
        )
        ax2.set_ylabel(r"$K_{dp}$ (deg km$^{-1}$)", color="r", fontsize=7)
        ax2.tick_params(axis="y", labelcolor="r", labelsize=7)

        # Set tick intervals dynamically based on the range of kdp values
        kdp_max = np.max(data[dict_elevation][kdp_key][row_index])

        if kdp_max < 2:
            ax2.yaxis.set_major_locator(MultipleLocator(0.2))
        elif kdp_max <= 4:
            ax2.yaxis.set_major_locator(MultipleLocator(0.4))
        else:
            ax2.yaxis.set_major_locator(MultipleLocator(1))

        ax2.grid(True, which="both", axis="y")

        lines = [line1, line2]
        # Optionally plot phi_raw on the same axis as phi_corr
        if i == 0:
            (line3,) = ax1.plot(
                data[dict_elevation]["phi_raw"][row_index],
                label="phi_raw",
                color="g",
                linestyle="--",
                zorder=2,
            )
            lines = [line1, line2, line3]

        # Set x-axis label and grid
        ax1.set_xlabel("Range (bin)", fontsize=7)
        ax1.set_xlim(
            0, len(data[dict_elevation][phi_key]) - 1
        )  # Ensure x-axis starts from 0
        ax1.set_xticks(range(0, len(data[dict_elevation][phi_key][row_index]), 200))
        ax1.tick_params(axis="x", labelsize=7)
        ax1.grid(True, which="both", axis="x")

        # Combine legends
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=10)

    # Save the plot if required
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    if plot_data:
        plt.show()


def plot_polar_on_map(
    img,
    map,
    par,
    img_title="",
    suptitle="",
    product_unit="General",
    vmin=None,
    vmax=None,
    node=None,
):
    """
    Plots a radar image on a geographic map projection, overlaying the radar data on a map of Italy or a specific region.

    Parameters:
        img: The radar image data to be plotted.
        map: The map projection object for geographic visualization.
        par: Parameters used for georeferencing the radar data.
        img_title: Title for the individual radar image plot. Default is an empty string.
        suptitle: Main title for the entire plot. Default is an empty string.
        product_unit: The unit of the radar product to display on the colorbar. Default is "General".
        vmin: Minimum value for the color scale. If None, inferred from `img`. Default is None.
        vmax: Maximum value for the color scale. If None, inferred from `img`. Default is None.
        node: Calibration metadata for colorbar configuration. Default is None.

    Outputs:
        Displays a radar image overlaid on a map with geographic regions and an appropriate colorbar.
    """
    projection = map.mapProj
    if vmin is None:
        vmin = np.nanmin(img[img != -np.inf])
    if vmax is None:
        vmax = np.nanmax(img[img != np.inf])

    destlines = img.shape[-2]
    destcols = img.shape[-1]
    y = np.arange(destlines).reshape(-1, 1) * np.ones((1, destcols))
    x = np.ones((destlines, 1)) * np.arange(destcols).reshape(1, -1).astype(int)

    y, x = dpg.map.lincol_2_yx(lin=y, col=x, params=par, set_center=True)
    lat, lon = dpg.map.yx_2_latlon(y, x, map)

    if projection.name == "tmerc":
        ll_lat = 35
        ur_lat = 47
        ll_lon = 6.5
        ur_lon = 20
    else:
        buffer = 0.5  # Buffer per creare una zona di interesse più ampia attorno al sito radar
        ll_lat = lat.min() - buffer
        ur_lat = lat.max() + buffer
        ll_lon = lon.min() - buffer
        ur_lon = lon.max() + buffer

    calibration_data = get_calibration_parname(node)
    if calibration_data is None:
        return
    parname = calibration_data.get("parname", "default")
    max_val = calibration_data.get("maxVal", None)
    min_val = calibration_data.get("offset", None)
    max_val = float(max_val) if max_val is not None else None
    min_val = float(min_val) if min_val is not None else None
    cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks = (
        configure_colorbar(parname, min_val=min_val, max_val=max_val)
    )
    height, width = img.shape
    dpi = 80
    figsize = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    m = Basemap(
        projection=projection.name,
        llcrnrlat=ll_lat,
        urcrnrlat=ur_lat,
        llcrnrlon=ll_lon,
        urcrnrlon=ur_lon,
        lat_0=map.p0lat,
        lon_0=map.p0lon,
        resolution="i",
        ax=ax,
    )

    m.readshapefile(get_italian_region_shapefile(), "italy_regions")

    x, y = m(lon, lat)

    c = m.pcolormesh(
        x,
        y,
        img,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        snap=True,
        linewidths=0,
    )
    c.set_edgecolor("face")

    ax.set_title(img_title, fontsize=8)

    # Create a new axis for the colorbar
    # cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # Adjust the position as necessary
    cbar = fig.colorbar(c)
    cbar.ax.set_title(product_unit, fontsize=7)

    plt.suptitle(suptitle, fontsize=20, y=0.99)

    plt.show()


def plot_data(data, suptitle, vmin, vmax):
    """
    Displays a single radar product as a 2D image with a colorbar.

    Parameters:
        data: The 2D radar data array to be plotted.
        suptitle: The title displayed above the plot.
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.

    Outputs:
        Displays a plot of the radar data with the given color range and a colorbar.
    """
    fig = plt.figure(figsize=(10, 12))

    # Create a GridSpec with 1 row and 2 columns
    gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.05)

    # Create the main axis for the image
    ax_img = fig.add_subplot(gs[0, 0])
    im = ax_img.imshow(data, cmap="jet", vmin=vmin, vmax=vmax)
    ax_img.set_title(f"{suptitle}")

    # Create the axis for the colorbar
    plt.colorbar(im, fraction=0.015, pad=0.04)


def _plot_images_together(
    img1,
    img2,
    img1_title="",
    img2_title="",
    suptitle="",
    product_unit="",
    vmin=0,
    vmax=0,
    lat0=0,
    lon0=0,
    par=None,
    map=None,
    cmap="jet",
    norm=None,
    ticks=None,
    slope=0,
):
    img1_title, img2_title, suptitle = reformat_title_and_paths(
        img1_title, img2_title, suptitle
    )

    destlines = img1.shape[-2]
    destcols = img1.shape[-1]
    y = np.arange(destlines).reshape(-1, 1) * np.ones((1, destcols))
    x = np.ones((destlines, 1)) * np.arange(destcols).reshape(1, -1).astype(int)

    y, x = dpg.map.lincol_2_yx(lin=y, col=x, params=par, set_center=True)
    lat, lon = dpg.map.yx_2_latlon(y, x, map)

    if map.mapProj.name == "tmerc":
        ll_lat = 35
        ur_lat = 47
        ll_lon = 6.5
        ur_lon = 20
    else:
        buffer = 0.5  # Buffer per creare una zona di interesse più ampia attorno al sito radar
        ll_lat = lat.min() - buffer
        ur_lat = lat.max() + buffer
        ll_lon = lon.min() - buffer
        ur_lon = lon.max() + buffer

    fig = plt.figure()
    plt.figure(1).set_size_inches(20.940000 / 2.54, 18.750000 / 2.54, forward=True)

    ax1 = fig.add_axes(
        [0.09137, 0.5503, 0.3057, 0.3623]
    )  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.5893, 0.5503, 0.3057, 0.3623])

    img1_title = "(a) Python"
    img2_title = "(b) IDL"

    # Main plotting loop
    for ax, img, title in zip([ax1, ax2], [img1, img2], [img1_title, img2_title]):
        m = Basemap(
            projection=map.mapProj.name,
            llcrnrlat=ll_lat,
            urcrnrlat=ur_lat,
            llcrnrlon=ll_lon,
            urcrnrlon=ur_lon,
            lat_0=lat0,
            lon_0=lon0,
            resolution="i",
            ax=ax,
        )

        m.readshapefile(get_italian_region_shapefile(), "italy_regions")

        # Converto le coordinate geografiche in coordinate della mappa
        x, y = m(lon, lat)

        # Sovrappongo l'immagine radar alla mappa con maggiore trasparenza
        c = m.pcolormesh(
            x,
            y,
            img,
            shading="auto",
            cmap=cmap,
            norm=norm,
            vmin=None if norm else vmin,
            vmax=None if norm else vmax,
            snap=True,
            linewidths=0,
        )
        c.set_edgecolor("face")

    # Create a new axis for the colorbar
    cbar_ax = fig.add_axes(
        [0.4832, 0.5503, 0.02, 0.3647]
    )  # Adjust the position as necessary
    cbar = fig.colorbar(c, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_title(product_unit)

    # if legend_data and "label" in legend_data and legend_data["label"]:
    #     legend_elements = create_custom_legend(legend_data)
    #     ax1.legend(handles=legend_elements, title="Pioggia", loc='lower left', bbox_to_anchor=(-0.6, 0), frameon=True)

    # Manually create the two axes with specified positions
    ax3 = fig.add_axes(
        [0.09137, 0.09201, 0.3057, 0.3623]
    )  # [left, bottom, width, height]
    ax4 = fig.add_axes([0.5893, 0.09201, 0.3057, 0.1598])

    img1 = np.nan_to_num(img1, neginf=0)
    img2 = np.nan_to_num(img2, neginf=0)
    img_diff = np.abs(img1 - img2)
    img_diff[img_diff <= slope] = 0

    m = Basemap(
        projection=map.mapProj.name,
        llcrnrlat=ll_lat,
        urcrnrlat=ur_lat,
        llcrnrlon=ll_lon,
        urcrnrlon=ur_lon,
        lat_0=lat0,
        lon_0=lon0,
        resolution="i",
        ax=ax3,
    )

    m.readshapefile(get_italian_region_shapefile(), "italy_regions")
    x, y = m(lon, lat)
    c = m.pcolormesh(
        x,
        y,
        img_diff,
        shading="auto",
        cmap="hot_r",
        vmin=np.min(img_diff),
        vmax=np.max(img_diff),
        snap=True,
    )

    img_dif_flatten = img_diff.flatten()
    differences_no_zero = img_dif_flatten[img_dif_flatten != 0]
    ax4.hist(differences_no_zero, color="blue", bins=30, alpha=0.7)
    ax4.set_xlim(0)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

    _, _, _, height_hist = ax4.get_position().bounds
    cbar_ax = fig.add_axes([0.4832, 0.08955, 0.02, 0.3647])
    cbar = fig.colorbar(c, cax=cbar_ax)

    plt.figure(1).text(0.2455, 0.5275, "(a) Python", ha="center")
    plt.figure(1).text(0.7414, 0.5275, "(b) IDL", ha="center")
    plt.figure(1).text(0.2455, 0.0678, "(c) Absolute Difference", ha="center")
    plt.figure(1).text(0.7414, 0.0446, "(d) Histogram of Differences", ha="center")

    # plt.show()


class PlotterManager:

    @staticmethod
    def plot_all_together(gt_path: str, prd_path: str):
        """
        Plots radar data from ground truth (GT) and produced (PRD) images side by side
        for visual comparison, including calibration, color mapping, and geographical overlays.

        Args:
            gt_path: Path to the ground truth data file.
            prd_path: Path to the produced data file.

        Returns:
            None
        """
        gt_path = Path(gt_path)
        prd_path = Path(prd_path)
        tree_path = prd_path
        prd_node = dpg.tree.createTree(str(prd_path))
        calibration_data = get_calibration_parname(prd_node)
        if calibration_data is None:
            return
        parname = calibration_data.get("parname", "default")
        max_val = calibration_data.get("maxVal", None)
        min_val = calibration_data.get("offset", None)
        max_val = float(max_val) if max_val is not None else None
        min_val = float(min_val) if min_val is not None else None
        slope = float(calibration_data.get("slope", 1))
        cmap, norm, vmin, vmax, null_color, void_color, discrete, ticks = (
            configure_colorbar(parname, min_val=min_val, max_val=max_val)
        )
        legend_file_path = build_legend_file_path(parname)
        if legend_file_path.exists():
            legend_data = get_legend_data(legend_file_path)
        else:
            legend_data = None
        if check_navigation(prd_node):
            prd_image = dpb.dpb.get_data(prd_node, numeric=True)
            gt_node, gt_image = get_data_with_nodes(gt_path)
            product_unit, product_min, product_max = get_product_unit_max_min(
                prd_node, prd_image, gt_image, prd_path
            )
        else:
            prd_name = dpg.tree.getNodeName(prd_node)
            tree_path = prd_path.parent
            prd_node = dpg.tree.createTree(str(tree_path))
            prd_image = dpb.dpb.get_data(prd_node.getSon(prd_name), numeric=True)
            gt_node, gt_image = get_data_with_nodes(gt_path)
            product_unit, product_min, product_max = get_product_unit_max_min(
                prd_node, prd_image, gt_image, prd_path
            )
        map, par, dim, _ = prd_node.checkMap()
        _, lat0, lon0 = map.get_map_info()

        img1_title = f"Produced"
        img2_title = f"Ground Truth"

        if len(prd_image.shape) == 3:  # eg (11, 360, 200)
            prd_image = [prd_image[i] for i in range(prd_image.shape[0])]
            gt_image = [gt_image[i] for i in range(gt_image.shape[0])]
        else:
            prd_image = [prd_image]
            gt_image = [gt_image]

        for idx, (prd_image_item, gt_image_item) in enumerate(zip(prd_image, gt_image)):
            suptitle = "TEST"
            _, par, _, _, _, _, hoff, hres = dpg.navigation.get_geo_info(prd_node)
            if len(par) > 10:
                eloff = par[10]
                suptitle += f" @ elevation {eloff} deg"
            elif hoff > 0. and hres > 0.:
                suptitle += f" @ altitude {hres * idx + hoff} m"
            else:
                eloff = None

            _plot_images_together(
                prd_image_item,
                gt_image_item,
                img1_title=f"{img1_title}+{idx}",
                img2_title=f"{img2_title}+{idx}",
                suptitle=suptitle,
                product_unit=product_unit,
                vmin=product_min,
                vmax=product_max,
                lat0=lat0,
                lon0=lon0,
                map=map,
                par=par,
                cmap=cmap,
                norm=norm,
                ticks=ticks,
                slope=slope,
            )

    @staticmethod
    def plot_product(
        node_path: str, polar: bool = True, vmin: float = None, vmax: float = None
    ):
        """
        Plots a single radar data product with optional polar or Cartesian representation
        and configurable data scaling.

        Args:
            node_path: Path to the radar product data file.
            polar: Whether to display the plot in polar coordinates. Default is True.
            vmin: Minimum value for data scaling. Default is None.
            vmax: Maximum value for data scaling. Default is None.

        Returns:
            None
        """
        node = dpg.tree.createTree(str(node_path))

        image = dpb.dpb.get_data(node)

        calibrationData, _ = node.getValues()

        if not isinstance(calibrationData, list):
            calibrationData = [calibrationData]

        for attr in calibrationData:
            if "UNIT" in attr.pointer.keys():
                product_unit = attr.pointer["UNIT"]
                product_min = attr.pointer["OFFSET"]
                product_max = np.nanmax(image)
            else:
                product_unit = "General"
                product_min = np.nanmin(image)
                product_max = np.nanmax(image)

        if vmin is not None:
            product_min = vmin
        if vmax is not None:
            product_max = vmax

        _map, par, dim, _ = node.checkMap()
        _, lat0, lon0 = _map.get_map_info()

        suptitle = Path(node_path).name
        if suptitle.isnumeric() or suptitle == "MOSAIC":
            suptitle = f"{Path(node_path).parent.name}/{Path(node_path).name}"

        if polar:
            plot_polar_on_map(
                image,
                _map,
                par,
                img_title=Path(node_path).parts[-5:],
                suptitle=suptitle,
                product_unit=product_unit,
                vmin=product_min,
                vmax=product_max,
                node=node,
            )

        else:
            plot_data(
                image, suptitle=Path(node_path).name, vmin=product_min, vmax=product_max
            )

    @staticmethod
    def plot_products_comparison(
        node_1_path: str,
        node_2_path: str,
        vmin_img1: float = None,
        vmax_img1: float = None,
        vmin_img2: float = None,
        vmax_img2: float = None,
        polar: bool = True,
        level=None,
    ):
        """
        Plots a comparison of two radar data products side by side with options for
        polar or Cartesian representations and individual data scaling ranges.

        Args:
            node_1_path: Path to the first radar product data file.
            node_2_path: Path to the second radar product data file.
            vmin_img1: Minimum value for scaling the first image. Default is None.
            vmax_img1: Maximum value for scaling the first image. Default is None.
            vmin_img2: Minimum value for scaling the second image. Default is None.
            vmax_img2: Maximum value for scaling the second image. Default is None.
            polar: Whether to display the plots in polar coordinates. Default is True.
            level: Specific data level to visualize in 3D datasets. Default is None.

        Returns:
            None
        """

        prd_node_1 = dpg.tree.createTree(str(node_1_path))
        prd_node_2 = dpg.tree.createTree(str(node_2_path))

        prd_image_1 = PlotterManager._get_data_from_node(prd_node_1)
        prd_image_2 = PlotterManager._get_data_from_node(prd_node_2)

        if level is not None:
            prd_image_1 = prd_image_1[level]
            prd_image_2 = prd_image_2[level]

        map_img_1, par_img_1, _, _ = prd_node_1.checkMap()
        map_img_2, par_img_2, _, _ = prd_node_2.checkMap()

        suptitle_img1 = Path(node_1_path).name
        if suptitle_img1.isnumeric() or suptitle_img1 == "MOSAIC":
            suptitle_img1 = f"{Path(node_1_path).parent.name}/{Path(node_1_path).name}"

        suptitle_img2 = Path(node_2_path).name
        if suptitle_img2.isnumeric() or suptitle_img2 == "MOSAIC":
            suptitle_img2 = f"{Path(node_2_path).parent.name}/{Path(node_2_path).name}"

        plot_comparison_polar_on_map(
            prd_image_1,
            prd_image_2,
            par_img_1,
            par_img_2,
            map_img_1,
            map_img_2,
            img1_title=Path(node_1_path).parts[-5:],
            img2_title=Path(node_2_path).parts[-5:],
            suptitle=f"{suptitle_img1} and {suptitle_img2}",
            img1_unit="General",
            img2_unit="General",
            vmin_img1=vmin_img1,
            vmax_img1=vmax_img1,
            vmin_img2=vmin_img2,
            vmax_img2=vmax_img2,
        )

    @staticmethod
    def _get_data_from_node(prd_node):
        if check_navigation(prd_node):
            return dpb.dpb.get_data(prd_node)
        else:
            prd_name = dpg.tree.getNodeName(prd_node)
            tree_path = prd_node.parent
            prd_node = dpg.tree.createTree(str(tree_path))
            return dpb.dpb.get_data(prd_node.getSon(prd_name))


def get_nearests(array, choosen, string):
    """
    Finds and returns the nearest value to a given target from an array of values,
    while providing context in the output.

    Args:
        array: The array of values to search within.
        choosen: The target value to find the nearest match for.
        string: A descriptive label for the data being searched.

    Returns:
        idx: The index of the nearest value found in the array.
    """
    idx = np.abs(np.array(array) - choosen).argmin()
    print("Nearest " + str(string) + " available is: " + str(array[idx]))
    return idx


def return_product_data(
    pkl_data, elevation, elev_in_deg, magnitude, which_magnitude, azimuth
):
    """
    Returns radar product data for a specific azimuth and elevation, accounting for
    magnitude and elevation dataset configurations.

    Args:
        pkl_data: The radar product data in dictionary format.
        elevation: The elevation value or index to search.
        elev_in_deg: Whether the elevation is provided in degrees. Default is False.
        magnitude: The data magnitude (e.g., reflectivity, velocity).
        which_magnitude: The specific index of the magnitude to extract. Default is 1.
        azimuth: The azimuth value to extract data for.

    Returns:
        The extracted radar product data based on the specified conditions (e.g., a 2D array).
    """

    navigation_dict = "navigation_dict_" + str(magnitude)
    if isinstance(which_magnitude, int):
        which_magnitude -= 1
    elevation_dataset = str(magnitude) + "_" + str(which_magnitude)

    magnitudes = []
    first_elev = list(pkl_data.keys())[0]
    if elevation_dataset not in pkl_data[first_elev].keys():
        print("Magnitude not available.")
        magnitudes = [
            key
            for key in pkl_data[list(pkl_data.keys())[0]].keys()
            if key.startswith(str(magnitude) + "_")
            or key.endswith("_" + str(which_magnitude))
        ]
        magnitudes_split = [str(key).split("_") for key in list(magnitudes)]
        all_magnitudes = []

        for elm in magnitudes_split:
            prefix = elm[0]
            suffix = elm[1]
            for key in list(pkl_data[list(pkl_data.keys())[0]].keys()):
                if key.startswith(str(prefix) + "_") or key.endswith("_" + str(suffix)):
                    if key not in all_magnitudes:
                        all_magnitudes.append(key)

        print("All magnitudes available are: " + str(all_magnitudes))
        return False

    print("Elevation dataset: " + str(elevation_dataset))
    if isinstance(elevation, float):
        elevation = int(elevation)
    if elev_in_deg:
        elevations = []
        for elev_ in pkl_data.keys():
            elevations.append(pkl_data[elev_][navigation_dict]["elevation_off"])
        elevation = get_nearests(elevations, elevation, "elevation")

    if ("elev_" + str(elevation)) in pkl_data.keys():
        print("Elevation index set: " + str(elevation))
        elev_set = "elev_" + str(elevation)
    else:
        print("Elevation value is not valid.")
        return

    azimuth_idx = get_nearests(
        pkl_data[elev_set][navigation_dict]["az_coords"], azimuth, "azimuth"
    )
    return pkl_data[elev_set][elevation_dataset][azimuth_idx]


def plot_att_and_kdp(data, azimuth=0, elevation=0, save_path=None, plot_data=False):
    """
    Plots the evolution of specific differential phase shift (Kdp) and path-integrated attenuation (PIA) for a given
    azimuth and elevation.

    Args:
        data: A nested dictionary containing radar data and navigation information.
        azimuth: The azimuth angle (degrees) for which data is to be plotted. Default is 0.
        elevation: The elevation angle (degrees) to locate the elevation data. Default is 0.
        save_path: File path to save the plot. If None, the plot is not saved. Default is None.
        plot_data: If True, displays the plot interactively. Default is False.

    Returns:
        Creates a plot showing the evolution of Kdp and PIA for the selected azimuth and elevation.
    """

    dict_elevation = find_elevation_by_el_coords_uz(data, 0.5)
    nav_dict = data[dict_elevation]["navigation_dict_uz"]
    row_index = dpg.map.get_az_index(
        azimuth, nav_dict["azimut_res"], nav_dict["azimut_off"], nav_dict["az_coords"]
    )
    # Extract the keys with the "_corr" suffix for plotting

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Plot phi values on the left y-axis
    (line1,) = ax1.plot(
        data[dict_elevation]["kdp"][row_index], label="kdp", color="b", zorder=3
    )
    ax1.set_ylabel(r"$K_{dp}$ (deg km$^{-1}$)", color="b", fontsize=7)
    ax1.tick_params(axis="y", labelcolor="b", labelsize=7)

    # Plot kdp values on the right y-axis
    (line2,) = ax2.plot(
        data[dict_elevation]["att"][row_index], label="PIA", color="r", zorder=4
    )
    ax2.set_ylabel(r"PIA (dBZ)", color="r", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="r", labelsize=7)

    lines = [line1, line2]
    # Optionally plot phi_raw on the same axis as phi_corr
    # Set x-axis label and grid
    ax1.set_xlabel("Range (bin)", fontsize=7)
    ax1.set_xlim(0, len(data[dict_elevation]["kdp"]) - 1)  # Ensure x-axis starts from 0
    ax1.set_xticks(range(0, len(data[dict_elevation]["att"][row_index]), 200))
    ax1.tick_params(axis="x", labelsize=7)
    ax1.grid(True, which="both", axis="x")
    ax1.set_ylim(bottom=0)

    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, which="both", axis="y")
    ax2.set_ylim(bottom=-0)

    # Combine legends
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    plt.tight_layout(h_pad=2.0)
    fig.suptitle(
        r"$K_{dp}$ and PIA | Azimuth: " + str(azimuth) + f"° | Elevation: {elevation}",
        fontsize=10,
    )

    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.95)
    # plt.show()
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "att_kdp_evolution.png")

    if plot_data:
        plt.show()

    plt.close()


def plot_uz_and_uzatt(data, azimuth=0, elevation=0, save_path=None, plot_data=False):
    """
        Plots the evolution of UZ and UZ corrected after PIA for a given azimuth and elevation from radar data.

    Args:
        data: A nested dictionary containing radar data and navigation information.
        azimuth: The azimuth angle (degrees) for which data is to be plotted. Default is 0.
        elevation: The elevation angle (degrees) to locate the elevation data. Default is 0.
        save_path: File path to save the plot. If None, the plot is not saved. Default is None.
        plot_data: If True, displays the plot interactively. Default is False.

    Outputs:
        Creates a plot showing UZ and UZ corrected after PIA evolution for the selected azimuth and elevation.
    """

    dict_elevation = find_elevation_by_el_coords_uz(data, 0.5)
    nav_dict = data[dict_elevation]["navigation_dict_uz"]
    row_index = dpg.map.get_az_index(
        azimuth, nav_dict["azimut_res"], nav_dict["azimut_off"], nav_dict["az_coords"]
    )
    # Extract the keys with the "_corr" suffix for plotting

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    # ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
    # ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Plot phi values on the left y-axis
    (line1,) = ax1.plot(
        data[dict_elevation]["uz_corr"][row_index]
        - data[dict_elevation]["att"][row_index],
        label="UZ",
        color="b",
        zorder=3,
    )
    ax1.set_ylabel(r"dBZ", color="b", fontsize=7)
    ax1.tick_params(axis="y", labelcolor="b", labelsize=7)
    ax1.set_ylim(bottom=-10)

    # Plot kdp values on the right y-axis
    (line2,) = ax1.plot(
        data[dict_elevation]["uz_corr"][row_index], label="UZ_CORR", color="r", zorder=4
    )
    # ax1.set_ylabel(r'dBZ', color='r', fontsize=7)
    # ax1.tick_params(axis='y', labelcolor='r', labelsize=7)
    # ax1.set_ylim(bottom=-10)

    lines = [line1, line2]
    # Optionally plot phi_raw on the same axis as phi_corr
    # Set x-axis label and grid
    ax1.set_xlabel("Range (bin)", fontsize=7)
    ax1.set_xlim(0, len(data[dict_elevation]["kdp"]) - 1)  # Ensure x-axis starts from 0
    ax1.set_xticks(range(0, len(data[dict_elevation]["att"][row_index]), 200))
    ax1.tick_params(axis="x", labelsize=7)
    ax1.grid(True, which="both", axis="x")

    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.grid(True, which="both", axis="y")

    # Combine legends
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    plt.tight_layout(h_pad=2.0)
    fig.suptitle(
        r"UZ and UZ after PIA | Azimuth: "
        + str(azimuth)
        + f"° | Elevation: {elevation}",
        fontsize=10,
    )

    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.95)
    # plt.show()
    if save_path:
        plt.savefig(save_path / "uz_uzatt_evolution.png")

    if plot_data:
        plt.show()

    plt.close()


def plot_att_and_phidp(data, azimuth=0, elevation=0, save_path=None, plot_data=False):
    """
    Plots the evolution of PIA and $PHI_{dp}$ for a given azimuth and elevation from radar data.

    Args:
        data: A nested dictionary containing radar data and navigation information.
        azimuth: The azimuth angle (degrees) for which data is to be plotted. Default is 0.
        elevation: The elevation angle (degrees) to locate the elevation data. Default is 0.
        save_path: File path to save the plot. If None, the plot is not saved. Default is None.
        plot_data: If True, displays the plot interactively. Default is False.

    Returns:
        Creates a plot showing PIA and $PHI_{dp}$ evolution for the selected azimuth and elevation.
    """

    dict_elevation = find_elevation_by_el_coords_uz(data, 0.5)
    nav_dict = data[dict_elevation]["navigation_dict_uz"]
    row_index = dpg.map.get_az_index(
        azimuth, nav_dict["azimut_res"], nav_dict["azimut_off"], nav_dict["az_coords"]
    )
    # Extract the keys with the "_corr" suffix for plotting

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # Plot phi values on the left y-axis
    (line1,) = ax1.plot(
        data[dict_elevation]["phidp"][row_index], label="phi", color="b", zorder=3
    )
    ax1.set_ylabel(r"$PHI$ (deg km$^{-1}$)", color="b", fontsize=7)
    ax1.tick_params(axis="y", labelcolor="b", labelsize=7)

    # Plot kdp values on the right y-axis
    (line2,) = ax2.plot(
        data[dict_elevation]["att"][row_index], label="PIA", color="r", zorder=4
    )
    ax2.set_ylabel(r"PIA (dBZ)", color="r", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="r", labelsize=7)

    lines = [line1, line2]
    # Optionally plot phi_raw on the same axis as phi_corr
    # Set x-axis label and grid
    ax1.set_xlabel("Range (bin)", fontsize=7)
    ax1.set_xlim(
        0, len(data[dict_elevation]["phidp"]) - 1
    )  # Ensure x-axis starts from 0
    ax1.set_xticks(range(0, len(data[dict_elevation]["att"][row_index]), 200))
    ax1.tick_params(axis="x", labelsize=7)
    ax1.grid(True, which="both", axis="x")
    ax1.set_ylim(bottom=0)

    ax2.yaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, which="both", axis="y")
    ax2.set_ylim(bottom=-0)

    # Combine legends
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=10)

    plt.tight_layout(h_pad=2.0)
    fig.suptitle(
        r"PIA and $PHI_{dp}$ | Azimuth: "
        + str(azimuth)
        + f"° | Elevation: {elevation}",
        fontsize=10,
    )

    # Adjust layout to make room for the title
    plt.subplots_adjust(top=0.95)
    # plt.show()
    if save_path:
        plt.savefig(save_path / "att_phidp_evolution.png")

    if plot_data:
        plt.show()

    plt.close()
