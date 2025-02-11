"""
Stima la probabilità di occlusione del fascio radar utilizzando una mappa stagionale di occorrenze
e, opzionalmente, una valutazione teorica basata sul modello digitale del terreno (DEM).
Bassi valori di frequenza indicano alta probabilità di occlusione, con un impatto sulla qualità del dato modulato
da pesi configurabili. Il coefficiente di visibilità, calcolato in funzione di quota e ampiezza del fascio,
aggiorna la qualità corrente, limitandola tra 0 e 1. Si consiglia di preferire l’analisi effettiva
rispetto a quella teorica per maggiore accuratezza.
"""
from datetime import datetime
import os

import numpy as np
from dateutil.relativedelta import relativedelta

import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message


def smooth_data_v2(matrix, kernel_size):
    """
    This function smooths a 2D matrix by applying a mean filter over a sliding kernel window,
    handling `NaN` values gracefully.

    Args:
        - matrix: 2D array containing the input data, potentially with `NaN` values.
        - kernel_size: Size of the square kernel (must be odd).

    Returns:
        - filled_matrix: 2D array where `NaN` values are replaced by local mean values, and non-NaN values are smoothed.

    """
    log_message(
        "DA VERIFICARE FUNZIONE smooth_data_v2()!", level="ERROR", all_logs=True
    )
    # Create an array to store the filled values
    filled_matrix = np.copy(matrix)

    # Define the padding size
    pad_size = kernel_size // 2

    # Iterate over each NaN value
    for i in range(pad_size, matrix.shape[0] - pad_size):
        for j in range(pad_size, matrix.shape[1] - pad_size):
            # Extract the kernel
            kernel = matrix[
                     i - pad_size: i + pad_size + 1, j - pad_size: j + pad_size + 1
                     ]
            # Compute the mean of non-NaN values in the kernel
            if np.isnan(np.sum(kernel)):
                filled_value = np.nan
            else:
                filled_value = np.nanmean(kernel)
            # Replace NaN with the computed mean
            filled_matrix[i, j] = filled_value

    return filled_matrix


def getSeasonClass(prodId, update=False):
    """
    This feature retrieves information about the season.

    First it uses the get_par method in dpb, then if there is no value present it retrieves the date from the node
    via the get_time method, finally it returns the season based on the month.

    Args:
        prodId (list of Node or Node): Node(s) where the parameters.txt file resides.
        update (bool, optional): flag used to update the retrieved date. Defaults to False.

    Returns:
        str: Season information. e.g. "summer" or "winter".
    """
    season_class, _ = dpb.dpb.get_par(prodId, "class", "")
    if season_class != "":
        return season_class

    date, _, _ = dpg.times.get_time(prodId)
    str_date = dpg.times.checkDate(date)
    formatted_date = datetime.strptime(str_date, "%d-%m-%Y")
    date = formatted_date.date()
    mm = formatted_date.month

    if update:
        mm = (formatted_date + relativedelta(months=1)).month

    if 4 < mm < 11:
        return "summer"

    return "winter"


def getOcclusionTest(
        visScanId,
        clutterScanId,
        minSamples,
        elevation,
        range_res,
        site_height,
        up_thresh,
        up_spread,
        down_thresh,
        down_spread,
        maxVal,
        static_thresh,
        smooth,
        max_height,
):
    """
    Performs an occlusion test on visibility scan data, applying thresholds and corrections based on clutter maps and
    elevation.

    Args:
        visScanId (str or dpg.node__define.Node): The identifier for the visibility scan node to be tested
        clutterScanId (str): The identifier for the clutter scan node used for correction
        minSamples (int): Minimum number of samples required to perform the test.
        elevation (float): The elevation angle of the radar scan
        range_res (float): The range resolution of the radar
        site_height (float): The height of the radar site above sea level
        up_thresh (float): Upper threshold for the visibility test
        up_spread (float): Spread value for adjusting visibility above the upper threshold
        down_thresh (float): Lower threshold for the visibility test
        down_spread (float): Spread value for adjusting visibility below the lower threshold
        maxVal (float): Maximum value for the visibility map scaling
        static_thresh (float): Threshold for static clutter correction
        smooth (int): Smoothing factor to be applied to the visibility map
        max_height (float): Maximum height above which radar data is considered invalid

    Returns:
        np.ndarray: A 2D array representing the occlusion-corrected visibility map.
    """
    pArray, out_dict = dpg.array.get_array(visScanId)
    if out_dict is not None:
        dim = out_dict["dim"]
    if pArray is None:
        return None
    _, calib, out_dict = dpg.calibration.get_array_values(
        visScanId, getMinMaxVal=True, to_not_create=True
    )
    if "maxVal" not in out_dict.keys():
        return None
    else:
        counter = out_dict["maxVal"]
    if counter < minSamples:
        return None

    c_map = (maxVal * pArray / counter).astype(np.float32)
    max_samples = 65535
    ind = np.where(pArray >= max_samples)
    c_map[ind] = maxVal

    out = c_map.copy()
    out[:] = 1

    ind = np.where(c_map < down_thresh)
    if len(ind[0]):
        if down_spread > 0:
            out[ind] = (c_map[ind] - down_thresh + down_spread) / down_spread
        else:
            out[ind] = 0

    ind = np.where(c_map > up_thresh)
    if len(ind[0]) > 0:
        static_map = dpb.dpb.get_data(clutterScanId, numeric=True)
        if np.size(static_map) == np.size(c_map):
            static_map[ind] /= static_thresh
            ind2 = np.where(static_map < 0.5)
            static_map[ind2] = 0.5
            ind2 = np.where(static_map > 1)
            static_map[ind2] = 1
            c_map[ind] *= static_map[ind]

        if up_spread > 0:
            out[ind] = (up_thresh + up_spread - c_map[ind]) / up_spread
        else:
            out[ind] = 1

    if smooth > 0:
        out = smooth_data_v2(out, smooth)

    ind = np.where(out < 0)
    out[ind] = 0

    ind = np.where(out > 1)
    out[ind] = 1

    out *= maxVal

    heightBeams = dpg.access.get_height_beams(
        elevation, dim[-1], rangeRes=range_res, site_height=site_height
    )
    ind = np.where(heightBeams > max_height)
    count = len(ind[-1])

    if count > 0:
        ind = ind[0][0]
        for aaa in range(dim[0]):
            out[aaa, ind:] = out[aaa, ind - 1]

    return out


def occlusion_test(prodId):
    """
    Performs an occlusion test on radar scans by applying visibility thresholds and clutter corrections.
    The results are stored in an output volume associated with the product ID.

    Args:
        prodId (Node): The product identifier associated with the radar data.

    Returns:
        None: The function processes radar scan data and updates the product with occlusion-corrected maps,
        but does not return a value.

    Notes:
        The function retrieves radar scan data up to max_el and applies occlusion corrections using visibility
        thresholds and clutter maps.
        It gathers parameters like up_thresh, down_thresh, and static_thresh,
        then computes visibility maps for each scan using getOcclusionTest.
        If available, a static clutter map is used for further adjustments. The corrected data is then stored in the
        output volume.
    """
    prevMap = None

    max_el, _ = dpb.dpb.get_par(prodId, "max_el", 5.0)

    volId = dpb.dpb.get_volumes(prodId=prodId, any=True)
    scans_dict = dpb.dpb.get_scans(volId=volId, max_el=max_el)
    scans = scans_dict["scans"]
    coord_set = scans_dict["coord_set"]
    site_coords = scans_dict["site_coords"]
    scan_dim = scans_dict["scan_dim"]
    best_scan_ind = scans_dict["best_scan_ind"]

    count = len(scans)
    if count <= 0:
        return

    up_thresh, site_name = dpb.dpb.get_par(prodId, "up_thresh", 50.0)
    up_spread, _ = dpb.dpb.get_par(prodId, "up_spread", 10.0, prefix=site_name)
    down_thresh, _ = dpb.dpb.get_par(prodId, "down_thresh", 1.0, prefix=site_name)
    down_spread, _ = dpb.dpb.get_par(prodId, "down_spread", 0.0, prefix=site_name)
    static_thresh, _ = dpb.dpb.get_par(prodId, "static_thresh", 0.0, prefix=site_name)
    minSamples, _ = dpb.dpb.get_par(prodId, "minSamples", 100, prefix=site_name)
    max_height, _ = dpb.dpb.get_par(prodId, "max_height", 6000.0, prefix=site_name)
    smooth, _ = dpb.dpb.get_par(prodId, "smooth", 5, prefix=site_name)

    season_class = getSeasonClass(prodId)

    dim = [count] + scan_dim

    outPointer, par, _ = dpg.radar.check_in(
        scans[best_scan_ind], prodId, dim=dim, type=4, filename="volume.dat"
    )
    if outPointer is None:
        return

    nEl = dim[-1] * dim[-2]
    maxVal = 100.0
    site_height = site_coords[2]
    site_attr, _ = dpg.radar.get_par_attr(prodId, parFile=site_name + ".txt")
    visRoot = dpg.access.getClutterMapRoot(site_name=site_name, class_name=season_class)

    if static_thresh > 0:
        clutterRoot = dpg.access.getClutterMapRoot(
            site_name=site_name, class_name="map"
        )

    for sss in range(len(scans)):
        elevation = coord_set[sss]
        visScanId = dpg.access.getClutterMapNode(scans[sss], visRoot)
        par_dict = dpg.navigation.get_radar_par(scans[sss])
        range_res = par_dict["range_res"]
        if static_thresh > 0:
            clutterScanId = dpg.access.getClutterMapNode(scans[sss], clutterRoot)
        if isinstance(site_attr, dpg.attr__define.Attr):
            prefix = round(range_res, 2)  # TODO: TO CHECK
            up_thresh, _, exists = dpg.radar.get_par(prodId, "up_thresh", up_thresh, prefix=prefix, attr=site_attr,
                                                     only_with_prefix=True)
            if exists:
                up_spread, _, _ = dpg.radar.get_par(prodId, "up_spread", up_spread, prefix=prefix, attr=site_attr,
                                                    only_with_prefix=True)
                down_thresh, _, _ = dpg.radar.get_par(prodId, "down_thresh", down_thresh, prefix=prefix, attr=site_attr,
                                                      only_with_prefix=True)
                down_spread, _, _ = dpg.radar.get_par(prodId, "down_spread", down_spread, prefix=prefix, attr=site_attr,
                                                      only_with_prefix=True, )
                static_thresh, _, _ = dpg.radar.get_par(prodId, "static_thresh", static_thresh, prefix=prefix,
                                                        attr=site_attr, only_with_prefix=True, )
                log_message(f"Using {site_name}.txt", level="INFO", all_logs=False)

        c_map = getOcclusionTest(
            visScanId,
            clutterScanId,
            minSamples,
            elevation,
            range_res,
            site_height,
            up_thresh,
            up_spread,
            down_thresh,
            down_spread,
            maxVal,
            static_thresh,
            smooth,
            max_height,
        )

        if c_map is not None:
            out = dpg.warp.warp_map(visScanId, prodId, source_data=c_map, regular=True)
        else:
            out = None

        if np.size(out) == nEl:
            if np.size(prevMap) == nEl:
                ind = np.where(out < prevMap)
                out[ind] = prevMap[ind]
            prevMap = out.copy()
            outPointer[sss, :, :] = prevMap
        else:
            outPointer[sss, :, :] = np.nan
            log_message(
                f"Elevation {np.round(elevation, 2)}: few samples!", level="WARNING"
            )

        dpg.tree.removeTree(visRoot)
        dpg.tree.removeTree(clutterRoot)

    dpg.radar.check_out(
        outId=prodId, pointer=outPointer, par=par, el_coords=coord_set, smoothBox=0
    )


def occlusion_update(destNode, sampleNode, threshold, reset=None, date=None, time=None, site_name=None):
    # TODO: Aggiungere head comment
    _, out_dict = dpg.array.get_array(sampleNode)

    if out_dict is None:
        return

    inDim = out_dict["dim"]

    out_dict = dpg.navigation.get_radar_par(sampleNode)
    par = out_dict["par"]
    site_coords = out_dict["site_coords"]
    range_res = out_dict["range_res"]
    elevation = out_dict["elevation_off"]

    outPointer, out_dict = dpg.array.get_array(destNode)

    if np.size(outPointer) > 1:
        outDim = out_dict["dim"]
        values, _, out_dict = dpg.calibration.get_array_values(destNode,
                                                               to_not_create=True,
                                                               getMinMaxVal=True)
        counter = out_dict["maxVal"]
        if np.size(values) > 1:
            # TODO: RAMO NON TESTATO
            log_message("Ramo non testato".upper(), "WARNING", all_logs=True)
            dpg.calibration.remove_array_values(destNode)
        out_dict = dpg.navigation.get_radar_par(sampleNode)
        res = out_dict["range_res"]
        if res != range_res:
            log_message(f"No match resolution! Cannot use {dpg.tree.getNodePath(sampleNode)}",
                        "WARNING",
                        all_logs=True)
            return
        if outDim[1] != inDim[1]:
            if outDim[1] > inDim[1] or reset is None:
                log_message(f"No match dimension! Cannot use {dpg.tree.getNodePath(sampleNode)}",
                            "WARNING",
                            all_logs=True)
                return
            log_message(f"No match dimension! Resetting!", "WARNING", all_logs=True)
            outPointer = None
        if outDim[0] != 360:
            outPointer = None

    if np.size(outPointer) <= 1:
        type = 12
        outDim = np.asarray(inDim)
        outDim[0] = 360
        outPointer = dpg.array.create_array(destNode, dtype=type, dim=outDim, filename='counter.dat')
        counter = 0
        azimut_off = 0.
        azimut_res = 1.
        range_off = 0.
        elevation_res = 0.
        elevation, _ = dpg.access.roundClutterElev(elevation)
        dpg.navigation.set_radar_par(outDim[1],
                                     node=destNode,
                                     site_coords=site_coords,
                                     site_name=site_name,
                                     range_off=range_off,
                                     range_res=range_res,
                                     azimut_off=azimut_off,
                                     azimut_res=azimut_res,
                                     elevation_off=elevation,
                                     elevation_res=elevation_res,
                                     par=par)

    current = dpg.warp.warp_map(sampleNode, destNode, numeric=True, regular=True)

    if np.shape(current) != np.shape(outPointer):
        return

    ind = np.where(current > threshold)
    outPointer[ind] += 1

    counter += 1
    if counter > 50000:
        log_message(f"Counter too high ! : {str(counter)}", "WARNING", all_logs=True)

    ind = np.where(outPointer > counter)
    outPointer[ind] = 0

    in_dict = {"parname": "Counter",
               "unit": "number",
               "scaling": 0,
               "offset": 0,
               "slope": 1,
               "bitplanes": 16,
               "maxVal": counter,
               }

    dpg.calibration.set_array_values(destNode, in_dict=in_dict)
    dpg.times.set_time(destNode, date, time)

    dpg.tree.saveNode(destNode, only_current=True)

    return


def updateOcclusion(prodId):
    # TODO: Aggiungere head comment
    max_el, _ = dpb.dpb.get_par(prodId, 'max_el', 5.)
    reset, _ = dpb.dpb.get_par(prodId, 'reset', 0)
    threshold, _ = dpb.dpb.get_par(prodId, 'threshold', 10.)

    seasonClass = getSeasonClass(prodId, update=True)
    moment, _ = dpb.dpb.get_par(prodId, 'moment', '')
    if moment == '':
        moment, _ = dpb.dpb.get_par(prodId, 'measure', 'UZ')

    volId = dpb.dpb.get_volumes(prodId, moment=moment)
    site_name = dpg.navigation.get_site_name(volId)
    if site_name == '':
        return

    ret_dict = dpb.dpb.get_scans(volId, max_el=max_el)
    scans = ret_dict["scans"]

    samplesRoot = dpg.access.getClutterMapRoot(site_name, seasonClass)
    date, time, _ = dpg.times.get_time(volId)

    if not isinstance(samplesRoot, dpg.node__define.Node):
        log_message(f"Raw Data not found for {site_name}", "WARNING")
        return

    volPath = dpg.tree.getNodePath(volId)
    last, _ = dpb.dpb.get_par(samplesRoot, 'last', '')
    if last == volPath:
        log_message(f"Volume already used: {volPath}", "WARNING")
        return

    if reset > 0:
        path = dpg.tree.getNodePath(samplesRoot)
        if not os.path.exists(path):
            seconds1 = 0.
        else:
            info = os.stat(path)
            seconds1 = info.st_ctime
        seconds2 = dpg.times.convertDate(date, time)[0]
        ndays = (seconds2 - seconds1) / (24. * 3600.)
        if ndays > reset:
            dpg.tree.removeTree(samplesRoot, directory=True)
            log_message(f"Removed {path}", "INFO")
            samplesRoot = dpg.tree.createTree(path)

    for sss in scans:
        samplesNode = dpg.access.getClutterMapNode(sss, samplesRoot)
        if isinstance(samplesNode, dpg.node__define.Node):
            occlusion_update(samplesNode, sss, threshold, reset=reset, date=date, time=time, site_name=site_name)
            dpg.tree.removeNode(samplesNode)

    dpg.radar.set_par(samplesRoot, 'last', volPath, only_current=True, to_save=True)
    dpg.tree.removeTree(samplesRoot)

    return


def occlusion(
        prodId,
        update=False,
        test=False,
        warp_node=None,
        nSamples=0,
        site_coords=None,
        par=None,
        beam_width=None,
        data=None,
):
    """
    NAME:
    OCCLUSION

    Description:
        Procedure that contains all services related to the occurrence map,
        including statistics on clutter, total and/or partial beam blockage.
        It allows updating the occurrence map, activating the occlusion test,
        and accessing the occurrence map.

    Parameters:
        prodId (str): The node of the current product, from which optional parameters in parameters.txt can be accessed.
        update (bool, optional): Keyword that activates the procedure to update statistics with the current volume
        data (default is False).
        test (bool, optional): Keyword that activates the procedure to run the occlusion test (default is False).
        warp_node (optional): A node that contains geographical information for warping (only if data is present).
        nSamples (int, optional): The number of samples for the current statistics (default is 0).
        site_coords (optional): Coordinates of the site.
        par (optional): Geographical parameters of the site.
        beam_width (optional): The width of the beam.


    Output:
        None
    """
    if update:
        updateOcclusion(prodId)

    if test:
        occlusion_test(prodId)

    if data is not None:
        occlusion_map(
            prodId,
            warp_node=warp_node,
            nSamples=nSamples,
            site_coords=site_coords,
            par=par,
            beam_width=beam_width,
            data=data,
        )
