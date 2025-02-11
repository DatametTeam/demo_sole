import numpy as np
import os
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg


def saveLights(path, DATE=None, TIMES=None, VALUES=None, LATS=None, LONS=None):
    # Not yet implemented
    pass


def splitLights(strings):
    """
    Function to split and validate light-related data strings.

    Args:
        strings (list): List of strings, where each string contains 7 space-separated columns.
                        The columns should represent: date, time, latitude, longitude, and associated values.

    Returns:
        tuple: If the data is valid, the function returns:
               - dates (list): List of validated dates.
               - times (list): List of validated times.
               - values (ndarray): Array of numerical values.
               - lats (ndarray): Array of latitudes.
               - lons (ndarray): Array of longitudes.
               If validation fails, it returns five `None`.

    Notes:
        The function checks for the presence of 7 columns in each string,
        using external functions to validate dates and times.
    """
    nc = len(strings)
    if nc <= 1:
        return None, None, None, None, None

    chan = strings[0].split()
    if np.size(chan) != 7:
        return None, None, None, None, None

    dates, times, lats, lons, values = [], [], [], [], []
    for line in strings:
        chan = line.split()
        if np.size(chan) == 7:
            dates.append(dpg.times.checkDate(chan[0]))
            times.append(dpg.times.checkTime(chan[1])[0])
            lats.append(float(chan[2]))
            lons.append(float(chan[3]))
            values.append(float(chan[4]))

    lats = np.asarray(lats)
    lons = np.asarray(lons)
    values = np.asarray(values)
    return dates, times, values, lats, lons


def count_lights(lat, lon, outDim, outMap, outPar):
    """
    Counts the number of light occurrences based on given latitude and longitude data,
    while considering the output dimensions and other parameters.

    Args:
        lat (list or array): A list or array of latitude values.
        lon (list or array): A list or array of longitude values.
        outDim (list or array): A 2-element list or array representing the output dimensions (e.g., [height, width]).
        outMap (array): A map or matrix that stores light data or results.
        outPar (dict or object): Additional parameters that influence the counting process.

    Returns:
        None: If the output dimensions are invalid (e.g., if they don't have exactly two elements or if the first
        dimension is less than or equal to zero), the function returns `None`.

    """
    if len(outDim) != 2:
        return None

    if outDim[0] <= 0:
        return None

    outImage = np.zeros((outDim[0], outDim[1]), dtype=np.uint8)

    nv = len(lat)
    if nv <= 0:
        return None

    y, x = dpg.map.latlon_2_yx(lat, lon, map=outMap)
    lin, col = dpg.map.yx_2_lincol(y=y, x=x, params=outPar, dim=outDim)

    for vvv in range(nv):
        if col[vvv] >= 0 and lin[vvv] >= 0:
            outImage[lin[vvv], col[vvv]] += 1

    return outImage


def initLGT(prodId, outFile=None):
    # Not yet implemented
    pass


def updateLGT(outShape, dates, times, values, lats, lons):
    # Not yet implemented
    pass


def lightning(prodId):
    """
    Manages lightning data by reading and decoding ASCII files that contain
    lightning positions and intensities received every ten minutes.

    Args:
        prodId (str): The product node identifier, which accesses the following optional parameters
            contained in the parameters.txt file:
            - inpath (str): Path where the raw files are received (default: '/data1/SENSORS/RX/LAMPI').
            - saveValues (str): Name of the child node where the ASCII records containing values
              and positions (value, latitude, longitude) will be saved.
            - saveCount (str): Name of the child node where the counts matrix will be saved
              (default: '') with the resolution defined in the navigation file.
            - shpfile (str): Optional name for the shapefile to be created (the shapefile is of type point).

    Returns:
        None: The function does not return any value. It saves data to the specified locations based on
        the parameters provided.

    Notes:
        - The input path (`inpath`) is checked and validated to ensure it has the proper format.
        - The function logs information about the files being processed.
        - The file deletion step is commented out for testing purposes and would typically be enabled in production.

    """
    inpath, _, _ = dpg.radar.get_par(prodId, 'inpath',
                               'datamet_data/SENSORS/RX/LAMPI')  # QUESTO PATH VA SISTEMATO, CHIEDERE A MIMMO
    saveValues, _, _ = dpg.radar.get_par(prodId, 'saveValues', '')
    saveCount, _, _ = dpg.radar.get_par(prodId, 'saveCount', '')

    inpath = dpg.path.checkPathname(inpath, with_separator=False)
    outpath = dpg.tree.getNodePath(prodId)
    files, count = dpg.utility.getFilesDir(inpath)

    all = []
    for filename in files:
        position = filename.find('.TXT')
        if position > 1:
            data, _ = dpg.io.read_strings(inpath, os.path.basename(filename))
            if len(data) > 0:
                log_message(f'Using {filename}', 'INFO')
            if len(all) == 0:
                all = data
            else:
                all.append(data)
        # Per ragioni legate a testing NON cancello il file
        # come verrebbe invece fatto in production
        # dpg.utility.deleteFile(filename)

    dates, times, values, lats, lons = splitLights(all)

    if saveValues != '':
        log_message("TODO: saveValues ancora da implementare", level='ERROR')

    if saveCount != '':
        outNode, _ = dpg.tree.addNode(prodId, saveCount)
        outMap, _, outDim, outPar, _, _, _, _, _ = dpg.navigation.check_map(outNode, destMap=True)
        outImage = count_lights(lats, lons, outDim, outMap, outPar)
        dpb.dpb.put_data(outNode, outImage)
