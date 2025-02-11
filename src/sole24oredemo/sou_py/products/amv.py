import numpy as np
import os
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg

"""
Funzioni ancora da portare
FUNCTION InitAMV 
PRO UpdateAMV
FUNCTION ComputePolygon
PRO ComputeMaxAMV
"""


def readLastAMV(pathName, as_is=False):
    def readLastAMV(pathName, as_is=False):
        """
        Read and parse the last AMV (Atmospheric Motion Vector) data from a file.

        This function reads strings from a specified file using `dpg.io.read_strings` and extracts atmospheric
        motion vector data (latitude, longitude, direction, and velocity). The data is processed and returned
        in separate arrays. By default, the function converts directions and scales velocity values for
        standard use, but this behavior can be overridden by setting `as_is=True`.

        Args:
            pathName (str): The path to the file containing the AMV data.
            as_is (bool, optional): If `True`, the direction and velocity values are returned as-is.
                If `False` (default), the directions are adjusted by +180 degrees (mod 360) and velocities
                are converted to km/h.

        Returns:
            nc (int): The number of valid records (rows) read from the file.
            lats (numpy.ndarray): Array of latitude values (dtype=np.float32).
            lons (numpy.ndarray): Array of longitude values (dtype=np.float32).
            dir (numpy.ndarray): Array of direction values (dtype=np.float32).
                Adjusted by +180 degrees and wrapped around if `as_is=False`.
            vel (numpy.ndarray): Array of velocity values (dtype=np.float32).
                Converted to km/h (multiplied by 1.852) if `as_is=False`.
        """

    strings, _ = dpg.io.read_strings("", "", pathname=pathName)
    nc = np.size(strings)
    if nc <= 2:
        return 0

    nc -= 2
    lats = np.zeros(nc, dtype=np.float32)
    lons = np.zeros(nc, dtype=np.float32)
    dir = np.zeros(nc, dtype=np.float32)
    vel = np.zeros(nc, dtype=np.float32)

    for ccc in range(nc):
        chan = strings[ccc + 2].split()
        if np.size(chan) == 4:
            lats[ccc] = chan[0]
            lons[ccc] = chan[1]
            dir[ccc] = chan[2]
            vel[ccc] = chan[3]

    if not as_is:
        dir += 180.0
        ind = np.where(dir >= 360.0)
        if np.size(ind) > 0:
            dir[ind] -= 360.0
        vel *= 1.852

    return nc, lats, lons, dir, vel


def computeMeanAMV(vel, dir):
    """
    Compute the mean velocity and direction based on the median velocity.

    This function takes an array of velocities (`vel`) and their corresponding directions (`dir`),
    and computes the velocity (`vvv`) and direction (`ddd`) associated with the median velocity
    from the sorted array of velocities.

    Parameters:
        vel (array): A list or array of velocity values.
        dir (array): A list or array of direction values corresponding to each velocity.

    Returns:
        vvv (float): The velocity associated with the median velocity.
        ddd (float): The direction associated with the median velocity.
    """
    nV = np.size(vel)
    if nV == 1:
        vvv = vel[0]
        ddd = dir[0]
        return vvv, ddd

    ind = np.argsort(vel)
    nV //= 2
    vvv = vel[ind[nV]]
    ddd = dir[ind[nV]]

    return vvv, ddd


def checkNearValids(velArray, indNull, destSize):
    """
    Identify valid neighboring values around a given null value in a 2D velocity array.

    This function checks the 8 neighboring cells surrounding a given null index (`indNull`) in a 2D velocity
    array (`velArray`). It returns the indices of the neighboring cells that contain finite values, along
    with the count of such valid neighbors. The function ensures that the null index is within valid bounds
    of the array dimensions (`destSize`).

    Parameters:
        velArray (array): A 2D array representing velocity values, where some values may be NaN or infinite.
        indNull (tuple): A tuple `(x, y)` representing the coordinates of the null value in the array.
        destSize (tuple): A tuple `(rows, cols)` representing the size of the 2D array.

    Returns:
        indices (tuple of arrays): A tuple `(valid_x, valid_y)` containing the x and y coordinates of the valid
            neighbors. If no valid neighbors are found, returns `None`.
        count (int): The number of valid neighboring values. Returns `-1` if `indNull` is out of bounds.
    """
    x0 = indNull[0]
    if x0 <= 0 or x0 >= destSize[0] - 1:
        return None, -1

    y0 = indNull[1]
    if y0 <= 0 or y0 >= destSize[1] - 1:
        return None, -1

    ind_x = np.array([x0 - 1, x0 - 1, x0 - 1, x0, x0, x0, x0 + 1, x0 + 1, x0 + 1])
    ind_y = np.array([y0 - 1, y0, y0 + 1, y0 - 1, y0, y0 + 1, y0 - 1, y0, y0 + 1])
    ind = (ind_x, ind_y)
    tmpInd = np.where(np.isfinite(velArray[ind]))
    if np.size(tmpInd) <= 0:
        return None, -1

    indices = (ind[0][tmpInd], ind[1][tmpInd])
    count = np.size(tmpInd)

    return indices, count


def amv(prodId, attr=None, shp=None, all=None, refData=None):
    """
    Extract and process the Atmospheric Motion Vectors (AMV) from a EUMETSAT AMV product.

    Args:
        prodId (Node): The product node containing the pair of output matrices (velocity and direction).
        attr (Attr, optional): Additional attributes (not implemented in this function).
        shp (bool, optional): If set, a shapefile will be created (currently not implemented).
        all ( bool ,optional): If set, processes all data (not implemented yet).
        refData (optional): Reference data for further processing (not used in the current implementation).

    Returns:
        None: Outputs are saved directly to files or as part of the product node.
    """

    as_is = None
    path, searchFile = dpg.access.search_raw_path(prodId)
    if path == "":
        log_message("Cannot Find Raw Data", "WARNING")
        dpg.tree.removeNode(prodId, directory=True)
        return

    path = os.path.normpath(os.path.join(path, searchFile))
    count, lats, lons, dir, vel = readLastAMV(path, as_is=as_is)

    if count <= 0:
        log_message("Cannot Read Raw Data", "WARNING")
        dpg.tree.removeNode(prodId, directory=True)
        return

    log_message(f"Using {path}", "INFO")

    if all:
        log_message("TODO: not yet implemented", "ERROR")
        return

    outMap, _, destSize, destPar, _, _, _, _, _ = dpg.navigation.check_map(
        prodId, mosaic=shp, destMap=True
    )

    y, x = dpg.map.latlon_2_yx(lats, lons, map=outMap)
    lin, col = dpg.map.yx_2_lincol(y=y, x=x, params=destPar, dim=destSize)

    velArray = np.zeros(destSize, dtype=np.float32)
    velArray[:] = np.nan
    dirArray = velArray.copy()
    valids = np.where((lin >= 0) & (col >= 0))[0]

    for valid in valids:
        ind = np.where((lin == lin[valid]) & (col == col[valid]))
        if np.size(ind) > 0:
            vvv, ddd = computeMeanAMV(vel[ind], dir[ind])
            velArray[lin[valid], col[valid]] = vvv
            dirArray[lin[valid], col[valid]] = ddd

    indNull, countNull, _, _ = dpg.values.count_invalid_values(velArray)
    tmpVel = velArray.copy()

    for index in np.stack((indNull[0], indNull[1]), axis=1):
        ind, count = checkNearValids(tmpVel, index, destSize)
        if count > 0:
            vvv, ddd = computeMeanAMV(tmpVel[ind], dirArray[ind])
            velArray[index[0], index[1]] = vvv
            dirArray[index[0], index[1]] = ddd

    if not shp:
        pointer = velArray
        dpg.array.set_array(prodId, pointer, filename="amv.dat", to_save=True)
        pointer = dirArray
        dpg.array.set_array(prodId, pointer, aux=True, filename="amv.aux", to_save=True)
        return

    log_message("TODO: not yet implemented", "ERROR")
