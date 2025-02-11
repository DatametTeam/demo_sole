"""
Genera la mappa del Top delle nubi, indicando la massima quota per cui l’eco radar
supera una soglia definita. Applica filtri per rimuovere punti sotto valori minimi e utilizza
un ampio smoothing per migliorare la qualità visiva, riducendo artefatti concentrici.
"""
import sys
import time

import numpy as np
import sou_py.dpg as dpg
from sou_py.dpg.log import log_message


def compute_etm(volume, heightBeams, zthresh, minH, maxH):
    """
    Calculates the echo top matrix (ETM) based on the volume data and specified parameters.

    This function calculates the ETM from radar volume data. For each point in the 2D
    grid, it determines the maximum height at which the radar reflectivity exceeds a
    given threshold (zthresh). The ETM is only computed for heights between `minH` and
    `maxH`. Values below `minH` are set to negative infinity. The result is a 2D array
    numpy representing the echo top heights for each grid point.

    Args:
        volume (np.ndarray): A 3D numpy array representing radar volume data, with dimensions corresponding to (height, y, x).
        heightBeams (np.ndarray): A 2D numpy array representing the height of each beam, with dimensions corresponding to (x, height).
        zthresh (float): The reflectivity threshold used to determine the echo top.
        minH (float): The minimum height to consider for echo top calculation.
        maxH (float): The maximum height to consider for echo top calculation.

    Returns:
        np.ndarray: A 2D numpy array of dtype np.float32, representing the echo top matrix for the given radar volume data.

    Note:
       The function assumes that the input volume data is properly scaled and calibrated.
       Heights where the reflectivity does not exceed `zthresh` between `minH` and `maxH`
       are set to -np.inf, indicating no significant echo was detected at those locations.
    """

    """
    # PREVIOUS IMPLEMENTATION
    dim = volume.shape
    nX = dim[2]
    nY = dim[1]
    etm = np.zeros((nY, nX), dtype=np.float32)
    for xxx in range(nX):
        hhh = np.where(heightBeams[xxx, :] <= maxH)
        last = len(hhh[0])
        if last > 0:
            for yyy in range(nY):
                ind = np.where(volume[0:last, yyy, xxx] >= zthresh)
                count = len(ind[0])
                if count > 0:
                    etm[yyy, xxx] = heightBeams[xxx, ind[0][-1]]
    etm = np.where(etm <= minH, -np.inf, etm)
    return etm
    """

    etm = support_etm(heightBeams, maxH, volume, zthresh)
    etm = etm.T
    etm = np.where(etm <= minH, -np.inf, etm)
    return etm


def support_etm(heightBeams, maxH, volume, zThresh):
    """
    Support function used for the computation of etm.

    This function computes ETM using the provided radar data and various thresholds.
    The ETM matrix represents the maximum vertical extent of the radar reflectivity.

    Args:
        heightBeams (np.ndarray): A 2D array containing the heights of the radar beams for each vertical level.
        maxH (float): The maximum height to be considered for calculations.
        volume (np.ndarray): 3D array representing the radar volume data.
        zThresh (float): Value under this threshold are ignored.

    Returns:
        np.ndarray: 2D array containing the integrated liquid water (VIL) content in g/m³.
    """
    dim0, dim1, dim2 = volume.shape
    etm = np.zeros((dim2, dim1), dtype=np.float32)

    # Number of valid element for each row
    n_of_valid_element = (heightBeams <= maxH).sum(axis=1)

    # Valid element for volume
    depth_range = np.arange(volume.shape[0])[:, None, None]
    idx_mask = depth_range < n_of_valid_element
    valid_element_volume = (volume >= zThresh) & idx_mask

    # Find last 'true' element along 0 axis
    reversed_volume = valid_element_volume[::-1]
    idx = np.argmax(reversed_volume, axis=0)
    any_true = np.any(valid_element_volume, axis=0)
    last_indices = np.where(any_true, dim0 - 1 - idx, -1).flatten()

    # Populate etm
    etm_mask = np.where(any_true)
    etm_res = heightBeams[etm_mask[1], last_indices[last_indices >= 0]]
    etm[etm_mask[1], etm_mask[0]] = etm_res
    return etm


def ETM(prodId, volume, main, etm=None):
    """
    Calculates the Echo Top Matrix (ETM) for radar volume data based on specified parameters.
    This function computes the ETM using radar volume data, taking into account various
    parameters such as the radar's height, range resolution, and echo top thresholds.
    It first validates the dimensions of the input volume data and the type of the 'main'
    argument. Parameters like 'zthresh', 'minH', 'maxH', and radar site coordinates are
    retrieved using specific functions from the 'dpg.radar' and 'dpg.navigation' modules.
    The height of radar beams is calculated and transposed to fit the expected dimensions
    for the 'compute_etm' function, which is then called to compute the ETM.

    Args:
        prodId (Node): Identifier for the radar product.
        volume (np.ndarray): A 3D numpy array representing the radar volume data.
        main (Node): The main node or context object. Expected to be of type 'dpg.node__define.Node'.

    Returns:
        np.ndarray or None: A 2D numpy array representing the ETM for the given radar
                            volume data, or None if the input volume data is not valid.

    Note:
       The function assumes that the input volume data is in the correct format and
       that the 'main' argument is of the correct type ('dpg.node__define.Node').
       If these conditions are not met, the function returns None or exits, respectively.
       Parameters for ETM calculation are retrieved dynamically based on 'prodId' and
       the 'main' context.
    """

    if etm is not None:
        log_message(f"ETM already evaluated @ node: {prodId.path}", level='WARNING')
        return etm

    scan_dim = volume.shape
    if len(scan_dim) < 2:
        return None

    if isinstance(main, dpg.node__define.Node):
        node = main
    else:
        sys.exit("The argument passed to ETM is not a node")

    # zThresh: Soglia di riflettivita'. Il valore rilevato al Top dovra' essere superiore a tale soglia (default = 10 dbZ)
    zthresh, site_name, _ = dpg.radar.get_par(prodId, "zthresh", 18.0)
    # minH: Quota minima. Il Top dovra' essere superiore a tale quota (default = 2000 m)
    minH, _, _ = dpg.radar.get_par(prodId, "minH", 2000.0, prefix=site_name)
    maxH, _, _ = dpg.radar.get_par(prodId, "maxH", 16000.0, prefix=site_name)

    out = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    site_coords = out["site_coords"]
    range_res = out["range_res"]
    range_off = out["range_off"]
    coord_set = out["el_coords"]

    heightBeams = dpg.access.get_height_beams(
        coord_set,
        scan_dim[2],
        range_res,
        site_height=site_coords[2],
        range_off=range_off,
        projected=True,
    )
    heightBeams = heightBeams.T

    etm = compute_etm(volume, heightBeams, zthresh, minH, maxH)

    return etm
