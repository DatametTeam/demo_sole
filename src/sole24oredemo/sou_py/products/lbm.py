import numpy as np
import os
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg
import sou_py.preprocessing as preprocessing


def computeLBM(values, qualities, heights, threshQual, no_check=False):
    """
    Compute the Lowest Beam Meeting (LBM) quality threshold.
    This function identifies the first height where the quality exceeds a specified threshold (`threshQual`)
    and retrieves the corresponding value, quality, and height.

    Args:
        values (np.ndarray): Array of data values corresponding to different heights.
        qualities (np.ndarray): Array of quality scores associated with the data values.
        heights (np.ndarray): Array of heights corresponding to the data values.
        threshQual (float): The quality threshold; the function selects the first height where the quality exceeds this threshold.
        no_check (bool, optional): If True, skips checks for finite values in `values`. Defaults to False.

    Returns:
        tuple:
            - value (float): The first data value where quality meets the threshold. Returns `np.nan` if no quality meets the threshold.
            - quality (float): The quality score at the selected height.
            - height (float): The height at which the quality meets the threshold. Returns `np.nan` if no quality meets the threshold.
    """
    ind = np.where(qualities >= threshQual)[0]
    if np.size(ind) <= 0:
        q = qualities[0]
        if no_check:
            h = heights[0]
            return values[0], q, h
        h = np.nan
        return np.nan, q, h

    iii = ind[0]
    q = qualities[iii]
    h = heights[iii]
    if not no_check:
        if not np.isfinite(values[iii]):
            h = -np.inf

    return values[iii], q, h


def LBM(prodId,
        volume,
        qVolume,
        moment=None,
        sampled=None,
        put=False,
        remove_on_error=False,
        node=None,
        as_is=None,
        no_check=None):
    """
    Computes the Lowest Beam Meeting (LBM) based on quality thresholds.
    This algorithm creates a synthetic PPI where each cell contains the lowest value in height that exceeds a given
    quality threshold. For each cell, it also returns the corresponding quality value and height. Requires a quality volume
    (`qVolume`) of the same dimensions as the input data volume (`volume`).

    Args:
        prodId (Node): The product node, providing access to optional parameters from `parameters.txt`:
            - `min_el`: Minimum elevation to consider. If `min_el >= max_el`, all elevations are used.
            - `max_el`: Maximum elevation to consider. Defaults to `min_el`.
            - `threshQuality`: Minimum quality threshold; data below this value is ignored. Defaults to 0.
        volume (np.ndarray): 3D polar data volume, typically from the SAMPLING procedure.
        qVolume (np.ndarray): Quality volume of the same dimensions as `volume`.
        moment (str, optional): Moment to be analyzed. Defaults to None.
        sampled (int, optional): Flag to indicate if data is sampled. Defaults to None.
        put (bool, optional): If True, stores the resulting LBM in the product node. Defaults to False.
        remove_on_error (bool, optional): If True, removes the node on error. Defaults to False.
        node (Node, optional): The sampled volume node, used for accessing volume properties. Defaults to None.
        as_is (bool, optional): Flag to indicate if data should be stored as is. Defaults to None.
        no_check (bool, optional): If True, skips checks for finite values. Defaults to None.

    Returns:
        tuple:
            - LBM (np.ndarray): A 2D array [range, azimuth] of type float. Contains the lowest values exceeding the quality threshold.
                                Returns None where no values satisfy the quality conditions.
            - quality (np.ndarray): A 2D array of the quality of the selected cells (float).
            - heights (np.ndarray): A 2D array of the heights of the selected cells (float).
    """
    threshQual, _, _ = dpg.radar.get_par(prodId, 'threshQuality', 0.)

    dim = np.shape(volume)
    if len(dim) < 2:
        log_message("Ramo non testato in products.lbm.LBM".upper(), "WARNING")
        if sampled is None:
            sampled, _, _ = dpg.radar.get_par(prodId, 'sampled', 1)
        here = 1 - sampled
        volume, _, node = preprocessing.sampling.sampling(prodId,
                                                          sampled=sampled,
                                                          moment=moment,
                                                          here=here,
                                                          remove_on_error=remove_on_error,
                                                          projected=True)
        dim = np.shape(volume)
        if len(dim) < 2:
            return None, None, None
        qVolume, _, node = preprocessing.sampling.sampling(prodId,
                                                          sampled=sampled,
                                                          moment='Quality',
                                                          here=here,
                                                          remove_on_error=remove_on_error,
                                                          projected=True)

    if np.size(volume) != np.size(qVolume):
        return None, None, None

    par_dict = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    par = par_dict["par"]
    range_res = par_dict["range_res"]
    el_coords = par_dict["el_coords"]
    site_coords = par_dict["site_coords"]

    nEl = np.size(el_coords)
    if nEl <= 0:
        return None, None, None

    LBM = np.zeros(dim[1:], dtype=np.float32)
    quality = np.zeros(dim[1:], dtype=np.float32)
    heights = np.zeros(dim[1:], dtype=np.float32)
    heightBeams = np.zeros((nEl, dim[2]), dtype=np.float32)

    for eee in range(nEl):
        heightBeams[eee, :] = dpg.access.get_height_beams(el_coords[eee],
                                                          dim[2],
                                                          range_res,
                                                          site_height=site_coords[2],
                                                          projected=True)

    for rrr in range(dim[2]):
        hBeam = heightBeams[:, rrr].copy()
        for aaa in range(dim[1]):
            LBM[aaa, rrr], quality[aaa, rrr], heights[aaa, rrr] = computeLBM(volume[:, aaa, rrr],
                                                                             qVolume[:, aaa, rrr],
                                                                             hBeam,
                                                                             threshQual,
                                                                             no_check=no_check)

    if put:
        dpb.dpb.put_data(prodId, LBM, main=node, as_is=as_is)
        _, _, _, calib, _, _ = dpb.dpb.get_values(node=node)
        dpb.dpb.put_values(prodId, calib)

    return LBM, quality, heights