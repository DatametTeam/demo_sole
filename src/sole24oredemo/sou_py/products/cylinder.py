import numpy as np
import os
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg
import sou_py.preprocessing as preprocessing
import sou_py.products as products
import warnings


def cylinder(volume, heights, height, cylinder=None, unlinearize=False, max_delta=None, pseudo=None):
    """
    Compute multiple horizontal sections from a 3D volume through interpolation.
    This algorithm calculates horizontal sections (cylinder) at specified altitudes from a 3D volume.
    Each section is obtained by interpolating between two consecutive PPI layers surrounding the desired altitude.

    Args:
        volume (np.ndarray): 3D volume to process, typically the output of the `SAMPLING` procedure.
        heights (np.ndarray): Altitudes of the sampled volume, typically obtained from `GET_HEIGHTS`.
        height (float or list[float]): One or more altitudes (in meters) for the sections to compute.
        cylinder (np.ndarray, optional): Pre-allocated output array. Defaults to None, and a new array will be created.
        unlinearize (bool, optional): If True, the output matrix will be converted back to dB scale. Defaults to False.
        max_delta (float, optional): Maximum vertical distance for interpolation (default: 3000 m). If `pseudo` is True, this is ignored.
        pseudo (bool, optional): If True, ignores `max_delta`. Defaults to False.

    Returns:
        cylinder(np.ndarray): 3D polar matrix (cylinder) of type float containing the computed sections.
    """

    dim = volume.shape
    if np.size(dim) <= 1:
        return None

    nH = np.size(height)
    if nH <= 0:
        return  None

    if nH == 1:
        hhh = [height]
        cylinder = np.zeros((nH,)+heights[0].shape)
    else:
        hhh = height
        cylinder = np.zeros((nH,)+dim[1:])

    if np.size(dim) == 2:
        for h in range(nH):
            cylinder[h, :, :] = volume
        if unlinearize:
            cylinder = dpg.prcs.unlinearizeValues(cylinder, scale=2)
        return cylinder

    if max_delta is None:
        max_delta = 3000.
    if pseudo:
        max_delta = 0.

    for xxx in range(dim[2]):
        h = heights[:, 0, xxx]
        sp = np.searchsorted(h, hhh)
        s = sp -1
        sp[np.where(sp > np.size(h)-1)] = np.size(h)-1
        s[np.where(s < 0)] = 0
        dH = hhh -h[s]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # RuntimeWarning: divide by zero encountered in divide
            deltaH = dH / (h[sp] - h[s])
        ind = np.where(~np.isfinite(deltaH))
        deltaH[ind] = 0.
        if max_delta > 0.:
            ind = np.where(np.abs(dH) > max_delta)
            deltaH[ind] = np.nan
        for yyy in range(dim[1]):
            v = volume[:, yyy, xxx]
            d = v[sp] - v[s]
            ind = np.where(~np.isfinite(d))
            d[ind] = 0.
            cylinder[:, yyy, xxx] = deltaH * d + v[s]

    if nH == 1:
        cylinder = np.squeeze(cylinder)
    if unlinearize:
        cylinder = dpg.prcs.unlinearizeValues(cylinder, scale=2)

    return cylinder