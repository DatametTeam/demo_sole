import numpy as np
import os
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg
import sou_py.preprocessing as preprocessing
import sou_py.products as products


def cappi(prodId, schedule=None, volume=None, node=None, moment=None, height=None, linear=None, cappi=None, put=False, planes=None):
    """
    Computes Constant Altitude Plan Position Indicator (CAPPI) from a polar volume.
    This function calculates horizontal sections (CAPPI) at specified altitudes from a polar volume.
    The sections are obtained by interpolating between consecutive PPI scans near the desired altitude,
    using the `CYLINDER` algorithm.

    Args:
        prodId (Node): Node of the product containing configuration parameters.
        schedule(Node, optional): Optional schedule node for accessing radar volumes.
        volume (np.ndarray, optional): Pre-loaded polar volume. If None, it will be retrieved or sampled.
        node: Node of the radar data associated with the volume.
        moment (str, optional): Radar moment (e.g., 'CZ', 'UZ'). Defaults to 'CZ' if not provided.
        height (float, optional): Altitude for CAPPI extraction. If not provided, uses `hoff` from parameters.
        linear (bool, optional): Flag indicating whether to use linear scale. Defaults to True for 'CZ' and 'UZ'.
        cappi (np.ndarray, optional): Output array to store CAPPI data. If None, a new array is created.
        put (bool, optional): If True, stores the result in `prodId`. Defaults to False.
        planes (np.ndarray, optional): Predefined altitude planes for CAPPI calculation. Defaults to computed values.

    Returns:
        None: The function modifies `cappi` or stores the result in `prodId` depending on the `put` flag.

    Notes:
        - Default parameter values are read from the associated configuration file (`parameters.txt`):
          - `hoff` (float): Minimum altitude for CAPPI planes (default: 2000 m).
          - `hres` (float): Altitude step between planes (default: 1000 m).
          - `nplanes` (int): Number of CAPPI planes to compute (default: 1).
          - `max_delta` (float): Maximum vertical distance for interpolation (default: 3000 m).
          - `pseudo` (bool): If set, `max_delta` is ignored (default: 1).
    """

    if volume is None:
        if moment is None:
            moment, _, _ = dpg.radar.get_par(prodId, 'moment', 'CZ')
        if linear is None:
            linear = (moment == 'CZ') or (moment == 'UZ')
        sampled, _, _ = dpg.radar.get_par(prodId, 'sampled', 1)
        if sampled <= 0:
            volume, _, node = preprocessing.sampling.sampling(prodId,
                                                              moment=moment,
                                                              projected=True,
                                                              linear=linear)
        else:
            volume, node = dpb.dpb.get_last_volume(prodId, moment, linear=linear, projected=True)
            #volume, node = schedule.get_curr_volume(prodId, moment, linear=linear, projected=True)
            if np.size(volume) <= 0:
                volume, _, node = preprocessing.sampling.sampling(prodId,
                                                                  moment=moment,
                                                                  projected=True,
                                                                  linear=linear)

    sourceDim = volume.shape
    if len(sourceDim) <= 2:
        dpg.tree.removeNode(prodId, directory=True)
        return

    hoff, _, _ = dpg.radar.get_par(prodId, 'hoff', 2000.)
    hres, _, _ = dpg.radar.get_par(prodId, 'hres', 1000.)
    nplanes, _, _ = dpg.radar.get_par(prodId, 'nplanes', 1)
    max_delta, _, _ = dpg.radar.get_par(prodId, 'max_delta', 3000.)
    pseudo, _, _ = dpg.radar.get_par(prodId, 'pseudo', 1)

    if height is not None:
        hoff = height
        nplanes = 1
    if nplanes == 1:
        hres = 0.

    planes = np.arange(nplanes, dtype=np.float32)*hres + hoff
    heights, _ = dpb.dpb.get_heights(node, projected=True)
    cylinder = products.cylinder.cylinder(volume,
                                          heights,
                                          planes,
                                          pseudo=pseudo,
                                          unlinearize=linear,
                                          max_delta=max_delta)

    if cappi is not None:
        if np.size(cylinder) > 1:
            cappi = cylinder
    else:
        if put is False:
            put = True

    if put:
        dpb.dpb.put_data(prodId, cylinder, main=node, no_copy=True, medianFilter=True)
        par_dict = dpg.navigation.get_radar_par(prodId)
        par = par_dict["par"]
        range_res = par_dict["range_res"]
        azimut_res = par_dict["azimut_res"]
        azimut_off = par_dict["azimut_off"]
        dpb.dpb.put_radar_par(prodId,
                              par=par,
                              range_res=range_res,
                              azimut_off=azimut_off,
                              azimut_res=azimut_res,
                              h_off=hoff,
                              h_res=hres,
                              remove_coords=True)
        if nplanes == 1 and height is None:
            dpg.radar.set_par(prodId, 'height', hoff)

    return