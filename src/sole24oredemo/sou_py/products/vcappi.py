import numpy as np
import os
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg
import sou_py.preprocessing as preprocessing
import sou_py.products as products
import warnings


def compute_vcappi(volume, heights, planes, max_delta, elevations, quality):
    """
    Computes the Vertical Cross-Azimuth Plan Position Indicator (VCAPPI) for a given 3D polar volume.
    This algorithm interpolates data from a 3D polar volume to generate vertical cross-sections (VCAPPI)
    at specified height planes (`planes`). It identifies the closest elevation beam for each height
    and applies quality filters if provided.

    Args:
        volume (np.ndarray): 3D polar volume data, with dimensions [elevation, azimuth, range].
        heights (np.ndarray): Heights corresponding to the volume, with dimensions [elevation, azimuth, range].
        planes (np.ndarray): Array of height planes (in meters) for which to compute the VCAPPI.
        max_delta (float): Maximum allowable vertical distance (in meters) between a height plane and the nearest
                           elevation beam. Values exceeding this threshold are ignored.
        elevations (np.ndarray): Elevation angles corresponding to the volume, with dimensions [elevation].
        quality (np.ndarray): Quality volume of the same dimensions as `volume`. Optional.

    Returns:
        tuple:
            - vcappi (np.ndarray): 3D array [height plane, azimuth, range] containing the interpolated data.
            - elev (np.ndarray): 3D array with the corresponding elevation angles for each point in the VCAPPI.
            - qual (np.ndarray): 3D array with quality values for each point in the VCAPPI.
            - azim (np.ndarray): 3D array with azimuth indices for each point in the VCAPPI.
    """
    dim = np.shape(volume)
    nX = dim[2]
    nY = dim[1]
    nZ = np.size(planes)
    vcappi = np.squeeze(np.zeros((nZ, nY, nX), dtype=np.float32))
    qual = np.zeros_like(vcappi)
    azim = np.zeros_like(vcappi)
    elev = np.zeros_like(vcappi)
    azim[:] = np.nan
    elev[:] = np.nan
    angles = np.arange(nY, dtype=np.float32)

    for zzz in range(nZ):
        for xxx in range(nX):
            dH = np.abs(heights[:, 0, xxx] - planes[zzz])
            ind = np.where(dH > max_delta)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # RuntimeWarning: All-NaN slice encountered
                dH[ind] = np.nan
            minD = np.nanmin(dH)
            if np.isfinite(minD):
                ind = np.nanargmin(dH)
                vvv = volume[ind, :, xxx].copy()
                nnn = np.where(~np.isfinite(vvv))
                vvv[nnn] = 0.
                vcappi[:, xxx] = vvv
                elev[:, xxx] = elevations[ind]
                azim[:, xxx] = angles
                if np.size(quality) == np.size(volume):
                    qual[:, xxx] = quality[ind, :, xxx]

    return vcappi, elev, qual, azim


def vcappi(prodId, volume=None, node=None, moment=None, height=None, put=False, planes=1, qual=False):
    """
    Computes horizontal sections from a polar sampled volume using the VCAPPI algorithm.
    Unlike the CAPPI algorithm, VCAPPI selects the PPI closest to the desired height for each horizontal section.
    Additionally, it provides information about the selected elevation, corresponding height, and quality.
    It is particularly used for the "V" quantity.

    Args:
        prodId (Node): Node of the product, used to access parameters in `parameters.txt`:
            - hoff (float): Minimum height (default = 2000 m).
            - hres (float): Height step (default = 1000 m).
            - nplanes (int): Number of height planes (default = 1).
            - max_delta (float): Maximum vertical distance (absolute value, default = 1000 m).
        volume (np.ndarray, optional): 3D polar volume, usually the output of the sampling procedure. Defaults to None.
        node: The associated node of the sampled volume.
        moment (str, optional): Radar moment to process. Defaults to "V".
        height (float, optional): Single height to process (in meters). Overrides `hoff` if provided.
        put (bool, optional): If True, saves the result to `prodId`. Defaults to False.
        planes (int, optional): Number of planes. Defaults to 1.
        qual (bool, optional): If True, processes the quality volume. Defaults to False.

    Returns:
        tuple: Contains:
            - cappi (np.ndarray): 2D polar array (or 3D cylinder) of type float.
            - elev (np.ndarray): Array containing the nearest elevation.
            - qual (np.ndarray): Array containing the quality values.
            - azim (np.ndarray): Array containing the azimuth values.
    """

    if volume is None:
        if moment is None:
            moment, _, _ = dpg.radar.get_par(prodId, 'moment', 'V')
        sampled, _, _ = dpg.radar.get_par(prodId, 'sampled', 1)
        if sampled <= 0:
            log_message("Ramo non testato in products.vcappi".upper(), "WARNING")
            volume, _, node = preprocessing.sampling.sampling(prodId,
                                                              moment=moment,
                                                              projected=True)
            if qual:
                quality, _, _ = preprocessing.sampling.sampling(prodId,
                                                                moment='Quality',
                                                                projected=True)
        else:
            volume, node = dpb.dpb.get_last_volume(prodId, moment, projected=True)
            if np.size(volume) <= 1:
                log_message("Ramo non testato in products.vcappi".upper(), "WARNING")
                volume, _, node = preprocessing.sampling.sampling(prodId,
                                                                  moment=moment,
                                                                  projected=True)
                if qual:
                    quality, _, _ = preprocessing.sampling.sampling(prodId,
                                                                    moment='Quality',
                                                                    projected=True)
            else:
                if qual:
                    quality, _ = dpb.dpb.get_last_volume(prodId, "Quality", projected=True)

    sourceDim = np.shape(volume)
    if len(sourceDim) < 2:
        dpg.tree.removeNode(prodId, directory=True)

    hoff, _, _ = dpg.radar.get_par(prodId, 'hoff', 2000.)
    hres, _, _ = dpg.radar.get_par(prodId, 'hres', 1000.)
    nplanes, _, _ = dpg.radar.get_par(prodId, 'nplanes', 1)
    max_delta, _, _ = dpg.radar.get_par(prodId, 'max_delta', 1000.)

    if height is not None:
        hoff = height
        nplanes = 1

    if nplanes == 1:
        hres = 0

    planes = np.arange(nplanes, dtype=np.float32) * hres + hoff
    heights, el_coords = dpb.dpb.get_heights(node, projected=True)

    cappi, elev, qual, azim = compute_vcappi(volume, heights, planes, max_delta, el_coords,
                                             quality)

    if put:
        dpb.dpb.put_data(prodId, cappi, main=node, medianFilter=True)
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

    return cappi, elev, qual, azim