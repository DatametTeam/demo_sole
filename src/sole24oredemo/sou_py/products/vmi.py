import sys

import numpy as np
import sou_py.dpg as dpg


def VMI(prodId, volume, qVolume, main, node=None, attr=None, void=None, no_put=None):
    """
    Algorithm for calculating the Vertical Maximum Intensity (VMI) starting from a generic 3D radar volume.
    Each pixel of the VMI map represents the maximum value recorded on the vertical, with the possibility of filtering
    the data based on altitude, elevation and quality. Value below a predetermined threshold are ignored.
    In the presence of anomalous propagation, the ETM product can be used to filter out clutter, applying a more
    stringent quality threshold.The quality of the VMI derives from the quality of the cell with maximum reflectivity,
    corrected according to specific parameters.

    Args:
        threshVoid (float): Below this threshold (default = 0) all values are considered Void (not significant).
        threshQuality (float): Minimum quality threshold; data with lower quality are not considered (default = 0).
        min_el (float): Minimum elevation to consider (if min_el >= max_el all elevations are considered).
        max_el (float): Maximum elevation to consider (by default max_el = min_el)
        max_height (float): Maximum altitude above which the data are not considered (default = 12000 m).
        smoothBox (int): Polar smoothing expressed as box width, i.e. number of cells in azimuth and range
                         (smooth * smooth) (default = 0)
        medianfilter: Median filter (alternative to smoothing filter) (default = 0)
        checkVoid: If non-zero the output matrix is filtered using the VOID matrix
        maxQifZero: By default, with zero reflectivity, the quality value coincides with the quality of the lowest elevation;
                    with maxQifZero = 1, the maximum quality value is reported on the vertical;
                    with maxQifZero > 1, the maximum quality value among the first maxQifZero elevations is reported.
        volume (np.ndarray): Volume to process.
        qVolume (np.ndarray): Quality voume.
        main (Node): Sampled volume node.


    Keywords:
        ATTR: Attributes for product coding (process.txt)
        VOID: Optional matrix to filter out non-significant cells


    Returns:
        VMI (np.ndarray): Array containing the polar matrix of the maximum
        QUALITY (np.ndarray): Array containing the quality of the selected cells
        HEIGHTS (np.ndarray): Array containing the heights of the selected cells
    """

    if isinstance(main, dpg.node__define.Node):
        node = main

    dim = volume.shape
    if len(dim) < 2:
        return

    out = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    site_coords = out["site_coords"]
    range_res = out["range_res"]
    el_coords = out["el_coords"]

    nEl = len(el_coords)
    if nEl <= 0:
        return

    threshVoid, site_name, _ = dpg.radar.get_par(prodId, "threshVoid", 0.0)
    threshQual, _, _ = dpg.radar.get_par(prodId, "threshQuality", 0.0, prefix=site_name)
    min_el, _, _ = dpg.radar.get_par(prodId, "min_el", -1.0, prefix=site_name)
    max_el, _, _ = dpg.radar.get_par(prodId, "max_el", min_el, prefix=site_name)
    max_height, _, _ = dpg.radar.get_par(prodId, "max_height", 12000.0, prefix=site_name)
    smoothBox, _, _ = dpg.radar.get_par(prodId, "smooth", 0, prefix=site_name)
    medianFilter, _, _ = dpg.radar.get_par(prodId, "medianFilter", 0, prefix=site_name)
    checkVoid, _, _ = dpg.radar.get_par(prodId, "checkVoid", 0, prefix=site_name)
    maxQifZero, _, _ = dpg.radar.get_par(prodId, "maxQifZero", 0, prefix=site_name)

    indEl = np.arange(nEl)
    if max_el > min_el:
        # Serve solo nel caso della KDP, per prendere un numero limitato di elevation
        indEl = np.where((el_coords >= min_el) & (el_coords <= max_el))[0]
        nEl = len(indEl)
        if nEl <= 0:
            return None, None, None

    # endif
    dtype = volume.dtype  # size(volume, /TYPE)
    eee = indEl[0]
    vmi = volume[eee, :, :].copy()

    if threshVoid > 0.0:
        vmi = np.where(vmi < threshVoid, -np.inf, vmi)
    if qVolume.size == volume.size:
        quality = qVolume[eee, :, :].copy()
        if threshQual > 0.0:
            vmi = np.where(quality < threshQual, np.nan, vmi)
        if maxQifZero > 0:
            if maxQifZero == 1:
                maxQifZero = nEl
            if maxQifZero > nEl:
                maxQifZero = nEl
            for iii in range(1, maxQifZero):
                tmpQ = qVolume[indEl[iii], :, :].copy()
                quality = np.where(quality < tmpQ, tmpQ, quality)
    else:
        if qVolume.size > 1:
            print("Error: Invalid Quality...resetting!")
            return
    # endif

    heights = np.zeros(dim[1:3])
    heightBeams = dpg.access.get_height_beams(
        el_coords[eee], dim[2], range_res, site_height=site_coords[2], projected=True
    )
    for aaa in range(dim[1]):
        heights[aaa, :] = heightBeams.copy()

    tmpH = heights.copy()

    for iii in range(1, nEl):
        eee = indEl[iii]
        tmp = volume[eee, :, :].copy()
        if threshVoid > 0.0:
            tmp = np.where(tmp < threshVoid, -np.inf, tmp)
        # endif
        if qVolume.size == volume.size:
            tmpQ = qVolume[eee, :, :].copy()
            if threshQual > 0.0:
                tmp = np.where(tmpQ < threshQual, np.nan, tmp)
            # endif
        # endif
        heightBeams = dpg.access.get_height_beams(
            el_coords[eee],
            dim[2],
            range_res,
            site_height=site_coords[2],
            projected=True,
        )
        for aaa in range(dim[1]):
            tmpH[aaa, :] = heightBeams.copy()

        # endif
        tmp = np.where(tmpH > max_height, np.nan, tmp)

        x, y = np.where(np.isnan(vmi) & np.logical_not(np.isnan(tmp)))

        vmi[x, y] = tmp[x, y].copy()
        heights[x, y] = tmpH[x, y].copy()
        quality[x, y] = tmpQ[x, y].copy()

        x, y = np.where(tmp > vmi)
        vmi[x, y] = tmp[x, y].copy()
        heights[x, y] = tmpH[x, y].copy()
        quality[x, y] = tmpQ[x, y].copy()
    # endfor

    if checkVoid > 0 and void is not None:
        if void.size == vmi.size:
            vmi = np.where(np.isnan(void), -np.inf, vmi)
        # endif
    # endif

    heights = np.where(np.isnan(vmi), -np.inf, heights)
    return vmi, quality, heights
