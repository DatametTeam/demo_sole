import numpy as np
import sou_py.dpg as dpg


def correctDistQuality(quality, res, rMax, rMin):
    """
        This function modifies the radar data quality array based on the distance of range points (rMin and rMax).
        The quality is scaled such that values farther from the maximum range (rMax) have lower quality.

    Args:
        quality:    The initial array of radar data quality
        res:        The spatial resolution of the radar data
        rMax:       The maximum range beyond which the quality is reduced
        rMin:       The minimum range below which the quality is reduced

    Returns:
        quality:    The quality array adjusted based on the distance

    Note:
        The function first calculates the polar range using the resolution and the number of bins.
        It then scales the quality and applies this correction to all quality data along the polar dimension.
    """
    if rMax <= rMin:
        return quality

    dim = quality.shape

    range_pol = dpg.map.getPolarRange(res, nBins=dim[1])

    qD = (rMax - range_pol) / (rMax - rMin)

    qD = np.where(qD < 0, 0, qD)
    qD = np.where(qD > 1, 1, qD)

    qD = np.sqrt(qD)

    for ddd in range(dim[0]):
        quality[ddd, :] *= qD

    return quality


def correctDEMQuality(quality, valuesHeight, dem, hMax, hMin):
    """
    This function adjusts the radar data quality array based on the distance of range points (rMin and rMax).
    The quality is scaled such that points farther from the maximum range (rMax) and closer to the minimum range (rMin) are assigned lower quality.

    Args:
        quality (array): The initial array representing radar data quality.
        res (float): The spatial resolution of the radar data.
        rMax (float): The maximum range; quality decreases for points beyond this range.
        rMin (float): The minimum range; quality decreases for points below this range.

    Returns:
        array: The adjusted quality array based on the distance.

    Note:
        The function first computes the polar range using the resolution and the number of bins.
        It then applies a scaling factor to adjust the quality values, ensuring the correction is applied
        to all data along the polar dimension.
    """

    if hMax <= hMin:
        return quality

    if valuesHeight.shape != quality.shape:
        return quality

    if dem is None:
        return quality

    elif dem.shape != quality.shape:
        return quality

    delta = valuesHeight - dem

    qH = (hMax - delta) / (hMax - hMin)
    qH = np.where(qH < 0, 0, qH)
    qH = np.where(qH > 1, 1, qH)
    qH = np.nan_to_num(qH, nan=0)

    quality *= qH

    return quality


def quality_adj(prodId, quality, data, heights=None):
    """
    This algorithm calculates adjustments to the radar data quality based on various parameters.
    It applies specific thresholds to reduce quality for invalid data and makes corrections
    based on height and distance.

    Args:
        prodId (str):    The radar product ID used to access configured parameters.
        quality (array): The initial array of radar data quality.
        data (array):    The radar data array used for threshold and validity checks.
        heights (array): The array of heights corresponding to the radar data.

    Returns:
        array: The quality array containing the polar matrix of maxima.
    """

    if quality is None:
        return

    rMin, site_name, _ = dpg.radar.get_par(prodId, "rMin", 0.0)
    rMax, _, _ = dpg.radar.get_par(prodId, "rMax", 0.0, prefix=site_name)
    rangeRes, _, _ = dpg.radar.get_par(prodId, "rangeRes", 1000.0, prefix=site_name)
    hMin, _, _ = dpg.radar.get_par(prodId, "hMin", 2000.0, prefix=site_name)
    hMax, _, _ = dpg.radar.get_par(prodId, "hMax", 0.0, prefix=site_name)
    reduceVoidQuality, _, _ = dpg.radar.get_par(prodId, "reduceVoidQuality", 0.0, prefix=site_name)
    threshVoid, _, _ = dpg.radar.get_par(prodId, "threshVoid", 0.0, prefix=site_name)
    absVoid, _, _ = dpg.radar.get_par(prodId, "absVoid", 0.0, prefix=site_name)

    par = dpg.navigation.get_radar_par(prodId)
    if par["range_res"] is not None and par["range_res"] != 0:
        rangeRes = par["range_res"]

    quality = correctDistQuality(quality, rangeRes, rMax, rMin)

    if len(data) > 0:
        if reduceVoidQuality > 0:
            if absVoid > 0:
                indVoid_x, indVoid_y = np.where(
                    np.logical_or(abs(data) <= threshVoid, np.isfinite(data) <= 0)
                )
            else:
                indVoid_x, indVoid_y = np.where(data <= threshVoid)

            if len(indVoid_x) > 0:
                quality[indVoid_x, indVoid_y] *= reduceVoidQuality

        ix, iy = np.where(np.isnan(data))
        quality[ix, iy] = 0

    if hMax > hMin:
        if np.size(heights) == np.size(quality):
            dim = heights.shape
            outMap, sourceMap, _, par, ispolar, isvertical, _, _, _ = (
                dpg.navigation.check_map(prodId, destMap=True)
            )
            dem = dpg.warp.get_dem(outMap, par, dim, numeric=True)
            quality = correctDEMQuality(quality, heights, dem, hMax, hMin)

    return quality
