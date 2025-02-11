import numpy as np
import sou_py.dpg as dpg

"""
Funzioni ancora da portare
FUNCTION GetZoneAlert 
FUNCTION LoadCurrentZones 
PRO CreateMaskedArray 
PRO SaveCurrentZones 
FUNCTION GetPercentile         // UNUSED
"""


def computePercentile(
    values: np.ndarray,
    perc: np.ndarray,
    thresh: float = None,
    count_thresh: int = None,
    valid_thresh: float = None,
    up_thresh: float = None,
) -> np.ndarray:
    """
    Calculates specified percentiles of a given array of values, applying optional thresholds.

    This function computes the percentiles of an array of values. It can filter the values
    based on given thresholds and returns percentiles as specified. The function handles
    empty arrays and threshold conditions, and can compute maximum, minimum, and mean values.

    Args:
        values (np.ndarray): An array of values for which percentiles are to be calculated.
        perc (np.ndarray of float): The percentiles to compute (values between 0 and 100).
        thresh (float, optional): The lower threshold for filtering values. Defaults to None.
        count_thresh (int, optional): The minimum count of values above 'thresh' required
                                      to compute percentiles. Defaults to None.
        valid_thresh (float, optional): The minimum percentage of valid values (non-NaN)
                                        required to compute percentiles. Defaults to None.
        up_thresh (float, optional): The upper threshold for filtering values. Defaults to None.

    Returns:
        np.ndarray: An array of calculated percentile values. If conditions are not met
                    for calculation, returns an array of NaNs or zeros.

    Note:
        The function filters out NaN values and applies optional thresholds to determine
        which values to include in the percentile calculation. It handles special cases
        where the requested percentile is 0 or 100, and the case where the minimum and
        maximum values are equal. The function can return early with NaNs or zeros if the
        specified thresholds are not met.
    """
    # Set initial values for maximum, minimum, and mean to NaN
    maximum = np.nan
    minimum = np.nan
    mean = np.nan

    nP = np.size(perc)
    if nP <= 0:
        return np.nan

    # Set default percentile to NaN
    percentile = np.full(nP, np.nan)

    # Filter non-NaN values
    valid_values = values[np.isfinite(values)]
    if len(valid_values) == 0:
        return percentile

    maximum = np.nanmax(valid_values)
    minimum = np.nanmin(valid_values)
    mean = np.nanmean(valid_values)

    if up_thresh is not None:
        valid_values = valid_values[valid_values < up_thresh]

    if thresh is not None:
        ct = count_thresh if count_thresh is not None else 0
        valid_values = valid_values[valid_values >= thresh]
        if len(valid_values) <= ct:
            return percentile

    if valid_thresh is not None and valid_thresh > 0.0:
        valids = (len(valid_values) * 100.0) / len(values)
        if valids < valid_thresh:
            return np.zeros(nP)

    if nP == 1:
        if perc[0] >= 100.0:
            return maximum
        if perc[0] <= 0.0:
            return minimum
        if minimum >= maximum:
            return minimum

    sorted_indices = np.argsort(valid_values)
    th = (perc * len(valid_values) / 100.0).astype(int)

    return valid_values[sorted_indices[th]]


def computeROIStatistics(
    data: np.ndarray,
    mask: np.ndarray = None,
    values: list = None,
    xres: float = None,
    yres: float = None,
    simmetric: bool = False,
    thresh_filter: float = None,
) -> dict:
    """
    Computes statistical measures of a Region of Interest (ROI) in a data array, applying a mask and thresholds.

    This function calculates various statistical measures for a given data array,
    considering a provided mask, value thresholds, and spatial resolution. It handles
    null and void values, applies threshold filters, and computes statistics like mean,
    standard deviation, sum, and a histogram of the data within the ROI.

    Args:
        data (np.ndarray): The data array for which statistics are to be computed.
        mask (np.ndarray, optional): A mask array to specify the ROI. Non-zero values
                                     are considered valid. Defaults to None.
        values (list, optional): A list of values used in value mapping. Defaults to None.
        xres (float, optional): The x-resolution of the data in meters. Used for area
                                calculations. Defaults to None.
        yres (float, optional): The y-resolution of the data in meters. Used for area
                                calculations. Defaults to None.
        simmetric (bool, optional): Indicates whether the data is symmetric. Defaults
                                    to False.
        thresh_filter (float, optional): A threshold value for filtering the data.
                                         Defaults to None.

    Returns:
        dict: A dictionary containing computed statistical measures, including 'n_samples',
              'sum', 'minimum', 'maximum', 'mean', 'stdev', 'sum_of_squares', 'variance',
              'hist' (histogram), and 'locations' (bin locations for the histogram).

    Note:
        The function applies the mask to filter the data, considers null and void values,
        and applies threshold filters if provided. It computes statistics only on the
        valid data as defined by the mask and thresholds. If spatial resolutions are
        provided, it calculates the total and valid areas of the ROI. The function uses
        NumPy's histogram function to compute the histogram and bin locations.
    """

    discarded = 0
    first_valid = 0
    if data.size <= 0:
        return

    array, ind_null, null_values, ind_void, count_void = dpg.values.get_valued_array(
        data, values, simmetric=simmetric
    )

    if mask is not None:
        area_samples = np.sum(mask != 0)
    else:
        area_samples = data.size

    if (
        null_values > 0
        or len(ind_null) > 0
        or thresh_filter is not None
        or count_void > 0
    ):
        if mask is None:
            mask = np.ones_like(data, dtype=np.uint8)
        if null_values > 0:
            mask[ind_null] = 0
        if len(ind_void) > 0:
            mask[ind_void] = 0
        if thresh_filter is not None:
            discarded_indices = np.where(array < thresh_filter)
            mask[discarded_indices] = 0

    if array.size <= 1:
        return

    # Compute statistics using numpy functions
    if mask is not None:
        n_samples = np.sum(mask)
        valid_data = array[mask != 0]
        sum_val = np.sum(valid_data)
        minimum = np.min(valid_data)
        maximum = np.max(valid_data)
        mean = np.mean(valid_data)
        stdev = np.std(valid_data, ddof=1)
        sum_of_squares = np.sum(valid_data**2)
        variance = stdev**2
    else:
        n_samples = array.size
        sum_val = np.sum(array)
        minimum = np.min(array)
        maximum = np.max(array)
        mean = np.mean(array)
        stdev = np.std(array, ddof=1)
        sum_of_squares = np.sum(array**2)
        variance = stdev**2

    if xres is not None and yres is not None:
        xr = abs(xres) / 1000.0
        yr = abs(yres) / 1000.0
        valid_area = n_samples * xr * yr
        total_area = area_samples * xr * yr

    # Histogram and other statistics
    hist, bin_edges = np.histogram(
        array, bins=256, range=(thresh_filter, maximum), density=False
    )
    locations = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    return {
        "n_samples": n_samples,
        "sum": sum_val,
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "stdev": stdev,
        "sum_of_squares": sum_of_squares,
        "variance": variance,
        "hist": hist,
        "locations": locations,
    }
