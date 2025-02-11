import numpy as np
from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg


def GetSeverityIndex(data, minV, maxV, www, no_max=False):
    """
    Calculate a severity index by normalizing input data and applying scaling.

    This function computes a normalized severity index based on the range defined by `minV`
    (minimum value) and `maxV` (maximum value). The resulting values are scaled and optionally
    clamped between 0 and 1, with an additional weighting factor applied.

    Args:
        data (array): Input data to be normalized.
        minV (float): Minimum value for normalization.
        maxV (float): Maximum value for normalization.
        www (float): Weighting factor to scale the normalized output.
        no_max (bool, optional): If `True`, values greater than 1 are not clamped; defaults to `False`.

    Returns:
        array: The scaled severity index.
    """
    out = (data - minV) / (maxV - minV)
    out[((~np.isfinite(out)) | (out < 0.0))] = 0.0

    if not no_max:
        out[out > 1.0] = 1.0

    if www == 1.0:
        return out

    return www * out


def SeverityIndex(prodId, features, names, parFile=None):
    """
    Procedure for calculating severity indices.

    Args:
        prodId (str):    Product node used to access the file containing the required parameters.
        features (array): A 3D array of type float containing the feature data.
        names (list):     Names of the features.
        parfile (str):    Name of the file containing the range limits and weights for each component (e.g., `ssi.par` or `hri.par`).

    Returns:
        s_index (array): An array containing the severity index values. By default, the severity index is normalized to 1,
                         unless the `maxNorm` parameter is defined (default set to the sum of weights).
    """
    s_index = None

    if len(features.shape) < 3:
        log_message("Undefined feature volume", level="WARNING")
        return s_index

    nFeat = np.size(names)  # o len(names)?
    dim = features.shape
    if nFeat != dim[0]:
        log_message("Invalid feature volume", level="WARNING")
        return s_index

    totW = 0.0
    for ppp in range(nFeat):
        minV , _, _= dpg.radar.get_par(prodId, "min", 0.0, prefix=names[ppp], parFile=parFile)
        maxV, _, _ = dpg.radar.get_par(prodId, "max", 0.0, prefix=names[ppp], parFile=parFile)
        if maxV > minV:
            www, _, _ = dpg.radar.get_par(prodId, "weight", 1.0, prefix=names[ppp], parFile=parFile)
            no_max, _, _ = dpg.radar.get_par(prodId, "no_max", 0.0, prefix=names[ppp], parFile=parFile)
            totW += www
            data = features[ppp, :, :]
            if s_index is None:
                s_index = GetSeverityIndex(data, minV, maxV, www, no_max=no_max)
            else:
                s_index += GetSeverityIndex(data, minV, maxV, www, no_max=no_max)

    maxNorm, _, _ = dpg.radar.get_par(prodId, "maxNorm", totW)

    if maxNorm != totW:
        s_index *= maxNorm / totW

    s_index[s_index > maxNorm] = maxNorm

    return s_index
