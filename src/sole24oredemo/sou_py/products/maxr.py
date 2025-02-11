import time

import numpy as np
import sou_py.dpg as dpg
import sou_py.dpb as dpb
from numba import njit
import warnings


@njit()
def computeMaxR(values, qualities, heights, maxQifzero, lbmQual, avg=0):
    # TODO: verificare la correttezza del commento
    """
    Method that execute a statistic analisys on a polar coordinates dataset (values).


    Args:
        values: R volume usually output of the SAMPLING procedure with /PROJECTED option and transformed with MARSHALL_PALMER.
        qualities: Quality volume of the same R volume size.
        heights: Height of radar beams for each scan in an elevation set.
        maxQifzero: Value of "maxQifZero" for the analyzed node.
        lbmQual: Value of "lbmQuality" for the analyzed node.

    Keywords:
        AVG: Value of "avg" for the analyzed node.

    Returns:
       totR/totQ: Result of the division of (maximum values * quality of maximum values) and (quality of maximum values).
       q: Quality of maximum values.
       h: Height of maximum values.
    """

    ind = np.isfinite(values)
    if not np.any(ind):
        return np.nan, np.nan, np.nan
    (ind,) = np.where(ind)
    maxInd = np.argmax(values[ind])  # max(values[ind], maxInd, /NAN)
    maxR = values[ind[maxInd]]
    if maxR <= 0.0:
        if maxQifzero == 0:
            q = qualities[ind[0]]
        elif maxQifzero == 1:
            q = np.nanmax(qualities[ind])
        else:
            to = maxQifzero if maxQifzero < qualities.size else qualities.size
            q = np.nanmax(qualities[0:to])
        # end
        h = heights[ind[0]]
        return maxR, q, h
    # endif
    q = qualities[ind[maxInd]]
    h = heights[ind[maxInd]]

    if maxInd <= 0 or (not avg):
        return maxR, q, h

    if lbmQual >= 0.0:
        qT = q if q > lbmQual else lbmQual
    else:
        qT = q + lbmQual
        if qT < 0.0:
            qT = q
    # endif

    mmmInd = np.where(qualities[ind[0:maxInd]] > qT)
    count = mmmInd[0].size
    if count == 0:
        return maxR, q, h

    totR = maxR * q
    totH = h * q
    totQ = q

    ind = ind[mmmInd]
    r = values[ind] * qualities[ind]
    indFinite = np.isfinite(r)
    totR += np.sum(r[indFinite])
    r = heights[ind] * qualities[ind]
    indFinite = np.isfinite(r)
    totH += np.sum(r[indFinite])
    r = qualities[ind]
    indFinite = np.isfinite(r)
    totQ += np.sum(r[indFinite])

    q = totQ / (count + 1.0)
    h = totH / totQ

    return totR / totQ, q, h


@njit()
def compute_nested(
    dim,
    heightBeams,
    rVolume,
    qVolume,
    maxQifZero,
    lbmQuality,
    avg,
    outPointer,
    quality,
    heights,
    indEl,
):
    """
    Algorithm for estimating ground precipitation from a 3D volume of R. Also, for each cell, it returns the detected
    quality value and the corresponding quota.

    This function is used to support the main function MAXR, it iterates along the second and third axes of qVolume and
    rVolume to find the maximum value, and the corresponding height and quality.

    Args:
        dim: Dimension of the elevetion set.
        heightBeams: Height of radar beams for each scan in an elevation set.
        rVolume: Volume usually output of the SAMPLING procedure with /PROJECTED option and transformed with MARSHALL_PALMER.
        qVolume: Quality volume of the same R volume size.
        maxQifzero: By default, where it DOES NOT rain, the quality value coincides with the quality of the lowest elevation with
                    maxQifZero = 1, the maximum value of quality is reported on the vertical
                    with maxQifZero > 1, the maximum value of quality is reported among the first maxQifZero elevations.
        lbmQuality: Quality threshold lbm (default = 80) (if 0 is ignored).
        avg: Activate alternative method (default = 0).
        outPointer: Array containing the polar matrix R (float 2D) (the result is null if there are no values that meet the minimum conditions).
        quality: Quality of maximum values.
        heights: Height of radar beams for each scan in an elevation set.
        indEl: Index of coordinate values.

    Returns:
        outPointer: 2D matrix containing details about precipitation
        qualities:  2D matrix containing details about data quality
        heights:    2D matrix containing details about heights
    """

    for rrr in range(dim[2]):
        hBeam = heightBeams[:, rrr].copy()
        for aaa in range(dim[1]):
            qualities = qVolume[indEl, aaa, rrr]
            values = rVolume[indEl, aaa, rrr]
            (ind,) = np.where(np.isfinite(values))
            if not np.any(ind):
                outPointer[aaa, rrr], quality[aaa, rrr], heights[aaa, rrr] = (
                    np.nan,
                    np.nan,
                    np.nan,
                )
                continue
            maxInd = np.argmax(values[ind])  # max(values[ind], maxInd, /NAN)
            maxR = values[ind[maxInd]]
            if maxR <= 0.0:
                if maxQifZero == 0:
                    q = qualities[ind[0]]
                elif maxQifZero == 1:
                    q = np.nanmax(qualities[ind])
                else:
                    to = maxQifZero if maxQifZero < qualities.size else qualities.size
                    q = np.nanmax(qualities[0:to])
                # end
                h = hBeam[ind[0]]
                outPointer[aaa, rrr] = maxR
                quality[aaa, rrr] = q
                heights[aaa, rrr] = h
                continue
            # endif
            q = qualities[ind[maxInd]]
            h = hBeam[ind[maxInd]]

            if maxInd <= 0 or (not avg):
                outPointer[aaa, rrr], quality[aaa, rrr], heights[aaa, rrr] = maxR, q, h

            if lbmQuality >= 0.0:
                qT = q if q > lbmQuality else lbmQuality
            else:
                qT = q + lbmQuality
                if qT < 0.0:
                    qT = q
            # endif

            mmmInd = np.where(qualities[ind[0:maxInd]] > qT)
            count = mmmInd[0].size
            if count == 0:
                outPointer[aaa, rrr], quality[aaa, rrr], heights[aaa, rrr] = maxR, q, h
                continue
            totR = maxR * q
            totH = h * q
            totQ = q

            ind = ind[mmmInd]
            r = values[ind] * qualities[ind]
            indFinite = np.isfinite(r)
            totR += np.nansum(r[indFinite])
            r = hBeam[ind] * qualities[ind]
            r = r.flatten()
            indFinite = np.isfinite(r).flatten()
            totH += np.sum(r[indFinite])
            r = qualities[ind]
            indFinite = np.isfinite(r)
            totQ += np.sum(r[indFinite])

            q = totQ / (count + 1.0)
            h = totH / totQ

            outPointer[aaa, rrr], quality[aaa, rrr], heights[aaa, rrr] = (
                totR / totQ,
                q,
                h,
            )

    return outPointer, quality, heights


def find_max_indices_values(rVolume):
    """
    Description:
        Finds the maximum values and their indices along the first dimension of rVolume.
        If all values along the first dimension are NaN for a particular cell, the index is set to -1.

    Parameters:
        rVolume:     A 3D matrix where the first dimension represents depth, and the other two dimensions
                     represent the cells for which the maximum values and their indices are calculated.

    Returns:
        max_indices: A 2D matrix of indices indicating the position of the maximum value along the first
                     dimension for each cell. If all values are NaN, the index is -1.
        max_values:  A 2D matrix of the maximum values along the first dimension for each cell.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # RuntimeWarning: All-NaN slice encountered 
        #       max_values = np.nan_to_num(np.nanmax(rVolume, axis=0), nan=0.0)
        # Find max value along dim 0
        max_values = np.nanmax(rVolume, axis=0)
    # Create a mask for argmax, when all values along dim 0 are NaN put -1
    all_nan_mask = np.all(np.isnan(rVolume), axis=0)
    nan_inf = np.nan_to_num(rVolume, nan=-np.inf)
    max_indices = np.where(all_nan_mask, -1, np.nanargmax(nan_inf, axis=0))

    return max_indices, max_values


def qt_mask(max_indices, qT, qVolume):
    """
    Description:
        Generates a mask based on the given thresholds and indices.
        This function creates a mask where only the elements with an index lower than the corresponding value
        in `max_indices` and a value greater than the corresponding element in `qT` are set to True.

    Args:
        max_indices: Matrix of indices indicating the maximum valid index for each cell.
                     Elements with a value of -1 indicate no valid index.
        qT:          Matrix representing the threshold values for each cell.
        qVolume:     Volume where the first dimension represents the depth, and the other two dimensions
                     correspond to the cells for which the mask is being created.

    Returns:
        mask:        A 3D boolean volume where the elements are True if the corresponding element in `qVolume`
                     is greater than `qT` and the index is lower than the corresponding value in `max_indices`.
    """
    # Only those cells where at least one element is not NaN
    valid_indices_mask = max_indices > -1
    # 2D Matrix with max indices, when no valid ind put 0
    valid_indices = np.where(valid_indices_mask, max_indices, 0)
    # First mask: consider only indices lower than max_indices
    depth_range = np.arange(qVolume.shape[0])[:, None, None]
    idx_mask = depth_range < valid_indices
    # Second mask: element greater than qT (dim 0)
    qVolume_mask = qVolume > qT[None, :, :]
    # Return a mask where only elements with index lower than max_indices and with a value greater than qT are True
    return idx_mask & qVolume_mask


def calculate_qualities_heights_out(rVolume, qVolume, heightBeams, maxQifzero, lbmQual):
    """
    Description:
        Calculate three 2D matrices that return information on precipitation, data quality and altitude.
    Args:
        rVolume:     3D volume that gives information about precipitation
        qVolume:     Volume of the same dimension 'Volume', in each cell there is a percentage value that indicates the
                     quality of the respective Volume cell
        heightBeams: 2D matrix with information about heights
        maxQifzero:  By default, where it DOES NOT rain, the quality value coincides with the quality of the lowest
                     elevation with maxQifZero = 1, the maximum value of quality is reported on the vertical with
                     maxQifZero > 1, the maximum value of quality is reported among the first maxQifZero elevations.
        lbmQual:     Quality threshold lbm

    Returns:
        outPointer: 2D matrix containing details about precipitation
        qualities:  2D matrix containing details about data quality
        heights:    2D matrix containing details about heights
    """
    # Find max values along dim 0 and their indices
    max_indices, max_values = find_max_indices_values(rVolume)

    ind = min(qVolume.shape[0], maxQifzero)

    # Create a mask for max_values less than or equal to 0
    mask_neg = max_values <= 0

    # Initialize qualities and heights
    qualities = np.full(max_values.shape, np.nan)
    heights = np.full(max_values.shape, np.nan)

    # Find max value from 0 to 'ind'
    q_max_negative = np.nanmax(qVolume[:ind, :, :], axis=0)

    # Find first valid number along dim 0
    first_ind = np.apply_along_axis(lambda x: np.argmax(np.isfinite(x)), 0, rVolume)

    # Update qualities when max <= 0
    if maxQifzero == 0:
        qualities[mask_neg] = qVolume[
            first_ind[mask_neg], np.where(mask_neg)[1], np.where(mask_neg)[2]
        ]
    elif maxQifzero == 1:
        qualities[mask_neg] = np.nanmax(qVolume[:, :, :], axis=0)[mask_neg]
    else:
        qualities[mask_neg] = q_max_negative[mask_neg]

    # Update qualities when max > 0
    mask_pos = ~mask_neg
    qualities[mask_pos] = qVolume[
        max_indices[mask_pos], np.where(mask_pos)[0], np.where(mask_pos)[1]
    ]

    # max <= 0
    heights[mask_neg] = heightBeams[first_ind[mask_neg], np.where(mask_neg)[1]]

    # max > 0
    heights[mask_pos] = heightBeams[max_indices[mask_pos], np.where(mask_pos)[1]]

    # qT matrix
    if lbmQual >= 0.0:
        qT = np.where(lbmQual > qualities, lbmQual, qualities)
    else:
        qT = qualities + lbmQual
        qT = np.where(qT < 0.0, qualities, qT)

    # Initialize outPointer
    outPointer = max_values.copy()

    # mask for qT
    updated_mask = qt_mask(max_indices, qT, qVolume)

    # Update values only when max_values>0
    flat_mask = np.any(updated_mask, axis=0) & (max_values > 0)

    # Weighted average
    totR = max_values[flat_mask] * qualities[flat_mask]
    totH = heights[flat_mask] * qualities[flat_mask]
    totQ = qualities[flat_mask]

    r1 = np.where(updated_mask, rVolume * qVolume, 0)
    totR += np.sum(r1, axis=0)[flat_mask]
    r2 = np.where(updated_mask, qVolume, 0)
    totQ += np.sum(r2, axis=0)[flat_mask]
    count = np.sum(updated_mask, axis=0)[flat_mask]
    replicated_h = np.repeat(
        heightBeams[:, np.newaxis, :], updated_mask.shape[1], axis=1
    )
    r3 = np.where(updated_mask, replicated_h * qVolume, 0)
    totH += np.sum(r3, axis=0)[flat_mask]

    qualities[flat_mask] = totQ / (count + 1.0)
    outPointer[flat_mask] = totR / totQ
    heights[flat_mask] = totH / totQ

    return outPointer, qualities, heights


def MAXR(prodId, rVolume, qVolume, node, indVoid=None):
    """
    Algorithm for estimating ground precipitation from a 3D volume of R.
    Also, for each cell, it returns the detected quality value and the corresponding quota.
    You need the Quality volume.
    Research the maximum value along the vertical: the search is limited to high quality values and low altitude.
    There are 2 thresholds: lbmQuality and lbmHeigth.
    The alternative 'selective weighted average' (option avg=1) is provided for:
    in this mode' the weighted average (with quality') of all values with quality greater than maxq and altitude lower than Maxh is calculated,
    where maxq and Maxh are respectively the quality and dimension associated with the maximum value of R along the vertical.


    Args:
       prodId: Product node, from which you access the following optional parameters contained in parameters.txt.
       lbmQuality: Quality threshold lbm (default = 80) (if 0 is ignored).
       avg: Activate alternative method (default = 0).
       min_el: Minimum elevation to be considered (if min_el >= max_el all elevations are used).
       max_el: Maximum elevation to be considered (by default max_el=min_el).
       max_height: Maximum altitude, above which data are not considered (default = 12000 m).
       maxQifzero:  By default, where it DOES NOT rain, the quality value coincides with the quality of the lowest elevation
                    with maxQifZero = 1, the maximum value of quality is reported on the vertical
                    with maxQifZero > 1, the maximum value of quality is reported among the first maxQifZero elevations.
       threshQuality: Minimum quality threshold, lower quality data are not considered (default = 0).
       smooth: Smoothing Polare espresso come ampiezza del box, Ovvero numero di celle in Azimut e range (smooth * smooth) (default = 0).
       medianfilter: Median Filter (Alternative to Smoothing Filter) (default = 0).
       rVolume: R volume usually output of the SAMPLING procedure with /PROJECTED option and transformed with MARSHALL_PALMER.
       qVolume: Quality volume of the same R volume size.


    :keywords:
        - NODE: Sampled volume node (required to access volume properties).
        - ATTR: Attributes for product coding (process.txt).
        - INDVOID: Index vector to set to 0 (useful to cancel noise-affected cells).


    :return:
       - **Outpointer**: Array containing the polar matrix R (float 2D) (the result is null if there are no values that meet the minimum conditions).
       - **QUALITY**: Array containing the quality of the selected cells (float 2D).
       - **HEIGHTS**: Array containing selected cell quotas (2D float).

    Note:
       In case of avg=1, Quality and Heigths matrices are also mediated.
    """

    rVolume = np.asarray(rVolume)
    qVolume = np.asarray(qVolume)
    if rVolume.size <= 1 or qVolume.size <= 1:
        raise ValueError("multidimensional array expected")
    if qVolume.size != rVolume.size:
        raise ValueError("quality size is not compatible")
    if (not isinstance(prodId, dpg.node__define.Node)) or (
        not isinstance(node, dpg.node__define.Node)
    ):
        raise ValueError("object of class Node expected")

    out = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    site_coords = out["site_coords"]
    range_res = out["range_res"]
    el_coords = out["el_coords"]
    nEl = el_coords.size
    if nEl <= 0:
        return None

    threshVoid, site_name, _ = dpg.radar.get_par(prodId, "threshVoid", 0.0)
    threshQual, _, _ = dpg.radar.get_par(prodId, "threshQuality", 50.0, prefix=site_name)
    lbmQuality, _, _ = dpg.radar.get_par(prodId, "lbmQuality", threshQual, prefix=site_name)
    min_el, _, _ = dpg.radar.get_par(prodId, "min_el", -1.0, prefix=site_name)
    max_el, _, _ = dpg.radar.get_par(prodId, "max_el", min_el, prefix=site_name)
    max_height, _, _ = dpg.radar.get_par(prodId, "max_height", 12000.0, prefix=site_name)
    avg, _, _ = dpg.radar.get_par(prodId, "avg", 0, prefix=site_name)
    maxQifZero, _, _ = dpg.radar.get_par(prodId, "maxQifZero", 0, prefix=site_name)
    smoothBox, _, _ = dpg.radar.get_par(prodId, "smooth", 0, prefix=site_name)
    medianFilter, _, _ = dpg.radar.get_par(prodId, "medianFilter", 0, prefix=site_name)

    indEl = np.arange(nEl)
    if max_el > min_el:
        indEl = np.where(el_coords >= min_el and el_coords <= max_el)
        nEl = len(indEl)
        if nEl <= 0:
            return
        nEl = len(indEl[0])
        if nEl <= 0:
            return
    # endif

    dim = rVolume.shape
    dtype = rVolume.dtype
    # CHECK_IN, IN=node, OUT=prodId, DIM=dim[0:1], TYPE=type, POINTER=outPointer
    # if ptr_valid(outPointer) le 0 then return
    # outPointer = np.zeros((dim[1], dim[2]), dtype=dtype)

    # quality = np.zeros((dim[1], dim[2]))
    # heights = np.zeros((dim[1], dim[2]))
    heightBeams = np.zeros((nEl, dim[2]))

    for eee in range(nEl):
        heightBeams[eee, :] = dpg.access.get_height_beams(
            el_coords[indEl[eee]],
            dim[2],
            range_res,
            site_height=site_coords[2],
            projected=True,
        )
        rVolume[indEl[eee]] = np.where(
            heightBeams[eee, :] > max_height, np.nan, rVolume[indEl[eee]]
        )
    outPointer, quality, heights = calculate_qualities_heights_out(
        rVolume, qVolume, heightBeams, maxQifZero, lbmQuality
    )

    """for rrr in range(dim[2]):
        hBeam = heightBeams[:, rrr].copy()
        for aaa in range(dim[1]):
            x, q, h = computeMaxR(
                rVolume[indEl, aaa, rrr],
                qVolume[indEl, aaa, rrr],
                hBeam,
                maxQifZero,
                lbmQuality,
                avg=avg,
            )
            outPointer[aaa, rrr] = x
            quality[aaa, rrr] = q
            heights[aaa, rrr] = h
        # endfor
    # endfor"""

    if indVoid is not None:
        outPointer[indVoid] = 0

    return outPointer, quality, heights
