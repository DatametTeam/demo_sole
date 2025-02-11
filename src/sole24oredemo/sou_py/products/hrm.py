import numpy as np
import pandas as pd
from scipy.signal import correlate, correlate2d

import sou_py.dpg as dpg
import sou_py.dpb as dpb
import geopandas as gpd
import sou_py.products as products
from sou_py.dpg.log import log_message


def cross_correlation(x, y, L):
    """
    Computes the normalized cross-correlation between two input signals `x` and `y` with a specified lag `L`.

    Parameters:
        x (array): First signal or data series.
        y (array): Second signal or data series.
        L (int): Lag for cross-correlation. Positive values shift `x` to the right, negative values shift `x` to the
        left.

    Returns:
        float: Normalized cross-correlation value.
    """

    N = len(x)

    # Means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Normalization factors (denominators in the formula)
    denominator_x = np.sum((x - x_mean) ** 2)
    denominator_y = np.sum((y - y_mean) ** 2)

    # Compute the numerator
    if L < 0:
        numerator = np.sum((x[: N + L] - x_mean) * (y[-L:N] - y_mean))
    else:
        numerator = np.sum((x[L:N] - x_mean) * (y[: N - L] - y_mean))

    # Return the normalized cross-correlation value
    return numerator / np.sqrt(denominator_x * denominator_y)


def computeMotion(field_t0, field_t1, threshVel, minStep):
    """
    Computes the normalized cross-correlation between two input signals `x` and `y` with a specified lag `L`.

    Parameters:
        x (array): First signal or data series.
        y (array): Second signal or data series.
        L (int): Lag for cross-correlation. Positive values shift `x` to the right, negative values shift `x` to the
        left.

    Returns:
        float: Normalized cross-correlation value.
    """

    dim = np.shape(field_t0)
    winSize = np.shape(field_t1)

    ySteps = dim[0] - winSize[0]
    xSteps = dim[1] - winSize[1]

    corr = np.zeros((xSteps, ySteps), dtype=np.float32)

    thresh = threshVel * minStep / 60.0
    thresh *= thresh

    for lll in range(ySteps):
        yyy = lll - ySteps / 2.0
        for ccc in range(xSteps):
            xxx = ccc - xSteps / 2.0
            vvv = yyy * yyy + xxx * xxx
            if vvv <= thresh:
                corr[lll, ccc] = cross_correlation(
                    field_t0[
                    lll: lll + winSize[1],
                    ccc: ccc + winSize[0],
                    ],
                    field_t1,
                    0,
                )

    c_max = np.nanmax(corr)

    if not np.isfinite(c_max):
        vel = np.nan
        dir = np.nan
        return 0, vel, dir

    ind = np.where(corr >= c_max * 0.9)
    # col = ind[1] % xSteps
    # lin = ind[0] % xSteps
    col = ind[1] + winSize[1] / 2.0
    lin = ind[0] + winSize[0] / 2.0
    col = col - dim[1] / 2.0
    lin = lin - dim[0] / 2.0

    tmp = corr[ind]
    tmp *= tmp
    den = np.sum(tmp)
    xxx = np.sum(col * tmp) / den
    yyy = np.sum(lin * tmp) / den

    # dir = 450.0 - 180.0 / np.pi * np.arctan2(yyy, xxx)
    dir = 450.0 - 180.0 / np.pi * np.arctan2(yyy, -xxx)
    dir = np.float32(int(dir + 0.5))

    if dir <= 0:
        dir += 360
    if dir >= 360:
        dir -= 360

    vel = np.sqrt(yyy * yyy + xxx * xxx)
    vel *= 60.0 / minStep
    vel = np.float32(int(vel + 0.5))

    return c_max, vel, dir


def hrm_motion(
        points, currData, prevData, par, inRadius, outRadius, threshVel, minStep
):
    """
    Processes storm data to calculate motion (velocity and direction) for each storm in the current and previous frames.

    Parameters:
        points (array): Array of storm points.
        currData (array): Current frame data.
        prevData (array): Previous frame data.
        par (dict): Parameter dictionary.
        inRadius (int): Radius for input region.
        outRadius (int): Radius for output region.
        threshVel (float): Threshold for velocity.
        minStep (float): Minimum step size for motion calculation.

    Returns:
        tuple:
            - array: Array of correlation values for each storm.
            - array: Array of calculated velocities for each storm.
            - array: Array of calculated directions for each storm.
    """

    dim = points.shape
    if len(dim) <= 1:
        x = points[0]
        y = points[1]
    else:
        x = points[
            :, 0
            ]  # TODO: da controllare che le dimensioni siano giuste (forse al contrario)
        y = points[:, 1]

    nStorms = len(x)
    vel = np.zeros(nStorms, dtype=np.float32)
    dir = np.zeros(nStorms, dtype=np.float32)
    corr = np.zeros(nStorms, dtype=np.float32)

    if np.size(prevData) <= 1:
        return

    if np.size(prevData) != np.size(currData):
        return

    dim = np.shape(currData)

    ind = np.where(~np.isfinite(prevData))
    prevData[ind] = 0.0

    ind = np.where(~np.isfinite(currData))
    currData[ind] = 0.0

    y, x = dpg.map.yx_2_lincol(y, x, par, dim=dim, set_center=True)

    sh = x - outRadius
    ind = np.where(sh < 0)
    x[ind] -= sh[ind]

    sh = x + outRadius - dim[1]
    ind = np.where(sh > 0)
    x[ind] -= sh[ind]

    sh = y - outRadius
    ind = np.where(sh < 0)
    y[ind] -= sh[ind]

    sh = y + outRadius - dim[0]
    ind = np.where(sh > 0)
    y[ind] -= sh[ind]

    for sss in range(nStorms):
        pData = prevData[
                int(y[sss] - outRadius): int(y[sss] + outRadius),
                int(x[sss] - outRadius): int(x[sss] + outRadius),
                ]
        cData = currData[
                int(y[sss] - inRadius): int(y[sss] + inRadius),
                int(x[sss] - inRadius): int(x[sss] + inRadius),
                ]
        corr[sss], vvv, ddd = computeMotion(pData, cData, threshVel, minStep)
        vel[sss] = vvv
        dir[sss] = ddd

    return corr, vel, dir


def computeMeanMotion(
        currDir, currVel, currCorr, prevDir, prevVel, prevCorr, threshCorr
):
    """
    Processes storm data to calculate motion (velocity and direction) for each storm in the current and previous frames.

    Parameters:
        points (array): Array of storm points.
        currData (array): Current frame data.
        prevData (array): Previous frame data.
        par (dict): Parameter dictionary.
        inRadius (int): Radius for input region.
        outRadius (int): Radius for output region.
        threshVel (float): Threshold for velocity.
        minStep (float): Minimum step size for motion calculation.

    Returns:
        tuple:
            - array: Array of correlation values for each storm.
            - array: Array of calculated velocities for each storm.
            - array: Array of calculated directions for each storm.
    """
    if not np.isfinite(currVel) or currVel < 1 or currDir < 0:
        meanDir = prevDir
        meanCorr = prevCorr
        return meanDir, prevVel, meanCorr

    if not np.isfinite(prevVel) or prevVel < 1.0 or prevDir < 0.0:
        meanDir = currDir
        meanCorr = currCorr
        return meanDir, currVel, meanCorr

    if currCorr < threshCorr:
        meanDir = prevDir
        meanCorr = prevCorr
        return meanDir, prevVel, meanCorr

    if currCorr > prevCorr + 0.3:
        meanDir = currDir
        meanCorr = currCorr
        return meanDir, currVel, currCorr

    if currVel < prevVel / 10.0:
        meanDir = prevDir
        meanCorr = prevCorr
        return meanDir, prevVel, meanCorr

    sumCorr = currCorr + prevCorr

    meanVel = (currCorr * currVel + prevCorr * prevVel) / sumCorr
    meanVel = float(int(meanVel + 0.5))

    meanCorr = (currCorr * currCorr + prevCorr * prevCorr) / sumCorr

    if meanVel < 5:
        meanDir = prevDir
        return meanDir, meanVel, meanCorr

    sector = abs(currDir - prevDir)
    ccc = currDir
    ppp = prevDir

    if sector > 180.0:
        sector = 360.0 - sector
        if ppp > ccc:
            ppp -= 360.0
        else:
            ccc -= 360.0

    meanDir = (currCorr * ccc + prevCorr * ppp) / sumCorr
    if meanDir < 0.0:
        meanDir += 360.0

    meanDir = float(int(meanDir + 0.5))
    if meanDir >= 360.0:
        meanDir = 0.0

    return meanDir, meanVel, meanCorr


def hrm_check(prodId, currPoints, currStorms, vel, dir, corr, threshVel, map):
    """
    Validates and adjusts storm information by comparing the current storm data with previous data. It checks the
    distance between current and previous storm points and updates the storm parameters based on correlation and
    velocity.

    Parameters:
        prodId (Node): Production ID for the current node.
        currPoints (array): Current storm points.
        currStorms (DataFrame): DataFrame containing current storm data.
        vel (array): Array of current velocities.
        dir (array): Array of current directions.
        corr (array): Array of current correlation values.
        threshVel (float): Threshold for velocity.
        map (GeoDataFrame): Geographic map data.

    Returns:
        tuple:
            - DataFrame: Updated current storm data.
            - array: Updated correlation values.
    """

    threshDist, _, _ = dpg.radar.get_par(prodId, "threshDist", 15000.0)
    threshCorr, _, _ = dpg.radar.get_par(prodId, "threshCorr", 0.5)
    minCheck, _, _ = dpg.radar.get_par(prodId, "minCheck", 5)
    prevNode, _, _ = dpg.times.getPrevNode(prodId, minCheck)

    log_message(
        f"Performing HRM_Check with node: {dpg.tree.getNodePath(prevNode)}",
        level="INFO",
    )

    nCurr = len(vel)
    nPrev, prevPoints = dpg.coords.get_points(prevNode, destMap=map, center=True)
    prev, nFeat, parnames = dpg.coords.get_shape_info(prevNode, names=True)

    if nFeat <= 0:
        dpg.tree.removeTree(prevNode)
        return

    pd = currStorms["Dir"].copy()
    pv = currStorms["Vel"].copy()
    pc = np.zeros(nCurr) + 1

    if nPrev > 0:
        xxx = prevPoints[:, 0]
        yyy = prevPoints[:, 1]
        xxx, yyy = dpg.map.translatePoints(xxx, yyy, prev["Vel"], prev["Dir"], minCheck)
        scores = np.zeros((nPrev, nCurr), dtype=np.float32)
        for ccc in range(nCurr):
            dX = xxx - currPoints[ccc, 0]
            dY = yyy - currPoints[ccc, 1]
            scores[:, ccc] = np.sqrt((dX * dX) + (dY * dY))
            minV = np.min(scores[:, ccc])
            ppp = np.argmin(scores[:, ccc])
            if minV < 3 * threshDist:
                pd[ccc] = prev.loc[ppp]["Dir"]
                pv[ccc] = prev.loc[ppp]["Vel"]
                pc[ccc] = prev.loc[ppp]["Cor"]

        minV = np.min(scores)
        ind = np.where(scores == minV)
        while minV < threshDist:
            ccc = ind[1]
            ppp = ind[0]
            name = (currStorms.loc[ccc, "Name"]).values[0]
            name = (prev.loc[ppp, "Name"]).values[0][:7] + name[7:]
            currStorms.loc[ccc, "Name"] = name
            pd.loc[ccc] = prev.loc[ppp, "Dir"].values[0]
            pv.loc[ccc] = prev.loc[ppp, "Vel"].values[0]
            pc[ccc] = prev.loc[ppp, "Cor"].values[0]
            scores[:, ccc] = threshDist
            scores[ppp, :] = threshDist
            minV = np.min(scores)
            ind = np.where(scores == minV)

    for sss in range(nCurr):
        meanDir, meanVel, meanCorr = computeMeanMotion(
            dir[sss], vel[sss], corr[sss], pd[sss], pv[sss], pc[sss], threshCorr
        )
        if meanVel > threshVel:
            meanVel = threshVel
        currStorms.loc[sss, "Vel"] = meanVel
        currStorms.loc[sss, "Dir"] = meanDir
        corr[sss] = meanCorr

    dpg.tree.removeTree(prevNode)
    return currStorms, corr


def hrm(prodId, hrd, main):
    """
    Main function that coordinates storm motion and validation, and outputs the results to a shapefile. It retrieves
    storm data, computes motion and correlation, and updates storm properties based on predefined thresholds.

    Parameters:
        prodId (Node): Production ID for the current node.
        hrd (Node): HRD (historical data) node containing storm data.
        main (Node): Main node for data retrieval.

    Returns:
        None
    """
    valids = False

    if not isinstance(prodId, dpg.node__define.Node):
        return

    threshold, _, _ = dpg.radar.get_par(prodId, "threshold", 3.0)
    threshIndex, _, _ = dpg.radar.get_par(prodId, "threshIndex", threshold)
    minStep, _, _ = dpg.radar.get_par(prodId, "minStep", 15.0)
    inRadius, _, _ = dpg.radar.get_par(prodId, "inRadius", 30)
    outRadius, _, _ = dpg.radar.get_par(prodId, "outRadius", 90)
    threshVel, _, _ = dpg.radar.get_par(prodId, "threshVel", 160.0)
    shpfile, _, _ = dpg.radar.get_par(prodId, "shpfile", "hrm.shp")

    storms, nFeat, parnames = dpg.coords.get_shape_info(hrd, names=True)
    if nFeat <= 0:
        return

    map, par = dpb.dpb.get_geo_info(main, map=True)

    nStorms, points = dpg.coords.get_points(hrd, destMap=map, center=True)

    if nStorms > 0:
        indSSI = parnames.index("SSI")
        indHRI = parnames.index("HRI")

        if indHRI >= 0:
            maxIndex = storms[["SSI", "HRI"]].max(axis=1)
        else:
            maxIndex = storms["SSI"]

        valids = np.where(maxIndex > threshIndex)
        nStorms = len(valids)

    if nStorms > 0:
        points = points[valids]
        storms = storms.loc[valids]
        currFrame = dpb.dpb.get_data(main)
        prevFrame = dpb.dpb.get_prev_data(main, min_step=minStep)
        corr, vel, dir = hrm_motion(
            points, currFrame, prevFrame, par, inRadius, outRadius, threshVel, minStep
        )
        storms, corr = hrm_check(
            prodId, points, storms, vel, dir, corr, threshVel, map=map
        )
        _, _, _, _, _, _, _, _, ent = dpg.navigation.check_map(hrd, coords=True)

    parnames = parnames + ["Cor"]
    nFeat += 1

    outPath = dpg.tree.getNodePath(prodId)
    outFile = dpg.path.getFullPathName(outPath, shpfile)

    storms["Cor"] = corr
    storms = storms.map(
        lambda x: (
            int(x * 1000) / 1000 if isinstance(x, float) and not np.isnan(x) else x
        )
    )
    storms.to_file(outFile)

    dpg.tree.replaceAttrValues(
        prodId,
        dpg.cfg.getGeoDescName(),
        ["coordfile", "format"],
        [shpfile, "shp"],
        only_current=True,
        to_save=True,
    )
    shpfile = shpfile[:-3] + "dbf"
    dpg.tree.replaceAttrValues(
        prodId,
        dpg.cfg.getArrayDescName(),
        ["datafile", "format"],
        [shpfile, "dbf"],
        only_current=True,
        to_save=True,
    )

    return
