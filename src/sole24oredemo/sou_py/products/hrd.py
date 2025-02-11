import os

import fiona
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely import LineString, Polygon
from skimage.draw import polygon
from skimage.measure import find_contours

import sou_py.dpg as dpg
import sou_py.products as products
from sou_py.dpg.log import log_message
import warnings


def hrd_detect(feature, threshold, par):
    """
    Detects and transforms contours based on a threshold in a 2D feature array.

    This function detects contours in the provided 2D feature array (`feature`) that correspond to a specified
    threshold value. It then converts the contour coordinates from a linear index system to geographic coordinates.
    The contours are returned as transformed coordinates.

    Args:
        feature (numpy.ndarray): A 2D array containing the feature data, where the contours are to be detected.
        threshold (float): The threshold value used to detect contours. Contours are formed at this level.
        par (list): Parameters that define the mapping transformation used to convert contour coordinates.

    Returns:
        contours_xy (list of numpy.ndarray): A list of 2D arrays where each array contains the transformed
                                              (x, y) coordinates of a contour. Each contour corresponds to
                                              a detected region in the `feature` array.

    """
    dim = feature.shape

    tmp = feature.copy()

    indNull = np.where(~np.isfinite(tmp))
    tmp[indNull] = 0

    contours = find_contours(
        feature, level=threshold, fully_connected="high", positive_orientation="low"
    )

    contours_xy = contours.copy()
    for idx, contour in enumerate(contours):
        datay, datax = dpg.map.lincol_2_yx(
            contour[:, 0], contour[:, 1], par, set_center=True
        )
        contours_xy[idx][:, 0] = datax
        contours_xy[idx][:, 1] = datay

    return contours_xy


def HRD_init(prodId, names):
    """
    Initializes and prepares a GeoDataFrame for storing HRD (High Resolution Data) features.

    Args:
        prodId (Node): Node which is used to access configuration parameters for the shapefile and other settings.
        names (list of str): A list of feature names for which data will be stored in the GeoDataFrame.

    Returns:
        outShape (GeoDataFrame): An empty GeoDataFrame initialized with the necessary columns to store HRD feature data.
        getMax (numpy.ndarray): An array of integers representing the maximum values for each feature.
        outFile (str): The full file path for the output shapefile where the data will be saved.
    """

    shpfile, _, _ = dpg.radar.get_par(prodId, "shpfile", "hrd.shp")
    currPath = dpg.tree.getNodePath(prodId)
    outFile = dpg.path.getFullPathName(currPath, shpfile)

    # outShape = fiona.open(outFile, mode='w')
    outShape = gpd.GeoDataFrame()
    outShape.__setattr__("output_path", outFile)

    outShape["Name"] = pd.Series([], dtype="string")
    outShape["Time"] = pd.Series([], dtype="string")
    outShape["Area"] = pd.Series([], dtype="float64")
    # outShape["n_vertices"] = pd.Series(dtype="Int64")

    nFeat = len(names)
    getMax = np.zeros(nFeat, dtype="int")

    for idx, ppp in enumerate(names):
        outShape[ppp] = pd.Series([], dtype="float64")
        gm, _, _ = dpg.radar.get_par(prodId, "getMax", 1, prefix=ppp)
        getMax[idx] = gm

    return outShape, getMax, outFile


def HRD_GetName(index, point, strTime):
    """
    Generates a formatted string for a given HRD feature based on the provided index, point, and time.

    This function constructs a string identifier for a High Resolution Data (HRD) feature, including the feature's
    index, time, and product name (if available). The function formats the time and index into a specific format,
    and appends the product name and its provenance if available, ensuring the final string does not exceed 30 characters.

    Args:
        index (int): The index of the feature, used to generate a unique identifier.
        point (tuple): The coordinates (or point) for the HRD feature, used to retrieve product information.
        strTime (str): The timestamp string (in the format YYYY-MM-DD HH:MM:SS) used to extract the time part of the identifier.

    Returns:
        str: A formatted string containing the time, index, product name (if available), and provenance (if available).

    """

    ddd = strTime.strip()
    len_d = len(ddd)
    ttt = ddd[len_d - 5 : len_d - 5 + 2]
    ttt += ddd[len_d - 2 :]
    str_i = str(index + 1).strip()
    if len(str_i) == 1:
        str_i = "0" + str_i
    str_i = ttt + str_i

    name, _, prov, _ = products.commanager.commanager(point, prov=True, name=True)

    if np.size(name) != 1:
        return str_i

    str_i += " " + name
    if len(str_i) > 30:
        str_i = str_i[:30]
    if prov != "":
        str_i += f" ({prov})"

    return str_i


def HRD_setAttr(
    outShape,
    features,
    path_xy,
    sourceMap,
    getMax,
    par,
    min_area,
    max_area,
    HRI,
    threshIndex,
    strTime,
    names,
):
    """
    Sets the attributes of a GeoDataFrame for HRD (High Resolution Data) features, including geometric
    information (polygons) and associated data such as the area and feature values.

    Args:
        outShape (GeoDataFrame): A GeoDataFrame to store the results, including geometric data and feature attributes.
        features (numpy.ndarray): A 3D array containing feature data (e.g., radar data) used to populate the GeoDataFrame.
        path_xy (list): A list of arrays, where each array contains XY coordinates defining the boundaries of HRD features.
        sourceMap (Map): A map that provides the mapping between XY and latitude/longitude coordinates.
        getMax (numpy.ndarray): An array indicating how to compute maximum or mean values for features (1 for max, 2 for mean).
        par (list): Parameters for the calculation, including resolution or other geographical data.
        min_area (float): The minimum area threshold for a feature to be considered valid.
        max_area (float): The maximum area threshold for a feature to be considered valid.
        HRI (numpy.ndarray): A 2D array of HRI (Human Readable Index) values used to filter valid features.
        threshIndex (float): The threshold HRI value below which the feature is discarded.
        strTime (str): A string representing the time of the HRD feature for inclusion in the GeoDataFrame.
        names (list): A list of feature names (e.g., radar parameters) corresponding to each feature in the `features` array.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the updated HRD features, including geometry and computed attributes,
                      or `None` if no valid features are found.

    """

    valids = 0
    nInfo = len(path_xy)
    if nInfo <= 0:
        log_message("No Storms", level="WARNING+")
        return None

    if outShape is None:
        return None

    dim = features.shape
    nEl = dim[-1] * dim[-2]
    nF = dim[0]
    dim = dim[-2:]
    path_ll = [
        np.array(
            [
                (lon, lat)
                for lat, lon in [
                    dpg.map.yx_2_latlon(y, x, sourceMap) for x, y in array
                ]
            ]
        )
        for array in path_xy
    ]

    www = 0
    fact = np.abs(par[1] * par[3]) / 1000000.0

    for iii in range(nInfo):
        lin, col = dpg.map.yx_2_lincol(
            path_xy[iii][:, 1], path_xy[iii][:, 0], par, dim=dim
        )

        rr, cc = polygon(lin, col, dim)
        mask = np.zeros(dim, dtype=np.uint8)
        mask[rr, cc] = 1

        ind = np.where(mask > 0)
        count = len(ind[0])
        if count > 0:
            count = count * fact
            maxHRI = np.nanmax(HRI[ind])
            www = np.nanargmax(HRI[ind])
            if maxHRI <= threshIndex:
                count = 0
        if min_area < count <= max_area:
            # outShape.at[iii, "n_vertices"] = len(path_xy[iii])
            outShape.at[iii, "geometry"] = Polygon(path_ll[iii])
            # outShape = outShape.set_geometry([Polygon(path_ll[iii])])
            indP = (ind[0][www], ind[1][www])
            outShape.at[iii, "Name"] = HRD_GetName(valids, indP, strTime)
            outShape.at[iii, "Time"] = strTime
            outShape.at[iii, "Area"] = count
            ind1 = ind
            for fff in range(nF):
                mf = features[fff][indP]
                if getMax[fff] == 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # RuntimeWarning: All-NaN slice encountered
                        mf = np.nanmax(features[fff][ind1])
                if getMax[fff] == 2:
                    mf = np.nanmean(features[fff][ind1])
                if ~np.isfinite(mf):
                    mf = -9999
                outShape.at[iii, names[fff]] = mf
                # outShape[names[fff]] = outShape[names[fff]].apply(
                #     lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else x)
                # outShape[names[fff]] = pd.to_numeric(outShape[names[fff]], errors='coerce')
            valids += 1

    if len(outShape) == 0:
        log_message("No Storms", level="WARNING")

    outShape = outShape.reset_index(drop=True)
    return outShape


def recomputeIndex(attr, minV, maxV, www, replaceMax, normFact, ind):
    if replaceMax is not None and len(replaceMax) == 2:
        log_message("TODO: da fare questa parte", level="ERROR")
        pippo

    nC = len(ind[0])
    data = np.zeros(nC, dtype=np.double)
    for ccc in range(nC):
        data[ccc] = attr.iloc[ind[0][ccc]]

    iii = np.where(np.isnan(data))
    if len(iii[0]) > 0:
        data[iii] = minV[iii]

    vvv = (data - minV) / (maxV - minV)
    iii = np.where(vvv < 0)
    vvv[iii] = 0
    iii = np.where(vvv > 1)
    vvv[iii] = 1

    return np.sum(www * vvv) * normFact


def HRD_Recompute(prodId, outShape, valids, indexName, names, attr_set=None):
    """
    Recomputes the HRD (High Resolution Data) index for each storm feature based on the provided radar data
    and associated parameters. The function recalculates the index (e.g., HRI) for each feature by applying a
    formula that considers the minimum, maximum, and weight values for each feature. Additionally, the function
    can apply corrections based on a specified replacement index.

    Args:
        prodId (Node): The product node used to retrieve radar parameters and configuration settings for recomputing the index.
        outShape (GeoDataFrame): A GeoDataFrame containing the attributes of the features that will be updated with the new index.
        valids (int): The number of valid features to process.
        indexName (str): The name of the index to recompute (e.g., HRI).
        names (list): A list of feature names that correspond to radar parameters.
        attr_set (GeoDataFrame, optional): A fallback GeoDataFrame containing feature attributes, used if `outShape` is `None`.

    Returns:
        GeoDataFrame or None: The updated `outShape` GeoDataFrame with the recalculated HRD index for each feature, or `None` if no valid features are processed.

    """

    replaceMax = None

    if np.size(valids) < 0 or valids is None:
        return

    hriIndex = names.index(indexName)
    if np.size(hriIndex) != 1:
        return

    parFile = indexName.lower() + ".par"
    nFeat = len(names) + 3
    minV = np.zeros(nFeat, dtype=np.float32)
    maxV = np.zeros(nFeat, dtype=np.float32)
    www = np.zeros(nFeat, dtype=np.float32)

    for fff in range(3, nFeat):
        name = names[fff - 3]
        vvv, _, _ = dpg.radar.get_par(prodId, "min", 0.0, prefix=name, parFile=parFile)
        minV[fff] = vvv
        vvv, _, _ = dpg.radar.get_par(prodId, "max", 0.0, prefix=name, parFile=parFile)
        maxV[fff] = vvv
        vvv, _, _ = dpg.radar.get_par(prodId, "weight", 1.0, prefix=name, parFile=parFile)
        www[fff] = vvv
        rep, _, _ = dpg.radar.get_par(prodId, "replaceMax", "", prefix=name, parFile=parFile)
        if rep != "":
            toRep = np.where(names == rep)
            if len(toRep[0]) == 1:
                replaceMax = [fff, toRep[0] + 3]

    ind = np.where(minV < maxV)
    if len(ind[0]) <= 0:
        return

    hriIndex = hriIndex + 3
    minV = minV[ind]
    maxV = maxV[ind]
    www = www[ind]
    totW = np.sum(www)
    maxNorm, _, _ = dpg.radar.get_par(prodId, "maxNorm", totW)
    normFact = maxNorm / totW

    for eee in range(valids):
        if eee < len(outShape):
            attr = outShape.loc[eee].copy()
        else:
            attr = attr_set.loc[eee]
        newIndex = recomputeIndex(attr, minV, maxV, www, replaceMax, normFact, ind=ind)

        if attr.iloc[hriIndex] < newIndex:
            log_message(f"{indexName} = {attr.iloc[hriIndex]}")
            log_message(f"... new = {newIndex}")
            if outShape is not None:
                attr.iloc[hriIndex] = newIndex
                outShape.iloc[eee, hriIndex] = newIndex
            else:
                attr_set.at[eee, hriIndex] = newIndex

    return outShape


def hrd(prodId, features, names, HRI, main=None, indexName=None):
    """
    Processes radar feature data to compute HRD indices (HRI, SSI) and stores results in a shapefile.
    The function detects storms, sets feature attributes, and recalculates indices based on radar parameters.

    Args:
        prodId (Node): Product node for retrieving radar parameters and data.
        features (np.ndarray): Array of feature data.
        names (list): List of feature names.
        HRI (ndarray): High-resolution index data.
        main (Node, optional): Main node for map and parameter retrieval.
        indexName (str, optional): Name of the index to recompute (defaults to "HRI" and "SSI").

    Returns:
        None: Saves the results to a shapefile and updates attributes in the tree.
    """

    dim = features.shape

    if len(dim) < 3:
        log_message("Invalid Feature Volume", level="WARNING+")

    nEl = dim[-1] * dim[-2]

    threshold, _, _ = dpg.radar.get_par(prodId, "threshold", 2.0)
    threshIndex, _, _ = dpg.radar.get_par(prodId, "threshIndex", threshold)
    min_area, _, _ = dpg.radar.get_par(prodId, "min_area", 20.0)
    max_area, _, _ = dpg.radar.get_par(prodId, "max_area", 5000.0)
    featureIndex, _, _ = dpg.radar.get_par(prodId, "", -1)

    if isinstance(main, dpg.node__define.Node):
        _, sourceMap, _, par, _, _, _, _, _ = dpg.navigation.check_map(main)
    else:
        _, sourceMap, _, par, _, _, _, _, _ = dpg.navigation.check_map(
            prodId, par=True, sourceMap=True, mosaic=True
        )

    if 0 <= featureIndex < dim[0]:
        log_message("CASO DA FIXARE", level="ERROR")
        path_xy = hrd_detect()
    else:
        path_xy = hrd_detect(HRI, threshold, par)

    outShape, getMax, outFile = HRD_init(prodId, names)

    date, time, _ = dpg.times.get_time(prodId)
    strTime = dpg.times.checkDate(date, sep="-", year_first=True)
    time, _, _ = dpg.times.checkTime(time)
    strTime = strTime + "T" + time

    outShape = HRD_setAttr(
        outShape=outShape,
        features=features,
        getMax=getMax,
        sourceMap=sourceMap,
        par=par,
        min_area=min_area,
        max_area=max_area,
        path_xy=path_xy,
        HRI=HRI,
        threshIndex=threshIndex,
        strTime=strTime,
        names=names,
    )

    valids = len(outShape)

    if indexName is not None:
        outShape = HRD_Recompute(prodId, outShape, valids, indexName, names)
    else:
        outShape = HRD_Recompute(prodId, outShape, valids, "HRI", names)
        outShape = HRD_Recompute(prodId, outShape, valids, "SSI", names)

    outShape = outShape.map(
        lambda x: (
            int(x * 1000) / 1000 if isinstance(x, float) and not np.isnan(x) else x
        )
    )
    outShape.to_file(outFile)

    shpfile = os.path.basename(outFile)
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
