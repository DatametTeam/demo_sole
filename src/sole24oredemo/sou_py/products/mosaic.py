"""
Realizza la mosaicatura dei dati radar utilizzando algoritmi flessibili.
L'algoritmo predefinito applica una media ponderata selettiva basata sulla qualità dei dati;
in assenza del dato di qualità, viene utilizzato il criterio del massimo.
La copertura del mosaico è configurabile tramite file dedicato o predefinita su una matrice 1400x1200 con
risoluzione di 1 km² per pixel, proiettata in Transverse Mercator (origine Roma, WGS84).
"""
import os
import shutil
import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb

from sou_py.dpg.log import log_message

computed_maps = {}


def get_product_node(mosaicNode):
    # TODO: verificare la correttezza del commento
    """
    Algorithm that takes as input a node of a tree structure that represents a mosaic (mosaicNode) and returns
    a node of the mosaic itself for which the element "attr" (name of the schedule) associated is defined.

    Args:
        mosaicNode: Node of a mosaic.

    Returns:
       - curr: First node below mosaicNode for which attr exists, otherwise twisted mosaicNode.
    """

    attr = dpg.tree.getAttr(mosaicNode, dpg.cfg.getProductDescName())
    name = dpg.cfg.getScheduleDescName()

    curr = mosaicNode
    next_node = mosaicNode.parent
    while dpg.tree.node_valid(next_node):
        attr = dpg.tree.getSingleAttr(next_node, name=name, only_current=True)
        if attr:
            return curr
        curr = next_node
        next_node = next_node.parent

    return mosaicNode


def CheckMosaicDesc(mosaicNode, nNodes, name=[], auto=0):
    # TODO: verificare la correttezza del commento
    """
    Algorithm that takes as input a node of a tree structure representing a mosaic (mosaicNode) and also
    the number of nodes of which it is composed and returns the geographical descriptor associated with the latter.

    Args:
        mosaicNode: Node of a mosaic.
        nNodes: Number of nodes in the tree structure representing the mosaic.

    :keywords:
       - name: Name of a .txt file related to an attribute.
       - auto: Default value associated with attribute name.

    Returns:
       - **idnav**: Geographic descriptor associated with the tree node passed to the algorithm.
       - **auto**: Value of "auto" attribute associated to the mosaic descriptor.
    """

    # in metodo viene riempito il navigation.txt di mosaic con il mosaic.txt della cartella padre
    # in IDL sembrano esserci molti parametri in più a seguito dell'aggiunta dei siti..

    navName = dpg.cfg.getGeoDescName()
    idnav = dpg.tree.getSingleAttr(mosaicNode, navName)

    if idnav is not None:
        return idnav

    if nNodes == 1:
        if len(name) != 1:
            name = "default.txt"
            auto = 1
    else:
        if len(name) != 1:
            name = dpg.cfg.getMosaicDescName()
            auto = 0

    # in questo punto viene caricato il mosaic.txt
    attr = dpg.tree.getSingleAttr(mosaicNode, name)

    if attr:
        from_node = mosaicNode
    else:
        from_node = dpg.tree.createTree("$NAVIGATIONS", shared=True)
        attr = dpg.tree.getSingleAttr(from_node, name)

    auto, _, _ = dpg.attr.getAttrValue(attr, "auto", auto)
    _, _ = dpg.tree.copyAttr(from_node, mosaicNode, name, dest_name=navName)
    idnav = dpg.tree.getSingleAttr(mosaicNode, navName)

    return idnav, auto


def GetNodesToMosaic(mosaicNode):
    # TODO: verificare la correttezza del commento
    """
    Identifies the nodes related to a mosaic and retrieves additional information about their relationships.

    Args:
        mosaicNode (Node): ode of a mosaic.

    Returns:
        tuple:
       - count (int): The number of valid nodes associated with the mosaic.
       - auto (int): Value associated to the "auto" attribute of the mosaic descriptor.
       - find_quality (int or None): it's indicating whether quality nodes were found (1) or not (None).
       - siteNodes(list[Node]): A list of valid nodes associated with the mosaic.
    """

    # in questo metodo viene fuori che nNodes = 20 e invece (probabilmente) dovrebbero essere di più
    prod = get_product_node(mosaicNode)
    find_quality = None
    nNodes = 0

    if prod == mosaicNode or prod == mosaicNode.parent:
        nNodes, siteNodes = dpg.tree.getBrothers(mosaicNode)
    else:
        nodes = dpg.tree.findAllDescendant(prod, dpg.tree.getNodeName(mosaicNode))
        if np.size(nodes) <= 0:
            return 0
        siteNodes = [nnn for nnn in nodes if nnn != mosaicNode]
        nNodes = len(siteNodes)
        find_quality = 1

    if nNodes <= 0:
        return 0, None, None, None

    nav, auto = CheckMosaicDesc(mosaicNode, nNodes)

    sites, _, _ = dpg.attr.getAttrValue(nav, "site", "")
    names = []
    for nnn in range(nNodes):
        names.append(dpg.tree.getNodeName(siteNodes[nnn]))
        ind = np.where(sites == names[nnn])[0]
        count = len(ind)
        if count > 0:
            names[nnn] = ""

    ind = [i for i in range(len(names)) if names[i] != ""]
    count = len(ind)
    if count > 0:
        siteNodes = [siteNodes[index] for index in ind]
        names = [names[index] for index in ind]
        ret = dpg.attr.replaceTags(
            nav, ["site" for i in range(count)], names, to_add=True
        )

    return count, auto, find_quality, siteNodes


def mosaic3d(data, col, lin, outPointer, nPlanes):
    """
    Creates a 3D mosaic by merging input data into the existing output array based on specified conditions

    Args:
        data (np.ndarray): The input data array of shape (nPlanes, nPoints), where nPlanes is the number of planes
                           and nPoints is the number of data points per plane
        col (np.ndarray): Column indices corresponding to the input data
        lin (np.ndarray): Line indices corresponding to the input data
        outPointer (np.ndarray): The output 3D array of shape (nPlanes, nRows, nCols) where the mosaic is stored
        nPlanes (int): The number of planes in the 3D data

    Returns:
        np.ndarray: The updated `outPointer` array with the merged data

    Notes:
        - The function updates `outPointer` plane by plane. For each plane, it flattens the plane, determines the
          current mosaic values, and compares them with the corresponding values in the input `data`
        - Values in `data` replace those in `outPointer` if the corresponding value in `outPointer` is smaller or
          if it is NaN (not a number)
        - The mosaic operation is performed on indices calculated using the `lin` and `col` arrays, ensuring that
          only specified positions are updated
        - After updating the current plane, the flattened data is reshaped back to its original dimensions and stored
          in `outPointer`

    """
    dim = np.shape(outPointer)
    index = lin * dim[2] + col

    for lll in range(nPlanes):
        tmp1 = outPointer[lll].copy()
        tmp1 = tmp1.flatten()
        currMos = tmp1[index].copy()
        tmp2 = data[lll, :]
        minInd = np.where((currMos < tmp2) | np.isnan(currMos))
        currMos[minInd] = tmp2[minInd]
        tmp1[index] = currMos
        outPointer[lll] = tmp1.reshape(dim[-2:])

    return outPointer


def mosaicMaxEqual(data, quality, col, lin, outPointer, outQuality):
    """
    Updates the output mosaic and quality matrices by comparing the input data and quality values

    Args:
        data (np.ndarray): Input data array representing values to integrate into the mosaic
        quality (np.ndarray): Input quality array corresponding to `data`, indicating quality scores for each value
        col (np.ndarray): Column indices where the data will be placed in the output mosaic
        lin (np.ndarray): Line indices where the data will be placed in the output mosaic
        outPointer (np.ndarray): Output data array representing the mosaic to be updated
        outQuality (np.ndarray): Output quality array representing the quality of each value in the mosaic

    Returns:
        tuple:
            - outPointer (np.ndarray): Updated mosaic array with integrated data
            - outQuality (np.ndarray): Updated quality array with integrated quality scores
    """
    if outQuality is not None and np.size(quality) == np.size(data):
        currMos = outPointer[lin, col]
        currQual = outQuality[lin, col]
        minInd = np.where(currQual < quality)
    else:
        currMos = outPointer[lin, col]
        # minInd = where(currMos lt data or finite(currMos, /NAN), count)
        minInd = np.where((currMos < data) | np.isnan(currMos))

    if np.size(minInd) <= 0:
        return outPointer, outQuality

    currMos[minInd] = data[minInd]
    outPointer[lin, col] = currMos

    if np.size(currQual) <= 0:
        return outPointer, outQuality

    currQual[minInd] = quality[minInd]
    outQuality[lin, col] = currQual

    return outPointer, outQuality


def checkCurrMap(destPar, destMap, destDim):
    # TODO: verificare la correttezza del commento
    """
    Algorithm that deals with returning the complete coordinates of longitude and latitude inherent in the elements
    in the map passed in input.

    Args:
        destPar: Parameters of the map.
        destMap: Map obtained from mosaicNode passed to the mosaic algorithm.
        destDim: DestMap size.

    Returns:
        - **M_lon**: Vector containing the longitudes of the elements in the map.
        - **M_lat**: Vector containing latitudes of the elements in the map.
    """

    key = (tuple(destPar), destMap.mapProj.definition, tuple(destDim))
    if key in computed_maps.keys():
        # If yes, return the stored results
        return computed_maps[key]

    M_proj_name = ""
    M_proj_lat = 0
    M_proj_lon = 0
    M_par = np.zeros(4)

    if (M_par != destPar).all():  # TODO: check
        M_par = destPar

    pName, proj, p0Lat, p0Lon = dpg.map.getMapName(destMap)

    if pName != "":
        if M_proj_name != pName or M_proj_lat != p0Lat or M_proj_lon != p0Lon:
            M_proj_name = pName
            M_proj_lat = p0Lat
            M_proj_lon = p0Lon

    if len(destDim) == 3:
        lin = (np.ones((destDim[2], destDim[1])) * np.arange(destDim[1])).T
        col = np.ones((destDim[1], destDim[2])) * np.arange(destDim[2])
    else:
        lin = (np.ones((destDim[1], destDim[0])) * np.arange(destDim[0])).T
        col = np.ones((destDim[0], destDim[1])) * np.arange(destDim[1])

    y, x = dpg.map.lincol_2_yx(lin, col, destPar, set_center=True)
    M_lat, M_lon = dpg.map.yx_2_latlon(y, x, map=destMap)

    computed_maps[key] = (M_lon, M_lat)
    return M_lon, M_lat


def get_mosaic_comp(
        siteNode,
        destMap,
        destPar,
        destDim,
        data=None,
        nPlanes=None,
        threshvoid=10,
        find_quality=None,
):
    """
    Maps data and quality values from a source node to a target mosaic using geographic parameters.

    Args:
       siteNode (Node): The source node containing the data to be mapped
       destMap (Map): Map obtained from mosaicNode passed to the mosaic algorithm.
       destPar (list): Geographic parameters for the target map
       destDim (tuple): Dimensions of the target map (rows, columns)
       data (np.ndarray, optional): Placeholder for mapped data. Defaults to None.
       nPlanes (int, optional): Number of planes for 3D data. Defaults to None.
       threshvoid (float, optional): Threshold below which data values are adjusted. Defaults to 10.
       find_quality (bool, optional): Flag to search for quality values in parent nodes. Defaults to None.

    Returns:
        tuple
       - data (np.ndarray): Mapped data values from the source node to the target mosaic
       - quality (np.ndarray or None): Quality values associated with the mapped data, if available.
       - destLin (np.ndarray): Row indices of the mapped data in the target mosaic.
       - destCol (np.ndarray): Column indices of the mapped data in the target mosaic.
       - checkvoid (int): Flag set to 1 if `threshvoid` is applied, otherwise 0.
    """

    checkvoid = 0
    destLin, destCol = None, None

    polarPointer, out_dict = dpg.array.get_array(siteNode, replace_quant=True)
    if isinstance(out_dict, dict):
        sourceDim = out_dict["dim"]
    if polarPointer is None:
        return data, None, destLin, destCol, checkvoid
    _, sourceMap, _, sourcePar, ispolar, isvertical, _, _, _ = dpg.navigation.check_map(
        siteNode, sourceMap=True
    )

    if len(sourcePar) < 4:
        return data, None, destLin, destCol, checkvoid

    box1, _, _, _ = dpg.map.get_box_from_par(
        map=sourceMap, par=sourcePar, dim=sourceDim, regular=True
    )
    box2, _, _, _ = dpg.map.get_box_from_par(map=destMap, par=destPar, dim=destDim, regular=True)

    if len(box1) < 4 or len(box2) < 4:
        return data, None, destLin, destCol, checkvoid

    box1 = dpg.map.get_dest_box(
        box1, sourceMap=sourceMap, destMap=destMap
    )
    box = dpg.map.get_inner_box(box1, box2)

    lin1, col1 = dpg.map.yx_2_lincol(box[1], box[0], params=destPar, set_center=True)
    lin2, col2 = dpg.map.yx_2_lincol(box[3], box[2], params=destPar, set_center=True)
    lin1, lin2, col1, col2 = int(lin1), int(lin2), int(col1), int(col2)

    if len(destDim) == 3:
        check, lin1, lin2 = dpg.map.check_points(lin1, lin2, destDim[1])
    else:
        check, lin1, lin2 = dpg.map.check_points(lin1, lin2, destDim[0])
    if check <= 0:
        return data, None, destLin, destCol, checkvoid
    if len(destDim) == 3:
        check, col1, col2 = dpg.map.check_points(col1, col2, destDim[2])
    else:
        check, col1, col2 = dpg.map.check_points(col1, col2, destDim[1])
    if check <= 0:
        return data, None, destLin, destCol, checkvoid

    subsize = [lin2 - lin1 + 1, col2 - col1 + 1]
    if subsize[0] <= 0:
        return data, None, destLin, destCol, checkvoid
    if subsize[1] <= 0:
        return data, None, destLin, destCol, checkvoid

    M_lon, M_lat = checkCurrMap(destPar=destPar, destMap=destMap, destDim=destDim)

    destLin = (np.ones((subsize[1], subsize[0])) * np.arange(subsize[0])).T
    destCol = np.ones((subsize[0], subsize[1])) * np.arange(subsize[1])

    destLin += lin1
    destCol += col1

    destLin = destLin.astype(int)
    destCol = destCol.astype(int)

    y_lat = M_lat[destLin, destCol]
    x_lon = M_lon[destLin, destCol]

    y, x = dpg.map.latlon_2_yx(y_lat, x_lon, sourceMap)
    lin, col = dpg.map.yx_2_lincol(
        y, x, params=sourcePar, dim=sourceDim, set_center=True
    )

    ind_x, ind_y = np.where(np.logical_and(lin >= 0, destLin >= 0))
    if len(ind_x) <= 0:
        return data, None, destLin, destCol, checkvoid

    lin = lin[ind_x, ind_y]
    col = col[ind_x, ind_y]
    destLin = destLin[ind_x, ind_y]
    destCol = destCol[ind_x, ind_y]

    if nPlanes is not None and len(sourceDim) == 3:
        nL = min(nPlanes, sourceDim[0])
        data = np.zeros((nL,) + col.shape)
        data[:] = np.nan
        for lll in range(nL):
            tmp = polarPointer[lll, :, :].copy()
            data[lll, :] = tmp[lin, col]
        return data, None, destLin, destCol, checkvoid

    nEl = len(polarPointer.flatten())
    data = polarPointer[lin, col]
    if threshvoid is not None:
        checkvoid = 1
        ind = np.where(data < threshvoid)
        data[ind] = threshvoid - 0.1

    qNode = dpg.tree.getSon(siteNode, "Quality")
    if not isinstance(qNode, dpg.node__define.Node) and find_quality is not None:
        qNode = dpg.tree.getSon(siteNode.getParent(), "Quality")

    if not isinstance(qNode, dpg.node__define.Node):
        return data, None, destLin, destCol, checkvoid

    polarPointer, _ = dpg.array.get_array(qNode, replace_quant=True)
    if polarPointer is None:
        return data, None, destLin, destCol, checkvoid

    if len(polarPointer.flatten()) != nEl:
        return data, None, destLin, destCol, checkvoid

    quality = polarPointer[lin, col]
    quality = np.nan_to_num(quality)

    return data, quality, destLin, destCol, checkvoid


def mosaicmax(
        data,
        col,
        lin,
        outPointer,
        quality=None,
        nPlanes=False,
        outQuality=None,
        abs=None,
        thresh=None,
):
    """
    Updates the output mosaic and quality matrices by merging input data and quality values based on specified conditions
    outQuality and outPointer update algorithm based on input node data.

    Args:
        data (np.ndarray): Input data array related to the current node treated by other algorithms.
        col (np.ndarray): Matrix where each column contains a sequence of integers.
        lin (np.ndarray): Matrix where each column contains a sequence of integers.
        outPointer (np.ndarray): Matrix in which each element represents the weighted sum accumulated data on.
        quality (np.ndarray, optional): ExternQual value associated with the current node treated by other algorithms.
        nPlanes (bool, optional): Flag indicating whether the mosaic involves multiple planes. Defaults to False
        outQuality (np.ndarray, optional): Vector containing the information of the node "Quality" child of the father of the current node
        treated by other algorithms
        abs (bool, optional): If True, comparisons are performed using absolute values. Defaults to None.
        thresh (float, optional): Value associated with the externThresh parameter of the current node treated by other algorithms.. Defaults to None.

    Returns:
        tuple:
        - outPointer (np.ndarray): Array associated to the node current created by other algorithms.
        - outQuality (np.ndarray, optional): Node "Quality" child of the current node treated by other algorithms.
    """

    currQual = None

    if nPlanes:
        outPointer = mosaic3d(data, col, lin, outPointer, nPlanes)
        return outPointer, None

    currMos = outPointer[lin, col]
    if quality is not None and len(quality) == len(data):
        currQual = outQuality[lin, col]
        eqInd = np.where((currMos == data) & (currQual < quality))
        if len(eqInd[0]) > 0:
            currQual[eqInd] = quality[eqInd]
            outQuality[lin, col] = currQual

        # Vecchia implementazione, cambiata con una più semplce
        # currQual = np.where(
        #     np.logical_and(currMos == data, currQual < quality), quality, currQual
        # )
        # outQuality[lin, col] = currQual

    if abs:
        minInd = np.where(
            np.logical_or(np.abs(currMos) < np.abs(data), np.isnan(currMos))
        )
    else:
        minInd = np.where((currMos < data) | (np.isnan(currMos)))

    if len(minInd[0]) <= 0:
        return outPointer, outQuality

    currMos[minInd] = data[minInd]
    outPointer[lin, col] = currMos

    if quality is not None and outQuality is not None:
        currQual = outQuality[lin, col]

    if currQual is None:
        return outPointer, outQuality

    if len(quality) == 1:
        ind = np.where(np.isnan(currQual[minInd]))
        currQual[minInd[ind]] = 40
        ind = np.where((currMos[minInd] > thresh) & (currQual[minInd] < quality))
        currQual[minInd[ind]] = quality
        ind = np.where((currMos == data) & (currQual < quality))
        currQual[ind] = quality
    else:
        currQual[minInd] = quality[minInd]

    outQuality[lin, col] = currQual

    return outPointer, outQuality


def mosaic_extern(
        mosaicNode,
        outPointer,
        outQuality,
        destMap,
        destPar,
        destSize,
        threshVoid,
        path=None,
):
    # TODO: verificare la correttezza del commento
    """
    Algorithm for loading data from external sources (specified in path) and merging within an existing mosaic.

    Args:
        mosaicNode: Mosaic node treated by the algorithm.
        outPointer: Array associated with the current node treated by other algorithms.
        outQuality: Vector containing the information of the node "Quality" child of the father of the current node
        treated by other algorithms.
        destMap: Target map determined.
        destPar: List of parameters associated with the mosaic of which the mosaicNode is part.
        destSize: Dimensions of the mosaic of which the mosaicNode.
        threshvoid: ThreshVoid parameter associated with mosaicNode.

    :keywords:
        - path: External source path from which to load additional data.

    Returns:
        - **outPointer**: Array associated with the current node treated by other algorithms.
        - **outQuality**: Node "Quality" child of the current node treated by other algorithms.
    """

    if path is None:
        dpg.globalVar.GlobalState.update(
            "externPath", dpg.radar.get_par(mosaicNode, "externPath", "")[0]
        )
        externPath = dpg.globalVar.GlobalState.externPath
    if not isinstance(externPath, list):
        dpg.globalVar.GlobalState.update("externPath", [externPath])
        externPath = dpg.globalVar.GlobalState.externPath
    externQual, _, _ = dpg.radar.get_par(mosaicNode, "externQual", 60.0)
    externThresh, _, _ = dpg.radar.get_par(mosaicNode, "externThresh", 0.0)
    date, time, _ = dpg.times.get_time(mosaicNode)

    if not ((externPath is None) or (externPath == "") or (len(externPath) > 0 and externPath[0] == "")):
        for eee in externPath:
            last_path = dpg.times.search_path(
                eee, date=date, time=time, minStep=None, nMin=15
            )
            if last_path != "":
                lastNode = dpg.tree.createTree(last_path)
                data, quality, lin, col, checkvoid = get_mosaic_comp(
                    lastNode, destMap, destPar, destSize, threshvoid=threshVoid
                )
                if data is not None:
                    externFact, _, _ = dpg.radar.get_par(mosaicNode, "externFact", 0.0, prefix="")
                    if externFact != 0:
                        data *= externFact
                    outPointer, outQuality = mosaicmax(
                        data, col, lin, outPointer, quality=externQual, thresh=externThresh
                    )

    return outPointer, outQuality


def mosaicAvgMaxQual(data, lin, col, quality, qual, totmos, totqual, totwei):
    # TODO: verificare la correttezza del commento
    """
    Algorithm for aggregation and calculation of statistics for a mosaic data set.

    Args:
        data:       Data related to the current node treated by other algorithms.
        col:        Matrix where each column contains a sequence of integers.
        lin:        Matrix where each column contains a sequence of integers.
        quality:    Data and metadata associated with the node "Quality" child of the father of the current node
        treated by other algorithms.
        qual:       Vector containing the information of the node "Quality" child of the father of the current node
        treated by other algorithms.
        totmos:     Matrix for total of data.
        totqual:    Matrix for total quality.
        totwei:     Matrix for total of weights.

    Returns:
        - **totmos**: Matrix for total of data.
        - **totqual**: Matrix for total quality.
        - **totwei**: Matrix for total of weights.
    """

    tmp = qual[lin, col]
    ind = np.where(np.logical_and(quality > 0, quality >= tmp))
    if len(ind) <= 0:
        return
    q = quality[ind]
    w = q * q

    tmp = totqual[lin, col]
    tmp[ind] += q * w
    totqual[lin, col] = tmp

    tmp = totmos[lin, col]
    tmp[ind] += data[ind] * w
    totmos[lin, col] = tmp

    tmp = totwei[lin, col]
    tmp[ind] += w
    totwei[lin, col] = tmp

    return totmos, totqual, totwei


def MOSAIC(
        mosaicNode,
        nPlanes=None,
        mode=None,
        smoothBox=None,
        counter=None,
        get_mosaic_data=False,
        plot=False,
        remove=None,
        remove_on_error=False,
):
    """
    Mosaic algorithm: generates the mosaic of any set of products.
    Algorithm based on selective weighted average criterion.
    The mosaic products must all reside in the same tree at the same level as the output node.
    The input data can be in polar and/or cartesian format, provided correctly georeferenced.

    :Params:
        mosaicNode: Node where the mosaic will be generated.

    :keywords:
        - nPlanes: --
        - mode: --
        - smoothBox: --
        - counter: --
        - get_mosaic_data: --
        - plot: Flag for permission to display a plot.
        - remove: Flags for removing folder generated by the algorithm.
        - remove_on_error: Flag for permission to remove mosaicNode.

    Returns:
        - **outpointer**: Data of the mosaic.
    """

    err = 1
    destSize = None
    print("Computing Mosaic...")

    if not isinstance(mosaicNode, dpg.node__define.Node):
        return

    if nPlanes:
        if nPlanes <= 1:
            nPlanes = None

    # in questo punto torna un numero di siti errato
    nSites, auto, find_quality, siteNodes = GetNodesToMosaic(mosaicNode)
    if nSites <= 0:
        log_message(
            "Cannot find Nodes to mosaic in " + dpg.tree.getNodePath(mosaicNode),
            level="WARNING",
            all_logs=True,
        )
        if remove_on_error:
            dpg.tree.removeNode(mosaicNode, directory=True)
        return

    if nSites == 1 and auto:
        log_message(
            "TODO: caso in cui nSites del mosaic sono == 1",
            level="ERROR",
            all_logs=True,
        )
        # TODO: IMPLEMENTARE

    destMap, _, destSize, destPar, ispolar, isvertical, _, _, _ = dpg.navigation.check_map(
        mosaicNode, destMap=True
    )

    if nPlanes:
        destSize = np.append(nPlanes, destSize)

    outPointer = dpg.array.check_array(mosaicNode, type=4, dim=destSize, value=np.nan)
    threshVoid, _, exists = dpg.radar.get_par(mosaicNode, "threshVoid", 0)
    if not exists:
        threshVoid = None
    qualityNode = dpg.tree.getSon(mosaicNode, "Quality")
    outQuality = None

    if not isinstance(qualityNode, dpg.node__define.Node) and find_quality:
        qualityNode = dpg.tree.getSon(mosaicNode.parent, "Quality")

    if isinstance(qualityNode, dpg.node__define.Node):
        outQuality = dpg.array.check_array(qualityNode, type=4, dim=destSize, value=0)
        if outQuality is not None:
            outQuality[:] = 0.

    if mode is None:
        mode = 1
    if smoothBox is None:
        smoothBox = 3

    if counter is not None:
        counter = np.zeros(destSize, dtype=np.uint8)

    siteNodes = sorted(siteNodes, key=lambda obj: obj.name)

    for sss in siteNodes:
        data, quality, lin, col, checkvoid = get_mosaic_comp(
            sss,
            destMap=destMap,
            destPar=destPar,
            destDim=destSize,
            threshvoid=threshVoid,
            nPlanes=nPlanes,
            find_quality=find_quality,
        )

        if data is not None:
            if len(data) > 0:
                if err > 0:
                    dpg.values.copy_values_info(
                        sss, mosaicNode, only_if_not_exists=True
                    )

                if mode == 2:
                    outPointer, outQuality = mosaicMaxEqual(data, quality, col, lin, outPointer, outQuality)

                else:
                    outPointer, outQuality = mosaicmax(
                        data,
                        outPointer=outPointer,
                        lin=lin,
                        col=col,
                        quality=quality,
                        outQuality=outQuality,
                        nPlanes=nPlanes,
                        abs=(mode > 3)
                    )

                err = 0
                if counter:
                    counter[lin, col] += 1

    if outQuality is not None:
        if mode < 2:
            TOTMOS = np.zeros(destSize)
            TOTQUAL = np.zeros(destSize)
            TOTWEI = np.zeros(destSize)

            for sss in siteNodes:
                data, quality, lin, col, checkvoid = get_mosaic_comp(
                    sss,
                    destMap=destMap,
                    destPar=destPar,
                    destDim=destSize,
                    threshvoid=threshVoid,
                )

                if data is None:
                    continue

                if len(data) > 0 and quality is not None:
                    TOTMOS, TOTQUAL, TOTWEI = mosaicAvgMaxQual(
                        data,
                        lin=lin,
                        col=col,
                        quality=quality,
                        qual=outQuality,
                        totmos=TOTMOS,
                        totqual=TOTQUAL,
                        totwei=TOTWEI,
                    )

            ind_x, ind_y = np.where(TOTWEI > 0)
            if len(ind_x) > 0:
                TOTMOS[ind_x, ind_y] /= TOTWEI[ind_x, ind_y]
                TOTQUAL[ind_x, ind_y] /= TOTWEI[ind_x, ind_y]

            ind_x, ind_y = np.where(TOTWEI <= 0)
            if len(ind_x) > 0:
                TOTMOS[ind_x, ind_y] = np.nan
                TOTQUAL[ind_x, ind_y] = np.nan

            outQuality[:] = TOTQUAL
            outPointer[:] = TOTMOS

        else:
            ind = np.where(np.isnan(outPointer))
            outQuality[ind] = np.nan

    outPointer, outQuality = mosaic_extern(
        mosaicNode, outPointer, outQuality, destMap, destPar, destSize, threshVoid
    )

    if checkvoid:
        ind = np.where(outPointer < threshVoid)
        outPointer[ind] = -np.inf

    if smoothBox:
        outPointer[:] = dpg.prcs.smooth_data(outPointer, smoothBox, opt=1, no_null=None)
        if outQuality is not None:
            outQuality[:] = dpg.prcs.smooth_data(
                outQuality, smoothBox, opt=1, no_null=None
            )

    if remove is None:
        remove, _ = dpb.dpb.get_par(mosaicNode, 'remove_sites', default=1)
    if remove:
        folders = [
            f
            for f in os.listdir(mosaicNode.parent.path)
            if os.path.isdir(os.path.join(mosaicNode.parent.path, f)) and f != "MOSAIC"
        ]
        folders.sort()
        print(f"{mosaicNode.parent.name} phase2 end: deleting folders: {folders}")
        for folder in folders:
            if folder.lower() != "mosaic":
                folder_path = os.path.join(mosaicNode.parent.path, folder)
                shutil.rmtree(folder_path)

    if get_mosaic_data:
        # return outPointer
        pass

    attr = dpg.navigation.fill_nav(mosaicNode)
    parname, _, _ = dpg.attr.getAttrValue(attr, 'parname', '')
    if parname != '':
        dpg.calibration.set_values(mosaicNode, attr)

    # TODO: da implementare
    # dpg.array.clean_nodes(siteNodes)

    return outPointer
