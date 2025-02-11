import numpy as np
import sou_py.dpg as dpg
from sou_py.dpg.log import log_message
from sou_py.dpg.map__define import Map

"""
Funzioni ancora da portare
FUNCTION GetMapCoverage 
PRO CreateParFromVertex 
PRO CUT_MAP
PRO POLAR_TO_CART
PRO WARP_MAP_FROM_COORDS 
FUNCTION GetParFromVertex     // UNUSED
PRO CREATE_DEM                // UNUSED
PRO CREATE_TILED_DEM          // UNUSED
"""

"""
; NAME:
; WARP_MAP
;
; :Description:
;    Algoritmo che effettua la riproiezione di una mappa.
;    Data una immagine georeferenziata, ne cambia il sistema di riferimento e adatta il contenuto alla nuova proiezione.
;    E' necessario che la matrice di input sia correttamente definita (navigation.txt) nel nodo origine.
;    E' necesario che sia definito il nuovo sistema di riferimento nel nodo destinazione:
;       Proiezione e origine (prj_lat, proj_lon)
;       I 4 parametri fondamentali che delimitano il box
;       Dimensioni della mappa di output
;
; :Params:
;    source        Nodo sorgente: deve contenere una matrice a 2 o 3 dimensioni con i parametri di navigazione.
;    destNode      Nodo destinazione: deve contenere i parametri di navigazione destinazione.
;    outData       Variabile che conterra' la matrice di output.
;
; :Keywords:
;    NUMERIC       Keyword che abilita la conversione in float (nel caso in cui la matrice di dati fosse codificata)
;    LINEAR        Keyowrd che abilita la conversione in scala lineare (nel caso in cui la matrice di dati fosse 
logaritmica)
;    LEVEL         Eventuale livello di una matrice a 3 dimensioni.
;    MOSAIC        Keyword: se settata, la georeferenziazione di output viene letta dal file mosaic.txt
;    SOURCE_DATA   Nel caso sia presente, sostituisce la matrice di input (in questo caso il nodo source puo' 
contenere il solo file navigation.txt)
;    AUTO          Definisce automaticamente dimensioni e copertura cartesiana in caso di matrice polare
;    PUT           Keyword: se settata, la matrice di output viene assegnata al nodo destNode.
"""


def warp_map(
        source,
        destNode,
        outData=None,
        source_data=None,
        aux=None,
        mosaic=False,
        numeric=False,
        linear=False,
        regular=False,
        auto=False,
        level=None,
        put=False,
        remove=False,
        calib=None,
):
    """
    Map transformation function that remaps data from a source map to a destination map.

    Args:
        source (str or dpg.node__define.Node): source node containing the map data to be transformed
        destNode (dpg.node__define.Node): destination node where the transformed data will be stored
        outData (np.ndarray, optional): output data array
        source_data (np.ndarray, optional): source data array, if available
        aux (optional): auxiliary data or parameters
        mosaic (bool): flag to indicate if the output is a mosaic of maps
        numeric (bool): flag to indicate if the transformation should be in numeric format
        linear (bool): flag to indicate if the transformation should be linear
        regular (bool): flag to indicate if the transformation should be regular
        auto (bool): flag to indicate if the transformation parameters should be set automatically
        level (int, optional): specific level to transform, if applicable
        put (bool): flag to indicate if the transformed data should be stored in the destination node
        remove (bool): flag to indicate if the source node should be removed after transformation
        calib (optional): calibration parameters

    Returns:
        np.ndarray: the transformed data based on the provided coordinates and specifications, or None if
        transformation fails
    """

    check = 0

    if source is None:
        outData = None
        return outData

    if isinstance(source, dpg.node__define.Node):
        sourceNode = source
    else:
        if dpg.io.type_py2idl(type(source)) != 7 or source == "":
            outData = None
            return outData

        sourceNode = dpg.tree.createTree(source)

    if source_data is not None:
        sourceArray = source_data
        IDLtype = dpg.io.type_py2idl(source_data.dtype)
        sourceDim = np.shape(source_data)
        to_reset = 1
    else:
        sourceArray, out_dict = dpg.array.get_array(sourceNode, aux=aux)
        if sourceArray is None or out_dict is None:
            sourceDim = ()
        else:
            sourceDim = out_dict["dim"]
            IDLtype = out_dict["type"]
        to_reset = 0

    if len(sourceDim) <= 1:
        if to_reset == 1:
            pass  # ptr_free, sourceArray
        outData = None
        return outData

    d2 = sourceDim
    sourceMap, _, d2, sourcePar, _, _, _, _, _ = dpg.navigation.check_map(
        sourceNode, destMap=True
    )

    out_dict = dpg.navigation.get_radar_par(
        sourceNode, regular=regular, get_el_coords_flag=True, get_az_coords_flag=True
    )
    source_az_coords = out_dict["az_coords"]
    source_el_coords = out_dict["el_coords"]

    destDim = 0
    if isinstance(destNode, dpg.node__define.Node):
        _, _, destDim, _, _, _, _, _ = destNode.getArrayInfo()
    if np.size(destDim) <= 0 or destDim is None:
        destDim = 0
    d1 = destDim
    destMap, _, d1, destPar, _, _, _, _, _ = dpg.navigation.check_map(destNode, destMap=True)
    if destMap is not None:
        check = 1
    out_dict = dpg.navigation.get_radar_par(
        destNode, get_el_coords_flag=True, get_az_coords_flag=True, regular=regular
    )
    dest_az_coords = out_dict["az_coords"]
    dest_el_coords = out_dict["el_coords"]

    if len(sourcePar) > 6 and auto:
        box, _, _, _ = dpg.map.get_box_from_par(map=sourceMap, par=sourcePar, dim=sourceDim)
        destDim = sourceDim * 2
        destDim[1] = destDim[0]
        destPar = dpg.map.get_par_from_box(box=box, dim=list(destDim), isotropic=True)
        if len(sourceDim) == 3:
            destDim[2] = sourceDim[2]
        if sourcePar[4] > 0 or sourcePar[5] > 0:
            destPar = [destPar, sourcePar[4], sourcePar[5]]

    if np.sum(destDim) <= 0 or mosaic:
        destDim = d1

    if np.sum(destDim) <= 0:
        if to_reset:
            sourceArray = None
        outData = None
        return outData

    lev = 0
    nplanes = 0

    if len(sourceDim) > len(destDim):
        if level is not None:
            if level < sourceDim[0]:
                lev = level
        else:
            if sourceDim[0] <= sourceDim[2]:
                nplanes = sourceDim[0]
            else:
                nplanes = sourceDim[2]
    else:
        if len(sourceDim) == 3:
            nplanes = sourceDim[2]
            destDim = destDim[0:1]

    if check > 0:
        if nplanes <= 0:
            if len(sourceArray.shape) == 3:
                outData = warp_image(
                    sourceArray[lev, :, :],
                    sourceMap=sourceMap,
                    sourcePar=sourcePar,
                    destMap=destMap,
                    destPar=destPar,
                    destSize=destDim,
                    az_coords=dest_az_coords,
                    el_coords=dest_el_coords,
                    source_az_coords=source_az_coords,
                )
            elif len(sourceArray.shape) == 2:
                outData = warp_image(
                    sourceArray[:, :],
                    sourceMap=sourceMap,
                    sourcePar=sourcePar,
                    destMap=destMap,
                    destPar=destPar,
                    destSize=destDim,
                    az_coords=dest_az_coords,
                    el_coords=dest_el_coords,
                    source_az_coords=source_az_coords,
                )
        else:
            if sourceDim[0] <= sourceDim[2]:
                outData = np.zeros(
                    [nplanes, destDim], dtype=dpg.io.type_idl2py(IDLtype)
                )
                for ppp in range(nplanes):
                    outData[ppp, :, :] = warp_image(
                        sourceArray[ppp, :, :],
                        sourceMap=sourceMap,
                        sourcePar=sourcePar,
                        destMap=destMap,
                        destPar=destPar,
                        destSize=destDim,
                        az_coords=dest_az_coords,
                        el_coords=dest_el_coords,
                        source_az_coords=source_az_coords,
                    )
            else:
                outData = np.zeros(
                    [destDim, nplanes], dtype=dpg.io.type_idl2py(IDLtype)
                )
                for ppp in range(nplanes):
                    outData[:, :, ppp] = warp_image(
                        sourceArray[:, :, ppp],
                        sourceMap=sourceMap,
                        sourcePar=sourcePar,
                        destMap=destMap,
                        destPar=destPar,
                        destSize=destDim,
                        az_coords=dest_az_coords,
                        el_coords=dest_el_coords,
                        source_az_coords=source_az_coords,
                    )

    else:
        outData = sourceArray
        dpg.navigation.copy_geo_info(sourceNode, destNode)

    if to_reset:
        del sourceArray

    if put:
        dpg.array.set_array(destNode, data=outData)
        if auto:
            dpg.navigation.set_geo_info(destNode, par=destPar)

    if numeric or linear:
        if IDLtype == 4 or IDLtype == 5:
            to_not_create = 1
        else:
            to_not_create = 0
        values, calib, out_dict = dpg.calibration.get_array_values(
            sourceNode, to_not_create=to_not_create
        )
        scale = out_dict["scale"]
        outData = dpg.calibration.convertData(outData, values, linear=linear)

    if not isinstance(source, dpg.node__define.Node) or remove:
        dpg.tree.removeTree(sourceNode)

    if auto or not mosaic or not check:
        return outData

    ret, new_attr = dpg.tree.copyAttr(
        destNode,
        destNode,
        dpg.cfg.getMosaicDescName(),
        dest_name=dpg.cfg.getGeoDescName(),
    )
    dpg.tree.removeAttr(destNode, name=dpg.cfg.getMosaicDescName(), delete_file=True)
    dpg.navigation.set_geo_info(destNode, par=destPar)
    dpg.navigation.fill_nav(destNode)
    if len(dest_el_coords) > 1 and len(dest_el_coords) > 20:
        dest_el_coords = dpg.navigation.set_el_coords(destNode)

    return


def get_dem(
        destMap: Map,
        destPar: list,
        destDim: tuple,
        path=None,
        hr=False,
        sourceMap=None,
        sourcePar=None,
        az_coords: tuple = (),
        el_coords=None,
        numeric=None,
):
    """
    Digital Elevation Model (DEM) retrieval and transformation function. This function retrieves a DEM, optionally
    remaps it from a source map to a destination map, and returns the transformed DEM.

    Args:
        destMap (map__define.Map): destination map that defines the coordinate system of the destination image
        destPar (list): additional parameters related to the destination map
        destDim (tuple): dimensions of the destination image
        path (str, optional): path to the DEM file; if None, a default DEM is used
        hr (bool): high resolution flag for DEM retrieval
        sourceMap (map__define.Map, optional): source map that defines the coordinate system of the source image; if
                                                None, it is retrieved from the DEM
        sourcePar (list, optional): additional parameters related to the source map
        az_coords (tuple, optional): azimuthal coordinates specific to the source image
        el_coords (tuple, optional): elevational or altitudinal coordinates
        numeric (bool, optional): flag indicating if the DEM should be returned in numeric format

    Returns:
        np.ndarray: the transformed DEM based on the provided coordinates and destination specifications, or None if
        retrieval fails
    """

    if path is not None:
        demId = dpg.tree.createTree(path)
    else:
        demId = dpg.navigation.get_dem_id(hr=hr)

    demArray, out_dict = dpg.array.get_array(demId)
    dim = out_dict["dim"]
    type = out_dict["type"]

    values, calib, out_dict = dpg.calibration.get_array_values(demId)

    if demArray is None:
        return None

    log_message(f"Using {demId.path}", level="INFO")

    sourceMap, _, dim, sourcePar, _, _, _, _, _ = dpg.navigation.check_map(
        demId, destMap=True
    )

    polarDem = dpg.warp.warp_image(
        demArray,
        sourceMap=sourceMap,
        sourcePar=sourcePar,
        destMap=destMap,
        destPar=destPar,
        destSize=destDim,
        az_coords=az_coords,
        el_coords=el_coords,
    )

    if numeric and type != 4:
        polarDem = values[polarDem]

    if path is not None:
        dpg.tree.removeTree(demId)

    return polarDem


def warp_coords(
        sourceMap,
        sourcePar,
        sourceSize,
        destMap,
        destPar,
        destSize,
        x,
        y,
        up_direction,
        az_coords,
        el_coords,
        source_az_coords,
):
    """
    Image transformation function, specifically remapping coordinates from a source map to a destination map.

    Args:
        sourceMap (map__define.Map): source map that defines the coordinate system of the source image
        sourcePar (list): additional parameters related to the source map
        sourceSize (tuple): dimensions of the source image
        destMap (map__define.Map): destination map that defines the coordinate system of the destination image
        destPar (list): additional parameters related to the destination map
        destSize (tuple): dimensions of the destination image
        x (np.ndarray): transformed coordinates along the x-axis
        y (np.ndarray): transformed coordinates along the y-axis
        up_direction (int or str): parameter that defines the "up" direction in the new image
        az_coords (tuple): azimuthal coordinates specific to the source image
        el_coords (tuple): elevational or altitudinal coordinates
        source_az_coords (tuple): azimuthal coordinates specific to the destination image

    Returns:
        tuple:
            - int: status code indicating the success (1) or failure (-1, 0) of the transformation
            - np.ndarray: transformed coordinates along the x-axis
            - np.ndarray: transformed coordinates along the y-axis
    """

    if len(destSize) <= 1:
        return -1, x, y

    if len(destSize) == 3:
        destlines_ind = 1
        destcols_ind = 2
    else:
        destlines_ind = 0
        destcols_ind = 1

    if len(sourceSize) == 3:
        sourcelines_ind = 1
        sourcecols_ind = 2
    else:
        sourcelines_ind = 0
        sourcecols_ind = 1

    if destSize[destlines_ind] <= 0 or destSize[destcols_ind] <= 0:
        log_message("Dimensions must be positive!", level="ERROR", all_logs=True)
        return -1, x, y
    if np.sum(~np.isnan(np.array(destPar))) < 4:
        log_message("Parameters cannot be NaN", level="ERROR", all_logs=True)
        return -1, x, y
    if 4 <= len(destPar) <= 6:
        if destPar[1] == 0 or destPar[3] == 0:
            log_message("Resolution cannot be null", level="ERROR", all_logs=True)
            return -1
    if 4 <= len(sourcePar) <= 6:
        if sourcePar[1] == 0 or sourcePar[3] == 0:
            log_message("Resolution cannot be null", level="ERROR", all_logs=True)
            return -1, x, y

    check = dpg.map.check_maps(sourceMap=sourceMap, destMap=destMap)
    if (
            check == 1
            and np.size(az_coords) == destSize[destlines_ind]
            and np.size(source_az_coords) == sourceSize[sourcelines_ind]
    ):
        azInd = dpg.map.get_az_index(az_coords, 0.0, 0.0, az_coords_in=source_az_coords)
        if len(sourcePar) > 10:
            in_el = sourcePar[10]
        if len(destPar) > 10:
            out_el = destPar[10]
        rngInd = dpg.beams.getRangeBeamIndex(
            sourceSize[sourcecols_ind], destSize[destcols_ind], sourcePar[7], destPar[7]
        )
        y = np.ones((1, destSize[destcols_ind]), dtype=int) * azInd.reshape(-1, 1)
        x = (rngInd.reshape(-1, 1) * np.ones((1, destSize[destlines_ind]), dtype=int)).T
        return 1, x, y

    y = np.arange(destSize[destlines_ind]).reshape(-1, 1) * np.ones(
        (1, destSize[destcols_ind])
    ).astype(int)
    x = np.ones((destSize[destlines_ind], 1)).astype(int) * np.arange(
        destSize[destcols_ind]
    ).reshape(1, -1)

    y, x = dpg.map.lincol_2_yx(
        lin=y,
        col=x,
        params=destPar,
        az_coords=az_coords,
        el_coords=el_coords,
        set_center=True,
    )
    if check == 0 and destMap is not None:
        lat, lon = dpg.map.yx_2_latlon(y, x, destMap)
        y, x = dpg.map.latlon_2_yx(lat, lon, sourceMap)

    y, x = dpg.map.yx_2_lincol(y, x, sourcePar, dim=sourceSize, set_center=True, az_coords=source_az_coords)

    ind = np.where(x >= 0)
    if len(ind) == 0:
        return 0
    ind = np.where(y >= 0)
    if len(ind) == 0:
        return 0
    return 1, x, y


def warp_image(
        image: np.ndarray,
        sourceMap: Map = None,
        sourcePar: tuple = (),
        destMap: Map = None,
        destPar: tuple = (),
        destSize: tuple = (),
        az_coords: tuple = (),
        source_az_coords: tuple = (),
        level: tuple = (),
        all: bool = False,
        x: tuple = (),
        set_null: tuple = (),
        img: tuple = (),
        el_coords: tuple = (),
        y: tuple = (),
        up_direction=None,
):
    """
    Image transformation function, specifically remapping coordinates from a source map to a destination map.

    Args:
        image (np.ndarray): source image that needs to be transformed

    :keywords:
        - sourceMap (map__define.Map): source map that defines how the points of the source image should be
        transformed into the new image
        - sourcePar (list): additional parameters related to the source map
        - destMap (map__define.Map): destination map that defines how the points of the source image should be
        transformed into the new image
        - destPar (list): additional parameters related to the destination map
        - destSize (tuple): size of the destination dimensions for the transformed image
        - az_coords (tuple): azimuthal coordinates specific to the source image
        - source_az_coords (tuple): azimuthal coordinates for the destination image
        - level (tuple): specifies a particular level of the image to transform if the image is 3D
        - all (boolean): indicates whether the transformation should be applied to all levels of the image
        - x (tuple): transformed coordinates along the x-axis
        - set_null (tuple): specifies how to handle null or missing values in the transformed image
        - img (tuple): resulting image after transformation
        - el_coords (NoneType): elevational or altitudinal coordinates
        - y (tuple): transformed coordinates along the y-axis
        - up_direction (integer): parameter that defines the "up" direction in the new image

    Returns:
        img (tuple): transformed image based on the provided coordinates and destination specifications
    """

    sourceDim = image.shape
    if len(sourceDim) <= 1:
        return None
    if len(destSize) <= 1:
        return image

    if len(destSize) == 3:
        nlines_ind = 1
        ncols_ind = 2
    else:
        nlines_ind = 0
        ncols_ind = 1

    sd = dpg.array.getBidimensionalDim(sourceDim)
    ret = 0

    if np.array_equal(sd[0:2], destSize[nlines_ind:]):
        ret = dpg.map.check_maps(
            sourceMap=sourceMap,
            destMap=destMap,
            source_par=sourcePar,
            dest_par=destPar,
            source_az_coords=source_az_coords,
            dest_az_coords=az_coords,
        )

    if ret == 0:
        codprc, x, y = warp_coords(
            sourceMap,
            sourcePar,
            sourceSize=sd,
            destMap=destMap,
            destPar=destPar,
            destSize=destSize,
            x=x,
            y=y,
            up_direction=up_direction,
            az_coords=az_coords,
            el_coords=el_coords,
            source_az_coords=source_az_coords,
        )

    type = image.dtype
    type = dpg.io.type_py2idl(type)
    nullV = np.nan
    if type != 4:
        nullV = 0

    if len(sourceDim) == 3:
        if len(level) == 0:
            if all:
                indNull = np.where(x < 0)
                countNull = len(indNull)
                if len(destSize) == 2:
                    log_message(
                        "Controllare che la dimensione sia corretta", level="ERROR"
                    )  # TODO da controllare
                    destSize = destSize + sourceDim[2]
                img = np.zeros(destSize, dtype=bytes)
                for ddd in range(destSize[2]):
                    tmp = np.squeeze(image[:, :, ddd])
                    tmp = tmp[x, y]
                    if countNull > 0:
                        tmp[indNull] = nullV
                    img[:, :, ddd] = tmp
                return image
            else:
                level = 0
        if len(img) <= 0:
            img = image[:, :, level]
            if len(x) > 0:
                img = np.reshape(img[x, y], destSize)
            else:
                img = np.reshape(img, destSize)
    else:
        if np.size(x) == destSize[nlines_ind] * destSize[ncols_ind]:
            img = np.reshape(image[y.astype(int), x.astype(int)], destSize[nlines_ind:])
        else:
            if np.size(image) != destSize[nlines_ind] * destSize[ncols_ind]:
                return -1
            img = np.reshape(image, destSize[nlines_ind:])

    if len(x) <= 0:
        return img

    offset = [y[0, 0], x[0, 0]]
    if y[0, 0] > y[destSize[nlines_ind] - 1, 0]:
        reverse = 1

    indNull = np.where(x < 0)
    countNull = len(indNull[0])
    if countNull <= 0:
        return img
    if type != 4:
        nullV = 0
        if len(set_null) == 1:
            img = np.fix(img, type=np.size(set_null))
            nullV = set_null

    img[indNull] = nullV

    return img
