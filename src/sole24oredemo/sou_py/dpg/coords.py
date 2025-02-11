import matplotlib
import numpy as np
from shapely import Polygon, MultiPolygon
from skimage.draw import polygon

import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION CheckShapeCoords 
FUNCTION GetCoords 
FUNCTION GetShapeCoords 
FUNCTION GetShapeEntity 
FUNCTION GetShapeEntityOpt 
FUNCTION GetZoneMask 
FUNCTION IDL_rv_get_points 
FUNCTION IDL_rv_get_shape_info 
FUNCTION TranslateRegion 
PRO CleanMask 
PRO ComputeEvolution 
PRO ComputeTrack 
PRO ComputeTrackDim 
PRO COPY_SHAPEFILE 
PRO GET_GEO 
PRO GET_GEO_COORDS 
PRO GET_MAP_COORDS 
PRO GetEvolutionCoeffs 
PRO LoadROI 
PRO SaveROI 
PRO SetEvolutionCoeffs 
PRO SLICE_VOLUME 
FUNCTION FilterCoords             // UNUSED
FUNCTION GET_LATITUDE             // UNUSED
FUNCTION GET_LONGITUDE            // UNUSED
FUNCTION GetDEMLevel              // UNUSED
FUNCTION GetLines                 // UNUSED
FUNCTION GetPoints                // UNUSED
FUNCTION GetValidEntities         // UNUSED
FUNCTION IDL_rv_get_shape_entity  // UNUSED
PRO CREATE_ALERT_FILES            // UNUSED
PRO GET_ALT_COORDS                // UNUSED
PRO LoadCursor                    // UNUSED
PRO SaveCursor                    // UNUSED
PRO SHP_TO_HML                    // UNUSED
PRO SPLIT_SHAPEFILE               // UNUSED
PRO TraslateShapeEntity           // UNUSED
"""


def get_shape_info(node, index=None, names=None):
    n_ent = 0
    pointer, out_dict = dpg.array.get_array(node, silent=True)
    dtype = out_dict["type"]

    pNames = dpg.tree.getAttr(node, "attr_names", to_not_load=True)[0]
    if pNames is None or pNames.pointer is None:
        log_message(
            f"Cannot find data in {dpg.tree.getNodePath(node)}", level="WARNING"
        )
        return None, None, names

    n_attr = len(pNames.pointer)
    if names is not None:
        names = pNames.pointer

    if pointer is None:
        if np.size(index) == 1:
            if index >= 0:
                return None
        return None, n_attr, names

    if dtype != 8:
        return None

    n_ent = len(pointer)

    if index is None:
        attr = pointer
        return attr, n_attr, names

    log_message("DA FARE PORTING DI QUESTA PARTE", level="ERROR")
    raise SystemExit("Porting da completare")


def get_points(
    node,
    center=True,
    attr=None,
    prefix=None,
    check_date=None,
    box=False,
    optim=False,
    destMap=None,
    index=None,
):
    if attr is None:
        nav = dpg.tree.getSingleAttr(node, dpg.cfg.getGeoDescName())
        modelDesc = dpg.tree.getAttr(
            node, dpg.cfg.getModelDescName(), only_current=True
        )
        itemDesc = dpg.tree.getAttr(node, dpg.cfg.getItemDescName(), only_current=True)
        # attr = [nav, modelDesc, itemDesc]
        attr = [
            elem
            for item in [nav, modelDesc, itemDesc]
            if item is not None
            for elem in (item if isinstance(item, list) else [item])
        ]

    coordfile, coords, format_ = dpg.navigation.check_coords_file(node, attr, prefix=prefix, check_date=check_date)

    if coordfile == "":
        return 0

    if format_.upper() == "TXT":
        log_message("DA IMPLEMENTARE", level="ERROR")
        (
            nPoints,
            destMap,
        ) = dpg.navigation.get_points(coords)

    elif format_.upper() == "ASCII":
        log_message("DA IMPLEMENTARE", level="ERROR")
        (
            nPoints,
            destMap,
        ) = dpg.navigation.get_points(coords)

    elif format_.upper() == "SHP":
        _, sourceMap, _, _, _, _, _, _, _ = dpg.navigation.check_map(
            node, sourceMap=True
        )
        nPoints, points, polylines = dpg.coords.getShapeCoords(
            coords, sourceMap=sourceMap, destMap=destMap, index=index, optim=optim
        )
        if box and nPoints > 0:
            box = dpg.map.check_box(points[:, 0], points[:, 1], None)
        if optim:
            return nPoints, points

    elif format_.upper() == "COORDS":
        log_message("DA IMPLEMENTARE", level="ERROR")
        (
            nPoints,
            destMap,
        ) = dpg.navigation.get_points(coords)
    else:
        nPoints = 0

    return nPoints, points


def get_number_poly_parts(geometry):
    """
    Counts the total number of polygon parts in a geometry, including exterior and interior rings.

    Args:
        geometry: A Shapely geometry object, which can be a `Polygon` or `MultiPolygon`.

    Returns:
        The total number of parts:
            For a `Polygon`, this is 1 (for the exterior) plus the number of interior rings.
            For a `MultiPolygon`, this is the sum of the above for each polygon.
            Returns 0 if the input is not a `Polygon` or `MultiPolygon`.
    """

    if isinstance(geometry, Polygon):
        return 1 + len(geometry.interiors)
    elif isinstance(geometry, MultiPolygon):
        return sum(1 + len(polygon.interiors) for polygon in geometry)
    else:
        return 0


def get_all_parts(geometry):
    """
    Extracts all parts (exterior and interior rings) from a given geometry.

    Args:
        geometry: A Shapely geometry object, which can be a `Polygon` or `MultiPolygon`.

    Returns:
        parts: A list of all the parts in the geometry:
            For a `Polygon`, the list includes its exterior ring and all interior rings (holes).
            For a `MultiPolygon`, the list includes the exterior and interior rings for each polygon in the collection.
    """

    parts = []

    if isinstance(geometry, Polygon):
        # Add the exterior ring
        parts.append(geometry.exterior)

        # Add the interior rings (holes)
        parts.extend(geometry.interiors)

    elif isinstance(geometry, MultiPolygon):
        # Iterate over each Polygon in the MultiPolygon
        for polygon in geometry:
            # Add the exterior ring
            parts.append(polygon.exterior)

            # Add the interior rings (holes)
            parts.extend(polygon.interiors)

    return parts


def checkShapeCoords(x, y, data, polylines, curr, totElP, z):
    if np.size(x) == 1:
        data[curr, 0] = x
        data[curr, 1] = y

        if z is not None:
            data[curr, 2] = z

        curr += 1
        totElP += 1
        return 1, curr, totElP

    else:
        log_message("DA IMPLEMENTARE CHECK SHAPE COORDS", level="ERROR")


def getShapeEntity(
    entities,
    index,
    sourceMap,
    ndim=None,
    center=False,
    destMap=None,
    polylines=None,
    curr=None,
    totElP=None,
    data=None,
):
    z = None
    if ndim is None:
        ndim = 2

    nVerts = len(entities.pointer[index].exterior.coords)

    if nVerts <= 0:
        return

    nParts = len(entities.pointer[index].interiors) + get_number_poly_parts(
        entities.pointer[index]
    )
    if nParts == 0:
        log_message("NPARTS = 0: TODO DA IMPLEMENTARE", level="ERROR")

    parts = get_all_parts(entities.pointer[index])

    for j in range(len(parts)):
        offset = parts[j]
        if j < nParts - 1:
            nv = parts[j + 1] - 1
        else:
            nv = nVerts - 1
        if center:
            x, y = (
                entities.pointer[index].centroid.x,
                entities.pointer[index].centroid.y,
            )
        else:
            log_message(
                "CENTROID IS NOT SET: TO BE IMPLEMENTEDE AND CHECKED", level="ERROR"
            )
            x = [point[0] for point in entities.pointer[index].exterior.coords]
            y = [point[1] for point in entities.pointer[index].exterior.coords]

        if destMap is not None:
            lat, lon = dpg.map.yx_2_latlon(y, x, map=sourceMap)
            minLat = np.nanmin(lat)
            maxLat = np.nanmax(lat)
            if minLat > -100 and maxLat < 100:
                if ndim == 3:
                    log_message("TODO: DA FIXARE CASO NDIM == 3", level="ERROR")
                    y, x = dpg.map.latlon_2_yx(
                        lat=lat, lon=lon, map=destMap, z=z
                    )
                else:
                    y, x = dpg.map.latlon_2_yx(lat=lat, lon=lon, map=destMap)

        nc, curr, totElP = checkShapeCoords(x, y, data, polylines, curr, totElP, z=z)
        if center:
            return 1, curr, totElP

    return nVerts


def getShapeEntityOpt(entities, index, sourceMap, destMap, polylines):
    nVerts = len(entities.pointer[index].exterior.coords)
    if nVerts <= 0:
        return 0
    nParts = len(entities.pointer[index].exterior.coords) + sum(
        len(interior.coords) for interior in entities.pointer[index].interiors
    )

    globe = dpg.map.is_3D_map(destMap)
    if globe > 0:
        data = np.zeros((nVerts, 3), dtype=np.float32)
    else:
        data = np.zeros((nVerts, 2), dtype=np.float32)
    polylines = np.zeros(nVerts + nParts, dtype=np.float32)
    totElP = 0
    curr = 0

    if nParts == 0:
        log_message("TODO: nParts == 0: DA IMPLEMENTARE")
        print(todo)

    x, y = entities.pointer[index].exterior.xy
    if destMap is not None:
        lat, lon = dpg.map.yx_2_latlon(y, x, map=sourceMap)
        if globe > 1:
            print(todo)
        else:
            y, x = dpg.map.latlon_2_yx(lat, lon, map=destMap)

        data[:, 0] = x
        data[:, 1] = y

    return nVerts, data


def getShapeCoords(
    entities,
    index=None,
    destMap=None,
    optim=None,
    sourceMap=None,
    polylines=None,
    box=False,
):
    """
    Extracts coordinates and polyline data from geometric entities.

    Args:
        entities: List or single entity object containing geometry.
        index: Index of the specific entity to process (optional).
        destMap: Destination map for projection settings (optional).
        optim: Boolean to optimize for a specific index (optional).
        sourceMap: Source map for geometry reference (optional).
        polylines: Array to store polyline data (optional).
        box: Boolean flag to compute bounding box (optional).

    Returns:
        A tuple containing the number of entities, their coordinates, and polylines.
    """
    err = 1

    if isinstance(entities, list):
        entities = entities[0]

    if entities is None:
        return 0

    if entities.pointer is None:
        return 0
    nEntities = len(entities.pointer)

    if np.size(index) == 1 and index is not None:
        if index >= nEntities:
            return 0

    err = 0
    curr = 0
    totElP = 0
    nDim = 2
    proj = 0

    if destMap is not None:
        proj = destMap.mapProj

    if proj == -2:  # TODO: wat?
        nDim = 3

    totV = entities.pointer.apply(lambda poly: len(poly.exterior.coords)).sum()

    if not optim or index is None:
        data = np.zeros((totV, nDim))
        polylines = np.zeros(nDim * totV)

    if index is not None and np.size(index) == 1:
        if optim:
            nVerts, data = getShapeEntityOpt(
                entities,
                index,
                sourceMap=sourceMap,
                destMap=destMap,
                polylines=polylines,
            )
            if box:
                print(todo)
            return nVerts, data, polylines
    else:
        for h in range(nEntities):
            ret, curr, totElP = getShapeEntity(
                entities,
                h,
                sourceMap=sourceMap,
                ndim=nDim,
                center=True,
                destMap=destMap,
                polylines=polylines,
                curr=curr,
                totElP=totElP,
                data=data,
            )

    if curr <= 0 or curr > len(data):
        return 0

    data = data[0:curr, :]
    if totElP <= 0:
        if len(polylines) > 0:
            tmp = polylines.copy()
    else:
        polylines = polylines[0:totElP]

    if box:
        box = dpg.map.check_box(data[:, 0], data[:, 1], None)

    return nEntities, data, polylines


def computeTrack(
    pVertex,
    vel,
    deg,
    minutes,
    intensity,
    evolution,
    dim,
    par,
    incDir=None,
    get_track=False,
    step=None,
    data=None,
):
    """
    Simulates the movement of points and computes a track map.

    Parameters:
        pVertex: Initial positions of vertices.
        vel: Velocity for translation.
        deg: Direction in degrees for movement.
        minutes: Duration of the simulation in minutes.
        intensity: Intensity value for the track map.
        evolution: Scaling factors for each time interval.
        dim: Dimensions of the output map.
        par: Parameters for the projection or mapping function.
        incDir: Incremental change in direction (optional).
        get_track: Boolean to generate a track map (optional).
        step: Time step interval for data recording (optional).
        data: Data structure for storing simulation results (optional).

    Returns:
        If `get_track` is True, returns a track map.
        Otherwise, returns None.
    """
    if pVertex is None:
        return

    nD = np.shape(pVertex)
    x = pVertex[:, 0]
    y = pVertex[:, 1]

    if nD[0] == 3:
        z = pVertex[:, 2]
    totV = len(x)

    if len(evolution) <= 0:
        evolution = np.ones(6)
    if incDir is None:
        incDir = 0

    lastMinute = minutes
    ind = np.where(evolution <= 0)
    if len(ind[0]) > 0 and minutes > ind[0] * 10:
        lastMinute = ind[0] * 10

    if get_track:
        if len(dim) != 2:
            return
        nEl = dim[0] * dim[1]
        track = np.zeros(dim)
        lin, col = dpg.map.yx_2_lincol(y, x, par, dim=dim, force_limits=True)

        mask = np.zeros(dim, dtype=bool)
        rr, cc = polygon(lin, col, mask.shape)
        mask[rr, cc] = True
        maskInd = np.where(mask > 0)
        track[maskInd] += intensity

    if step is None:
        step = 10

    if data:
        nSteps = np.fix(lastMinute / step) + 1
        data = np.zeros((nD[0], nSteps * totV))
        polygons = np.zeros((nSteps * totV) + nSteps)
        ind = lindgen(totV)  # TODO: da fare
        ind2 = [totV, ind]  # TODO: da controllare
        last = 0
        data[0, ind] = x
        data[1, ind] = y
        polygons[last : last + totV] = ind2
        if nD[0] == 3:
            data[2, ind] = z

    angle = deg

    for mmm in range(1, lastMinute):
        scale = 1
        if np.mod(mmm, 5) == 1:
            scInd = mmm // 10
            if scInd < len(evolution):
                scale = np.sqrt(evolution[scInd])

        x, y = dpg.map.translatePoints(x, y, vel, angle, 1)
        x, y = dpg.map.scale_points(x, y, scale=scale)
        angle += incDir / 10
        if data is not None and np.size(data) > 0 and np.mod(mmm, step) == 0:
            last += totV + 1
            ind += totV
            ind2[1:] += totV
            data[ind, 0] = x
            data[ind, 1] = y
            if nD[0] == 3:
                data[ind, 2] = z
            polygons[last : last + totV] = ind2

        if get_track:
            lin, col = dpg.map.yx_2_lincol(
                y, x, par, dim=dim, force_limits=True, set_center=True
            )
            mask = np.zeros(dim, dtype=bool)
            rr, cc = polygon(lin, col, mask.shape)
            mask[rr, cc] = True
            maskInd = np.where(mask > 0)
            track[maskInd] += intensity

    return track
