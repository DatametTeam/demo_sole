import sys
from numbers import Number
import numpy as np
import math

import pyproj
from numba import njit

import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.io import get_idl_proj_from_python
from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message
from sou_py.dpg.map__define import Map
from pyproj import Proj, Transformer

"""
Funzioni ancora da portare
FUNCTION GET_DEST_SCALE 
FUNCTION GET_DIM_FROM_BOX 
FUNCTION GET_LOCAL_RES 
FUNCTION GET_OFFSETS 
FUNCTION GET_PROJ_CENTER 
FUNCTION GET_PROJ_CORNERS 
FUNCTION GET_PROJ_PAR 
FUNCTION GET_PROJ_PAR_FROM_DIAM 
FUNCTION GET_PROJ_PAR_SAMPLING 
FUNCTION GET_PROJ_SCALE 
FUNCTION GetAzIndex 
FUNCTION GetGeoTiffTags 
FUNCTION GetProjNames 
FUNCTION IS_3D_MAP 
FUNCTION SCALE_PAR 
PRO CHECK_PROJ_PAR 
PRO GEO_2_NORM 
PRO GET_ALTITUDES 
PRO GET_PROJ_RANGE 
FUNCTION CONVERT_PAR                    // UNUSED
FUNCTION GET_DEST_BOX_SAMPLING          // UNUSED
FUNCTION GET_DIAM_FROM_PROJ_PAR         // UNUSED
FUNCTION GET_NORM_FROM_PROJ_PAR         // UNUSED
FUNCTION GET_PROJ_PAR_FROM_NORM         // UNUSED
FUNCTION GET_SAMPLING                   // UNUSED
"""

transformer_cache = {}


def getMapName(map: Map = None, box=None):
    """
    Retrieves various properties of a map object, including its name, projection, and coordinates.

    This function examines a map object (presumed to be of type 'dpg.map__define.Map') and extracts its name,
    projection type, and coordinates. If 'map' is not an instance of 'dpg.map__define.Map', default values are
    returned. Additionally, if 'box_flag' is set to True, the function will also return the bounding box of the map.

    Args:
        map (dpg.map__define.Map, optional): The map object from which to extract information. Defaults to None.
        box_flag (bool, optional): A flag to determine whether the bounding box should be returned. Defaults to False.

    Returns:
        tuple: A tuple containing the following:
            - up_name (str): The name of the map or an empty string if 'map' is not valid.
            - proj (int): The projection type of the map, or 0 if 'map' is not valid.
            - p0Lat (float): The latitude of the reference point of the map, or 0.0 if 'map' is not valid.
            - p0Lon (float): The longitude of the reference point of the map, or 0.0 if 'map' is not valid.
            - box (list[float] or None): The bounding box of the map as [min_lon, min_lat, max_lon, max_lat],
            or None if 'box_flag' is False.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    up_name = ""
    if not isinstance(map, dpg.map__define.Map):
        proj = 0
        p0Lat = 0.0
        p0Lon = 0.0
        # if box_flag:
        #     box = [-180., -90., 180., 90.]
        return up_name, proj, p0Lat, p0Lon
    # endif

    if box:
        box = map.uv_box

    proj = map.mapProj
    p0Lat = map.p0lat
    p0Lon = map.p0lon
    up_name = map.up_name
    return up_name, proj, p0Lat, p0Lon


def getEarthRadius(real: bool = False) -> float:
    """
    Returns the Earth's radius in meters.

    This function provides the radius of the Earth based on the value of the 'real' flag. If 'real' is True,
    the function returns the actual average radius of the Earth. If 'real' is False, it returns a default
    larger value, which might be used for specific calculations or models.

    Args:
        real (bool, optional): A flag to determine which radius value to return. If True, the actual average
                               Earth radius is returned. Defaults to False.

    Returns:
        float: The radius of the Earth in meters. Returns the actual average Earth radius (6378137.0 meters)
               if 'real' is True, otherwise returns a default value (8500000.0 meters).

    Raises:
        None: This function does not raise any exceptions.
    """
    if real:
        return 6378137.0
    return 8500000.0


def getPolarRange(
        rangeRes: Number, nBins: int, range_off: Number = None, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Generates an array representing the range values for polar coordinates.

    This function calculates the range values for a polar coordinate system based on the specified range resolution,
    number of bins, and an optional range offset. The range values are calculated as an array of values, each
    representing the distance from the origin to the middle of a bin.

    Args:
        rangeRes (Number): The range resolution, which is the distance between consecutive range values.
        nBins (int): The number of bins or segments in the range.
        range_off (Number, optional): An optional offset to be added to each range value. Defaults to None.
        dtype (data-type, optional): The desired data-type for the array, for example, numpy.float32. Defaults to
        np.float32.

    Returns:
        numpy.ndarray: A NumPy array of length 'nBins', where each element represents a range value calculated based
                       on the range resolution and range offset.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter standard NumPy exceptions
        related to array operations.
    """
    sr = np.arange(nBins, dtype=dtype) + 0.5

    if rangeRes is None:
        log_message("RangeRes is None: ERROR", level="ERROR")
        return sr

    sr *= rangeRes
    if isinstance(range_off, Number):
        sr += range_off
    return sr


def slantRangeToHeight(
        slantRange: Number,
        elevationDeg: Number,
        ECC: Number = 1,
        site_height: Number = None,
) -> np.float32:
    """
    Calculates the height above the Earth's surface for a given slant range and elevation angle.

    This function computes the height at which an object is located above the Earth's surface given its slant range
    (the straight-line distance from the observer to the object) and elevation angle. It optionally considers the
    Earth's curvature correction (ECC) and the observer's height above the Earth's surface (site_height).
    For elevation angles greater than 80 degrees, or when ECC is 0, Earth's curvature is ignored, and the height is
    calculated as a simple trigonometric function of the slant range and elevation angle.

    Args:
        slantRange (Number): The slant range, i.e., the straight-line distance from the observer to the object.
        elevationDeg (Number): The elevation angle in degrees.
        ECC (Number, optional): The Earth's curvature correction factor. If set to 0, Earth's curvature is ignored.
        Defaults to 1.
        site_height (Number, optional): The observer's height above the Earth's surface. Defaults to None.

    Returns:
        numpy.float32: The height of the object above the Earth's surface, adjusted for Earth's curvature if applicable.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter standard NumPy exceptions
        related to mathematical operations.
    """
    sinEl = np.sin(elevationDeg * np.pi / 180)  # !DTOR * elevationDeg
    height = slantRange * sinEl
    if isinstance(elevationDeg, Number):
        if elevationDeg > 80.0 or ECC == 0.0:
            if isinstance(site_height, Number):
                height += site_height
            return height
        # endif
    # endif
    er = getEarthRadius()
    hhh = np.ceil(height * er * 2.0, dtype=np.float64)
    hhh += er * er
    hhh += np.square(slantRange)
    if isinstance(site_height, Number):
        er -= site_height
    height = np.sqrt(hhh) - er
    return height.astype(np.float32)


def total_finite(a) -> int:
    """
    Counts the number of finite (non-None, non-NaN, non-infinite) elements in an iterable.

    This function iterates over the elements of the given iterable 'a', and counts how many of them are finite values.
    A finite value in this context is defined as a value that is not None, not NaN (Not a Number), and not infinite.
    The function is useful for data sets where such non-finite values may be present and need to be excluded from
    certain calculations.

    Args:
        a (iterable): An iterable (like a list or tuple) containing elements to be checked for finiteness.

    Returns:
        int: The count of finite elements in the iterable.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'math.isnan' and 'math.isinf',
              which might raise exceptions if 'a' contains non-numeric types.
    """
    return sum(
        1 for x in a if x is not None and not math.isnan(x) and not math.isinf(x)
    )


def get_proj_direction(map: Map) -> float:
    """
    Determines the projection direction (north-up or south-up) for a given map object.

    This function calculates the projection direction of a map object by comparing the y-coordinates of two points:
    one at the map's reference latitude (p0Lat) and another one degree north of this reference point.
    If the y-coordinate of the northern point is smaller than that of the reference point, the function infers a
    south-up projection direction. Otherwise, it infers a north-up projection.

    Args:
        map (Map object): The map object for which the projection direction needs to be determined. The map object
        should
                          have attributes like 'up_name', 'projection', 'p0lat', 'p0lon', and a method 'latlon_2_yx' for
                          converting latitude-longitude to map coordinates.

    Returns:
        list[float]: A two-element list representing the projection direction. The first element is always 1.0. The
        second element is 1.0 for north-up projections and -1.0 for south-up projections.

    Raises:
        None: This function does not explicitly raise any exceptions but may depend on the behavior of the
        'getMapName' and 'latlon_2_yx' methods of the map object.
    """
    dir = np.array([1.0, 1.0], dtype=np.float32)

    if map is None:
        return dir

    up_name, proj, p0Lat, p0Lon = getMapName(map)

    lat0r = np.array([p0Lat, p0Lat + 1.0], dtype=np.float32)
    lon0r = np.array([p0Lon, p0Lon + 1.0], dtype=np.float32)

    y, x = latlon_2_yx(lat0r, lon0r, map=map)

    if y[0] > y[1]:
        dir[1] = -1

    return dir


def get_box_from_par(
        map: Map,
        par,
        dim,
        box=None,
        polar_dim: bool = None,
        up_direction: bool = None,
        regular: bool = None,
) -> list:
    """
    Calculates the bounding box of a map view based on various parameters including the map object, view parameters,
    and dimensions.

    This function determines the bounding box of a map view. If 'box' is not provided or if it doesn't contain 4
    elements,
    the function tries to obtain it from the map object. The function uses 'par' (parameters like offsets and
    resolutions)
    and 'dim' (dimensions) to calculate the bounding box. It also considers special cases like polar dimensions and
    projection
    directions.

    Args:
        map (Map object): The map object to use for getting default box values.
        par (list[float]): Parameters including offsets and resolutions in the format [xoff, xres, yoff, yres, ...].
        dim (list[int]): Dimensions of the view in the format [ydim, xdim].
        box (list[float], optional): The initial bounding box as [Xw, Yn, Xe, Ys]. Defaults to None.
        polar_dim (bool, optional): A flag indicating if polar dimensions are considered. Defaults to None.
        up_direction (bool, optional): A flag indicating the up direction of the map view. Defaults to None.
        regular (bool, optional): A flag to use regular bounding box calculations. Defaults to None.

    Returns:
        list[float]: The calculated bounding box as [X1, Y1, X2, Y2].

    Raises:
        None: This function does not explicitly raise any exceptions but may depend on the behavior of other
        functions like 'get_proj_direction'.
    """
    if box is None:
        box = np.array([], dtype=np.float32)
    view_defined = 0

    if len(box) != 4:
        if map is not None:
            name, proj, p0Lat, p0Lon = dpg.map.getMapName(map)
        else:
            box = 0

    if len(par) < 4:
        return box, None, None, None
    if len(par) <= 6:
        res = total_finite(par[0:4])
        if res != 4:
            return box, None, None, None

    if len(dim) < 2:
        return box, None, None, None
    if dim[0] <= 0 or dim[1] <= 0:
        if len(box) == 4:
            if total_finite(box) == 4:
                return box, None, None, None
        return 0, None, None, None

    xoff = par[0]
    yoff = par[2]
    xres = par[1]
    yres = par[3]

    if xres == 0 and yres == 0:
        if len(par) >= 8:
            xres = par[7]
        if xres <= 0:
            return box, None, None, None
        yres = xres
        if len(dim) == 3:
            xoff = np.float32(dim[2])
        else:
            xoff = np.float32(dim[1])
        yoff = xoff
        polar_dim = 1

    if len(dim) == 3:
        xdim = dim[2]
        ydim = dim[1]
    else:
        xdim = dim[1]
        ydim = dim[0]

    if polar_dim is not None:
        xdim *= 2
        ydim = xdim

    if xres == 0:
        ydim = xdim
        xres = yres

    if yres == 0:
        ydim = xdim
        yres = xres

    X1 = -xoff * xres
    Y1 = -yoff * yres
    X2 = np.float32(X1 + xres * xdim)
    Y2 = np.float32(Y1 + yres * ydim)

    dir = get_proj_direction(map)

    Xw = min([X1, X2])
    Xe = max([X1, X2])

    if dir[1] > 0:
        Ys = min([Y1, Y2])
        Yn = max([Y1, Y2])
    else:
        if up_direction is not None:
            appo = -Y2
            Y2 = -Y1
            Y1 = appo
        Yn = min([Y1, Y2])
        Ys = max([Y1, Y2])

    nw_se_box = np.array([Xw, Yn, Xe, Ys])
    ne_sw_box = np.array([Xe, Yn, Xw, Ys])

    if regular:
        X1, X2 = minmax([X1, X2])
        Y1, Y2 = minmax([Y1, Y2])

    view_defined = 1
    return np.array([X1, Y1, X2, Y2]), nw_se_box, ne_sw_box, view_defined


def minmax(elem):
    """
    Returns the minimum and maximum values from a given iterable.

    This function takes an iterable 'elem' and returns a tuple containing the minimum and maximum values found in it.
    It's a utility function meant to simplify finding both the smallest and largest elements in a collection of values.

    Args:
        elem (iterable): An iterable (like a list, tuple, etc.) containing comparable elements.

    Returns:
        tuple: A tuple containing two elements:
            - The first element is the minimum value in the iterable.
            - The second element is the maximum value in the iterable.

    Raises:
        ValueError: If 'elem' is an empty iterable.
    """
    return min(elem), max(elem)


def extractCoords(values: list):
    """
    Extracts coordinate values and indices from a list of strings, each representing a set of coordinates.

    This function processes a list of strings, where each string contains space-separated coordinate values.
    It splits these strings into individual coordinates, converts them to floats, and accumulates them in a list.
    Additionally, it generates a list of index ranges for each set of coordinates.

    Args:
        values (list[str]): A list of strings, each containing space-separated coordinate values.

    Returns:
        tuple: A tuple containing three elements:
            - nc (int): The number of strings (coordinate sets) processed.
            - coords (list[float]): A flattened list of all coordinate values extracted from the strings.
            - list_ (list[list]): A list where each element is a two-element list containing:
                - The number of coordinates in a set.
                - A list of indices representing the positions of these coordinates in the 'coords' list.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    nc = len(values)
    coords = []
    list_ = []
    currInd = 0

    for ccc in range(nc):
        chan = values[ccc].split()
        nb = len(chan)
        coords.extend(float(coord) for coord in chan)
        ind = np.arange(nb) + currInd
        list_.extend([nb, ind.tolist()])
        currInd += nb

    return nc, coords, list_


def get_proj_z_conv() -> list:
    """
    Returns a default conversion factor for z-coordinates in a projection.

    This function provides a fixed set of values that might be used as a default conversion factor for
    z-coordinates in certain map projections or transformations.

    Returns:
        list[float]: A two-element list containing default conversion factors for z-coordinates, [0.0, 1.0].

    Raises:
        None: This function does not raise any exceptions.
    """
    return [0.0, 1.0]


def rlon2rlons(plat: float, plon: float, pole: list) -> float:
    """
    Converts latitude and longitude coordinates to rotated longitude based on a specified pole.

    This function calculates the rotated longitude of a point given its latitude and longitude, and the latitude
    and longitude of a rotation pole. The rotation is performed around the given pole, and the function is
    typically used in rotated pole grid systems in meteorology and climatology.

    Args:
        plat (float): The latitude of the point in degrees.
        plon (float): The longitude of the point in degrees.
        pole (list[float]): A two-element list containing the latitude and longitude of the rotation pole, in degrees.

    Returns:
        float: The rotated longitude of the point.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter mathematical domain errors in
        functions like math.sin or math.atan2.
    """
    pollon = pole[0] * math.radians(1)
    pollat = pole[1] * math.radians(1)

    zsinpol = math.sin(pollat)
    zcospol = math.cos(pollat)

    zlat = plat * math.radians(1)
    zlon = plon * math.radians(1)

    zarg1 = -math.sin(zlon - pollon) * math.cos(zlat)
    zarg2 = -zsinpol * math.cos(zlon - pollon) * math.cos(zlat) + zcospol * (
        math.sin(zlat)
    )

    return math.atan2(zarg1, zarg2)


def rotateWindComponents(pus: list, pvs: list, plat: float, plon: float, pole: list):
    """
    Rotates wind component vectors based on a specified rotation pole.

    This function takes u and v wind components (pus and pvs) and rotates them according to the position of a specified
    pole. The rotation is calculated based on the latitude and longitude of each point in the wind data and the latitude
    and longitude of the rotation pole. It is commonly used in meteorological data processing to adjust wind vectors
    to a rotated grid.

    Args:
        pus (list[float]): The u-components of the wind vectors.
        pvs (list[float]): The v-components of the wind vectors.
        plat (float): The latitude of the wind data points in degrees.
        plon (float): The longitude of the wind data points in degrees.
        pole (list[float]): A two-element list containing the latitude and longitude of the rotation pole, in degrees.

    Returns:
        tuple: A tuple containing two lists:
            - The rotated u-components of the wind vectors.
            - The rotated v-components of the wind vectors.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter mathematical domain errors
        during calculations.
    """
    if pole[0] == 90.0:
        return pus, pvs

    pollon = pole[0] * math.radians(1)
    pollat = pole[1] * math.radians(1)
    zsinpol = math.sin(pollat)
    zlat = plat * math.radians(1)
    zlon = plon * math.radians(1)

    zlons = rlon2rlons(plat, plon, pole)

    zarg = -zsinpol * math.sin(zlon - pollon) * math.sin(zlons) - math.cos(
        zlon - pollon
    ) * math.cos(zlons)

    ind = [i for i, val in enumerate(zarg) if val > 1.0]
    for i in ind:
        zarg[i] = 1.0

    ind = [i for i, val in enumerate(zarg) if val < -1.0]
    for i in ind:
        zarg[i] = -1.0

    zbeta = [math.acos(val) for val in zarg]

    zpollond = pollon
    if zpollond < 0.0:
        zpollond += 360.0

    y = -(plon - (zpollond - 180.0))

    ind = [i for i, val in enumerate(y) if val >= 0.0]
    comp = [i for i in range(len(y)) if i not in ind]
    nComp = len(comp)

    for i in ind:
        zbeta[i] = abs(zbeta[i])

    for i in comp:
        zbeta[i] = -abs(zbeta[i])

    tmpU = pus
    tmpV = pvs
    pus = [
        tmpU[i] * math.cos(zbeta[i]) - tmpV[i] * math.sin(zbeta[i])
        for i in range(len(tmpU))
    ]
    pvs = [
        tmpU[i] * math.sin(zbeta[i]) + tmpV[i] * math.cos(zbeta[i])
        for i in range(len(tmpV))
    ]

    return pus, pvs


def rotatePoints(x: np.ndarray, y: np.ndarray, deg: float):
    """
    Rotates points (x, y) around the origin by a specified degree.

    This function rotates a set of points specified by their x and y coordinates around the origin (0, 0) by a given
    angle
    in degrees. The rotation is counterclockwise. If the rotation degree is 0, the function returns the original
    coordinates.

    Args:
        x (np.ndarray): An array of x-coordinates.
        y (np.ndarray): An array of y-coordinates.
        deg (float): The angle in degrees by which to rotate the points.

    Returns:
        tuple: A tuple containing two arrays:
            - The x-coordinates after rotation.
            - The y-coordinates after rotation.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter mathematical domain errors
        during calculations.
    """
    if deg == 0.0:
        return x, y

    alpha = np.arctan2(y, x)
    ro = np.sqrt(y * y + x * x)
    ind = np.where(np.isfinite(alpha) == 0)[0]
    if len(ind) > 0:
        alpha[ind] = 0.0

    rot = (360.0 - deg) * math.radians(1)
    alpha += rot
    x = ro * np.cos(alpha)
    y = ro * np.sin(alpha)

    return x, y


def translatePoints(
        x: np.ndarray, y: np.ndarray, vel: float, deg: float, minutes: float
):
    """
    Translates points (x, y) by a specified velocity and direction over a given time interval.

    This function moves each point in the (x, y) coordinate space based on a velocity and a direction for a specified
    amount of time (in minutes). The direction is given in degrees, with 0 degrees translating directly eastward and
    increasing counterclockwise.

    Args:
        x (np.ndarray): An array of x-coordinates.
        y (np.ndarray): An array of y-coordinates.
        vel (float): The velocity at which to move the points.
        deg (float): The direction in which to move, in degrees.
        minutes (float): The time interval over which to move the points, in minutes.

    Returns:
        tuple: A tuple containing two arrays:
            - The x-coordinates after translation.
            - The y-coordinates after translation.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter mathematical domain errors
        during calculations or invalid finite checks.
    """
    fact = (minutes * 1000.0) / 60.0
    alpha = (450.0 - deg) * math.radians(1)

    dx = vel * fact * np.cos(alpha)
    dy = vel * fact * np.sin(alpha)

    ind = np.where((np.isfinite(dx) == 0) | (vel < 0.0))[0]
    if len(ind) > 0:
        dx[ind] = 0.0
        dy[ind] = 0.0

    x += dx
    y += dy

    return x, y


def rotateGrid(lat: np.ndarray, lon: np.ndarray, rotation: list):
    """
    Rotates a grid of latitude and longitude coordinates based on specified rotation angles.

    This function rotates a grid defined by latitude and longitude arrays using a specified rotation specified by
    two angles: the rotation around the x-axis (fi) and the rotation around the z-axis (teta). The rotation is
    performed in three-dimensional space, treating the Earth as a sphere.

    Args:
        lat (np.ndarray): An array of latitudes in degrees.
        lon (np.ndarray): An array of longitudes in degrees.
        rotation (list[float]): A two-element list specifying the rotation angles in degrees. The first element
                                is the rotation around the x-axis, and the second element is the rotation around the
                                z-axis.

    Returns:
        tuple: A tuple containing two arrays:
            - The latitudes after rotation.
            - The longitudes after rotation.

    Raises:
        None: This function does not explicitly raise any exceptions but may encounter mathematical domain errors
        during calculations.
    """
    fi = rotation[0] * math.radians(1)
    teta = rotation[1] * math.radians(1)

    xarray = lon * math.radians(1)
    yarray = lat * math.radians(1)

    cosx = np.cos(xarray)
    cosy = np.cos(yarray)
    sinx = np.sin(xarray)
    siny = np.sin(yarray)
    costeta = np.cos(teta)
    cosfi = np.cos(fi)
    sinteta = np.sin(teta)
    sinfi = np.sin(fi)

    x1 = cosx * cosy
    y1 = sinx * cosy
    z1 = siny

    xxx = costeta * cosfi * x1 + sinfi * y1 + sinteta * cosfi * z1
    yyy = -costeta * sinfi * x1 + cosfi * y1 - sinteta * sinfi * z1
    zzz = -sinteta * x1 + costeta * z1

    lat = np.arcsin(zzz) * math.degrees(1)
    lon = np.arctan2(yyy, xxx) * math.degrees(1)

    return lat, lon


def rotateGrid_inverse(y: np.ndarray, x: np.ndarray, map: Map = None):
    """
    Performs the inverse rotation of a grid of latitude and longitude coordinates based on a map's rotation settings.

    This function applies an inverse rotation to a set of latitude and longitude coordinates. It first retrieves the
    world offset (pole position) from the map object, if provided, and then uses this to reverse the rotation applied
    to the grid. The function is useful for converting coordinates from a rotated grid system back to a standard
    geographical grid system.

    Args:
        y (np.ndarray): An array of latitudes in degrees.
        x (np.ndarray): An array of longitudes in degrees.
        map (Map object, optional): The map object containing rotation settings. Defaults to None.

    Returns:
        tuple: A tuple containing two arrays:
            - The latitudes after inverse rotation.
            - The longitudes after inverse rotation.

    Raises:
        None: This function does not explicitly raise any exceptions but depends on the 'getWorldOffset' and
        'rotateGrid' functions.
    """
    pole = getWorldOffset(map)
    lat = y
    lon = x

    if pole is not None or len(pole) < 2:
        return lat, lon
    if pole[0] == 0.0 and pole[1] == 0.0:
        return lat, lon

    rotation = [pole[0], 90.0 + pole[1]]
    lat, lon = rotateGrid(lat, lon, -rotation)

    return lat, lon


def rotateGrid_forward(y: np.ndarray, x: np.ndarray, map: Map = None):
    """
    Rotates a grid of latitude and longitude coordinates forward based on a map's rotation settings.

    This function applies a forward rotation to a set of latitude and longitude coordinates. It uses the world offset
    (pole position) from the map object, if provided, to perform the rotation. This is useful for converting standard
    geographical coordinates to a rotated grid system as used in some map projections.

    Args:
        y (np.ndarray): An array of latitudes in degrees.
        x (np.ndarray): An array of longitudes in degrees.
        map (Map object, optional): The map object containing rotation settings. Defaults to None.

    Returns:
        tuple: A tuple containing two arrays:
            - The latitudes after forward rotation.
            - The longitudes after forward rotation.

    Raises:
        None: This function does not explicitly raise any exceptions but depends on the 'getWorldOffset' and
        'rotateGrid' functions.
    """
    lat = y
    lon = x
    pole = getWorldOffset(map)

    if pole is not None or len(pole) < 2:
        return lat, lon
    if pole[0] == 0.0 and pole[1] == 0.0:
        return lat, lon
    rotation = [pole[0], 0.0]
    lat, lon = rotateGrid(lat, lon, rotation)
    rotation = [0.0, 90.0 + pole[1]]
    lat, lon = rotateGrid(lat, lon, rotation)

    return lat, lon


def check_box(datax: np.ndarray, datay: np.ndarray, dataz: np.ndarray = None):
    """
    Calculates the bounding box and height range of given data points.

    This function computes the bounding box for a set of data points represented by their x and y coordinates,
    and also determines the range of values (height range) for the z-axis data. It handles NaN values by ignoring
    them in the calculation of the maxima and minima.

    Args:
        datax (np.ndarray): An array of x-coordinates.
        datay (np.ndarray): An array of y-coordinates.
        dataz (np.ndarray): An array of z-coordinates (height data).

    Returns:
        tuple: A tuple containing two elements:
            - A list representing the bounding box as [X1, Y1, X2, Y2], where X1 and Y1 are the minimum x and y
              coordinates, and X2 and Y2 are the maximum x and y coordinates.
            - A list representing the height range as [minz, maxz], where minz and maxz are the minimum and maximum
              z-coordinates.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on NumPy's nanmin and nanmax functions,
              which might return NaN if the entire array contains only NaNs.
    """
    X1 = 0.0
    X2 = 0.0
    Y1 = 0.0
    Y2 = 0.0
    hRange = [0.0, 0.0]

    if np.ndim(datax) > 0:
        X2 = np.nanmax(datax)
        X1 = np.nanmin(datax)
    if np.ndim(datay) > 0:
        Y2 = np.nanmax(datay)
        Y1 = np.nanmin(datay)
    if np.ndim(dataz) > 0:
        maxz = np.nanmax(dataz)
        minz = np.nanmin(dataz)
        hRange = [minz, maxz]

    return [X1, Y1, X2, Y2], hRange


def scale_points(x: np.ndarray, y: np.ndarray, scale: float):
    """
    Scales the given points (x, y) relative to their bounding box, preserving the center.

    This function scales the x and y coordinates of a set of points by a specified scale factor. The scaling is done
    relative to the bounding box of the points, with the center of the bounding box remaining fixed. If the scale factor
    is 1.0 or less than or equal to 0.0, the function returns the original coordinates.

    Args:
        x (np.ndarray): An array of x-coordinates.
        y (np.ndarray): An array of y-coordinates.
        scale (float): The scale factor by which to scale the points.

    Returns:
        tuple: A tuple containing two arrays:
            - The scaled x-coordinates.
            - The scaled y-coordinates.

    Raises:
        None: This function does not explicitly raise any exceptions but depends on the 'check_box' function to
        determine
              the bounding box of the points.
    """
    if scale == 1.0:
        return x, y
    if scale <= 0.0:
        return x, y

    box, _ = check_box(x, y)

    xr = box[2] - box[0]
    yr = box[3] - box[1]

    if xr > 0.0:
        x = -0.5 + (x - box[0]) / xr
        x *= scale * xr
        x += (box[0] + box[2]) / 2.0

    if yr > 0.0:
        y = -0.5 + (y - box[1]) / yr
        y *= scale * yr
        y += (box[1] + box[3]) / 2.0

    return x, y


def check_points(p1: int, p2: int, dim: int):
    """
    Validates and adjusts a pair of points against a specified dimension limit.

    This function checks and adjusts two points, p1 and p2, to ensure they fall within a valid range based on
    a given dimension (dim). It swaps the points if p1 is greater than p2, ensures the points are within the
    range [0, dim-1], and sets an indicator (inp) to specify if the points are within a valid range.

    Args:
        p1 (int): The first point.
        p2 (int): The second point.
        dim (int): The dimension limit against which the points are to be validated.

    Returns:
        tuple: A tuple containing three elements:
            - inp (int): An indicator (1 or 0) specifying if the points are within a valid range. 1 indicates valid,
            0 indicates invalid.
            - p1 (int): The adjusted first point, ensured to be within the valid range.
            - p2 (int): The adjusted second point, ensured to be within the valid range.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    inp = 1

    if p1 > p2:
        app = p1
        p1 = p2
        p2 = app

    if p2 < 0:
        p1 = 0
        p2 = 0
        inp = 0

    if p1 < 0:
        p1 = 0

    if p1 >= dim:
        p1 = dim - 1
        p2 = dim - 1
        inp = 0

    if p2 >= dim:
        p2 = dim - 1

    return inp, p1, p2


def translateLevel(lev: int, par: list = None, el_coords: float = None) -> float:
    """
    Translates a level index to a corresponding elevation value based on provided parameters or elevation coordinates.

    This function computes the elevation value corresponding to a given level index ('lev'). If a list of elevation
    coordinates ('el_coords') is provided and 'lev' is within the range of 'el_coords', the function returns the
    elevation at that index. Otherwise, it calculates the elevation using the height offset ('hoff') and resolution
    ('hres') provided in 'par'. If 'hres' is non-positive and 'par' has more than 9 elements, an alternate method
    using other elements in 'par' is used for calculation.

    Args:
        lev (int): The level index for which the elevation is to be calculated.
        par (list[float], optional): Parameters including height offset and resolution. Defaults to None.
        el_coords (list[float], optional): A list of predefined elevation coordinates. Defaults to None.

    Returns:
        float: The calculated or retrieved elevation value for the specified level.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    hoff = 0.0
    hres = 1.0

    if par is not None and len(par) > 4:
        hoff = par[4]
    if par is not None and len(par) > 5:
        hres = par[5]
    if par is not None and len(par) > 9 and hres <= 0.0:
        hres = par[8]
        hres = par[9]

    if lev < len(el_coords):
        return el_coords[lev]

    return (lev * hres) + hoff


def get_up_dir_par(par: list, dir: list = None, order: int = None):
    """
    Adjusts parameters based on the up direction and optionally sets the ordering.

    This function modifies a copy of the given parameters ('par') based on the up direction ('dir'). If the
    y-component of 'dir' is negative, it negates the third element of 'par'. If the product of the y-component
    of 'dir' and the third element of 'par' is negative, it sets 'order' to 1. The function is useful for
    adjusting parameters in scenarios where directionality affects parameter values.

    Args:
        par (list[float]): The original parameters to be adjusted.
        dir (list[float], optional): The up direction, typically as [x, y] components. Defaults to None.
        order (int, optional): A variable to set the ordering based on the direction. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - A list of adjusted parameters.
            - The potentially modified 'order' value.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    UpPar = par.copy()

    if dir is None or len(dir) < 2:
        return UpPar

    if dir[1] < 0:
        UpPar[3] = -par[3]

    if dir[1] * par[3] < 0:
        order = 1

    return UpPar, order


def get_rotated_par(par: list, dim: list = None):
    """
    Adjusts parameters based on rotation and flips along the x and y axes.

    This function computes the rotation needed and adjusts the parameters ('par') based on the given dimensions
    ('dim'). It considers flipping along the x and/or y axes. If 'par[1]' or 'par[3]' is negative, indicating
    a flip, the function adjusts the corresponding offset parameters ('par[0]' or 'par[2]') and negates the
    resolution ('par[1]' or 'par[3]'). The function returns a rotation code and the adjusted parameters.

    Args:
        par (list[float]): The original parameters to be adjusted.
        dim (list[int], optional): The dimensions of the grid. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - rotation (int): The rotation code, indicating the type of rotation/flip applied.
            - rotatedPar (list[float]): The adjusted parameters after considering rotation and flip.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    rotation = 0
    rotatedPar = par.copy()

    if dim is None or len(dim) < 2:
        return 0

    flip = 0
    flop = 0

    if par[1] < 0 and dim[0] > 0:
        rotatedPar[0] = dim[0] - par[0]
        rotatedPar[1] = -par[1]
        flip = 1

    if par[3] < 0 and dim[1] > 0:
        rotatedPar[2] = dim[1] - par[2]
        rotatedPar[3] = -par[3]
        flop = 1

    if flip == 1 and flop == 1:
        rotation = 2
    if flip == 1 and flop == 0:
        rotation = 5
    if flip == 0 and flop == 1:
        rotation = 7

    return rotation, rotatedPar


def init_lincol(
        dim: list, precision: float, float_flag: bool = False, plus_1_flag: bool = False
):
    """
    Initializes linearly spaced column and line indices based on specified dimensions and precision.

    This function generates arrays of column ('col') and line ('lin') indices for a grid defined by 'dim'
    and scaled by 'precision'. The function allows options to adjust the final dimension size by 1 and to
    choose between floating-point or integer indices.

    Args:
        dim (list[int]): The original dimensions of the grid as [columns, lines].
        precision (float): The scaling factor for the grid dimensions.
        float_flag (bool, optional): If True, the indices are returned as floating-point numbers. If False,
                                they are floored and returned as integers. Defaults to False.
        plus_1_flag (bool, optional): If True, 1 is added to the number of columns. Defaults to False.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - col (np.ndarray): An array of column indices.
            - lin (np.ndarray): An array of line indices.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on NumPy array operations.
    """
    outDim = [int(d * precision) for d in dim]
    cdim = dim[0] / float(outDim[0])
    ldim = dim[1] / float(outDim[1])

    if plus_1_flag:
        outDim[0] += 1

    col = np.arange(outDim[0]) * cdim
    lin = np.arange(outDim[1]) * ldim

    if not float_flag:
        col = np.floor(col).astype(int)
        lin = np.floor(lin).astype(int)

    return col, lin


def set_regular_xy_array(datax: np.ndarray, datay: np.ndarray) -> np.ndarray:
    """
    Creates a regular grid array from separate x and y coordinate arrays.

    This function constructs a 2D grid array ('data_xy') where each element is a pair of x and y coordinates.
    The x coordinates ('datax') are repeated for each y coordinate ('datay'). The resulting array has a shape
    that combines the lengths of 'datax' and 'datay'.

    Args:
        datax (np.ndarray): An array of x-coordinates.
        datay (np.ndarray): An array of y-coordinates.

    Returns:
        np.ndarray: A 2D NumPy array where each element is a pair [x, y] of coordinates from 'datax' and 'datay'.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on NumPy's array operations.
    """

    col = datax
    lin = np.repeat(1, len(datax))

    data_xy = np.transpose([col, lin])

    return data_xy


def findSharedMap(projection: str = "", p0lat=0, p0lon=0):
    """
    Args:
        attr: istanza della classe attr_define.Attr
        projection:

    Returns:
        pMap: istanza della classe map__define.Map (?)

    """

    if dpg.globalVar.GlobalState.SHARED_MAPS:
        for map in dpg.globalVar.GlobalState.SHARED_MAPS:
            if isinstance(map, dpg.map__define.Map):
                if map.mapProj is not None:
                    proj = map.mapProj
                    if get_idl_proj_from_python(proj.name).upper() == projection.upper():
                        lon_0, lat_0 = get_lon0_and_lat0(proj)

                        if lat_0 == p0lat and lon_0 == p0lon:
                            # log_message(f"Found shared map {elem.up_name.upper()}: p0lat = {p0lat} and p0lon = {
                            # p0lon}", level='INFO')
                            return map

    return None


def get_lon0_and_lat0(proj_elem, double=False):
    """
    Extracts the central longitude (lon_0) and latitude (lat_0) from a projection element's SRS string.

    Args:
        proj_elem: An object containing the SRS (spatial reference system) string.

    Returns:
        lon_0 (float32): The central longitude value.
        lat_0 (float32): The central latitude value.
    """
    if isinstance(proj_elem, str):
        return None, None

    params = proj_elem.srs.split()
    for param in params:
        if param.startswith("+lat_0"):
            lat_0 = float(param.split("=")[1])
        if param.startswith("+lon_0"):
            lon_0 = float(param.split("=")[1])
    if double:
        return lon_0, lat_0
    else:
        return np.float32(lon_0), np.float32(lat_0)


def get_coord_index(coord: np.ndarray, coordVect: np.ndarray, tolerance=None, angle: bool = False):
    """
    Finds the index of the closest coordinate in a vector to each coordinate in an array, with an optional tolerance.

    This function identifies the index of the coordinate in 'coordVect' that is closest to each 'coord' element.
    If 'angle' is True, the function handles coordinates as angles in degrees, adjusting differences to account
    for the circular nature of angles. An optional 'tolerance' parameter defines the maximum allowed difference
    for a match; if the closest coordinate is outside this tolerance, -1 is returned.

    Args:
        coord (np.ndarray): The coordinate (or angle) to match.
        coordVect (np.ndarray): An array of coordinates (or angles) to search through.
        tolerance (float or int, optional): The maximum allowed difference between 'coord' and its closest match
                                             in 'coordVect'. Defaults to None, which ignores tolerance.
        angle (bool, optional): If True, coordinates are treated as angles in degrees. Defaults to False.

    Returns:
        np.ndarray: The index array of the closest coordinate in 'coordVect' to each 'coord' element.
            Returns -1 if no suitable match is found within 'tolerance'.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on NumPy's array operations.
    """

    diff = coordVect - coord[:, None]

    if angle:
        ind = np.where(diff < -180.0)
        diff[ind] += 360.0

        ind = np.where(diff > 180.0)
        diff[ind] -= 360.0

        if tolerance is None:
            tolerance = 2

    diff = np.abs(diff)
    index = np.nanargmin(diff, axis=1)
    minima = np.nanmin(diff, axis=1)

    if tolerance is None:
        return index

    index[minima >= tolerance] = -1

    return index


# TODO Controllare se mai usata
def isValidPar(par) -> bool:
    """
    Checks if the given parameter array is valid based on specific criteria.

    This function validates a parameter array ('par') by checking its length and the finiteness of its first four
    elements. It ensures that 'par' has at least four elements, all four are finite numbers, and neither the second
    nor the fourth element is zero.

    Args:
        par (list or np.ndarray): The parameter array to be validated.

    Returns:
        bool: Returns True if 'par' meets the validation criteria, False otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if len(par) < 4:
        return False
    if np.sum(np.isfinite(par[:4])) != 4:
        return False
    if par[1] == 0.0 or par[3] == 0.0:
        return False
    return True


def setStationaryCoordConv(
        box: list,
        user_offset: list,
        scale: list = None,
        log: bool = None,
        dimensions: list = None,
        location: list = None,
) -> int:
    """
    Sets up a stationary coordinate conversion based on the given parameters.

    This function configures a stationary coordinate conversion system using a bounding box, user offsets,
    and optional scaling. It initializes 'location' and 'dimensions' for this conversion. The function checks
    for valid lengths of 'user_offset' and 'box', and sets defaults if necessary. It calculates a range based
    on the bounding box and applies the user offset and scaling to determine 'location' and 'dimensions'.

    Args:
        box (list[float]): The bounding box as [min_x, min_y, max_x, max_y].
        user_offset (list[float]): The user-specified offsets in x and y directions.
        scale (list[float], optional): Scaling factors for x and y dimensions. Defaults to None.
        log (bool, optional): If True, applies a logarithmic scale to the range. Defaults to None.
        dimensions (list[float], optional): The dimensions to be calculated. Defaults to None.
        location (list[float], optional): The location to be calculated. Defaults to None.

    Returns:
        int: Returns 1 if the setup is successful, 0 otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if len(user_offset) < 2:
        return 0
    if len(box) < 4:
        box = [0.0, 0.0, 100000.0, 100000.0]
    location = [0.0, 0.0]

    if dimensions is not None and len(dimensions) == 2:
        rapp = dimensions[1] / float(dimensions[0])
    else:
        rapp = None

    dimensions = [0.0, 0.0]

    range1 = box[2] - box[0]
    range2 = box[3] - box[1]
    _range = range if range1 > range2 else range2

    if log:
        _range = np.sqrt(_range * 100000.0)

    offset = user_offset[0]
    offset *= range1
    location[0] = box[0] + offset
    sc = 0.1
    if scale and len(scale) > 0:
        sc = scale[0]

    dimensions[0] = _range * sc

    offset = user_offset[1]
    offset *= range2
    location[1] = box[1] + offset
    if scale and len(scale) > 1:
        sc = scale[1]
    if rapp is not None:
        sc *= rapp

    dimensions[1] = _range * sc

    return 1


def getWorldOffset(map: Map) -> np.ndarray:
    """
    Retrieves the world offset for a given map object.

    This function calculates the world offset based on the properties of the provided map object. It extracts
    information such as the map name, projection type, and reference latitude/longitude. Depending on the
    projection type and reference coordinates, it returns an appropriate offset as a NumPy array.

    Args:
        map (Map object): The map object from which to extract the world offset.

    Returns:
        np.ndarray: A NumPy array representing the world offset. The array format is [longitude, latitude,
        additional parameter].

    Raises:
        None: This function does not explicitly raise any exceptions but depends on the 'getMapName' function to
        retrieve map information.
    """
    name, proj, p0lat, p0lon, box = getMapName(map, box=True)

    if name == "":
        return np.array([0.0, 0.0, 0.0])
    if proj == -3:
        return np.array([p0lon, p0lat, 0.0])
    if proj != -2:
        return np.array([0.0, 0.0, 0.0])
    if p0lon == 0.0 and p0lat == 0.0:
        return np.array([0.0, 0.0, 0.0])

    sph_coord = [p0lon, p0lat, box[3]]
    offset = None
    # TODO offset = float(CV_COORD(FROM_SPHERE=sph_coord, /TO_RECT, /DEGREES))

    return offset


def geostaz_inverse(y: np.ndarray, x: np.ndarray, map: Map = None):
    """
    Performs the inverse geostationary transformation of coordinates.

    This function converts x and y coordinates from a geostationary projection to geographic coordinates (latitude
    and longitude). It accounts for the Earth's oblateness by using different radii for the equator and poles. The
    transformation takes into account the satellite height and the reference longitude of the projection, which can
    be provided through a map object.

    Args:
        y (np.ndarray): The y-coordinates (in degrees) in the geostationary projection.
        x (np.ndarray): The x-coordinates (in degrees) in the geostationary projection.
        map (Map object, optional): The map object containing the reference longitude. Defaults to None.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - The latitudes in degrees.
            - The longitudes in degrees.

    Raises:
        None: This function does not explicitly raise any exceptions but involves complex mathematical operations
        that could lead to domain errors.
    """
    # TODO rivedere dove ci sono i where, che nel codice IDL ci sarebbero degli IF
    h = 42164.000
    re = 6378.169
    rp = 6356.5838
    conv = np.deg2rad(-36.0 / 5000000.0)
    cosx = np.cos(np.deg2rad(x))  # sarebbe x*conv, è la stessa cosa?
    cosy = np.cos(np.deg2rad(y))  # sarebbe x*conv, è la stessa cosa?
    sinx = np.sin(np.deg2rad(x))  # sarebbe x*conv, è la stessa cosa?
    siny = np.sin(np.deg2rad(y))  # sarebbe x*conv, è la stessa cosa?
    rap = (re * re) / (rp * rp)

    A = h * cosx * cosy
    B = cosy * cosy + rap * siny * siny
    C = A * A - B * (h * h - re * re)

    ind = np.where(np.isfinite(C))
    C[ind] = -1.0
    ind = np.where(C < 0.0)
    C[ind] = 0.0
    nv = np.where(C < 0.0)

    sd = np.sqrt(C)
    sn = (A - sd) / B
    s1 = h - sn * cosx * cosy
    s2 = -sn * sinx * cosy
    s3 = sn * siny
    cosx = 0
    cosy = 0
    sinx = 0
    siny = 0
    sxy = np.sqrt(s1 * s1 + s2 * s2)
    lon = np.rad2deg(np.arctan2(s2, s1))
    lat = np.rad2deg(np.arctan2(rap * s3, sxy))

    ind = np.where((lon < -90.0) | (lon > 90.0))
    lon[ind] = np.nan
    lat[ind] = np.nan
    ind = np.where((lat < -90.0) | (lat > 90.0))
    lon[ind] = np.nan
    lat[ind] = np.nan

    _, _, _, p0lon, _ = getMapName(map)
    if p0lon != 0.0:
        lon += p0lon

    return lat, lon


def geostaz_forward(lat: np.ndarray, lon: np.ndarray, proj: Proj = None):
    """
    Performs a forward geostationary transformation of geographic coordinates.

    This function converts geographic coordinates (latitude and longitude) to x and y coordinates in a
    geostationary projection. It accounts for the Earth's oblateness and the reference longitude of the
    geostationary projection, which can be provided through a proj object. The transformation is suitable
    for visualizing data in a geostationary projection.

    Args:
        lat (np.ndarray): The latitudes in degrees.
        lon (np.ndarray): The longitudes in degrees.
        proj (Proj object, optional): The proj object containing the reference longitude. Defaults to None.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - The y-coordinates in the geostationary projection.
            - The x-coordinates in the geostationary projection.

    Raises:
        None: This function does not explicitly raise any exceptions but involves complex mathematical operations
        that could lead to domain errors.
    """
    # TODO rivedere dove ci sono i where, che nel codice IDL ci sarebbero degli IF
    h = 42164.000
    re = 6378.169
    rp = 6356.584

    dlon = np.asarray(lon, dtype='float32')
    lat = np.asarray(lat, dtype='float32')
    p0lon = proj.crs.to_dict()['lon_0'] if proj is not None else 0.
    if p0lon != 0.0:
        dlon -= p0lon
        ind = np.where(dlon < -180.0)
        if np.size(ind) > 0:
            dlon[ind] += 360.0

    rap = (rp * rp) / (re * re)
    clat = np.arctan(rap * np.tan(np.deg2rad(lat)))
    cosclat = np.cos(clat)
    sinclat = np.sin(clat)

    ind = np.where((dlon < -76.0) | ((dlon > 76.0) & (dlon < 270.0)))  # ex 90
    if np.size(ind) > 0:
        nv = ind
    ind = np.where((lat < -76.0) | (lat > 76.0))  # ex 90
    if np.size(ind) > 0:
        if "nv" in locals():
            nv = np.concatenate((nv, ind))
        else:
            nv = ind

    dlon *= np.deg2rad(1.0)

    A = 1.0 - (1.0 - rap) * cosclat * cosclat
    rl = rp / np.sqrt(A)
    r1 = h - rl * cosclat * np.cos(dlon)
    r2 = -rl * cosclat * np.sin(dlon)
    r3 = rl * sinclat

    B = r1 * r1 + r2 * r2 + r3 * r3
    rn = np.sqrt(B)

    x = np.arctan2(r2, r1)
    y = np.arcsin(r3 / rn)

    conv = np.rad2deg(-5000000.0 / 36.0).astype('float32')
    x *= conv
    y *= conv
    if "nv" in locals():  # poco ottimizzato
        x[nv] = np.nan
        y[nv] = np.nan

    return x, y


def world_inverse(
        y: np.ndarray,
        x: np.ndarray,
        z: np.ndarray = None,
        Map: Map = None,
        vert_exageration: float = None,
):
    """
    Performs the inverse world coordinate transformation.

    This function transforms x, y, and optionally z coordinates from a world coordinate system back to
    spherical coordinates (latitude, longitude, altitude). It applies a world offset obtained from the
    provided map object and handles vertical exaggeration if specified.

    Args:
        y (np.ndarray): The y-coordinates in the world coordinate system.
        x (np.ndarray): The x-coordinates in the world coordinate system.
        z (np.ndarray, optional): The z-coordinates (altitude) in the world coordinate system. Defaults to None.
        Map (Map object, optional): The map object from which to obtain the world offset. Defaults to None.
        vert_exageration (float, optional): The factor for vertical exaggeration. Defaults to None.

    Returns:
        tuple: A tuple containing three NumPy arrays:
            - The latitudes in degrees.
            - The longitudes in degrees.
            - The altitudes, if z-coordinates are provided.

    Raises:
        None: This function does not explicitly raise any exceptions but depends on the 'getWorldOffset' function.
    """
    offset = getWorldOffset(Map)
    rect_coord = np.zeros((3, len(y)))
    rect_coord[0, :] = offset[0]
    rect_coord[2, :] = y + offset[2]
    rect_coord[1, :] = x + offset[1]
    if z is not None:
        ex = vert_exageration if vert_exageration is not None else 1.0
        if ex != 1.0 and ex != 0.0:
            rect_coord[0, :] += z / ex
        else:
            rect_coord[0, :] += z

    sph_coord = (
        None  # TODO sph_coord = CV_COORD(FROM_RECT=rect_coord, /TO_SPHERE, /DEGREES)
    )
    # TODO manca parte
    lon = sph_coord[0]
    lat = sph_coord[1]
    alt = sph_coord[2]

    return lat, lon, alt


def world_forward(
        lat: np.ndarray,
        lon: np.ndarray,
        ALT: np.ndarray = None,
        Map: Map = None,
        vert_exageration: float = None,
):
    """
    Performs the forward world coordinate transformation.

    This function transforms spherical coordinates (latitude, longitude, altitude) to a world coordinate
    system. It takes into account any vertical exaggeration specified and applies a world offset from the
    provided map object. The function is used to convert geographic coordinates to a format suitable for
    certain types of map projections or visualization purposes.

    Args:
        lat (np.ndarray): The latitudes in degrees.
        lon (np.ndarray): The longitudes in degrees.
        ALT (np.ndarray, optional): The altitudes. Defaults to None.
        Map (Map object, optional): The map object from which to obtain the world offset. Defaults to None.
        vert_exageration (float, optional): The factor for vertical exaggeration. Defaults to None.

    Returns:
        tuple: A tuple containing three NumPy arrays:
            - The y-coordinates in the world coordinate system.
            - The x-coordinates in the world coordinate system.
            - The z-coordinates (elevation) in the world coordinate system.

    Raises:
        None: This function does not explicitly raise any exceptions but depends on the 'getWorldOffset' function.
    """
    sph_coord = np.zeros((3, len(lat)))
    sph_coord[0, :] = lon
    sph_coord[1, :] = lat
    sph_coord[2, :] = Map.uv_box[3] if Map is not None else 0.0
    if ALT is not None:
        ex = vert_exageration if vert_exageration is not None else 1.0
        if ex != 1.0 and ex != 0.0:
            sph_coord[2, :] += ALT * ex
        else:
            sph_coord[2, :] += ALT

    rect_coord = (
        None  # TODO rect_coord = CV_COORD(FROM_SPHERE=sph_coord, /TO_RECT, /DEGREES)
    )
    offset = getWorldOffset(Map)
    z = rect_coord[0, :] - offset[0]
    x = rect_coord[1, :] - offset[1]
    y = rect_coord[2, :] - offset[2]

    return y, x, z


def get_par_from_box(
        box: list = None,
        dim: list = None,
        isotropic: bool = False,
        outer: bool = False,
        minus1: bool = False,
) -> list:
    """
    Calculates parameters for a grid based on a specified bounding box and dimensions.

    This function computes the offsets and resolutions for a grid based on the given bounding box ('box') and
    grid dimensions ('dim'). It supports isotropic scaling, where the resolutions in both dimensions are made
    equal, and can adjust calculations based on whether the box represents the outer or inner limits of the grid.
    The 'minus1' flag alters the calculation to accommodate grid definitions where the number of intervals, rather
    than the number of grid points, is specified.

    Args:
        box (list[float]): The bounding box as [X1, Y1, X2, Y2].
        dim (list[int]): The dimensions of the grid as [height, width].
        isotropic (bool, optional): If True, ensures isotropic scaling. Defaults to False.
        outer (bool, optional): If True, considers the bounding box as the outer limits of the grid. Defaults to False.
        minus1 (bool, optional): If True, adjusts the resolution for a grid defined by intervals instead of points.
        Defaults to False.

    Returns:
        list: A list of parameters as [xoffset, xresolution, yoffset, yresolution].

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    X1, Y1, X2, Y2 = box

    if dim[0] == 0 or dim[1] == 0:
        if np.isfinite(X1):
            return [X1, 0.0, Y1, 0.0]
        return [X2, 0.0, Y2, 0.0]

    if np.sum(np.isfinite(box)) != 4:
        return [np.nan, 0.0, np.nan, 0.0]

    if minus1:
        if dim[0] > 1:
            yres = (Y2 - Y1) / float(dim[0] - 1)
        else:
            yres = (Y2 - Y1) / float(dim[0])

        if dim[1] > 1:
            xres = (X2 - X1) / float(dim[1] - 1)
        else:
            xres = (X2 - X1) / float(dim[1])
    else:
        yres = (Y2 - Y1) / float(dim[0])
        xres = (X2 - X1) / float(dim[1])

    if isotropic:
        ysign = -1.0 if yres < 0 else 1.0
        xsign = -1.0 if xres < 0 else 1.0

        if not outer:
            xres = min([abs(xres), abs(yres)])
        else:
            xres = max([abs(xres), abs(yres)])

        yres = xres * ysign
        xres *= xsign

        Y1 = (Y1 + Y2 - yres * dim[0]) / 2.0
        X1 = (X1 + X2 - xres * dim[1]) / 2.0

    yoff = int(round(-Y1 / yres)) if yres != 0 else np.nan
    xoff = int(round(-X1 / xres)) if xres != 0 else np.nan

    return [xoff, xres, yoff, yres]


def get_inner_box(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Calculates the inner bounding box from two given bounding boxes.

    This function computes the intersection of two bounding boxes, resulting in a box that represents the
    overlapping region. If the second box has fewer than four elements, the function returns a copy of the first box.

    Args:
        box1 (np.ndarray): The first bounding box as [X1, Y1, X2, Y2].
        box2 (np.ndarray): The second bounding box as [X1, Y1, X2, Y2].

    Returns:
        np.ndarray: The inner bounding box representing the overlapping region of 'box1' and 'box2'.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    box = np.copy(box1)
    if len(box2) < 4:
        return box
    box[0] = max(box1[0], box2[0])
    box[1] = max(box1[1], box2[1])
    box[2] = min(box1[2], box2[2])
    box[3] = min(box1[3], box2[3])

    return box


def get_outer_box(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Calculates the outer bounding box encompassing two given bounding boxes.

    This function computes a box that covers both 'box1' and 'box2', essentially creating the smallest box
    that contains both of the input boxes. If the second box has fewer than four elements, the function
    returns a copy of the first box.

    Args:
        box1 (np.ndarray): The first bounding box as [X1, Y1, X2, Y2].
        box2 (np.ndarray): The second bounding box as [X1, Y1, X2, Y2].

    Returns:
        np.ndarray: The outer bounding box encompassing both 'box1' and 'box2'.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    box = np.copy(box1)
    if len(box2) < 4:
        return box
    box[0] = min(box1[0], box2[0])
    box[1] = min(box1[1], box2[1])
    box[2] = min(box1[2], box2[2])
    box[3] = min(box1[3], box2[3])

    return box


def get_box_from_vertex(vertex: np.ndarray, Xe: float, Yn: float) -> np.ndarray:
    """
    Creates a bounding box from vertex coordinates and specified eastern and northern bounds.

    This function constructs a bounding box using the vertex coordinates and the specified easternmost
    (Xe) and northernmost (Yn) bounds. It finds the westernmost (Xw) and southernmost (Ys) bounds from
    the vertex coordinates to complete the box.

    Args:
        vertex (np.ndarray): An array of vertex coordinates, typically as [X, Y].
        Xe (float): The easternmost bound.
        Yn (float): The northernmost bound.

    Returns:
        np.ndarray: The bounding box as [Xw, Ys, Xe, Yn].

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    Xw = np.minimum(vertex[0, :], Xe)
    Ys = np.minimum(vertex[1, :], Yn)
    return np.array([Xw, Ys, Xe, Yn])


def get_cardinal_points(vertex: np.ndarray) -> np.ndarray:
    """
    Calculates the cardinal points (North, East, South, West) from vertex coordinates.

    This function determines the cardinal points using a given set of vertex coordinates. It computes
    the North, East, South, and West points based on the vertex array, which should contain the coordinates
    of these points. The function assumes that the appropriate indices for each cardinal direction are
    already determined.

    Args:
        vertex (np.ndarray): An array of vertex coordinates.

    Returns:
        np.ndarray: An array containing the coordinates of the cardinal points in the order [North, East, South, West].

    Raises:
        None: This function does not explicitly raise any exceptions but relies on correct indexing within the
        'vertex' array.
    """
    westInd, Xe, southInd, Yn, northInd, eastInd = (
        0  # TODO capire da dove vengono questi valori
    )
    Xw = np.minimum(vertex[0, :], westInd)
    Xw = np.minimum(Xw, Xe)

    Ys = np.minimum(vertex[1, :], southInd)
    Ys = np.minimum(Ys, Yn)

    N = np.array([vertex[0, northInd], Yn])
    E = np.array([Xe, vertex[1, eastInd]])
    S = np.array([vertex[0, southInd], Ys])
    W = np.array([Xw, vertex[1, westInd]])

    return np.array([N, E, S, W])


def get_alt_range(par: list, dim: list) -> np.ndarray:
    """
    Calculates the altitude range from parameters and dimensions.

    This function computes the range of altitudes based on parameters and dimensions provided. It uses the
    altitude offset ('hoff') and resolution ('hres') specified in 'par'. The function also accounts for an
    optional parameter that can override the upper limit of the altitude range.

    Args:
        par (list[float]): Parameters including altitude offset and resolution.
        dim (list[int]): Dimensions that determine the range of the altitude calculation.

    Returns:
        np.ndarray: A NumPy array representing the altitude range [minimum altitude, maximum altitude].

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the length and values of 'par' and
        'dim'.
    """
    hoff = 0.0
    hres = 0.0
    if len(par) > 4:
        hoff = par[4]
    if len(par) > 5:
        hres = par[5]
    alt_range = [hoff, hoff]

    if len(par) > 10:
        if par[10] > 0:
            alt_range[1] = 10000.0
        return np.array(alt_range)

    if len(dim) > 1:
        alt_range[1] = hoff + hres * dim[1]
    if len(dim) > 2:
        alt_range[1] = hoff + hres * dim[2]

    return np.array(alt_range)


def map_proj_init(attr: dict, proj_name='') -> pyproj.Proj:
    """
    Initializes a map projection using specified projection parameters.

    This function sets up a map projection based on given parameters. It converts projection parameters from IDL
    (Interactive Data Language) to a corresponding format in Python using 'get_python_proj_from_idl' from 'dpg.io'.
    The function handles special cases, such as adjusting parameters for certain projection types, before initializing
    the projection using 'pyproj.Proj'.

    Args:
        projection_params (dict): A dictionary of projection parameters. Must include a 'proj' key specifying
                                  the name of the projection, along with other relevant parameters.

    Returns:
        pyproj.Proj: The initialized map projection object.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the 'pyproj.Proj' function
              and 'get_python_proj_from_idl' method.
    """

    proj_name, _, _ = dpg.attr.getAttrValue(attr, "projection", proj_name)
    orig_lat, _, _ = dpg.attr.getAttrValue(attr, "orig_lat", 0.0)
    orig_lon, _, _ = dpg.attr.getAttrValue(attr, "orig_lon", 0.0)
    p0lat, _, _ = dpg.attr.getAttrValue(attr, "prj_lat", 0.0)
    p0lon, exists, _ = dpg.attr.getAttrValue(attr, "prj_lon", 0.0)
    if exists <= 0:
        p0lat = orig_lat
        p0lon = orig_lon

    if proj_name == "":
        proj_name = dpg.cfg.getDefaultProjection()

    py_projection = dpg.io.get_python_proj_from_idl(proj_name)

    projection_params = {"proj": py_projection, "lat_0": p0lat, "lon_0": p0lon}

    if py_projection == "geos":
        # TODO: da gestire con mimmo per vedere caso specifico
        # proiezione satellite
        # provare il codice commentando queste variabili
        projection_params["h"] = 42164000.
        projection_params["a"] = 6378169.
        projection_params["b"] = 6356583.8

    if py_projection == "utm":
        projection_params["zone"], _, _ = dpg.attr.getAttrValue(attr, "Zone", 32)
    # Initialize the map projection
    map_proj = pyproj.Proj(projection_params)
    # map_proj = pyproj.Proj(f"+proj={py_projection} +lat_0="+str(p0lat)+" +lon_0="+str(p0lon)+" +datum=WGS84")

    return map_proj


def latlon_2_yx(lat, lon, map: dpg.map__define.Map):
    """
    Converts latitude and longitude coordinates to map projection coordinates.

    This function transforms geographic coordinates (latitude and longitude) into the corresponding x and y
    coordinates in the specified map projection. The transformation is performed using the provided map projection
    object.

    Args:
        lat (np.ndarray or float): The latitude(s) in degrees.
        lon (np.ndarray or float): The longitude(s) in degrees.
        map (dpg.map__define.Map): The map object containing the pyproj projection used for the conversion.

    Returns:
        tuple: A tuple containing two elements:
            - y (np.ndarray or float): The y-coordinate(s) in the map projection.
            - x (np.ndarray or float): The x-coordinate(s) in the map projection.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the 'pyproj.Proj' function.
    """
    if map is None:
        log_message("Invalid map", level='WARNING')
        return lat, lon

    proj = map.mapProj
    if proj.name == 'latlong' or proj.name == 'latlon':
        return lat, lon
    if proj.name == 'geos':
        x, y = geostaz_forward(lat, lon, proj)
    else:
        x, y = proj(longitude=lon, latitude=lat, inverse=False)

    return y, x


def get_transformer(map_proj_input: str | Proj,
                    map_proj_output: str | Proj):
    """
    Retrieve or create a Transformer for the specified projection parameters.
    """
    lon0, lat0 = get_lon0_and_lat0(map_proj_input)

    lon1, lat1 = get_lon0_and_lat0(map_proj_output)

    if isinstance(map_proj_output, pyproj.Proj):
        map_proj_output = map_proj_output.name
    if isinstance(map_proj_input, pyproj.Proj):
        map_proj_input = map_proj_input.name

    # Create a unique key for this transformation
    key = (map_proj_input, map_proj_output, lat0, lon0)

    if key not in transformer_cache:
        # transformer = get_transformer(map_proj.name, lat_0, lon_0)
        map_proj2 = pyproj.Proj(proj=map_proj_output, lat_0=lat1, lon_0=lon1)

        map_proj1 = Proj(proj=f"{map_proj_input}", lat_0=lat0, lon_0=lon0)

        # Create the transformer (assuming lat/lon as input and output)
        transformer_cache[key] = Transformer.from_proj(map_proj1, map_proj2, always_xy=False)

    return transformer_cache[key]


def yx_2_latlon(y, x, map: dpg.map__define.Map):
    """
    Converts map projection coordinates to latitude and longitude.

    This function transforms x and y coordinates in a specified map projection back into geographic coordinates
    (latitude and longitude). The transformation is performed using the provided map projection object.

    Args:
        y (np.ndarray or float): The y-coordinate(s) in the map projection.
        x (np.ndarray or float): The x-coordinate(s) in the map projection.
        map_proj (pyproj.Proj): The map projection object used for the conversion.

    Returns:
        tuple: A tuple containing two elements:
            - lat (np.ndarray or float): The latitude(s) in degrees.
            - lon (np.ndarray or float): The longitude(s) in degrees.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the 'pyproj.Proj' function.
    """
    if map is None:
        log_message("Invalid map", level='WARNING')
        return y, x

    map_proj = map.mapProj

    lon, lat = map_proj(longitude=x, latitude=y, inverse=True)

    # Prova per provare a velocizzare la conversione, al momento non sembra migliorare le perfomance
    # transformer = get_transformer(map_proj, 'latlong')
    # lon, lat = transformer.transform(x, y, direction="FORWARD")

    return lat, lon


# def read_projection_mapping_file(file_path):
#     """
#     Read the projection mapping from a text file and return a dictionary.
#     Each line in the file should be in the format: IDL Projection Name: pyproj Projection Name
#     """
#     projection_mapping = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 idl_projection, pyproj_projection = line.split(':')
#                 projection_mapping[idl_projection.strip()] = pyproj_projection.strip()
#     return projection_mapping


# def idl_proj_2_py(idl_projection, mapping_file):
#     """
#     Convert an IDL projection to its pyproj equivalent using the provided mapping file.
#     Returns the pyproj equivalent or 'NOT PRESENT' if not found.
#     """
#     projection_mapping = read_projection_mapping_file(mapping_file)
#     return projection_mapping.get(idl_projection, 'NOT PRESENT')


# def py_proj_2_idl(pyproj_projection, mapping_file):
#     """
#     Convert a pyproj projection to its IDL equivalent using the provided mapping file.
#     Returns the IDL equivalent or 'NOT PRESENT' if not found.
#     """
#     projection_mapping = read_projection_mapping_file(mapping_file)
#     # Reverse the dictionary for py_proj_2_idl
#     reverse_mapping = {v: k for k, v in projection_mapping.items()}
#     return reverse_mapping.get(pyproj_projection, 'NOT PRESENT')


def lincol_2_radyx(
        lin,
        col,
        par: dict,
        az_coords=np.array(()),
        el_coords: np.ndarray = None,
        set_az: bool = False,
        set_center: bool = False,
        lev=np.array(()),
        set_z: bool = False,
        radz=np.array(()),
):
    """
    Converts linear and column coordinates to radar coordinates in the azimuth and range dimensions.

    This function transforms linear and column indices (lin, col) into radar coordinates (rady, radx)
    based on specified parameters (par) and optional azimuth and elevation coordinates. It calculates
    the azimuth and range for each point and converts these into x and y coordinates in a radar-based
    coordinate system.

    Args:
        lin (np.ndarray or int): Linear indices or a single linear index.
        col (np.ndarray or int): Column indices or a single column index.
        par (dict): Parameters containing offsets and resolutions for the transformation.
        az_coords (np.ndarray, optional): Azimuth coordinates corresponding to the linear indices. Defaults to None.
        el_coords (np.ndarray, optional): Elevation coordinates. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - radx (np.ndarray or float): The x-coordinates in the radar coordinate system.
            - rady (np.ndarray or float): The y-coordinates in the radar coordinate system.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the provided 'par' dictionary and
        'az_coords'.
    """
    azres = 0
    polres = 0
    azoff = 0
    poloff = 0
    if az_coords is None:
        az_coords = np.array(())

    if len(par) >= 10:
        poloff = par[6]
        polres = par[7]
        azoff = par[8]
        azres = par[9]

    if polres == 0:
        radx = col
        rady = lin
        return

    az = azoff

    if len(az_coords) == 0:
        if azres != 0:
            az += lin * azres
    else:
        if len(lin) == 1:
            if 0 <= lin <= len(az_coords):
                az = az_coords[lin]
        else:
            az = az_coords[lin]

    if len(az) == 1 or set_az:
        azimuth = az
        if len(azimuth) == 1:
            if azimuth >= 360:
                azimuth -= 360

    ro = col * polres

    if set_center:
        if len(az_coords) == 0:
            az += azres / 2.0
        ro += polres / 2.0

    # Perchè gli angoli in polare partono da 0 a nord e vanno in senso orario
    # al contrario della trigonometria normale
    az = 450 - az  # rende antiorario

    # if n_elements(ro) eq 1 or keyword_set(set_range) eq 1 then range=ro

    if poloff > 0:
        ro += poloff

    az *= np.pi / 180

    # # Invertiamo così evitiamo istruzione precedente in cui togliamo 450
    # cosaz = np.sin(az)
    # sinaz = np.cos(az)

    cosaz = np.cos(az)
    sinaz = np.sin(az)

    radx = ro * cosaz
    rady = ro * sinaz

    if azres == 0 and len(lev) == 0:
        lev = lin
    if not set_z:
        return radx, rady
    if radz is not None:
        return radx, rady

    dpg.map.get_altitudes()

    return radx, rady


def lincol_2_yx(
        lin: np.ndarray,
        col: np.ndarray,
        params: list,
        dim=None,
        az_coords: tuple = (),
        el_coords=None,
        set_center: bool = False,
):
    """

    Args:
        lin:
        col:
        params: parametri del file NAVIGATION.txt
        az_coords: contenuto del file AZIMUTH.txt
        el_coords: contenuto del file ELEVATION.txt
        dim: parametro opzionale che serve solo nel caso della mosaicatura.

    Returns:

    """
    if dim is None:
        dim = []

    if len(params) >= 10:
        # Caso dati polari

        # if set_az:
        """
        TODO: qua manca una parte complicata del set_az, dovuta al fatto che gli azimuth
        non sempre sono ordinati, ma non partono necessariamente da 0 e non sempre hanno
        un passo regolare. Serve una lista chiamata az_coords che è un vettore la cui lunghezza
        coincide col numero di linee. Per ogni linea abbiamo l'angolo di azimuth corrispondente.
        Il pass in azimuth può avere delle oscillazioni, quindi serve sapere ogni singolo fascio
        dove sta puntando, che sta dentro az_coords. (nel caso di dati grezzi).
        Questa complicazione viene risolta nel momento in cui abbiamo i dati polari campionati.
        Viene riportato tutto l'array in una matriche che va a passi di 1° da 0 e 360, e si fa
        anche un campionamento in range, di solito di 1km. Quella grezza è del tutto variabile in
        base al radar. La risoluzione è variabile pure in elevazione.
        I vari array che corrispondo al ppi (giro a 360°) hanno una dimensione variabile, non sono
        tutte uguali. Quindi sono salvate in file diversi, l'albero dei dati grezzi è molto più
        profondo. Ogni singola matrice ha un numero di righe e di colonne diverso e una ris9oluzione
        che è anche essa diversa.
        Più i dati sono in elevazione, meno ci interessano, perchè si potrebbero andare a coprire
        parti che non ci interessano (oltre i 20km si scarta tutto).
        """

        x, y = lincol_2_radyx(
            lin, col, params, az_coords, el_coords, set_center=set_center
        )

    else:
        # Caso dati non polari
        xoff = params[0]
        xres = params[1]
        yoff = params[2]
        yres = params[3]

        if set_center:
            xoff -= 0.5
            yoff -= 0.5
        if xres == 0 or yres == 0:
            x = 0.0
            y = 0.0
            return

        x = (col - xoff) * xres
        y = (lin - yoff) * yres

        # TODO: caso particolare, poi nel caso lo facciamo. Da vedere se spsotare in un'altra fuznione esterna
        # if SET_Z:
        #     get_altitudes()

    if len(dim) != 2:
        return y, x

    # TODO: da controllare, è solo check sulle richieste fuori matrice
    ind = np.where((lin < 0) | lin >= dim[1])[0]

    if len(ind) > 0:
        notValid = ind

    ind = np.where((col < 0) | col >= dim[0])[0]
    if len(ind) > 0:
        if len(notValid) > 0:
            notValid = np.concatenate((notValid, ind))
        else:
            notValid = ind

    if len(notValid) <= 0:
        return

    y[notValid] = np.nan
    x[notValid] = np.nan

    return y, x


def yx_2_lincol(
        y: np.ndarray,
        x: np.ndarray,
        params: list,
        max_az_index: int = None,
        dim: list = None,
        force_limits: bool = None,
        set_center: bool = None,
        z: np.ndarray = None,
        as_is: bool = None,
        az_coords: np.ndarray = None,
):
    """
    Converts y and x coordinates to linear and column indices based on provided parameters.

    This function performs the conversion of y and x coordinates to linear and column indices for both polar
    and non-polar data formats. It uses the parameters provided in 'params' and considers optional arguments
    such as 'max_az_index', 'dim', and 'force_limits'. The function handles different cases based on the length
    of 'params' and the presence of 'dim' and 'max_az_index'.

    Args:
        y (np.ndarray): The y-coordinates.
        x (np.ndarray): The x-coordinates.
        params (list): Parameters for conversion, including offsets and resolutions.
        max_az_index (int, optional): The maximum azimuth index for polar data. Defaults to None.
        dim (list, optional): Dimensions of the data grid. Defaults to None.
        force_limits (bool, optional): If True, forces the indices within certain limits. Defaults to None.
        set_center (bool, optional): If True, adjusts the indices to the center of the grid cells. Defaults to None.
        z (np.ndarray, optional): The z-coordinates for 3D data. Defaults to None.
        as_is (bool, optional): If True, returns indices as they are without adjustment for invalid values. Defaults
        to None.

    Returns:
        tuple: A tuple containing two elements:
            - lin (np.ndarray): The computed linear indices.
            - col (np.ndarray): The computed column indices.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the provided 'params' and optional
        arguments for calculations.
    """

    # notValid_x = np.int64([])
    # notValid_y = np.int64([])
    notValid = None

    if dim is None:
        dim = []
    if max_az_index is None:
        max_az_index = []

    # gridDim viene usato per processare dati con dimensione > 2
    # vengono estratte le ultime due dimensioni contenute in dim
    # quindi gridDim = dim se len(dim) == 2
    gridDim = dim[-2:]

    if len(params) >= 10:
        # Caso dati polari
        if len(gridDim) >= 2 and len(max_az_index) != 1:
            max_az_index = gridDim[0]

        lin, col = radyx_2_lincol(
            y, x, params, max_az_index=max_az_index, set_center=set_center, az_coords=az_coords
        )

    else:
        # Caso dati non polari
        xoff = params[0]
        xres = params[1]
        yoff = params[2]
        yres = params[3]

        if set_center is not None:
            xoff -= 0.5
            yoff -= 0.5

        if yres != 0:
            lin = np.round(y / yres + yoff).astype(int)
        else:
            lin = 0
        if xres != 0:
            col = np.round(x / xres + xoff).astype(int)
        else:
            col = 0

    # if n_elements(notValid) gt 0 then tmp=temporary(notValid)
    if len(dim) < 2:
        return lin, col

    if len(dim) > 2 and z is not None and len(params) > 5:
        hres = params[5]
        if hres != 0:
            lev = round((z - params[4]) / hres)

    ind = np.where(lin < 0)
    notValid = ind

    if len(ind[0]) > 0:
        if force_limits is not None:
            print("TODO: case force_limits is not None")  # TODO

    ind = np.where(lin >= gridDim[0])
    notValid = tuple(
        np.concatenate((notValid[d], ind[d]), dtype=np.int64) for d in range(lin.ndim)
    )

    if len(ind[0]) > 0:
        if force_limits is not None:
            print("TODO: case force_limits is not None")  # TODO

    ind = np.where(col < 0)
    notValid = tuple(
        np.concatenate((notValid[d], ind[d]), dtype=np.int64) for d in range(col.ndim)
    )

    if len(ind[0]) > 0:
        if force_limits is not None:
            print("TODO: case force_limits is not None")  # TODO

    ind = np.where(col >= gridDim[1])
    notValid = tuple(
        np.concatenate((notValid[d], ind[d]), dtype=np.int64) for d in range(col.ndim)
    )

    if len(ind[0]) > 0:
        if force_limits is not None:
            print("TODO: case force_limits is not None")  # TODO
            # col[ind] = dim[0] - 1
            # ind1 = np.where(lin[ind] < 0)
            # if len(ind1[0]) > 0:
            #     lin[ind][ind1] = 0
            # ind1 = np.where(lin[ind] >= dim[1])
            # if len(ind1[0]) > 0:
            #     lin[ind][ind1] = dim[1] - 1

    if as_is is not None or force_limits is not None:
        return lin, col

    lin[notValid] = -1
    col[notValid] = -1

    return lin, col


def get_az_index(
        azimut: np.ndarray,
        azRes: float,
        azOff: float,
        az_coords_in: np.ndarray,
        max_index: int = None,
        set_center: bool = None,
        tolerance: float = None,
) -> np.ndarray:
    """
    Computes azimuth indices from azimuth angles based on resolution, offset, and an optional azimuth coordinate array.

    This function calculates azimuth indices for given azimuth angles ('azimut'). The calculation accounts for azimuth
    resolution ('azRes'), offset ('azOff'), and optionally uses a provided array of azimuth coordinates (
    'az_coords_in').
    The function adjusts angles based on whether they are set to the center of grid cells and whether a tolerance for
    coordinate matching is provided.

    Args:
        azimut (np.ndarray): An array of azimuth angles.
        azRes (float): The azimuth resolution.
        azOff (float): The azimuth offset.
        az_coords_in (np.ndarray): An optional array of azimuth coordinates for index lookup. Defaults to None.
        max_index (int): The maximum allowed index.
        set_center (bool, optional): If True, adjusts the angles to the center of grid cells. Defaults to None.
        tolerance (float, optional): A tolerance for matching coordinates in 'az_coords_in'. Defaults to None.

    Returns:
        np.ndarray: An array of computed azimuth indices.

    Raises:
        None: This function does not explicitly raise any exceptions but involves complex array manipulations and
        conditional checks.
    """
    nAz = len(az_coords_in) if az_coords_in is not None else 0
    az = azimut

    if set_center:
        az -= azRes / 2
    if nAz <= 0:
        az = az - azOff

    az = np.where(az < 0, az + 360, az)
    az = np.where(az >= 360, az - 360, az)

    nC = np.size(az)
    if nAz > 0:
        shape = np.shape(az)
        az = az.flatten()

        if nC == 1:
            return get_coord_index(az, az_coords_in, tolerance=tolerance, angle=True)

        unique_az, inverse = np.unique(az, return_inverse=True)
        index = get_coord_index(unique_az, az_coords_in, tolerance=tolerance, angle=True)
        index = index[inverse]
        index = index.reshape(shape)

        return index

    if azRes is not None and azRes < 0:
        ind = np.where(az > 0)[0]
        count = len(ind)
        if count > 0:
            az[ind] -= 360

    if azRes != 1:
        lin = np.floor(az / azRes).astype(int)
    else:
        lin = np.floor(az).astype(int)

    if max_index is not None:
        nC = max_index
    if nC > 1:
        lin = np.where(lin == nC, nC - 1, lin)
        lin = np.where(lin > nC, -1, lin)

    return lin


def radyx_2_lincol(
        rady: np.ndarray,
        radx: np.ndarray,
        par: list,
        el_coords: np.ndarray = None,
        lev: np.ndarray = None,
        radz: np.ndarray = None,
        az_coords: np.ndarray = None,
        max_az_index: int = None,
        set_center: bool = None,
):
    """
    Converts radar coordinates to linear and column indices for a given set of parameters.

    This function transforms radar coordinates (rady, radx) into linear and column indices based on the provided
    parameters (par). It handles azimuth and polar resolutions, offsets, and optional elevation coordinates
    (el_coords), azimuth coordinates (az_coords), and radar elevation (radz). The function also supports
    an optional maximum azimuth index and an option to set the center of the grid cells.

    Args:
        rady (np.ndarray): The radar y-coordinates.
        radx (np.ndarray): The radar x-coordinates.
        par (list): Parameters including offsets and resolutions for the transformation.
        el_coords (np.ndarray, optional): Elevation coordinates corresponding to linear indices. Defaults to None.
        lev (np.ndarray, optional): Level indices for elevation coordinates. Defaults to None.
        radz (np.ndarray, optional): Radar elevation coordinates. Defaults to None.
        az_coords (np.ndarray, optional): Azimuth coordinates for index lookup. Defaults to None.
        max_az_index (int, optional): The maximum azimuth index for polar data. Defaults to None.
        set_center (bool, optional): If True, adjusts the indices to the center of the grid cells. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - lin (np.ndarray): The computed linear indices.
            - col (np.ndarray): The computed column indices.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the provided parameters and
        optional arguments.
    """

    if el_coords is None:
        el_coords = []
    if az_coords is None:
        az_coords = []

    azres = 0
    polres = 0
    origazres = 0
    elevation = 0

    if len(par) >= 10:
        poloff = par[6]
        polres = par[7]
        azoff = par[8]
        azres = par[9]
        origazres = azres

    if len(par) > 10:
        elevation = par[10]

    if azres == 0 or polres == 0:
        col = radx.astype(int)
        lin = rady.astype(int)
        return lin, col

    az = np.rad2deg(np.arctan2(rady, radx))
    az = 450 - az

    az = np.where(az < 0, az + 360, az)
    az = np.where(az >= 360, az - 360, az)

    ro = np.sqrt(rady * rady + radx * radx)

    if len(el_coords) > 0:
        elevation = el_coords[0]
        if len(lev) == 1:
            if 0 <= lev < len(el_coords):
                elevation = el_coords[lev]

    if elevation > 0:
        cosEL = np.cos(np.pi * elevation / 180.)
        ro /= cosEL

    if poloff > 0:
        ro -= poloff

    lin = get_az_index(
        az,
        azRes=azres,
        azOff=azoff,
        az_coords_in=az_coords,
        max_index=max_az_index,
        set_center=set_center,
    ).astype(int)
    rng_corr = 0
    col = np.floor(ro / polres).astype(int)

    if radz is None:
        return lin, col

    hoff = par[4]
    hres = par[5]

    if hres == 0 or origazres != 0:
        return lin, col

    h = radz - hoff
    lin = np.floor(h / hres)

    return lin, col


def get_dest_box(
        sourceBox: list, sourceMap: dpg.map__define.Map, destMap: dpg.map__define.Map, regular: bool = None
) -> list:
    """
    Calculates the destination bounding box from a source box using source and destination map projections.

    This function transforms a bounding box defined in one map projection (sourceMap) to another map projection
    (destMap). It supports both regular and irregular transformations. A regular transformation assumes the box
    is a simple rectangle, while an irregular transformation takes into account the potential distortion caused by
    the projection change.

    Args:
        sourceBox (list[float]): The bounding box in the source projection as [X1, Y1, X2, Y2].
        sourceMap (dpg.map__define.Map): The source map projection object.
        destMap (dpg.map__define.Map): The destination map projection object.
        regular (bool, optional): If True, performs a regular transformation. Defaults to None for irregular
        transformation.

    Returns:
        list[float]: The bounding box in the destination projection as [X1, Y1, X2, Y2].

    Raises:
        None: This function does not explicitly raise any exceptions but relies on coordinate transformations between
        map projections.
    """
    if regular is not None:
        X = [sourceBox[0], sourceBox[2]]
        Y = [sourceBox[1], sourceBox[3]]
    else:
        X = [sourceBox[0], sourceBox[2], sourceBox[2], sourceBox[0]]
        Y = [sourceBox[1], sourceBox[3], sourceBox[1], sourceBox[3]]

    lat, lon = yx_2_latlon(Y, X, sourceMap)
    Y, X = latlon_2_yx(lat, lon, destMap)

    if regular is not None:
        box = [X[0], Y[1], X[1], Y[0]]
    else:
        minx = np.nanmin(X)
        miny = np.nanmin(Y)
        maxx = np.nanmax(X)
        maxy = np.nanmax(Y)
        box = [minx, miny, maxx, maxy]

    return np.array(box)


def getLLRange(attr: Attr):
    """
    Retrieves latitude and longitude ranges along with a reverse flag from attributes.

    This function attempts to extract geographical coordinate ranges (latitude and longitude) and a
    reverse flag from the provided attribute object. It checks for various attribute keys to find these values.

    Args:
        attr (Attr): An attribute object which contains geographical information.

    Returns:
        tuple of (list of float, list of float, int):
            - latRange: The range of latitude values [minLat, maxLat].
            - lonRange: The range of longitude values [minLon, maxLon].
            - reverse: An indicator flag that may reverse the coordinate order.

    The function checks for three sets of attributes in order of preference:
    'LL_lon', 'LL_lat', 'UR_lon', 'UR_lat', and 'reverse' keys for corners of a bounding box and a reverse flag;
    'minLon', 'maxLon', 'minLat', 'maxLat' keys for minimum and maximum coordinate ranges;
    'firstLon', 'lastLon', 'firstLat', 'lastLat' keys for first and last coordinate ranges.
    If the specific keys are not found, it defaults the range to global coordinates with an optional reverse flag.
    """

    latRange = None
    lonRange = None
    reverse = None

    minLon, exists, _ = dpg.attr.getAttrValue(attr, "LL_lon", 0.0)
    if exists:
        minLat, _, _ = dpg.attr.getAttrValue(attr, "LL_lat", 0.0)
        maxLon, _, _ = dpg.attr.getAttrValue(attr, "UR_lon", 0.0)
        maxLat, _, _ = dpg.attr.getAttrValue(attr, "UR_lat", 0.0)
        reverse, _, _ = dpg.attr.getAttrValue(attr, "reverse", 1)
        latRange = [minLat, maxLat]
        lonRange = [minLon, maxLon]
        return latRange, lonRange, reverse

    minLon, exists, _ = dpg.attr.getAttrValue(attr, "minLon", 0.0)
    if exists:
        maxLon, _, _ = dpg.attr.getAttrValue(attr, "maxLon", 90.0)
        minLat, _, _ = dpg.attr.getAttrValue(attr, "minLat", -90.0)
        maxLat, _, _ = dpg.attr.getAttrValue(attr, "maxLat", 90.0)
        latRange = [minLat, maxLat]
        lonRange = [minLon, maxLon]
        return latRange, lonRange, reverse

    minLon, exists, _ = dpg.attr.getAttrValue(attr, "firstLon", 0.0)
    if exists:
        maxLon, _, _ = dpg.attr.getAttrValue(attr, "lastLon", 90.0)
        minLat, _, _ = dpg.attr.getAttrValue(attr, "firstLat", -90.0)
        maxLat, _, _ = dpg.attr.getAttrValue(attr, "lastLat", 90.0)
        latRange = [minLat, maxLat]
        lonRange = [minLon, maxLon]
        return latRange, lonRange, reverse

    return latRange, lonRange, reverse


def get_proj_par_from_local_res(
        destMap: Map,
        res: list,
        center: list,
        destdim: list,
        corner: bool = False,
        isotropic: bool = False,
) -> list:
    """
    Calculates projection parameters for a map based on local resolution.

    This function computes the parameters needed to project a map, given the local resolution,
    the center coordinates, and the dimensions of the destination map. It calculates the necessary
    transformations for converting geographic coordinates (lat and lon) to map projection
    coordinates (x, y) based on the specified destination map projection.

    Args:
        destMap (Proj): The destination map projection object
        res (list): A list containing the local resolutions [xres, yres] in degrees
        center (list): A list containing the center coordinates [latitude, longitude]
        destdim (list): A list containing the dimensions of the destination map [width, height]
        corner (bool, optional): If True, the function considers the bounding box corners. Defaults to False
        isotropic (bool, optional): If True, the function enforces isotropic scaling. Defaults to False

    Returns:
        list: A list of projection parameters [xoffset, xresolution, yoffset, yresolution].

    Note:
        - The function converts the local resolution from degrees to radians
        - It uses trigonometric functions to calculate the resolutions in meters
        - The function calculates the bounding box of the map based on the center coordinates and resolutions
        - Depending on the 'corner' parameter, it either calculates the bounding box from the center or from the corners
        - The resulting projection parameters can be used to project geographic coordinates onto the destination map
    """
    xres = res[0]
    yres = res[1]

    lat = center[0]
    lon = center[1]

    coslat = float(np.cos(np.radians(lat)))
    re = 6378137.0
    rp = re

    dlat = np.float32(yres / (np.radians(rp)))
    dlon = np.float32(xres / (np.radians(re * coslat)))
    lats = [lat, lat + dlat, lat]
    lons = [lon, lon, lon + dlon]

    y, x = latlon_2_yx(lats, lons, destMap)
    if destdim[0] == 0 or destdim[1] == 0:
        if math.isfinite(x[0]):
            return [x[0], 0.0, y[0], 0.0]
        return [x[1], 0.0, y[1], 0.0]

    cres = x[2] - x[0]
    lres = y[1] - y[0]
    dir = get_proj_direction(destMap)
    xr = dir[0] * cres * destdim[0] / 2
    yr = dir[1] * lres * destdim[1] / 2
    if corner:
        box = [x[0], y[0], x[0] + 2 * xr, y[0] + 2 * yr]
    else:
        box = [x[0] - xr, y[0] - yr, x[0] + xr, y[0] + yr]

    return get_par_from_box(box=box, dim=destdim, isotropic=isotropic)


def check_proj_par(
        destMap: Map,
        destPar: list,
        destDim: list,
        xres: float = None,
        yres: float = None,
        sampling: float = 0,
        sourceMap: Map = None,
        sourceScale: float = None,
        center: list = [],
        earthDiam: list = [],
        latRange: list = [],
        lonRange: list = [],
        isotropic: float = 0,
        corner: float = 0,
):
    """
    The check_proj_par method manages and calibrates the projection parameters and dimensions of a map (destMap)
    according to different parameters and specified conditions.

    Args:
        destMap (Map): The destination map on which the checks are made.
        destPar (list): The projection parameters of the destination map.
        destDim (list): The size of the destination map.

    :keywords:
        - xres (float): Vertical resolution of the map.
        - yres (float): Horizontal resolution of the map.
        - sampling (float): Whether sampling is active.
        - sourceMap (Map): Parameters of the source map.
        - sourceScale (float): Parameters of the scale map.
        - center (list): Center of the map.
        - earthDiam (list): Earth diameter.
        - latRange (list): Latitude range of the map.
        - lonRange (list): Longitude range of the map.
        - isotropic (float): Whether the map is isotropic.
        - corner (float): Corner of the map.

    :return:
        - ispolar (float): A flag indicating if the map is polar.
        - isvertical (float): A flag indicating if the map is vertical
        - destPar (list): I parametri di proiezione aggiornati o eventualmente calcolati per la mappa di destinazione.
        - xres (float): The horizontal resolution of the destination map.
        - yres (float): The vertical resolution of the destination map.
    """
    hPar = None
    isvertical = 0
    ispolar = 0

    if len(destPar) > 9:
        ispolar = 1
        if destPar[9] == 0:
            isvertical = 1

    if ispolar == 0 and destPar[1] != 0 and destPar[3] != 0:
        box, _, _, _ = get_box_from_par(map=destMap, par=destPar, dim=destDim)
        return ispolar, isvertical, destPar, xres, yres, box

    if xres is None:
        xres = 0
    if yres is None:
        yres = 0
    if len(destPar) > 5:
        hPar = destPar[4:]
    full = 0

    if sampling > 0:
        print("DA GESTIRE CASO SAMPLING > 0")
        log_message("DA GESTIRE CASO SAMPLING > 0", level="ERROR", all_logs=True)
        if sourceMap is None:
            sourceMap = map_proj_init(lat_0=0, lon_0=0, proj_name="GEOS")
            if sourceScale != 2:
                sourceScale = get_proj_scale(sourceMap)
            if len(center) > 0:
                cent = center
            destPar = GET_PROJ_PAR_SAMPLING()
    else:
        if xres > 0 and yres > 0:
            log_message(
                "DA GESTIRE CASO xres > 0 and yres > 0", level="ERROR", all_logs=True
            )
            destPar = get_proj_par_from_local_res()
        if len(earthDiam) == 2:
            log_message(
                "DA GESTIRE CASO len(earthDiam) == 2", level="ERROR", all_logs=True
            )
            sys.exit()

        if destPar[1] == 0 and destPar[3] == 0:
            if len(latRange) == 2 and len(lonRange) == 2:
                log_message(
                    "DA GESTIRE CASO len(latRange) == 2 and len(lonRange) == 2",
                    level="ERROR",
                    all_logs=True,
                )
                destPar = get_proj_par()

    if hPar is not None and len(destPar) == 4:
        destPar = destPar + hPar
    if isvertical:
        return ispolar, isvertical, destPar, xres, yres, None
    if destPar[1] != 0 and destPar[3] != 0 and full == 0 and ispolar == 0:
        return ispolar, isvertical, destPar, xres, yres, None
    if ispolar == 1:
        destDim = [destDim[0] * 2, destDim[0] * 2]
        if xres <= 0:
            xres = destPar[7]
        if yres <= 0:
            yres = destPar[7]

    destPar = get_proj_par_from_local_res(
        destMap, [xres, yres], center, destDim, isotropic=isotropic
    )
    if hPar is not None:
        destPar = destPar + hPar

    return ispolar, isvertical, destPar, xres, yres, None


def check_maps(
        sourceMap: Map,
        destMap: Map,
        dest_par: tuple = (),
        source_par: tuple = (),
        dest_az_coords: tuple = (),
        source_az_coords: tuple = (),
) -> int:
    """
    Method that verifies if two maps, sourceMap and destMap, are equivalent in terms of projection, projection
    parameters and azimuth coordinates.

    Args:
        sourceMap (Map): It represents the source map to compare with the destination map.
        destMap (Map): Represents the destination map to compare with the source map.

    :keywords:
        - dest_par (tuple): Projection parameters of the destination map. These parameters describe the projection
        configuration used to represent the map.
        - source_par (tuple): Projection parameters of the source map. Similarly to dest_par, these parameters
        describe the projection configuration used to represent the map.
        - dest_az_coords (tuple): Azimuth coordinates of the target map. Azimuth coordinates are angles measured
        against a specific reference direction, often used to determine the orientation of objects or points on a map.
        - source_az_coords (tuple): Azimuth coordinates of the source map. Similar to dest_az_coords, they represent
        the angles of orientation on a map.

    Returns:
        integer: Integer value indicating whether the two maps (sourceMap and destMap) are considered equivalent or not.
    """

    _, destProj, destP0Lat, destP0Lon = getMapName(destMap)
    _, sourceProj, sourceP0Lat, sourceP0Lon = getMapName(sourceMap)

    if sourceProj != destProj:
        return 0

    if np.abs(sourceP0Lon - destP0Lon) >= 0.001:
        return 0
    if np.abs(sourceP0Lat - destP0Lat) >= 0.001:
        return 0

    nd = len(dest_par)
    ns = len(source_par)

    if np.not_equal(ns, nd):
        return 0
    if ns > 0:
        if ns > 9:
            s = [e * 1000 for e in source_par[0:9]]
            d = [e * 1000 for e in dest_par[0:9]]
        else:
            s = [e * 1000 for e in source_par]
            d = [e * 1000 for e in dest_par]

        s = np.nan_to_num(s).astype(np.dtype("int64"))
        d = np.nan_to_num(d).astype(np.dtype("int64"))

        if not np.equal(s, d).all():
            return 0

    if dest_az_coords is None and source_az_coords is None:
        return 0

    nd = np.size(dest_az_coords)
    ns = np.size(source_az_coords)

    if ns != nd:
        return 0

    if ns > 0:
        s = source_az_coords * 10
        d = dest_az_coords * 10
        if np.not_equal(s, d).all():
            return 0

    return 1


def get_proj_corners(map, par, dim, regular=False):
    """
    Calculates the geographical corners and bounding box for a given projection.

    Parameters:
        map: The map object containing projection information.
        par: Parameters used to define the map's projection and resolution.
        dim: Dimensions of the map grid.
        regular (bool): Whether to compute a regular grid or not.

    Returns:
        [lat[0], lon[0], lat[1], lon[1]]: Top-left and bottom-right corners of the bounding box.
        nw_se_corners: Northwest and southeast corners of the bounding box.
        ne_sw_corners: Northeast and southwest corners of the bounding box.
        nw_se_box: Physical coordinates of the northwest-southeast box.
        range: Minimum and maximum latitude and longitude values of the bounding box.
    """
    range_n = np.empty(4)
    range_n[:] = np.nan

    nw_se_corners = range_n.copy()
    ne_sw_corners = range_n.copy()

    box, nw_se_box, ne_sw_box, view_defined = dpg.map.get_box_from_par(map=map, par=par, dim=dim, regular=regular)

    X = [box[0], box[2], nw_se_box[0], nw_se_box[2], ne_sw_box[0], ne_sw_box[2]]
    Y = [box[1], box[3], nw_se_box[1], nw_se_box[3], ne_sw_box[1], ne_sw_box[3]]

    lat, lon = dpg.map.yx_2_latlon(Y, X, map=map)

    minLat = np.min(lat)
    maxLat = np.max(lat)
    minLon = np.min(lon)
    maxLon = np.max(lon)

    nw_se_corners = [lat[2], lon[2], lat[3], lon[3]]
    ne_sw_corners = [lat[4], lon[4], lat[5], lon[5]]

    range = [minLat, minLon, maxLat, maxLon]

    return [lat[0], lon[0], lat[1], lon[1]], nw_se_corners, ne_sw_corners, nw_se_box, range


def get_proj_center(map, par, dim, latRange=np.array([]), lonRange=np.array([])):
    """
    Determines the center of a map projection in latitude and longitude.

    Parameters:
        map: The map object containing projection details.
        par: Parameters defining the map's characteristics.
        dim: Dimensions of the map grid.
        latRange (array): Optional latitude range to define the map center.
        lonRange (array): Optional longitude range to define the map center.

    Returns:
        centerLat: Latitude of the map's center.
        centerLon: Longitude of the map's center.
    """
    box, _, _, view_defined = dpg.map.get_box_from_par(map=map, par=par, dim=dim)
    if view_defined:
        X1 = box[0]
        Y1 = box[1]
        X2 = box[2]
        Y2 = box[3]
    else:
        if len(latRange) != 2 or len(lonRange) != 2:
            name, _, p0Lat, p0Lon = dpg.map.getMapName(map)
            return p0Lat, p0Lon

        Y1, X1 = dpg.map.latlon_2_yx(latRange[1], lonRange[0], map=map)
        Y2, X2 = dpg.map.latlon_2_yx(latRange[0], lonRange[1], map=map)

    y = (Y1 + Y2) / 2.
    x = (X1 + X2) / 2.

    centerLat, centerLon = dpg.map.yx_2_latlon(y, x, map=map)

    return [centerLat, centerLon]


def get_local_res(map, center, par):
    """
    Computes the local resolution of a map in meters at a given center point.

    Parameters:
        map: The map object containing projection and resolution information.
        center: [latitude, longitude] coordinates of the point of interest.
        par: Parameters defining resolution and grid spacing.

    Returns:
        [xres, yres]: Horizontal and vertical resolution in meters.
    """
    lat = center[0]
    lon = center[1]

    xres = np.nan
    yres = np.nan

    y, x = dpg.map.latlon_2_yx(lat, lon, map=map)

    xpoints = [x, x, x + par[1]]
    ypoints = [y, y + par[3], y]

    lats, lons = dpg.map.yx_2_latlon(ypoints, xpoints, map=map)

    if np.sum(np.isfinite(lats)) != 3:
        return [xres, yres]

    dlat = np.abs(lats[1] - lats[0])
    dlon = np.abs(lons[2] - lons[0])
    coslat = float(np.cos(lats[0] * np.pi / 180))

    re = 6378137.
    rp = re

    yres = np.pi / 180 * rp * dlat
    xres = np.pi / 180 * re * dlon * coslat

    return [xres, yres]


def is_3D_map(map):
    # TODO: da implementare per bene
    if map is not None:
        if len(map.dim) > 2:
            return 1
        else:
            return 0

    return 0
