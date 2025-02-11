import shutil
import os

import numpy as np

import sou_py.dpg as dpg
from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message
from sou_py.dpg.map__define import Map
from sou_py.dpg.node__define import Node

"""
Funzioni ancora da portare
FUNCTION GET_POINTS 
FUNCTION IDL_rv_get_dem_id 
FUNCTION IDL_rv_get_geo_info 
FUNCTION IDL_rv_get_heights 
PRO IDL_rv_get_site_coords 
FUNCTION GetSeaLevel                   // UNUSED
FUNCTION IDL_rv_replace_coords         // UNUSED
PRO IDL_rv_add_alt_in_navigation       // UNUSED
PRO IDL_rv_add_coords_in_navigation    // UNUSED
PRO IDL_rv_move_mosaic_desc            // UNUSED
PRO IDL_rv_save_geo_info               // UNUSED
"""


def get_idnavigation(
        node, reload: bool = False, only_current: bool = False, mosaic: bool = False
):
    """
    Retrieves the navigation attribute ID from a node, with options for mosaic and geo descriptions.

    This function fetches the navigation attribute (either mosaic or geo description) associated with the given node.
    It supports reloading the attribute, fetching only the current attribute, and returning either the attribute ID
    or both the attribute name and ID.

    Args:
        node (Node object): The node from which to retrieve the navigation attribute.
        name (bool, optional): If True, returns both the attribute name and ID. Defaults to False.
        reload (bool, optional): If True, reloads the attribute from the node. Defaults to False.
        only_current (bool, optional): If True, retrieves only the current attribute. Defaults to False.
        mosaic (bool, optional): If True, retrieves the mosaic description; otherwise, retrieves the geo description.
        Defaults to False.

    Returns:
        Either the navigation attribute ID or a tuple of the attribute name and ID, depending on the 'Name' argument.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    if mosaic:
        name = dpg.cfg.getMosaicDescName()
    else:
        name = dpg.cfg.getGeoDescName()
    idnavigation = dpg.tree.getAttr(
        node, name, reload=reload, only_current=only_current
    )
    return idnavigation


def get_site_name(node) -> str:
    """
    Retrieves the site name associated with a given node.

    This function attempts to find the site name for a node from various attributes, including the navigation,
    geo description, and array description attributes. It searches the attributes in the node itself and, if not
    found, looks further down the node hierarchy.

    Args:
        node (Node object): The node for which to find the site name.

    Returns:
        str: The site name associated with the node, if found; an empty string otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    name = ""
    if not isinstance(node, dpg.node__define.Node):
        return name

    attr = get_idnavigation(node)
    name, _, _ = dpg.attr.getAttrValue(attr, "origin", "")
    if name != "":
        return name

    files = [dpg.cfg.getGeoDescName(), dpg.cfg.getArrayDescName()]

    found = False
    for fff in files:
        attr = node.getAttr(fff)
        name, exists, _ = dpg.attr.getAttrValue(attr, "origin", "")
        if name != "":
            return name
        if exists:
            return name
        if not isinstance(attr, type(None)):
            found = True
    # endfor
    if found:
        return ""

    for fff in files:
        attr = dpg.tree.findAttr(node, fff, all=True, down=True)
        name, exists, _ = dpg.attr.getAttrValue(attr, "origin", "")
        if exists:
            return name
    # endfor
    return ""


def get_el_coords(node, attr: Attr = None, get_exists: bool = False):
    """
    Retrieves elevation coordinates and related information from a node's attribute.

    This function fetches elevation (EL) coordinates from a given node. It can use a specific attribute if provided;
    otherwise, it retrieves the default navigation attribute. The function returns the elevation coordinates,
    offset, resolution, and filename associated with the EL data, as well as the used attribute.

    Args:
        node (Node object): The node from which to retrieve elevation coordinates.
        attr (Attribute object, optional): The specific attribute to use. Defaults to None.

    Returns:
        tuple: A tuple containing five elements:
            - el_coords (np.ndarray or None): The array of elevation coordinates.
            - eloff (float): The elevation offset value.
            - elres (float): The elevation resolution value.
            - filename (str): The filename of the elevation data.
            - attr (Attribute object): The attribute used for retrieval.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    el_coords = None
    eloff = None
    elres = None
    filename = ""
    exists = 0

    if attr is None:
        attr = get_idnavigation(node)
    if attr is None:
        if get_exists:
            return el_coords, eloff, elres, filename, attr, exists
        else:
            return el_coords, eloff, elres, filename, attr

    eloff = 0.0
    nEl = 0
    filename, _, _ = dpg.attr.getAttrValue(attr, "elfile", "")
    if filename != "":
        el_attr = node.getSingleAttr(filename, format="VAL")
        if el_attr is not None:
            el_coords = el_attr.pointer
    if el_coords is not None:
        ind = np.where(el_coords > 180)
        if np.size(ind) > 0:
            # TODO: questo if va lasciato commentato in questo modo. Le coordinate di elevazione, grazie all'istruzione
            # scritta qui dentro rischiavano di assumere valori negativi quando il valore di elevazione era più baso
            # di 360.
            # concettualmente non è sbagliato, ma è stata rimossa per mantenere coerenza con IDL nei file ELEVATION.txt

            # el_coords[ind] -= 360.0
            pass
        nEl = len(el_coords)
        if nEl > 180:
            eloff = np.median(el_coords)
        else:
            eloff = el_coords[0]
    # endif
    elres, _, _ = dpg.attr.getAttrValue(attr, "elres", 0.0)
    eloff, exists, _ = dpg.attr.getAttrValue(attr, "eloff", eloff, round_value=1)
    eloff = np.round(eloff, 1)
    if eloff > 180:
        eloff -= 360.0
    if nEl > 0:
        exists = 1
        if get_exists:
            return el_coords, eloff, elres, filename, attr, exists
        else:
            return el_coords, eloff, elres, filename, attr

    if elres == 0.0:
        if get_exists:
            return el_coords, eloff, elres, filename, attr, exists
        else:
            return el_coords, eloff, elres, filename, attr

    nPlanes, _, _ = dpg.attr.getAttrValue(attr, "nPlanes", 0)
    if nPlanes <= 0:
        if get_exists:
            return el_coords, eloff, elres, filename, attr, exists
        else:
            return el_coords, eloff, elres, filename, attr
    el_coords = np.arange(nPlanes)
    el_coords *= elres
    el_coords += eloff
    if get_exists:
        return el_coords, eloff, elres, filename, attr, exists
    else:
        return el_coords, eloff, elres, filename, attr


def get_az_coords(node, regular: bool = False, attr: Attr = None):
    """
    Retrieves azimuth coordinates and related information from a node's attribute.

    This function fetches azimuth (AZ) coordinates from a given node. If a specific attribute is provided, it
    uses that; otherwise, it retrieves the default navigation attribute. The function returns the azimuth coordinates
    and the filename associated with the AZ data, as well as the used attribute. It also supports generating regular
    azimuth coordinates based on offset and resolution if no specific coordinates are available.

    Args:
        node (Node object): The node from which to retrieve azimuth coordinates.
        regular (bool, optional): If True, generates regular azimuth coordinates if not explicitly available.
        Defaults to False.
        attr (Attribute object, optional): The specific attribute to use. Defaults to None.

    Returns:
        tuple: A tuple containing three elements:
            - az_coords (np.ndarray or None): The array of azimuth coordinates.
            - filename (str): The filename of the azimuth data.
            - attr (Attribute object): The attribute used for retrieval.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    az_coords = None
    filename = ""

    if attr is None:
        attr = get_idnavigation(node)
    if attr is None:
        return az_coords, filename, attr

    nAz = 0
    filename, exists, _ = dpg.attr.getAttrValue(attr, "azfile", "")
    if exists:
        az_attr = node.getSingleAttr(filename, format="VAL")
        if az_attr is not None:
            az_coords = az_attr.pointer
    if az_coords is not None:
        nAz = len(az_coords)
    # endif
    if nAz > 0:
        return az_coords, filename, attr
    if not regular:
        return None, filename, attr
    azOff, _, _ = dpg.attr.getAttrValue(attr, "azoff", 0.0)
    azRes, _, _ = dpg.attr.getAttrValue(attr, "azres", 0.0)
    if azRes == 0.0:
        return None, filename, attr
    nPlanes, _, _ = dpg.attr.getAttrValue(attr, "nLines", 0)
    if nPlanes <= 0:
        gen = dpg.array.get_idgeneric(node)
        nPlanes, _, _ = dpg.attr.getAttrValue(gen, "nLines", 0)
    if nPlanes <= 0:
        return None, filename, attr
    az_coords = np.arange(nPlanes, dtype=float)
    az_coords *= azRes
    az_coords += azOff + azRes / 2.0
    return az_coords, filename, attr


def check_map(
        node,
        mosaic: bool = False,
        coords: str = None,
        par: list = None,
        destMap: Map = None,
        sourceMap: Map = None,
        get_box: bool = False,
        dim: bool = None,
        box: bool = None
):
    """
    Checks and retrieves map-related information from a given node.

    This function examines a node to determine map characteristics and retrieves relevant information like
    map parameters, dimensions, whether it's a polar or vertical map, etc. It can handle both individual
    nodes and mosaic structures, and it optionally processes coordinates files.

    Args:
        node (Node object): The node to be examined.
        mosaic (bool, optional): If True, treats the node as a mosaic. Defaults to False.
        coords (str, optional): The path to a coordinates file to check. Defaults to None.
        par (list, optional): A list of parameters to retrieve or check. Defaults to None.
        destMap (Map object, optional): Destination map to set if applicable. Defaults to None.
        sourceMap (Map object, optional): Source map to set if applicable. Defaults to None.
        box (list, optional): A box defining map boundaries. Defaults to None.

    Returns:
        tuple: A tuple containing six elements:
            - destMap (Map object or None): The determined destination map.
            - sourceMap (Map object or None): The determined source map.
            - dim (list): The dimensions of the map.
            - par (list): The parameters of the map.
            - ispolar (int): 1 if the map is polar, 0 otherwise.
            - isvertical (int): 1 if the map is vertical, 0 otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    if par is None:
        par = np.array([], dtype=np.float32)

    isvertical = 0
    ispolar = 0
    map = None
    attr = None

    if isinstance(node, dpg.node__define.Node) and (not mosaic):
        map, par, dim, attr = node.checkMap(map=map)
        if coords:
            _, coords, _ = check_coords_file(node, attr)
    else:
        if isinstance(node, dpg.node__define.Node) and mosaic:
            attr = node.getSingleAttr(dpg.cfg.getMosaicDescName())
        map = dpg.map.findSharedMap(
            dpg.attr.getAttrValue(attr, "projection", "")[0],
            p0lat=dpg.attr.getAttrValue(attr, "prj_lat", 0.0)[0],
            p0lon=dpg.attr.getAttrValue(attr, "prj_lon", 0.0)[0],
        )
        if map is None:
            map, par, dim, _ = node.checkMap(map=True, attr=attr)
        if dim or box:
            ncols, _, _ = dpg.attr.getAttrValue(attr, "ncols", 0)
            nlines, _, _ = dpg.attr.getAttrValue(attr, "nlines", 0)
            dim = [nlines, ncols]
        if par or box:
            cres, _, _ = dpg.attr.getAttrValue(attr, "cres", 0.0)
            coff, _, _ = dpg.attr.getAttrValue(attr, "coff", np.nan)
            lres, _, _ = dpg.attr.getAttrValue(attr, "lres", 0.0)
            loff, _, _ = dpg.attr.getAttrValue(attr, "loff", np.nan)
            par = [coff, cres, loff, lres]

    if len(par) > 9:
        ispolar = 1
        if par[9] == 0.0:
            isvertical = 1

    if destMap is not None and map is not None:
        destMap = map

    if sourceMap is not None and map is not None:
        sourceMap = map

    if get_box:
        latRange, lonRange, reverse = dpg.map.getLLRange(attr)
        if latRange is not None and lonRange is not None:
            y, x = dpg.map.latlon_2_yx(latRange, lonRange, map=map)
            if reverse:
                y = y[::-1]
                pass
            box = [x[0], y[0], x[1], y[1]]
        else:
            box, _, _, _ = dpg.map.get_box_from_par(map, par, dim)

    ret = 1 if map else 0
    return destMap, sourceMap, dim, par, ispolar, isvertical, box, attr, coords


def get_radar_par(
        node: Node | None = None,
        map: Map = None,
        el_coords: np.ndarray = None,
        reload=False,
        par=None,
        get_az_coords_flag: bool = False,
        get_el_coords_flag: bool = False,
        regular: bool = False,
) -> dict:
    """
    Retrieves radar parameter information from a node.

    This function extracts various radar parameters such as site coordinates, elevation and azimuth coordinates,
    range resolution and offset, beam width, and other relevant information from the given node. It handles both
    the cases where the node is a defined Node object or when specific parameters are provided.

    Args:
        node (Node object): The node from which to extract radar parameters.
        map (Map object, optional): The map associated with the node. Defaults to None.
        el_coords (np.ndarray, optional): Elevation coordinates. Defaults to None.
        reload (bool, optional): If True, reloads the attribute information. Defaults to False.

    Returns:
        dict: A dictionary containing key radar parameters, or a list of parameters if specific conditions are met.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    site_coords = None
    mode = 0
    out = {"mode": mode,
           "map": map,
           "par": par,
           "h_off": None,
           "h_res": None,
           "range_res": None,
           "range_off": None,
           "azimut_off": None,
           "azimut_res": None,
           "elevation_off": None,
           "elevation_res": None,
           "el_coords": None,
           'site_coords': site_coords}

    if isinstance(node, dpg.node__define.Node):
        # global map, currPar, dim, box, attr
        map, _, dim, currPar, ispolar, isvertical, box, attr, _ = check_map(
            node, destMap=True, get_box=True
        )
        _, projection, p0Lat, p0Lon = dpg.map.getMapName(map)
        if par is None and currPar is not None:
            par = currPar
        centerLat, _, _ = dpg.attr.getAttrValue(attr, "centerLat", p0Lat)
        centerLon, _, _ = dpg.attr.getAttrValue(attr, "centerLon", p0Lon)
        lat, _, _ = dpg.attr.getAttrValue(attr, "orig_lat", centerLat)
        lon, _, _ = dpg.attr.getAttrValue(attr, "orig_lon", centerLon)
        alt, _, _ = dpg.attr.getAttrValue(attr, "orig_alt", 0.0)
        center_coords = np.array([centerLat, centerLon, alt])  # Penso mai usato
        site_coords = np.array([lat, lon, alt])
        beam_width, _, _ = dpg.attr.getAttrValue(attr, "beamWidth", 0.9)
        origin, _, _ = dpg.attr.getAttrValue(attr, "origin", "")

        # range_res = dpg.attr.getAttrValue(attr, 'rangeres', 0.)
        # range_off = dpg.attr.getAttrValue(attr, 'rangeoff', 0.)

        if get_az_coords_flag:
            az_coords, filename, attr = get_az_coords(node, regular=regular)
            out["az_coords"] = az_coords
        if get_el_coords_flag:
            el_coords, eloff, elres, filename, attr = get_el_coords(node)
            out["el_coords"] = el_coords

        out["site_coords"] = site_coords
        out["beam_width"] = beam_width
        out["origin"] = origin

    out["map"] = map
    out["par"] = par
    if par is None:
        return out

    if len(par) <= 0:
        return out

    if len(par) <= 4:
        return out

    h_off = par[4]
    h_res = par[5]
    out["h_off"] = h_off
    out["h_res"] = h_res
    if len(par) <= 6:
        return out

    range_off = par[6]
    range_res = par[7]
    out["range_res"] = range_res
    out["range_off"] = range_off
    if len(par) <= 8:
        return out

    azimut_off = par[8]
    azimut_res = par[9]
    if azimut_off >= 360.0:
        azimut_off -= 360.0
    if azimut_res == 0.0:
        mode = 2
    out["azimut_off"] = azimut_off
    out["azimut_res"] = azimut_res
    out["mode"] = mode
    if len(par) <= 10:
        return out

    elevation_off = par[10]
    elevation_res = par[11]
    out["elevation_off"] = elevation_off
    out["elevation_res"] = elevation_res
    if elevation_off >= 360.0:
        elevation_off -= 360.0
    if el_coords is None:
        # log_message("Elevation coords is None!", all_logs=True, level='WARNING')
        el_coords = elevation_off
        out["el_coords"] = el_coords

    out['site_coords'] = site_coords
    out["mode"] = 1

    return out


def get_corners(
        node,
        set_center: bool = None,
        corners: np.ndarray = None,
        cartesian: bool = None,
        format: str = None,
        type: type = None,
) -> np.ndarray:
    """
    Retrieves the corner coordinates of a node's associated array or map.

    This function computes the corner coordinates for the array or map associated with the given node.
    It can optionally adjust the coordinates based on the center, handle cartesian coordinates, and format
    or cast the coordinates to a specific type.

    Args:
        node (Node object): The node whose corner coordinates are to be determined.
        set_center (bool, optional): If set, adjusts the coordinates based on the center. Defaults to None.
        corners (np.ndarray, optional): Pre-existing array to store corners. Defaults to None.
        cartesian (bool, optional): If True, handles coordinates in cartesian format. Defaults to None.
        format (str, optional): Format string to format the corner coordinates. Defaults to None.
        type (type, optional): The type to cast the corner coordinates to. Defaults to None.

    Returns:
        np.ndarray: An array containing the corner coordinates.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if corners is not None:
        corners = np.empty_like(corners)
    else:
        corners = np.empty((8,))

    _, out_dict = dpg.array.get_array(node, silent=True)
    dim = out_dict["dim"]
    if len(dim) < 2:
        return

    check = check_map(node)  # , DESTMAP=map, PAR=par, DIM=dim)
    if check <= 0:
        return

    minus = float(set_center is not None)

    lin = [dim[1] - minus, 0.0, 0.0, dim[1] - minus]
    col = [0.0, 0.0, dim[0] - minus, dim[0] - minus]

    # lin, col = LINCOL_2_YX(lin, col, par, SET_center=SET_center)

    if len(lin) != 4 or len(col) != 4:
        return

    if cartesian is None:
        pass  # yyy, xxx = YX_2_LATLON(lin, col, MAP=map)
    else:
        yyy, xxx = lin, col

    corners = np.vstack((xxx, yyy)).T

    if format is not None:
        corners = [format.format(x) for x in corners]
    if type is not None:
        corners = np.fix(corners).astype(type)

    return corners


def put_corners(node, LL_lat: float, LL_lon: float, UR_lat: float, UR_lon: float):
    """
    Updates the corner coordinates of a node's geographical description.

    This function sets the lower-left and upper-right corner coordinates (latitude and longitude)
    for the geographical description of the given node. It updates these coordinates in the node's
    associated attribute using predefined variable names.

    Args:
        node (Node object): The node whose geographical description is to be updated.
        LL_lat (float): Latitude of the lower-left corner.
        LL_lon (float): Longitude of the lower-left corner.
        UR_lat (float): Latitude of the upper-right corner.
        UR_lon (float): Longitude of the upper-right corner.

    Returns:
        None: The function does not return anything but updates the node's geographical description.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    varnames = ["LL_lat", "LL_lon", "UR_lat", "UR_lon"]
    values = [LL_lat, LL_lon, UR_lat, UR_lon]

    _ = dpg.tree.replaceAttrValues(node, dpg.cfg.getGeoDescName(), varnames, values)


def save_coords(
        path: str,
        x,
        y,
        name: str = "coords.txt",
        xy: bool = False,
        origin: str = None,
):
    """
    Saves coordinate data to a specified file and updates associated attributes.

    This function creates a file containing coordinate data (either in XY or lat-lon format) and saves it
    to a given path. It also updates related attributes such as 'coordfile' and 'format', and optionally
    includes the 'origin' attribute.

    Args:
        path (str): The file path where coordinates are to be saved.
        x (list or np.ndarray): The x-coordinates or longitude values.
        y (list or np.ndarray): The y-coordinates or latitude values.
        name (str, optional): The name of the file to be created. Defaults to 'coords.txt'.
        xy (bool, optional): If True, saves coordinates in XY format; otherwise in lat-lon format. Defaults to False.
        origin (str, optional): The origin of the coordinates. Defaults to None.

    Returns:
        None: The function does not return anything but saves coordinates to a file and updates attributes.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    nCoords = len(x)
    if nCoords <= 0:
        return

    if not name:
        name = "coords.txt"

    varnames = []
    values = []

    if xy:
        if len(y) == nCoords:
            varnames.extend(["y"] * nCoords)
            varnames.extend(["x"] * nCoords)
            values.extend(y)
            values.extend(x)
        else:
            varnames.extend(["x"] * nCoords)
            values.extend(x)
    else:
        varnames.extend(["lat"] * nCoords)
        varnames.extend(["lon"] * nCoords)
        values.extend(y)
        values.extend(x)

    _ = dpg.attr.saveAttr(path, name, varnames, values)

    varnames.extend(["coordfile", "format"])
    values.extend([name, "txt"])

    if origin is not None:
        varnames.append("origin")
        values.append(origin)

    # Save attributes
    _ = dpg.attr.saveAttr(path, dpg.cfg.getGeoDescName(), varnames, values)


# Probabilmente UNUSED
def set_coords(node, x, y, name: str = "coords.txt"):
    """
    Sets coordinates for a given node and saves them to a specified file.

    This function takes arrays of latitude and longitude values, combines them, and
    assigns them to a node in a 'lat', 'lon' sequence. If no file name is provided,
    'coords.txt' is used as the default. The coordinates are then saved to this file.

    Args:
        node: The node to which the coordinates will be assigned.
        x (list of float): An array of latitude values.
        y (list of float): An array of longitude values.
        name (str, optional): The name of the file where coordinates are saved.
            Defaults to 'coords.txt'.

    Returns:
        None: This function does not return any value but updates the node's attributes.
    """

    if not name:
        name = "coords.txt"

    nCoords = len(x)
    varnames = ["lat"] * nCoords + ["lon"] * nCoords
    values = [y, x]

    _ = dpg.tree.replaceAttrValues(node, name, varnames, values)


# Probabilmente UNUSED
def remove_polar_par(node):
    """
    Removes specific polar parameter attributes from a given node.

    This function retrieves the geographic descriptor name using the `dpg.cfg.getGeoDescName`
    method. It then defines a list of polar parameters such as 'rangeoff', 'rangeres',
    'azoff', 'azres', 'eloff', and 'elres'. These parameters are removed from the specified
    node using the `dpg.tree.dpg.tree.removeAttrValues` method.

    Args:
        node: The node from which the polar parameters will be removed.

    Returns:
        None: This function does not return any value but updates the specified node by
        removing specified attributes.
    """
    name = dpg.cfg.getGeoDescName()
    pars = ["rangeoff", "rangeres", "azoff", "azres", "eloff", "elres"]

    _ = dpg.tree.dpg.tree.removeAttrValues(node, name, pars)


def set_geo_info(
        node,
        projection=None,
        p0lat=None,
        p0lon=None,
        par=None,
        dim=None,
        cov=None,
        site_coords=None,
        site_name=None,
        to_owner=None,
        mosaic=None,
):
    """
    Sets geographic information for a given node with various parameters.

    This function updates a node with geographic information such as projection type,
    site coordinates, site name, and other parameters. It modifies attributes based on
    the provided arguments and handles conditional logic for different scenarios.
    It also performs checks and updates related to map properties of the node.

    Args:
        node: The node to update with geographic information.
        projection (optional): The projection type.
        p0lat (optional): Latitude value for the projection.
        p0lon (optional): Longitude value for the projection.
        par (list, optional): A list of parameters related to the geographic information.
        dim (list, optional): Dimensions as a list containing up to three values.
        cov (optional): Coverage value.
        site_coords (list, optional): A list of site coordinates (latitude, longitude, and optionally altitude).
        site_name (str, optional): The name of the site.
        to_owner (optional): Specifies the owner of the node.
        mosaic (bool, optional): Indicates if mosaic description is to be used.

    Returns:
        None: This function does not return any value.
    """

    varnames = []
    values = []
    parnames = []
    if projection is not None:
        varnames.append("projection")
        values.append(str(projection))

    if site_name is not None:
        if varnames:
            varnames.append("origin")
            values.append(str(site_name))
        else:
            varnames = ["origin"]
            values = [str(site_name)]

    if p0lat is not None:
        if varnames:
            varnames.append("prj_lat")
            values.append(str(p0lat))
        else:
            varnames = ["prj_lat"]
            values = [str(p0lat)]

    if p0lon is not None:
        if varnames:
            varnames.append("prj_lon")
            values.append(str(p0lon))
        else:
            varnames = ["prj_lon"]
            values = [str(p0lon)]

    if par is not None:
        nPar = len(par)
        if nPar >= 4:
            parnames = ["coff", "cres", "loff", "lres"]
        if nPar >= 5:
            parnames.extend(["hoff", "hres"])
        if nPar >= 7:
            parnames.extend(["rangeoff", "rangeres"])
        if nPar >= 9:
            parnames.extend(["azoff", "azres"])
        if nPar >= 11:
            parnames.extend(["eloff", "elres"])

        if parnames:
            if varnames:
                varnames.extend(parnames[:nPar])
                values.extend([str(p) for p in par])
            else:
                varnames = parnames[:nPar]
                values = [str(p) for p in par]

    if cov is not None:
        if varnames:
            varnames.append("coverage")
            values.append(str(cov))
        else:
            varnames = ["coverage"]
            values = [str(cov)]

    if site_coords is not None and len(site_coords) >= 2:
        if len(site_coords) > 2:
            orignames = ["orig_lat", "orig_lon", "orig_alt"]
        else:
            orignames = ["orig_lat", "orig_lon"]

        if varnames:
            varnames.extend(orignames)
            values.extend([str(coord) for coord in site_coords])
        else:
            varnames = orignames
            values = [str(coord) for coord in site_coords]

    if dim is not None and 1 <= len(dim) <= 3:
        varnames.extend(["ncols"])
        if len(dim) > 1:
            varnames.extend(["nlines"])
        if len(dim) > 2:
            varnames.extend(["nplanes"])

        values.extend([str(d) for d in dim])

    if mosaic:
        geoDesc = dpg.cfg.getMosaicDescName()
    else:
        geoDesc = dpg.cfg.getGeoDescName()

    if nPar <= 10:
        _ = dpg.tree.removeAttrValues(
            node, geoDesc, ["eloff", "elres"], to_owner=to_owner
        )
    if nPar <= 8:
        _ = dpg.tree.removeAttrValues(
            node, geoDesc, ["azoff", "azres"], to_owner=to_owner
        )
    if nPar <= 6:
        _ = dpg.tree.removeAttrValues(
            node, geoDesc, ["rangeoff", "rangeres"], to_owner=to_owner
        )
    if nPar <= 4:
        _ = dpg.tree.removeAttrValues(
            node, geoDesc, ["hoff", "hres"], to_owner=to_owner
        )

    _ = dpg.tree.replaceAttrValues(node, geoDesc, varnames, values, to_owner=to_owner)

    if nPar >= 4:
        if par[1] == 0 and par[3] == 0:
            _ = dpg.tree.removeAttrValues(
                node,
                geoDesc,
                ["coff", "cres", "loff", "lres"],
                to_owner=to_owner,
            )

    if isinstance(node, dpg.node__define.Node):
        oMap = node.getProperty("map")
        if isinstance(oMap, dpg.map__define.Map):
            node.checkMap(reset=True)

    return


def set_geo_par(attr, dim: list, eloff=None, elres=None):
    """
    Sets and returns geographic parameters and origin coordinates based on the provided attributes.

    This function calculates geographic parameters (such as offsets and resolutions in various directions)
    and origin coordinates (latitude, longitude, altitude) based on the specified attributes. It handles
    various conditions and defaults for parameters like 'eloff' and 'elres'. The function ensures the parameters
    are in list format and fetches values from attributes using the 'dpg.attr.getAttrValue' method.

    Args:
        attr: The attribute from which to retrieve geographic values.
        dim (list): A list specifying dimensions.
        eloff (list or None, optional): Elevation offset values. Defaults to None.
        elres (list or None, optional): Elevation resolution values. Defaults to None.

    Returns:
        tuple: A tuple containing two lists:
               1. A list of calculated geographic parameters.
               2. A list representing the origin coordinates [latitude, longitude, altitude].

    Note:
        The function dynamically adjusts the length and contents of the return lists based on the existence
        and values of various attributes.
    """
    if not isinstance(eloff, list):
        if eloff is None:
            eloff = []
        else:
            eloff = [eloff]
    if not isinstance(elres, list):
        if elres is None:
            elres = []
        else:
            elres = [elres]

    lat, _, _ = dpg.attr.getAttrValue(attr, "orig_lat", 0.0)
    lon, _, _ = dpg.attr.getAttrValue(attr, "orig_lon", 0.0)
    alt, _, _ = dpg.attr.getAttrValue(attr, "orig_alt", 0.0)
    origin = [lat, lon, alt]

    ncols, _, _ = dpg.attr.getAttrValue(attr, "ncols", 0)
    nlines, _, _ = dpg.attr.getAttrValue(attr, "nlines", 0)

    if len(dim) < 2:
        dim = [nlines, ncols]
    if dim[0] == 0:
        dim[0] = nlines
    if dim[1] == 0:
        dim[1] = ncols

    cres, _, _ = dpg.attr.getAttrValue(attr, "cres", 0.0)
    coff, _, _ = dpg.attr.getAttrValue(attr, "coff", np.nan)

    lres, _, _ = dpg.attr.getAttrValue(attr, "lres", 0.0)
    loff, _, _ = dpg.attr.getAttrValue(attr, "loff", np.nan)

    par = [coff, cres, loff, lres]

    hres, _, _ = dpg.attr.getAttrValue(attr, "hres", 0.0)
    hoff, exists, _ = dpg.attr.getAttrValue(attr, "hoff", 0.0)

    if exists:
        par += [hoff, hres]

    rangeres, exists, _ = dpg.attr.getAttrValue(attr, "rangeres", 0.0)
    if not exists:
        rangeres, _, _ = dpg.attr.getAttrValue(attr, "polres", 0.0)

    rangeoff, _, _ = dpg.attr.getAttrValue(attr, "rangeoff", 0.0)
    rangeoff, _, _ = dpg.attr.getAttrValue(attr, "poloff", rangeoff)

    if rangeres > 0:
        if len(par) == 4:
            par += [0.0, 0.0]
        par += [rangeoff, rangeres]
        azres, exists, _ = dpg.attr.getAttrValue(attr, "azres", 0.0)
        azoff, _, _ = dpg.attr.getAttrValue(attr, "azoff", 0.0)
        if exists:
            par += [azoff, azres]
            if len(eloff) == 1:
                par += eloff
                if len(elres) == 1:
                    par += elres

    return np.array(par, dtype=np.float32), origin


def copy_geo_info(
        fromNode, toNode, to_save: bool = False, only_if_not_exists: bool = False
):
    """
    Copies geographic information from one node to another and optionally saves the related files.

    This function copies geographic attributes from 'fromNode' to 'toNode' using the `dpg.tree.copyAttr` method.
    It handles the copying of azimuth and elevation coordinates, as well as associated files if 'to_save' is True.
    If the relevant file names are found, it copies the specified files to the 'toNode' path using `shutil.copyfile`.
    Additional handling is done for '.shp' files to also copy related '.dbf' and '.shx' files.

    Args:
        fromNode: The source node from which to copy geographic information.
        toNode: The target node to which geographic information will be copied.
        to_save (bool, optional): Flag indicating whether to save related files. Defaults to True.

    Returns:
        None: This function does not return any value.

    Note:
        The function immediately returns if either 'fromNode' or 'toNode' is None.
        It also returns early if 'to_save' is False or if no filename is associated with the geographic attributes.
    """
    if fromNode is None or toNode is None:
        return

    if only_if_not_exists:
        idnavigation = get_idnavigation(node=toNode, only_current=True)
        if idnavigation is not None:
            return
    _, _ = dpg.tree.copyAttr(fromNode, toNode, dpg.cfg.getGeoDescName())
    filename = ""

    _, filename, attr = get_az_coords(fromNode)
    if filename != "":
        _, _ = dpg.tree.copyAttr(fromNode, toNode, filename)

    _, _, _, filename, attr = get_el_coords(fromNode)
    if filename != "":
        _, _ = dpg.tree.copyAttr(fromNode, toNode, filename)

    if not to_save:
        return

    filename, _, _ = dpg.attr.getAttrValue(attr, "coordfile", "")
    if filename == "":
        return

    oldName = dpg.tree.getNodePath(fromNode) + filename
    toPath = dpg.tree.getNodePath(toNode)
    shutil.copyfile(oldName, toPath)

    if ".shp" not in oldName:
        return

    oldName = oldName.replace(".shp", ".dbf")
    shutil.copyfile(oldName, toPath)

    oldName = oldName.replace(".shp", ".shx")
    shutil.copyfile(oldName, toPath)

    return


def fill_nav(node):
    """
    To be completed
    """
    _, _, dim, _, _, _, _, _ = node.getArrayInfo()
    if np.size(dim) <= 1:
        return
    if dim[0] <= 0:
        return

    _, sourcemap, _, par, _, _, _, attr, _ = dpg.navigation.check_map(
        node, sourceMap=True
    )

    box, nw_se_corners, ne_sw_corners, nw_se_box, range_values = (
        dpg.map.get_proj_corners(sourcemap, par, dim)
    )

    latRange = [nw_se_corners[0], nw_se_corners[2]]
    lonRange = [nw_se_corners[1], nw_se_corners[3]]
    varnames = ["NWLon", "SELon", "NWLat", "SELat"]
    values = [str(lonRange[0]), str(lonRange[1]), str(latRange[0]), str(latRange[1])]

    latRange = [ne_sw_corners[0], ne_sw_corners[2]]
    lonRange = [ne_sw_corners[1], ne_sw_corners[3]]
    varnames = varnames + ["NELon", "SWLon"]
    values = values + [str(lonRange[0]), str(lonRange[1])]
    varnames = varnames + ["NELat", "SWLat"]
    values = values + [str(latRange[0]), str(latRange[1])]

    centerLat, _, _ = dpg.attr.getAttrValue(attr, "centerLat", 0.0)
    centerLon, exists, _ = dpg.attr.getAttrValue(attr, "centerLon", 0.0)
    if not exists:
        center = dpg.map.get_proj_center(sourcemap, par, dim)
        varnames = varnames + ["centerLat", "centerLon"]
        values = values + [str(center[0]), str(center[1])]
    else:
        center = [centerLat, centerLon]

    maxLat, exists, _ = dpg.attr.getAttrValue(attr, "maxLat", 90.0)
    if not exists and len(range_values) >= 4:
        varnames = varnames + ["minLon", "maxLon"]
        values = values + [str(range_values[1]), str(range_values[3])]
        varnames = varnames + ["minLat", "maxLat"]
        values = values + [str(range_values[0]), str(range_values[2])]

    xres, exists, _ = dpg.attr.getAttrValue(attr, "xres", 0.0)
    if not exists:
        res = dpg.map.get_local_res(sourcemap, center, par)
        varnames = varnames + ["xres", "yres"]
        values = values + [str(res[0]), str(res[1])]

    if len(dim) == 3:
        hoff = 0.0  # Placeholder for hoff
        hres = 1000.0  # Placeholder for hres
        varnames = varnames + ["hoff", "hres"]
        values = values + [str(hoff), str(hres)]

    # Placeholder for ReplaceTags function
    replace_tags_result = dpg.attr.replaceTags(attr=attr, tags=varnames, values=values)
    return attr


def check_coords_file(
        node,
        attr: Attr,
        prefix: str = "",
        check_date=None,
        dim: list = None,
        reload: bool = False,
) -> str:
    """
    Checks and possibly reloads coordinate files for a given node based on specified attributes.

    This function checks for the presence of coordinate, data, and attribute files based on the given 'attr'.
    It handles different file types and loads the appropriate data into the node. The function can also reload
    the data based on the 'reload' flag. If specific coordinate files ('latfile', 'lonfile', 'altfile') exist,
    it processes these files separately.

    Args:
        node: The node for which the coordinate file is to be checked.
        attr: The attribute containing information about the coordinate file.
        format (optional): The format of the coordinate file. Defaults to None.
        prefix (str, optional): A prefix to be added to file names. Defaults to ''.
        check_date (optional): A date to check against for file updates. Defaults to None.
        dim (list, optional): The dimensions to consider while reading the file. Defaults to None.
        reload (bool, optional): Flag to indicate if the file should be reloaded. Defaults to False.
        coords (optional): Pre-existing coordinates to use. Defaults to None.

    Returns:
        str: The name of the coordinate file processed or an empty string if no file is found.

    Note:
        The function returns early with an empty string if 'attr' is None or empty,
        and it returns the filename immediately if 'node' is not an instance of 'dpg.node__define.Node'.
    """
    coords = None
    format_ = None
    if attr is None or len(attr) == 0:
        return "", coords, format_

    filename, _, _ = dpg.attr.getAttrValue(attr, "coordfile", "", prefix=prefix)
    datafile, _, _ = dpg.attr.getAttrValue(attr, "datafile", "", prefix=prefix)
    attrfile, _, _ = dpg.attr.getAttrValue(attr, "attrfile", "", prefix=prefix)

    if not isinstance(node, dpg.node__define.Node):
        return filename, None, None

    names = None
    to_not_load = 1
    if attrfile != "":
        filename = attrfile
        to_not_load = 0

    if filename != "" or datafile != "":
        if filename == "":
            filename = datafile
        if not reload:
            coords = dpg.tree.getAttr(
                node,
                filename,
                to_not_load=to_not_load,
                format=format_,
                check_date=check_date,
                stop_at_first=True,
            )
            format_, _, _ = dpg.attr.getAttrValue(attr, "format", "", prefix=prefix)

        if coords is None:
            pathname = os.path.dirname(filename)
            if len(pathname) <= 1:
                pathname = dpg.tree.getNodePath(node)
            format_, _, _ = dpg.attr.getAttrValue(attr, "format", "", prefix=prefix)
            data, _, coords, file_date, names = dpg.io.read_array(
                pathname,
                filename,
                0,
                0,
                format=format_,
                get_file_date=True,
                names=names,
            )
            if coords is None and data is not None:
                coords = data
                data = None
            _ = dpg.tree.addAttr(
                node, filename, coords, format=format_, file_date=file_date
            )
            if data is not None:
                if datafile != "":
                    _ = dpg.tree.addAttr(node, datafile, data, file_date=file_date)
                    if names is not None:
                        pNames = names
                        _ = dpg.tree.addAttr(
                            node, "attr_names", pNames, file_date=file_date
                        )
                else:
                    data = None
            # endif
        # endif
        return filename, coords, format_
    # endif

    latfile, exists_lat, _ = dpg.attr.getAttrValue(
        attr, "latfile", "", prefix=prefix
    )
    if exists_lat:
        lonfile, _, _ = dpg.attr.getAttrValue(attr, "lonfile", "", prefix=prefix)
        altfile, _, _ = dpg.attr.getAttrValue(attr, "altfile", "", prefix=prefix)
        filename = "coords"
        coords = dpg.tree.getAttr(
            node, filename, only_current=True, to_not_load=True, format=format_
        )
        pathname = os.path.dirname(filename)
        if len(pathname) <= 1:
            pathname = dpg.tree.getNodePath(node)
        if coords is None:
            format = "COORDS"
            coords, _, _, _ = dpg.io.read_array(
                pathname, [lonfile, latfile, altfile], 0, dim, format=format
            )
            _ = dpg.tree.addAttr(node, filename, coords, format=format)
        # endif
    # endif

    return filename, coords, format_


def remove_az_coords(node) -> bool:
    """
    Removes azimuth coordinate information from the specified node.

    This function attempts to remove azimuth coordinate data associated with a node. It first retrieves
    the navigation ID attribute of the node and then checks for the existence of the 'azfile' attribute.
    If the 'azfile' attribute exists, the function proceeds to remove the azimuth coordinate attribute
    from the node and deletes the corresponding file. It also removes the 'azfile' tag from the attribute list.

    Args:
        node: The node from which azimuth coordinates are to be removed.

    Returns:
        bool: True if the removal is successful, False otherwise.

    Note:
        The function returns False immediately if either the navigation ID attribute is not found or the
        'azfile' attribute does not exist.
    """
    result = True
    attr = get_idnavigation(node, only_current=True)
    if attr is None:
        result = False
        return result
    filename, exists, _ = dpg.attr.getAttrValue(attr, "azfile", "")
    if not exists:
        result = False
        return result
    result = dpg.tree.removeAttr(node, name=filename, delete_file=True)
    _ = dpg.attr.removeTags(attr, "azfile")  # RemoveTags(attr[0], 'azfile') #TODO ok?
    return result


def remove_el_coords(node) -> bool:
    """
    Removes elevation coordinates from a node and deletes the associated file.

    Parameters:
        node: The node from which to remove elevation coordinates.

    Returns:
        True if the operation was successful, False otherwise.
    """

    result = True
    attr = get_idnavigation(node, only_current=True)
    if attr is None:
        result = False
        return result
    filename, exists, _ = dpg.attr.getAttrValue(attr, "elfile", "")
    if not exists:
        result = False
        return result
    result = dpg.tree.removeAttr(node, name=filename, delete_file=True)
    _ = dpg.attr.removeTags(attr, "elfile")  # RemoveTags(attr[0], 'elfile') #TODO ok?
    return result


def check_proj_par(attr, projection, p0lat=0, p0lon=0, par=None, dim=None) -> dict:
    """
    Checks and retrieves projection parameters for a specified attribute and projection.

    This function finds a shared map using the given attribute and projection. It then retrieves
    various parameters related to the projection such as isotropic value, center latitude/longitude,
    min/max latitude/longitude, sampling rate, diameters, and resolutions. It handles different
    scenarios for obtaining these values based on the existence of certain attributes.

    Args:
        attr: The attribute from which to retrieve projection parameters.
        projection: The specific projection for which the parameters are to be checked.

    Returns:
        dict: A dictionary containing the retrieved projection parameters, including:
              - 'center': A list [latitude, longitude] for the center of the projection.
              - 'latRange' (optional): A list [minLat, maxLat] if latitude range exists.
              - 'lonRange' (optional): A list [minLon, maxLon] if longitude range exists.
              - 'earthDiam' (optional): A list [diamx, diamy] if earth diameters are set.
              - 'xres': The x-resolution.
              - 'yres': The y-resolution.
              - Other individual parameters as retrieved.

    Note:
        The function dynamically adjusts the returned dictionary based on the existence
        and values of various attributes. Some keys in the dictionary may not be present if
        their corresponding values are not set in the attributes.
    """

    map = dpg.map.findSharedMap(projection=projection, p0lat=p0lat, p0lon=p0lon)
    isotropic, _, _ = dpg.attr.getAttrValue(attr, "isotropic", 1)
    centerLat, _, _ = dpg.attr.getAttrValue(attr, "centerLat", p0lat)
    centerLon, centro, _ = dpg.attr.getAttrValue(attr, "centerLon", p0lon)
    if not centro:
        centerLat, _, _ = dpg.attr.getAttrValue(attr, "orig_Lat", p0lat)
        centerLon, centro, _ = dpg.attr.getAttrValue(
            attr, "orig_Lon", p0lon
        )

    minLon, exists, _ = dpg.attr.getAttrValue(attr, "minLon", -90.0)
    if exists:
        maxLon, _, _ = dpg.attr.getAttrValue(attr, "maxLon", 90.0)
        minLat, _, _ = dpg.attr.getAttrValue(attr, "minLat", -90.0)
        maxLat, _, _ = dpg.attr.getAttrValue(attr, "maxLat", 90.0)
    else:
        minLon, exists, _ = dpg.attr.getAttrValue(attr, "firstLon", -90.0)
        maxLon, _, _ = dpg.attr.getAttrValue(attr, "lastLon", 90.0)
        minLat, _, _ = dpg.attr.getAttrValue(attr, "firstLat", -90.0)
        maxLat, _, _ = dpg.attr.getAttrValue(attr, "lastLat", 90.0)

    if exists:
        latRange = [minLat, maxLat]
        lonRange = [minLon, maxLon]

    sampling, _, _ = dpg.attr.getAttrValue(attr, "sampling", 0.0)
    diamx, _, _ = dpg.attr.getAttrValue(attr, "diamx", 0)
    diamy, _, _ = dpg.attr.getAttrValue(attr, "diamy", 0)
    if diamx != 0 and diamy != 0:
        earthDiam = [diamx, diamy]

    if map is not None:
        centerLon, centerLat = dpg.map.get_lon0_and_lat0(map.mapProj, double=True)
    center = [centerLat, centerLon]

    xres, _, _ = dpg.attr.getAttrValue(attr, "xres", 0.0)
    yres, _, _ = dpg.attr.getAttrValue(attr, "yres", 0.0)

    ispolar, isvertical, par, xres, yres, box = dpg.map.check_proj_par(
        map,
        par,
        dim,
        sampling=sampling,
        center=center,
        isotropic=isotropic,
        xres=xres,
        yres=yres,
        corner=1 - centro,
    )

    return isotropic, ispolar, isvertical, par, xres, yres, box, projection, dim


def get_geo_info(
        node,
        attr: Attr = None,
        reload: bool = None,
        mosaic: bool = None,
        dim: list = None,
        check_date: bool = False,
        projection: str = None,
) -> list:
    """
    Retrieves geographic information and parameters for a given node.

    This function fetches various geographic attributes, such as area, method, coordinates, dimensions, and projection
    details, from the specified node. It also checks and processes coordinate files if necessary and calculates the
    required projection parameters.

    Args:
        node:                        The node from which to retrieve geographic information.
        attr:                        The attribute object containing geographic data. Defaults to None.
        reload (bool, optional):     If True, reloads the attribute information. Defaults to None.
        mosaic (bool, optional):     If True, handles the node as a mosaic. Defaults to None.
        dim (list, optional):        The dimensions of the geographic data. Defaults to None.
        coords (str, optional):      Path to the coordinates file. Defaults to None.
        format (str, optional):      Format of the coordinates file. Defaults to None.
        check_date (bool, optional): If True, checks the file date for updates. Defaults to False.
        projection (str, optional):  The projection type for the geographic data. Defaults to None.

    Returns:
        list:                        A list of geographic parameters.
        dim:                         Tuple of the dimension

    Note:
        - The function dynamically retrieves various geographic attributes based on the provided node and attribute.
        - It processes coordinate files and calculates projection parameters if required.
        - The returned list of parameters includes various offsets, resolutions, and coordinate details.
    """
    ret = 1
    if attr is None:
        attr = get_idnavigation(node, reload=reload, mosaic=mosaic)
    else:
        if isinstance(attr, list):
            if attr[0] is not None:
                attr = get_idnavigation(node, reload=reload, mosaic=mosaic)

    area, _, _ = dpg.attr.getAttrValue(attr, "area", "")
    method, _, _ = dpg.attr.getAttrValue(attr, "method", 0)
    automatic, _, _ = dpg.attr.getAttrValue(attr, "automatic", 0)

    ncols, _, _ = dpg.attr.getAttrValue(attr, "ncols", 0)
    nlines, _, _ = dpg.attr.getAttrValue(attr, "nlines", 0)

    if dim is None or mosaic is not None:
        dim = [nlines, ncols]
    if dim[0] == 0:
        dim[0] = nlines
    if dim[1] == 0:
        dim[1] = ncols

    filename, coords, format = check_coords_file(
        node=node,
        attr=attr,
        dim=dim,
        check_date=check_date,
    )

    if projection is None:
        projection = dpg.cfg.getDefaultProjection()
    projection_tmp, exists, _ = dpg.attr.getAttrValue(attr, "projection")
    if exists:
        if projection == "NONE":
            existProjection = 0
        projection = projection_tmp

    p0lat, _, _ = dpg.attr.getAttrValue(attr, "orig_lat", 0.0)
    p0lat, _, _ = dpg.attr.getAttrValue(attr, "prj_lat", p0lat)
    p0lon, _, _ = dpg.attr.getAttrValue(attr, "orig_lon", 0.0)
    p0lon, _, _ = dpg.attr.getAttrValue(attr, "prj_lon", p0lon)

    cres, _, _ = dpg.attr.getAttrValue(attr, "cres", 0.0)
    coff, _, _ = dpg.attr.getAttrValue(attr, "coff", np.nan)
    if cres == 0:
        cfac, _, _ = dpg.attr.getAttrValue(attr, "cfac", 0.0)
        if cfac != 0:
            cres = 65536000000.0 / (7.2 * cfac)

    lres, _, _ = dpg.attr.getAttrValue(attr, "lres", 0.0)
    loff, _, _ = dpg.attr.getAttrValue(attr, "loff", np.nan)
    if lres == 0:
        lfac, _, _ = dpg.attr.getAttrValue(attr, "lfac", 0.0)
        if lfac != 0:
            lres = 65536000000.0 / (7.2 * lfac)

    par = [coff, cres, loff, lres]

    hres, _, _ = dpg.attr.getAttrValue(attr, "hres", 0.0)
    hoff, exists, _ = dpg.attr.getAttrValue(attr, "hoff", 0.0)
    if exists:
        par = par + [hoff, hres]

    rangeres, exists, _ = dpg.attr.getAttrValue(attr, "rangeres", 0.0)
    if exists:
        rangeres_tmp, exists, _ = dpg.attr.getAttrValue(
            attr, "polres", 0.0
        )
        if exists:
            rangeres = rangeres_tmp

    rangeoff, _, _ = dpg.attr.getAttrValue(attr, "rangeoff", 0.0)
    rangeoff, _, _ = dpg.attr.getAttrValue(attr, "poloff", rangeoff)

    if rangeres > 0:
        if len(par) == 4:
            par = par + [0.0, 0.0]
        par = par + [rangeoff, rangeres]

        azres, exists, _ = dpg.attr.getAttrValue(attr, "azres", 0.0)
        azoff, _, _ = dpg.attr.getAttrValue(attr, "azoff", 0.0)
        if exists:
            par = par + [azoff, azres]
            el_coords, eloff, elres, _, attr, exists = get_el_coords(
                node, get_exists=True
            )
            if exists:
                par = par + [eloff, elres]

    stdPar1, _, _ = dpg.attr.getAttrValue(attr, "stdPar1", 0.0)
    stdPar2, _, _ = dpg.attr.getAttrValue(attr, "stdPar2", 0.0)
    stdPar = [stdPar1, stdPar2]
    satHeight, _, _ = dpg.attr.getAttrValue(attr, "satHeight", 36000000)
    tmp, _, _ = dpg.attr.getAttrValue(attr, "zone", 0)

    if tmp > 0:
        zone = tmp

    dim2 = dim

    isotropic, ispolar, isvertical, par, xres, yres, box, projection, dim = (
        check_proj_par(
            attr, projection=projection, p0lat=p0lat, p0lon=p0lon, par=par, dim=dim
        )
    )

    hRange = dpg.map.get_alt_range(par, dim2)

    return box, np.array(par, dtype=np.float32), dim, p0lat, p0lon, projection, hoff, hres


def get_dem_id(hr: bool = False):
    """
    Retrieves the DEM root node, creating it if necessary

    This function checks if the DEM root node is already initialized in the global state.
    If not, it creates the DEM tree based on the specified high resolution (hr) flag.
    The function returns the appropriate DEM root node, either high resolution or standard.

    Args:
        hr (bool, optional): Flag to indicate if the high resolution DEM is required. Defaults to False

    Returns:
        Node:                The root node of the DEM tree, either high resolution or standard

    Note:
        - If 'hr' is True, it checks or creates the high resolution DEM root node
        - If 'hr' is False, it checks or creates the standard resolution DEM root node
        - The function uses the global state to store and retrieve the root nodes for efficiency
    """
    if hr:
        if not isinstance(dpg.globalVar.GlobalState.DEM_HROOT, dpg.node__define.Node):
            hroot = dpg.tree.createTree(dpg.cfg.getDEMPath(hr=True))
            dpg.globalVar.GlobalState.update("DEM_HROOT", hroot)
        else:
            hroot = dpg.globalVar.GlobalState.DEM_HROOT
        return hroot

    if not isinstance(dpg.globalVar.GlobalState.DEM_ROOT, dpg.node__define.Node):
        root = dpg.tree.createTree(dpg.cfg.getDEMPath())
        dpg.globalVar.GlobalState.update("DEM_ROOT", root)
    else:
        root = dpg.globalVar.GlobalState.DEM_ROOT

    return root


def set_radar_par(
        nBins=np.array([]),
        node=None,
        site_coords=None,
        site_name=None,
        range_off=None,
        range_res=None,
        azimut_off=None,
        azimut_res=None,
        elevation_off=None,
        elevation_res=None,
        cartesian=False,
        projection=None,
        h_off=None,
        h_res=None,
        az_coords=None,
        el_coords=None,
        map=None,
        remove_coords=None,
        par=np.array([]),
):
    """
    Set radar parameters for processing and mapping.

    Args:
        nBins (int or list of int): Number of bins in the radar data.
        node (object): Radar node to which parameters are applied.
        site_coords (np.ndarray or list of float): Coordinates of the radar site [longitude, latitude].
        site_name (str): Name of the radar site.
        range_off (float or list of float): Offset for range measurement.
        range_res (float or list of float): Resolution for range measurement.
        azimut_off (float or list of float): Offset for azimuth measurement.
        azimut_res (float or list of float): Resolution for azimuth measurement.
        elevation_off (float or list of float): Offset for elevation measurement.
        elevation_res (float or list of float): Resolution for elevation measurement.
        cartesian (bool): Flag indicating if Cartesian coordinates are used.
        projection (str): Projection type for mapping.
        h_off (float or list of float): Height offset.
        h_res (float or list of float): Height resolution.
        az_coords (float or list of float): Azimuth coordinates.
        el_coords (float or list of float): Elevation coordinates.
        map (object): Map object for projection.
        remove_coords (bool): Flag to indicate if coordinates should be removed.
        par (list of float): Additional parameters.

    Returns:
        None
    """

    outPar = np.zeros(12)
    nP = len(par)
    if nP > 0:
        outPar[:nP] = par[:nP]  # o forse outPar[0:nP] = par
    else:
        # if n_elements(map) eq 0 and n_elements(site_coords) ge 2 then begin
        if map is None and np.size(site_coords) >= 2:
            if projection is None:
                projection = dpg.cfg.getDefaultProjection()
            map = dpg.map.findSharedMap(
                projection=projection,
                p0lat=site_coords[0],
                p0lon=site_coords[1],
            )
        if len(nBins) == 1:
            if nBins > 0:
                res = [range_res, range_res]
                dim = [nBins * 2, nBins * 2]
                locpar = dpg.map.get_proj_par_from_local_res(
                    map, res, site_coords, dim, isotropic=True
                )
                outPar[0: len(locpar)] = locpar
    last = 3
    if h_off is not None:
        last = 4
        if isinstance(h_off, list):
            if len(h_off) == 1:
                outPar[last] = h_off[0]
        else:
            outPar[last] = h_off
    if h_res is not None:
        last = 5
        if isinstance(h_res, list):
            if len(h_res) == 1:
                outPar[last] = h_res[0]
        else:
            outPar[last] = h_res
    if not cartesian:
        if range_off is not None:
            last = 6
            if isinstance(range_off, list):
                if len(range_off) == 1:
                    outPar[last] = range_off[0]
            else:
                outPar[last] = range_off
        if range_res is not None:
            last = 7
            if isinstance(range_res, list):
                if len(range_res) == 1:
                    outPar[last] = range_res[0]
            else:
                outPar[last] = range_res
        if azimut_off is not None:
            last = 8
            if isinstance(azimut_off, list):
                if len(azimut_off) == 1:
                    outPar[last] = azimut_off[0]
            else:
                outPar[last] = azimut_off
        if az_coords is not None:
            last = 8
            if isinstance(az_coords, list):
                if len(az_coords) == 1:
                    outPar[last] = az_coords[0]
            else:
                outPar[last] = az_coords
        if azimut_res is not None:
            last = 9
            if isinstance(azimut_res, list):
                if len(azimut_res) == 1:
                    outPar[last] = azimut_res[0]
            else:
                outPar[last] = azimut_res
        if elevation_off is not None:
            last = 10
            if isinstance(elevation_off, list):
                if len(elevation_off) == 1:
                    outPar[last] = elevation_off[0]
            else:
                outPar[last] = elevation_off
        if el_coords is not None:
            last = 10
            if isinstance(el_coords, list):
                if len(el_coords) == 1:
                    outPar[last] = el_coords[0]
            else:
                outPar[last] = el_coords
        if elevation_res is not None:
            last = 11
            if isinstance(elevation_res, list):
                if len(elevation_res) == 1:
                    outPar[last] = elevation_res[0]
            else:
                outPar[last] = elevation_res
        if el_coords is not None:
            if isinstance(el_coords, list):
                if len(el_coords) > 1:
                    last = 9
    par = outPar[0:last+1]
    if node is None:
        return
    set_geo_info(node, par=par, site_coords=site_coords, site_name=site_name)
    if remove_coords:
        remove_az_coords(node)
        remove_el_coords(node)
    if az_coords is not None:
        if isinstance(az_coords, list) and len(az_coords) > 1:
            set_az_coords(node, az_coords)
    if el_coords is not None:
        if isinstance(el_coords, list) and len(el_coords) > 1:
            set_el_coords(node, el_coords)


def set_az_coords(node, az_coords: list):
    # TODO: debuggare per testare di che tipo sono i parametri in ingresso al metodo
    """
    Set azimuth coordinates for a specific node passed to the methos.

    Args:
        node(Node): Node for which set the elevation coordinates.
        az_coords: List of azimuth coordinates.
    """

    name = dpg.cfg.getGeoDescName()
    if len(az_coords) == 1:
        _ = dpg.tree.replaceAttrValues(
            node, name, "azimut", az_coords, only_current=True
        )
        return
    attr = dpg.tree.getSingleAttr(node, name, only_current=True)
    filename, _, _ = dpg.attr.getAttrValue(attr, "azfile", "azimut.txt")
    ptr = az_coords
    _ = dpg.tree.addAttr(node, filename, ptr, format="VAL")
    _ = dpg.tree.replaceAttrValues(node, name, "azFile", filename, only_current=True)


def set_el_coords(node, el_coords: list):
    # TODO: debuggare per testare di che tipo sono i parametri in ingresso al metodo
    """
    Set elevation coordinates for a specific node passed to the methos.

    Args:
        node(Node): Node for which set the elevation coordinates.
        el_coords: List of elevetion coordinates.
    """

    name = dpg.cfg.getGeoDescName()
    if len(el_coords) == 1:
        remove_el_coords(node)
        _ = dpg.tree.replaceAttrValues(
            node, name, "eloff", el_coords, only_current=True
        )
        return

    attr = dpg.tree.getSingleAttr(node, name, only_current=True)
    filename, _, _ = dpg.attr.getAttrValue(attr, "elfile", "elevation.txt")
    if isinstance(el_coords, list):
        el_coords = np.array(el_coords)
    ptr = el_coords.copy()
    _ = dpg.tree.addAttr(node, filename, ptr, format="VAL")
    _ = dpg.tree.replaceAttrValues(node, name, "elfile", filename, only_current=True)

    return


def get_site_coords(node, attr=None, az_coords=None, el_coords=None):
    if attr is None:
        attr = get_idnavigation(node)

    lat = dpg.attr.getAttrValue(attr, 'orig_lat', 0.)
    lon = dpg.attr.getAttrValue(attr, 'orig_lon', 0.)
    alt = dpg.attr.getAttrValue(attr, 'orig_alt', 0.)
    origin = dpg.attr.getAttrValue(attr, 'origin', '')
    region = dpg.attr.getAttrValue(attr, 'region', '')

    site_coords = [lat, lon, alt]

    if az_coords is not None:
        az_coords, _, _ = get_az_coords(node, regular=True)
    if el_coords is not None:
        el_coords, _, _, _, _ = get_el_coords(node)

    return site_coords, az_coords, el_coords
