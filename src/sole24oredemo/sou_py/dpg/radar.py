import copy
from sys import prefix
import math
import numpy as np
import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION IDL_rv_get_elevations 
PRO IDL_rv_add_info 
PRO IDL_rv_set_mosaicked_node 
FUNCTION IDL_rv_find_par_attr       // UNUSED
FUNCTION IDL_rv_get_processed_sites // UNUSED
FUNCTION IDL_rv_get_site_nodes      // UNUSED
PRO CHECK_SMOOTH                    // UNUSED
PRO GetNeighboorStat                // UNUSED
PRO IDL_rv_copy_par                 // UNUSED
PRO SET_SAMPLED                     // UNUSED
"""


def set_raw_path(node, raw_path: list):
    """
    Sets the raw path parameter for a given node.

    This function assigns a raw path to a specified node. The raw path is set only if 'raw_path' contains
    exactly one element; otherwise, an empty string is assigned.

    Args:
        node (Node object): The node for which the raw path is to be set.
        raw_path (list[str]): A list containing the raw path. Only the first element is used if the list contains
        exactly one element.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    if raw_path is None:
        raw = ""
    else:
        raw = raw_path

    set_par(node, "rawPath", raw, only_current=True)


def set_par(
        node,
        name: str,
        val,
        parFile: str = None,
        only_current: bool = False,
        format: str = None,
        to_save: bool = False,
):
    """
    Sets a parameter for a given node, with options for persistence and formatting.

    This function assigns a value to a parameter specified by 'name' for the given 'node'. The function first
    retrieves existing attributes and then either updates an existing attribute or creates a new one. There are
    options to specify a parameter file, to apply the change only to the current instance, to format the value,
    and to save the changes for persistence.

    Args:
        node (Node object): The node for which the parameter is to be set.
        name (str): The name of the parameter to set.
        val (any): The value to assign to the parameter.
        parFile (str, optional): The parameter file to use. Defaults to None.
        only_current (bool, optional): If True, changes apply only to the current instance of the node. Defaults to
        False.
        format (str, optional): The format to use for the value. Defaults to None.
        to_save (bool, optional): If True, saves the attribute for persistence. Defaults to False.

    Returns:
        Attr object or None: The attribute object after setting the parameter, or None if no attribute was set.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'get_par_attr', 'replaceTags',
              and 'createAttr' methods from 'dpg' module.
    """
    attr, parFile = get_par_attr(node, parFile=parFile, only_current=only_current)

    if attr is not None:
        attr = dpg.attr.replaceTags(attr, name, val)  # TODO, FORMAT=format)
    else:
        attr = dpg.tree.createAttr(
            node=node, name=parFile, varnames=name, values=val, format=format
        )

    if to_save and isinstance(attr, dpg.attr__define.Attr):
        attr.save()

    return attr


def get_par_attr(node, parFile: str = "", only_current: bool = True):
    """
    Retrieves the attribute of a parameter for a given node.

    This function fetches the attribute associated with a parameter for a specified node. It ensures that the
    node is of the correct type and that a parameter file is specified. If no parameter file is specified,
    it defaults to using the parameter description name from the configuration.

    Args:
        node (Node object): The node for which the attribute is to be retrieved.
        parFile (str, optional): The parameter file to use. Defaults to an empty string, which will use the
                                 default parameter description name from the configuration.
        only_current (bool, optional): If True, retrieves the attribute only for the current instance of the node.
        Defaults to True.

    Returns:
        Attr object or None: The attribute object associated with the parameter, or None if the node is not of the
        correct type or no attribute is found.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'getSingleAttr' method from
        'dpg.tree'.
    """
    """
    Parametro ind rimosso perché non usato
    Parametro reload rimosso perché non usato
    Parametro oAttr rimosso perché restituito come output
    #TODO owner deve essere restituito, non preso come input
    """

    if not isinstance(node, dpg.node__define.Node):
        log_message("Error: node is not Node type", level="WARNING")
        return None
    if parFile is None or parFile == "":
        parFile = dpg.cfg.getParDescName()

    attr = dpg.tree.getSingleAttr(
        node, parFile, only_current=only_current, format="txt"
    )
    return attr, parFile


def get_par(
        node,
        name: str,
        default,
        prefix: str = "",
        parFile: str = "",
        attr: Attr = None,
        only_with_prefix: bool = False,
):
    """
    Retrieves the value of a parameter for a given node, with options for prefixes and default values.

    This function searches for a parameter specified by 'name' in the attributes of a given node or its
    ancestors. It can also handle a specific attribute directly if provided. The function allows for
    specification of a prefix, and can return the prefix along with the parameter value. If the parameter is
    not found, a default value is returned.

    Args:
        node (Node object): The node for which the parameter value is to be retrieved.
        name (str): The name of the parameter.
        default (any): The default value to return if the parameter is not found.
        prefix (str, optional): A prefix to apply to the parameter name. Defaults to an empty string.
        get_prefix (bool, optional): If True, returns the used prefix along with the parameter value. Defaults to False.
        parFile (str, optional): The parameter file to use. Defaults to an empty string.
        attr (Attr object, optional): A specific attribute to use. Defaults to None.
        only_with_prefix (bool, optional): If True, considers only parameters with the specified prefix. Defaults to
        False.

    Returns:
        any (or tuple): The value of the parameter, or a tuple of the value and the used prefix if 'get_prefix' is True.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'get_par_attr' and 'getAttrValue'
        methods.
    """

    exists = 0
    if type(default) == float:
        default = np.float32(default)

    if isinstance(node, dpg.node__define.Node):
        curr = node
    else:
        # log_message('Error: node is not Node type', level='WARNING')
        curr = None

    if prefix == "" or prefix is None:  # and get_prefix:
        prefix = dpg.navigation.get_site_name(node)

    if isinstance(attr, dpg.attr__define.Attr):
        val, exists, _ = dpg.attr.getAttrValue(
            attr, name, default, prefix=prefix, only_with_prefix=only_with_prefix
        )
        return val, prefix, exists

    while isinstance(curr, dpg.node__define.Node):
        currattr, _ = get_par_attr(
            curr, parFile=parFile
        )  # TODO dovrebbe restituire owner?
        val, exists, _ = dpg.attr.getAttrValue(
            currattr,
            name,
            default,
            prefix=prefix,
            only_with_prefix=only_with_prefix,
        )
        if exists:
            return val, prefix, exists

        curr = curr.parent
    val = default
    if type(val) == float:
        val = np.float32(val)

    return val, prefix, exists


def sort_attr(attr_list: list, prefix: bool = False):
    """
    Sorts a list of attributes based on their priority values.

    This function sorts a given list of attributes ('attr_list') based on their 'priority' values. If the 'prefix'
    flag is set, it uses the prefixed version of the 'priority' attribute. The function returns the indexes of the
    sorted attributes, allowing the original list to be reordered accordingly.

    Args:
        attr_list (list[Attr objects]): A list of attribute objects to be sorted.
        prefix (bool, optional): If True, uses the prefixed version of the 'priority' attribute for sorting. Defaults
        to False.

    Returns:
        list[int] or int: A list of indexes representing the sorted order of the attributes. Returns 0 if the
        'attr_list' contains one or no items.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the 'getAttrValue' method from
        'dpg.attr'.
    """
    if len(attr_list) == 1:
        return [0]

    nP = len(attr_list)
    if nP <= 1:
        return 0
    priority = []
    for i, attr in enumerate(attr_list):
        priority.append(dpg.attr.getAttrValue(attr, "priority", 10, prefix=prefix)[0])

    sorted_indexes = sorted(
        range(len(priority)), key=lambda k: priority[k], reverse=False
    )
    sorted_priority = sorted(priority, reverse=False)

    return sorted_indexes


def find_site_node(tree, name: str, origin: str = ""):
    """
    Searches for nodes with a specific name within a tree structure, optionally starting from a specified origin node.

    This function locates all nodes with the given 'name' in a tree structure ('tree'). If an 'origin' is specified,
    the search is confined to the descendants of the 'origin' node. If no 'origin' is provided, the search covers the
    entire tree.

    Args:
        tree (Tree object): The tree in which to search for nodes.
        name (str): The name of the nodes to search for.
        origin (str, optional): The name of the origin node from which to start the search. If not provided or empty,
                                the search starts from the root of the tree. Defaults to an empty string.

    Returns:
        list[Node objects] or None: A list of nodes matching the specified 'name'. Returns None if no matching nodes
        are found or if 'origin' is invalid.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on the 'findAllDescendant' method
        from 'dpg.tree'.
    """
    if origin is not None and origin != "":
        tmp = dpg.tree.findAllDescendant(tree, origin)
        if tmp is None and len(tmp) != 1:
            # LogMessage, 'Cannot find origin ' + origin
            return None
        pass
    else:
        tmp = copy.deepcopy(tree)
    nodes = dpg.tree.findAllDescendant(tmp, name)
    return nodes


def checkParAndDim(outId, par: list = None, dim: list = None):
    """
    Validates and adjusts parameter and dimension values based on a given node.

    This function checks and potentially adjusts parameters ('par') and dimensions ('dim') for a given output
    node ('outId'). It ensures the node is of the correct type and that dimensions are adequately specified.
    The function fetches and adjusts various parameters such as range, range resolution, and azimuth resolution
    based on values associated with the output node.

    Args:
        outId (Node object): The output node for which parameters and dimensions are to be validated and adjusted.
        par (list, optional): The list of parameters to be checked and potentially adjusted. Defaults to None.
        dim (list, optional): The dimensions to be validated. Defaults to None.

    Returns:
        tuple: A tuple containing two elements:
            - par (list or None): The potentially adjusted parameters.
            - dim (list or None): The potentially adjusted dimensions.
              Returns (None, None) if 'outId' is not a Node type or if dimensions are inadequately specified.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on various parameter retrieval and
        setting functions.
    """
    if not isinstance(outId, dpg.node__define.Node):
        return par, dim
    if dim is None or np.size(dim) < 2:
        return par, dim

    if len(dim) == 3:
        nlines_ind = 1
        ncols_ind = 2
    else:
        nlines_ind = 0
        ncols_ind = 1

    siteName = dpg.navigation.get_site_name(outId)
    range, _, _ = get_par(outId, "range", 0.0, prefix=siteName)
    rangeRes, _, _ = get_par(outId, "rangeRes", 0.0, prefix=siteName)
    rangeOff, _, _ = get_par(outId, "range_off", 0.0, prefix=siteName)
    # format='(F4.1)'

    if range <= 0.0 and rangeRes > 0.0 and len(par) > 7:
        range = dim[ncols_ind] * par[7] / 1000.0

    if range > 0.0 and rangeRes > 0.0 and len(par) > 7:
        if par[7] <= rangeRes:
            par[7] = rangeRes
        else:
            rangeRes = par[7]
        ddd = int((range + 0.5) * 1000 / rangeRes)
        if dim[ncols_ind] <= 0 or dim[ncols_ind] >= ddd:
            dim[ncols_ind] = ddd
        par[6] = rangeOff * 1000

    azRes, _, _ = get_par(outId, "azRes", 1.0, prefix=siteName)

    if azRes == 0.0:
        azRes = par[9]
    azRes = abs(azRes)
    if azRes > 0.0 and len(par) > 9:
        az_len, _, _ = get_par(outId, "az_len", 360.0, prefix=siteName)
        az_off, _, _ = get_par(outId, "az_off", 0.0, prefix=siteName)
        if az_len > 1.0:
            dim[nlines_ind] = int(az_len / azRes)
            par[8] = az_off
            par[9] = azRes
        _ = dpg.navigation.remove_az_coords(outId)

    return par, dim


def selectScan(
        scans,
        delta: float,
        elevation: float = None,
        min_flag: bool = False,
        max_flag: bool = False,
        index: int = None,
):
    """
    Selects a scan from a list based on specified criteria related to elevation.

    This function processes a list of scans and selects one based on elevation criteria. It uses the elevation
    offset of each scan for selection. The function can select the scan with the maximum or minimum elevation,
    the one closest to a specified elevation, or a specific scan based on its index.

    Args:
        scans (list or object): A list of scan objects or a single scan object.
        delta (float): The tolerance for elevation difference. Used when neither max nor min flags are set.
        elevation (float, optional): The target elevation to compare against. Defaults to None.
        min_flag (bool, optional): If True, selects the scan with the minimum elevation. Defaults to False.
        max_flag (bool, optional): If True, selects the scan with the maximum elevation. Defaults to False.
        index (int, optional): The index of a specific scan to select. Defaults to None.

    Returns:
        tuple: A tuple containing three elements:
            - The selected scan object or None if no suitable scan is found.
            - The nominal elevation of the selected scan or None if no scan is selected.
            - The actual elevation used for selection.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'get_radar_par' from
        'dpg.navigation'.
    """

    if isinstance(scans, list):
        nScans = len(scans)
    elif scans is not None:
        nScans = 1
        scans = [scans]
    else:
        nScans = 0

    if nScans == 0:
        return None, None, elevation

    diff = [None] * nScans
    nominal = [None] * nScans

    for sss in range(len(nScans)):
        radar_dict = dpg.navigation.get_radar_par(scans[sss])
        curr = radar_dict["elevation_off"]  # TODO output ancora non gestito
        if curr is not None:
            if index is not None:
                if sss == index:
                    elevation = curr
                    nominal = curr
                    return scans[sss], nominal, elevation
            nominal[sss] = curr
            if elevation is not None:
                if math.isfinite(elevation):
                    diff[sss] = abs(elevation - curr)
    # endfor
    if max_flag:
        minEl = np.amax(nominal)
        ind = np.argmax(nominal)
        elevation = minEl
    else:
        if min_flag:
            minEl = np.amin(nominal)
            ind = np.argmin(nominal)
            elevation = minEl
        else:
            minEl = np.amin(diff)
            ind = np.argmin(diff)
    # endelse
    if not math.isfinite(elevation):
        return scans, nominal, elevation
    nominal = nominal[ind]
    if delta is None or delta <= 0.0 or minEl <= delta:
        return scans[ind], nominal, elevation
    return None, None, elevation


def set_corrected(node, prodId: str = None, sampled: bool = False):
    """
    Marks a node as corrected and updates its attributes based on specified conditions.

    This function sets the 'corrected' attribute for a given node. Additionally, it modifies the attributes
    of the node based on the 'create_check' parameter associated with a specified product ID ('prodId').
    The function handles both sampled and non-sampled cases.

    Args:
        node (Node object):         The node to be marked as corrected.
        prodId (str, optional):     The product ID used to retrieve the 'create_check' parameter. Defaults to None.
        sampled (bool, optional):   Indicates if the node is sampled. Defaults to False.

    Raises:
        None:                       This function does not explicitly raise any exceptions but relies
                                    on 'set_array_info', 'get_par', 'replaceAttrValues',
                                    and 'getRoot' methods from 'dpg' modules.
    """
    if not sampled:
        dpg.array.set_array_info(node, corrected=1)

    create_check, _, _ = get_par(prodId, "create_check", "")
    if create_check == "":
        return

    # help, calls=info #TODO chiedere a Mimmo
    info = None
    proc = info[1]
    pos = proc.find(" ")
    if pos > -1:  # substringa trovata
        proc = proc[:pos]

    _ = dpg.tree.replaceAttrValues(
        dpg.tree.getRoot(node), create_check, proc, "1", to_add=True, to_save=True
    )


def check_in(
        node_in,
        node_out,
        type=None,
        force_type: bool = False,
        dim=None,
        filename: str = "",
        pointer=None,
        values=None,
        no_values: bool = False,
        par=None,
        no_par=None,
):
    """

    Retrieves information from one node and copies it to another, with optional modifications.

    This function retrieves array data and metadata from a source node (`node_in`) and copies it
    to a destination node (`node_out`). It can modify the type and dimensions of the array,
    optionally force the type, handle file associations, and manage other array parameters.

    Args:
        node_in (dpg.node__define.Node): The node from which the information is to be retrieved.
        node_out (dpg.node_define.Node): The node to which the information is to be copied.
        type (int, optional): The type of the array elements. If not specified, it uses the type from `node_in`.
        force_type (bool, optional): Whether to force the array type to the specified type. Defaults to False.
        dim (list, optional): The dimensions of the array. If not specified, it uses the dimensions from `node_in`.
        filename (str, optional): The filename associated with the array. Defaults to an empty string.
        pointer (np.ndarray, optional): A pointer to an array, typically used for setting data without copying.
                                        Defaults to None.
        values (optional): Parameter not used.
        no_values (bool, optional): If True, array values will not be copied. Defaults to False.
        par: Parameter not used.
        no_par: Parameter not used.

    Returns:
        tuple: A tuple containing:
            - pointer (np.ndarray): The pointer to the array in the destination node.
            - par (list of float): A list of parameters related to the geographic information.
    """
    data, array_dict = dpg.array.get_array(node_in)
    arrayType = array_dict["type"]
    inDim = array_dict["dim"]
    bitplanes = array_dict["bitplanes"]
    exists = data is not None
    dpg.navigation.copy_geo_info(node_in, node_out, only_if_not_exists=True)
    _, par, _, _, _, _, _, _ = dpg.navigation.get_geo_info(node_in)
    dpg.array.copy_array_info(node_in, node_out, only_if_not_exists=True)

    if not no_values:
        if not exists:
            to_not_create = 1
        else:
            to_not_create = arrayType == 4 or arrayType == 5
        dpg.values.copy_values_info(
            node_in, node_out, only_if_not_exists=True, to_not_create=to_not_create
        )
        values, _, _ = dpg.calibration.get_array_values(
            node_in, to_not_create=to_not_create
        )

    if filename is not None and filename != "":
        dpg.array.set_array_info(node_out, filename=filename)
    par, dim = checkParAndDim(node_out, par=par, dim=dim)

    if type is not None:
        if exists:
            if type == 1 and not force_type:
                type = arrayType
        if dim is not None:
            if len(dim) > 0:
                if sum(dim) <= 0:
                    dim = inDim

        pointer = dpg.array.create_array(
            node=node_out, dtype=type, dim=dim, pointer=pointer
        )
    else:
        if np.size(inDim) < 2:
            return None, None, None
        if inDim[1] < 2:
            return None, None, None
        pointer = dpg.array.copy_array(node_in, node_out)

    dpg.times.set_date(node_out, node_in)
    dpg.navigation.set_geo_info(node_out, par=par)

    return pointer, par, values


def check_out(
        outId=None,
        pointer: np.ndarray = np.array(()),
        par: np.ndarray = np.array(()),
        el_coords: np.ndarray = np.array(()),
        filename: str = None,
        smoothBox=None,
        attr=None,
        medianFilter=None,
        reset_values: bool = False,
        values: np.ndarray = np.array(()),
        site_name: str = None,
        site_coords: np.ndarray = np.array(()),
        projection=None,
        p0lat=None,
        p0lon=None,
):
    """
    Performs a series of operations on an output array based on provided parameters

    Args:
        outId (optional):                   The ID of the output array
        pointer (np.ndarray, optional):     The array data to be processed. Defaults to an empty array
        par (np.ndarray, optional):         Parameters for processing. Defaults to an empty array
        el_coords (np.ndarray, optional):   Element coordinates. Defaults to an empty array
        filename (str, optional):           The filename for setting array information. Defaults to None
        smoothBox (optional):               Parameter for smoothing the array. Defaults to None
        attr (optional):                    Attributes for calibration. Defaults to None
        medianFilter (optional):            Parameter for median filtering the array. Defaults to None
        reset_values (bool, optional):      Flag to reset values. Defaults to False
        values (np.ndarray, optional):      Values for calibration. Defaults to an empty array
        site_name (str, optional):          Site name for setting geo information. Defaults to None
        site_coords (np.ndarray, optional): Site coordinates for setting geo information. Defaults to an empty array
        projection (optional):              Projection information for setting geo information. Defaults to None
        p0lat (optional):                   Latitude for setting geo information. Defaults to None
        p0lon (optional):                   Longitude for setting geo information. Defaults to None

    Returns:
        None
    """
    if filename is not None:
        dpg.array.set_array_info(outId, filename=filename)

    if smoothBox is None:
        smoothBox, _ = dpb.dpb.get_par(outId, "smooth", 0, attr=attr)
    if medianFilter is None:
        medianFilter, _ = dpb.dpb.get_par(outId, "medianFilter", 0, attr=attr)

    if smoothBox != 0:
        if pointer is None or len(pointer) == 0:
            pointer = dpg.array.get_array(outId)
        if pointer is not None and len(pointer) > 0 and isinstance(pointer, np.ndarray):
            dtype = dpg.io.type_py2idl(pointer.dtype)
            to_not_create = 0 if dtype == 4 else 1
            if not reset_values:
                if len(values) == 0:
                    values, calib, out_dict = dpg.calibration.get_array_values(
                        outId, to_not_create=to_not_create
                    )
                    scale = calib["scale"]
                else:
                    _, calib, out_dict = dpg.calibration.get_array_values(
                        outId, to_not_create=to_not_create
                    )
            if medianFilter > 0:
                opt = medianFilter
            toNotLinearize, _ = dpb.dpb.get_par(outId, "toNotLinearize", 0)
            lin = 1 - toNotLinearize
            dpg.prcs.smoothArray()

    dpg.array.set_array(outId, pointer=pointer)

    last = len(par)
    if last >= 3:
        if pointer is not None and isinstance(pointer, np.ndarray):
            dim = pointer.shape
        if len(dim) > 1:
            cov = par[1] * dim[-1]
            if cov == 0 and last > 6:
                cov = par[7] * dim[-1]
            cov /= 1000.0
            if cov < 0:
                cov = -cov
            if cov == 0:
                tmp = cov.copy()
        if len(el_coords) > 0 and last > 9:
            last = 9
        dpg.navigation.set_geo_info(
            outId,
            par=par[0:last],
            cov=cov,
            site_name=site_name,
            site_coords=site_coords,
            projection=projection,
            p0lat=p0lat,
            p0lon=p0lon,
        )
        dpg.navigation.remove_az_coords(outId)
        if len(el_coords) > 0:
            dpg.navigation.set_el_coords(outId, el_coords)
        else:
            dpg.navigation.remove_el_coords(outId)

    if attr is not None:
        dpg.calibration.set_values(outId, attr)

    return
