import copy
import multiprocessing
import os
import time

from unicodedata import numeric
import numpy as np
import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
PRO get_radar_par 
PRO GET_PPI 
PRO GET_RHI 
PRO WARP_DATA 
PRO PUT_GEO_INFO 
PRO DPB 
PRO GET_TAGS        //UNUSED
PRO COPY_LAST_NODE  //UNUSED
PRO GET_ETNA_STATUS //UNUSED
PRO PUT_TIMES       //UNUSED
PRO HDF_IMPORT      //UNUSED
"""


def get_var(node, name):
    """
    Retrieves and computes the mean of a specified variable from one or more nodes.

    This function takes a node or a list of nodes and a variable name, then retrieves
    the value of that variable from each node. It computes the mean of the variable
    values, considering only finite elements, and returns both the variable values and
    their mean. If the variable is not found in a node or if the node list is empty,
    the function returns NaN for the mean.

    Args:
        node (dpg.node__define.Node or list of dpg.node__define.Node): The node or list of
                                                                       nodes from which the
                                                                       variable is to be
                                                                       retrieved.
        name (str): The name of the variable to retrieve and compute the mean of.

    Returns:
        1. var (np.ndarray): The numpy array of the variable values retrieved from
                            the nodes.
        2. mean (list): A list of mean values of the variable from each node. If
                               no valid values are found, the list contains [np.nan].

    Note:
        The function assumes that the input 'node' is either a single node or a list of
        nodes of type 'dpg.node__define.Node'. If a single node is provided, it is
        converted to a list. The function uses 'dpg.attr__define.Attr' to access the
        attribute corresponding to 'name' in each node. NaN is used to handle cases where
        the variable is not present or the list of nodes is empty.
    """
    mean = []
    var = None

    if isinstance(node, dpg.node__define.Node):
        node = [node]

    for nnn in node:
        cMean = np.nan
        pVar = nnn.getAttr(name, format="ASCII")

        if isinstance(pVar, list):
            pVar = pVar[0]
        if isinstance(pVar, dpg.attr__define.Attr):
            var = np.float32(pVar.pointer)
            ind = np.isfinite(var)
            cMean = np.mean(var[ind])
        # endif
        mean.append(cMean)
    # endfor

    if len(mean) == 0:
        mean = [np.nan]

    return var, mean


def get_data(
        node,
        name="",
        numeric=None,
        linear=False,
        schedule=None,
        date=False,
        n_hours=None,
        warp=False,
        mosaic=False,
        regular=False,
        maximize=False,
        silent=False,
        path="",
        interactive=False,
        site_name=None,
        aux=False,
        remove=False,
        noRemove=False,
        getMinMaxVal=False,
        main=False,
        time=None,
):
    """
    Retrieves data from the specified node with various optional processing and retrieval options.

    Args:
        node (Node): The node from which to retrieve the information.
        name (str, optional): The name of the data to retrieve. Defaults to ''.
        numeric (bool, optional): Flag to specify if the data should be numeric. Defaults to None.
        linear (bool, optional): Flag to indicate if the data should be linearized. Defaults to False.
        schedule (Any, optional): Schedule used for data retrieval. Defaults to None.
        date (bool, optional): Flag to specify if the date should be used in retrieval. Defaults to False.
        n_hours (int, optional): Number of hours to consider for data retrieval. Defaults to None.
        warp (bool, optional): Flag to indicate if the data should be warped. Defaults to False.
        mosaic (bool, optional): Flag to indicate if the data should be a mosaic. Defaults to False.
        regular (bool, optional): Flag to specify if regular processing should be applied. Defaults to False.
        maximize (bool, optional): Flag to indicate if the data should be maximized. Defaults to False.
        silent (bool, optional): Flag to suppress output messages. Defaults to False.
        path (str, optional): The path to the data tree. Defaults to ''.
        interactive (bool, optional): Flag to indicate if interactive mode is enabled. Defaults to False.
        site_name (str, optional): The site name to use for data retrieval. Defaults to None.
        aux (bool, optional): Flag to indicate if auxiliary data should be included. Defaults to False.
        remove (bool, optional): Flag to remove the data tree after processing. Defaults to False.
        noRemove (bool, optional): Flag to prevent the removal of the data tree. Defaults to False.
        getMinMaxVal (bool, optional): Flag to return the minimum and maximum values. Defaults to False.
        main (Node, optional): Main node. Defaults to False.
        time (Any, optional): Time information. Defaults to None.

    Returns:
        var (np.ndarray): The retrieved data array.
        Optional[Tuple[np.ndarray, float, float]]: If getMinMaxVal is True, returns a tuple of the data array,
                                            minimum value, and maximum value.
    """

    var = np.array([0], dtype=np.float32)
    tree = None
    son = None
    if isinstance(node, dpg.node__define.Node):
        son = node
    if schedule is not None:
        if n_hours is None:
            n_hours = 1
        if date is None:
            date, time, _ = dpg.times.get_time(node)
        ret, prod_path = dpg.access.get_aux_products(
            node,
            schedule=schedule,
            site_name=site_name,
            interactive=interactive,
            last_date=date,
            last_time=time,
            n_hours=n_hours,
        )
        if ret > 0:
            log_message(f"Found data @ {prod_path[0]}")
        if ret <= 0:
            return np.zeros(1)
        path = prod_path[0]

    if path is not None and path != "":
        tree = dpg.tree.createTree(path)
        if site_name is not None:
            son = dpg.radar.find_site_node(tree, site_name)
        else:
            sons = tree.getSons()
            if len(sons) != 1:
                son = tree

    if name is not None and name != "":
        sons = dpg.tree.findAllDescendant(son, name)
        if sons is None:
            sons = dpg.tree.findAllDescendant(son.parent, name)
            if sons is None:
                return None
        son = sons[0]
        if isinstance(sons, list):
            son = sons[0]
        else:
            son = sons
    if not isinstance(son, dpg.node__define.Node):
        return var

    if numeric is None:
        numeric = True

    if warp or mosaic:
        if not silent:
            log_message(f"Using node {son.path}")
        if maximize:
            pointer, _, _ = get_pointer(son, aux)
            if pointer is not None:
                pointer = dpg.prcs.maximize_data(pointer, maximize)
                # TODO: da controllare che effettivamente il
                #  pointer che torna la maximize_data sia uguale a quello dentro il son (i.e. son.pointer e pointer
                #  devono essere uguali per riferimento)
        var = dpg.warp.warp_map(
            son,
            node,
            numeric=numeric,
            linear=linear,
            mosaic=mosaic,
            aux=aux,
            regular=regular,
            remove=remove,
        )

        dpg.tree.removeTree(tree)
        return var

    pointer, out_dict = dpg.array.get_array(son, aux=aux, silent=silent)

    if pointer is None:
        dpg.tree.removeTree(tree)
        return var

    var = pointer.copy()
    if numeric or linear:
        # get_array_values deve restituire values e scale
        values, calib, out_dict = dpg.calibration.get_array_values(son)
        scale = out_dict["scale"]
        # convertData deve restituire var
        var = dpg.calibration.convertData(var, values, linear=linear, scale=scale)
        if getMinMaxVal:
            if len(values) > 0:
                minVal = np.nanmin(values[np.isfinite(values)])
                maxVal = np.nanmax(values[np.isfinite(values)])

    if not silent:
        pass
        # TODO: log

    if main is not None:
        if isinstance(main, dpg.node__define.Node):
            noRemove = True
            main = son

    if not (noRemove):
        dpg.tree.removeTree(tree)

    if getMinMaxVal:
        return var, minVal, maxVal

    return var


def project_volume(node, volume, get_heights=False):
    """
    Projects radar volume data based on node parameters and optionally retrieves heights.

    This function processes radar volume data based on parameters obtained from a specified
    node. It handles the volume data differently depending on its dimensions. The function
    also has the option to return height data related to the radar volume. If `get_heights`
    is set to True, it calls another function to retrieve height information.

    Args:
        node: The node containing parameters for processing the radar volume data (position, resolution, degrees of
        elevation).
        volume (np.ndarray): A numpy array representing radar volume data. The array
                             can have 2 or 3 dimensions.
        get_heights (bool, optional): A flag indicating whether to return height data.
                                      If True, height data is returned. Defaults to False.

    Returns:
        np.ndarray or None: If `get_heights` is True, returns a numpy array of heights.
                            Otherwise, the function does not return a value.

    Note:
        The function first checks the dimensions of the input volume data. It then
        retrieves radar parameters such as range resolution and elevation coordinates
        from the node. These parameters are used to process the volume data. The function
        handles missing data by setting it to either 0 or NaN, based on the dtype of the
        volume. It also adjusts the volume data based on the range beam index, which is
        calculated using `dpg.beams.getRangeBeamIndex`.
    """

    dim = volume.shape
    if np.size(dim) < 2:
        return volume

    if np.size(dim) == 2:
        nScans = 1
    else:
        nScans = dim[0]

    noData = 0
    if dpg.io.type_py2idl(volume.dtype) == 4:
        noData = np.nan

    ret = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    range_res = ret["range_res"]
    range_off = ret["range_off"]
    el_coords = ret["el_coords"]

    nEl = np.size(el_coords)
    for sss in range(nScans):
        if sss < nEl:
            rngInd = dpg.beams.getRangeBeamIndex(
                inDim=dim[2],
                outDim=dim[2],
                inRes=range_res,
                outRes=range_res,
                range_off=range_off,
                in_el=el_coords[sss],
            )
            ind = np.where(rngInd < 0)
            for aaa in range(dim[1]):
                volume[sss, aaa, :] = volume[sss, aaa, rngInd.astype(int)]
                if np.size(ind) > 0:
                    volume[sss, aaa, ind] = noData

    if get_heights:
        heights = get_heights(node, projected=True)
        return heights
    return volume


def get_heights(node, projected=False):
    """
    Populate a 3D array heights using different heights of radar beams.

    Args:
        node (Node): The node from which the array data is to be retrieved.
        projected (bool): N.B. Parameter not used.

    Returns:
        np.ndarrays or None
    """
    array = None
    array, out_dict = dpg.array.get_array(node)
    dim = out_dict["dim"]
    if len(array) < 2:
        return None, None

    par = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    site_coords = par["site_coords"]
    range_res = par["range_res"]
    range_off = par["range_off"]
    el_coords = par["el_coords"]
    nEl = np.size(el_coords)
    if nEl <= 0:
        return None, None

    heights = np.zeros_like(array, dtype=np.float32)
    nAz = dim[1]

    for eee in range(nEl):
        hBeam = dpg.access.get_height_beams(
            el_coords[eee],
            dim[2],
            range_res,
            range_off=range_off,
            site_height=site_coords[2],
            projected=True,
        )

        for aaa in range(nAz):
            heights[eee, aaa, :] = hBeam

    return heights, el_coords


def get_pointer(node, aux=False, reload=False):
    """
    Retrieves data array, its dimensions, and file type from a given node.

    This function obtains a data array from a specified node along with associated metadata,
    including its dimensions and the file type. It can be configured to use auxiliary
    information and to reload data. If data is successfully retrieved, the function returns
    the data array, its dimensions, and file type. Otherwise, it logs an appropriate
    message and returns None for all three outputs.

    Args:
        node: The node from which the data array and metadata are to be retrieved.
        aux (bool, optional): A flag to determine whether to use auxiliary information.
                              Defaults to False.
        reload (bool, optional): A flag to determine whether to reload the data. Defaults
                                 to False.

    Returns:
        1. data (np.ndarray or None): The retrieved data array, or None if no data is found.
        2. dim: The dimensions of the retrieved data array, or None if no data is found.
        3. file_type: The file type of the data, or None if no data is found.

    Note:
        The function uses 'dpg.array.get_array' to retrieve the data array and metadata.
        If data is not found, the function logs a message.
    """
    data, out_dict = dpg.array.get_array(
        node, replace_quant=True, reload=reload, aux=aux
    )

    if data is not None and out_dict is not None:
        # log_message(f'Using node: {dpg.tree.getNodePath(node)}')
        dim = out_dict["dim"]
        file_type = out_dict["type"]
        return data, dim, file_type
    else:
        return None, None, None


# TODO controllare funzione
def get_corners(
        node, set_center=None, cartesian=None, format_=None, type_=None, corners=None
):
    """
    Retrieves the corner coordinates of a geographic area from a node or its child node.

    This function attempts to obtain the corner coordinates of a geographic area
    represented by the given node. If the first attempt is unsuccessful or the
    retrieved corners do not meet the expected criteria (having at least 8 elements),
    it tries to retrieve the corners from the first child of the node. The coordinates
    can be returned in various formats and types, and with different reference centers,
    based on the provided arguments.

    Args:
        node: The node from which to retrieve the corner coordinates (that contains the file navigation.txt).
        set_center (optional): Specifies the reference center for the coordinates.
                               Defaults to None.
        cartesian (bool, optional): Determines whether the coordinates should be in
                                    Cartesian format. Defaults to None.
        format_ (optional): The format in which the coordinates should be returned.
                            Defaults to None.
        type_ (optional): The type of coordinates to be retrieved. Defaults to None.
        corners (list, optional): A pre-existing list of corner coordinates. Defaults to None.
                                  List of 8 coordinates, geographical
                                  ([LL_lon, LL_lat, UL_lon, UL_lat, UR_lon, UR_lat, LR_lon, LR_lat])
                                  or cartesian
                                  ([LL_xxx, LL_yyy, UL_xxx, UL_yyy, UR_xxx, UR_yyy, LR_xxx, LR_yyy])

    Returns:
        None: The function does not explicitly return a value. It either successfully
              retrieves the corner coordinates or exits if the criteria are not met.

    Note:
        The function uses 'dpg.navigation.get_corners' to attempt to retrieve the
        corner coordinates. If the initial attempt does not yield at least 8 corner
        elements, it tries to retrieve the corners from the first child node of the
        given node using the same function. The function does not return the corners
        but is expected to modify them in place or handle them internally.
    """
    corners = dpg.navigation.get_corners(
        node, set_center=set_center, cartesian=cartesian, format_=format_, type_=type_
    )

    if corners and len(corners) >= 8:
        return

    sons = dpg.tree.getSons(node)
    if len(sons) != 1:
        return

    son = sons[0]
    corners = dpg.navigation.get_corners(
        son, set_center=set_center, cartesian=cartesian, format_=format_, type_=type_
    )


def get_date(node, sep=None, year_first=None, timesep=None, seconds=None, currsec=None):
    """
    Retrieves and processes the date and time from a given node, optionally converting it to seconds.

    This function fetches the date and time from a specified node, performs checks and
    formatting based on the provided arguments, and optionally converts the date and
    time to seconds since the epoch. If the 'currsec' argument is provided, the current
    time in seconds is also obtained. If no date is returned from the node, the function
    sets 'seconds' to NaN (if it's provided) and exits.

    Args:
        node: The node from which to retrieve the date and time (containing the file generic.txt).
        sep (str, optional): The separator to use in the date string. Defaults to None.
        year_first (bool, optional): Flag indicating if the year is the first element
                                     in the date string. Defaults to None.
        timesep (str, optional): The separator to use in the time string. Defaults to None.
        seconds (bool, optional): If True, the function converts the date and time to
                                  seconds. Defaults to None.
        currsec (bool, optional): If True, the function also fetches the current time in seconds
                                  (starting from 01-01-1970). Defaults to None.

    Returns:
        None: The function does not explicitly return a value. It may modify the 'seconds'
              variable if provided.

    Note:
        The function uses 'dpg.times.get_time' to fetch the date and time, and
        'dpg.times.checkDate' and 'dpg.times.checkTime' for formatting. If the 'seconds'
        argument is provided, 'dpg.times.convertDate' is used to convert the date and
        time to seconds. The function handles cases where no date is returned from the node.
    """

    # Get current seconds if currsec argument is present
    if currsec is not None:
        currsec = time.time()  # Import the time module at the beginning of your script

    # Get the time from the node
    date, time_, _ = dpg.times.get_time(node)

    # If no date is returned, set seconds to NaN and return
    if not date:
        if seconds is not None:
            seconds = float("NaN")
        return

    # Check the date and time
    date = dpg.times.checkDate(date, sep=sep, year_first=year_first)
    if timesep:
        time_, _, _ = dpg.times.checkTime(time_, sep=timesep)

    # Convert the date and possibly time to seconds
    if seconds is not None:
        seconds = dpg.times.convertDate(date, time=time_)

    return date, time_, seconds


def get_prev_node(node, min_step, date=None, time=None):
    """
    Retrieves the previous node based on a specified minimum step and optionally a date and time.

    This function finds and returns the node that precedes the given node by at least the
    specified minimum step. The search for the previous node can be further refined by
    specifying a date and/or time. The function utilizes the 'dpg.times.getPrevNode'
    method to perform this operation.

    Args:
        node (Node): The current node from which to find the previous node.
        min_step (int or float): The minimum step size to consider when finding the
                                 previous node.
        date (str, optional): The specific date to consider when finding the previous node.
                              Defaults to None.
        time (str, optional): The specific time to consider when finding the previous node.
                              Defaults to None.

    Returns:
        prevNode (Node): The previous node that meets the specified criteria, or None if no such node is found.

    Note:
        The function assumes that 'node' is part of a sequence where nodes are ordered in
        some manner related to the provided 'min_step', 'date', and 'time' arguments.
    """

    prevNode, _, _ = dpg.times.getPrevNode(node, min_step, date=date, time=time)
    return prevNode


def get_next_node(node, min_step, date=None, time=None):
    """
    Retrieves the next node based on a specified minimum step and optionally a date and time.

    This function finds and returns the node that follows the given node by at least the
    specified minimum step. The search for the next node can be further refined by
    specifying a date and/or time. The function utilizes the 'dpg.times.getPrevNode'
    method from the 'dpg.times' module, with the 'next' parameter set to True, to
    perform this operation.

    Args:
        node (Node): The current node from which to find the next node.
        min_step (int or float): The minimum step size to consider when finding the
                                 next node.
        date (str, optional): The specific date to consider when finding the next node.
                              Defaults to None.
        time (str, optional): The specific time to consider when finding the next node.
                              Defaults to None.

    Returns:
        nextNode (Node): The next node that meets the specified criteria, or None if no such node is found.

    Note:
        The function is similar in operation to 'get_prev_node' but searches forward
        instead of backward. It assumes that 'node' is part of a sequence where nodes
        are ordered in some manner related to the provided 'min_step', 'date', and 'time'
        arguments.
    """

    nextNode, _, _ = dpg.times.getPrevNode(node, min_step, date=date, time=time, next=True)
    return nextNode


def get_prev_var(node, min_step, name, var):
    """
    Retrieves a variable from the previous node based on a specified minimum step.

    This function identifies the node that precedes the given node by at least a specified
    minimum step and then retrieves a variable with a given name from that previous node.
    After retrieving the variable, it removes the previous node from the tree structure.

    Args:
        node: The current node from which to find the previous node.
        min_step (int or float): The minimum step size to consider when finding the
                                 previous node.
        name (str): The name of the variable to retrieve from the previous node.
        var: The variable to be retrieved and updated.

    Returns:
        The value of the specified variable from the previous node.

    Note:
        The function uses 'dpg.times.getPrevNode' to find the previous node and 'get_var'
        to retrieve the variable from that node. After the variable is retrieved,
        'dpg.tree.removeTree' is called to remove the previous node, ensuring that the
        tree structure remains up-to-date.
    """
    prevNode, _, _ = dpg.times.getPrevNode(node, min_step)
    var = get_var(prevNode, name)
    dpg.tree.removeTree(prevNode)
    return var


def get_prev_par(node, min_step, name, prefix=None, default=None):
    """
    Retrieves a parameter from the previous node based on a specified minimum step.

    This function identifies the node that precedes the given node by at least the
    specified minimum step and then retrieves a parameter with a given name from that
    previous node. After retrieving the parameter, it removes the previous node from
    the tree structure.

    Args:
        node: The current node from which to find the previous node.
        min_step (int or float): The minimum step size to consider when finding the
                                 previous node.
        name (str): The name of the parameter to retrieve from the previous node.
        prefix (str, optional): An optional prefix to be used when retrieving the
                                parameter. Defaults to None.
        default: The default value to return if the parameter is not found.
                 Defaults to None.

    Returns:
        The value of the specified parameter from the previous node, or the default
        value if the parameter is not found.

    Note:
        The function uses 'dpg.times.getPrevNode' to find the previous node and
        'get_par' to retrieve the parameter from that node. After the parameter is
        retrieved, 'dpg.tree.removeTree' is called to remove the previous node,
        ensuring that the tree structure remains up-to-date and no unnecessary nodes
        are left in memory.
    """
    prevNode, _, _ = dpg.times.getPrevNode(node, min_step)
    par, _, _ = get_par(prevNode, name, default=default, prefix=prefix)
    dpg.tree.removeTree(prevNode)
    return par


def get_par(
        nodes,
        name,
        default="",
        start_with=None,
        prefix=None,
        attr=None,
        par_file=None,
):
    """
    Procedure used to access the values of a series of parameters contained in the parameters.txt

    Args:
        nodes (list of Node or Node): Node where the parameters.txt file resides (or one of the descendant nodes)
        name (str): Parameter name to read
        default (str): Default value to assign to par if the name parameter is not found.
                The type of the par variable will be the same as the type of the default variable. Default to ''.
        start_with: N.B. parameter not used.
        prefix (str,optional): Any prefix of the 'name' parameter (by default it is set equal to the name of the
                               current site). generally used to distinguish some parameters depending on the site
                               (e.g. CAPOFIUME.moment = UZ).
        attr (Attr, optional): A specific attribute to use. Defaults to None.
        par_file (str,optional): Alternative file name to parameters.txt. Default to None.
        get_prefix (bool): If True, returns the used prefix along with the parameter value. Defaults to False.

    Returns:
        par (list of any): list of parameter requested.
    """

    # Initialize par with default if no elements are present

    if not isinstance(nodes, list):

        if isinstance(nodes, dpg.node__define.Node):
            par, prefix, _ = dpg.radar.get_par(
                nodes,
                name,
                default,
                prefix=prefix,
                attr=attr,
                parFile=par_file,
            )
            return par, prefix

    par = []

    if default:
        par = [default] * len(nodes)

    for node in nodes:
        val, _, _ = dpg.radar.get_par(
            node, name, default, prefix=prefix, attr=attr, parFile=par_file
        )  # TODO start_with=start_with
        par.append(val)

    # If no elements have been added to par, set it to default
    if not par:
        par = [default]

    return par, prefix


def put_data(
        node,
        var,
        no_copy=1,
        main=None,
        linear=False,
        no_null=False,
        smoothBox=None,
        medianFilter=None,
        as_is=None,
        attr=None,
        filename=None,
        name=None,
        aux=None,
        to_save=False,
):
    """
    Processes and assigns data to a node with optional parameters for smoothing,
    linearization, and file saving.

    Args:
        node (Node): The node to which data will be assigned.
        var (np.ndarray): The data array to be assigned to the node.
        no_copy (int, optional): Flag to indicate whether to copy the data array (default is 1).
        main (Node, optional): Main node to copy geo info and calibration from, if provided.
        linear (bool, optional): Flag to indicate whether to linearize the data array (default is False).
        no_null (bool, optional): Flag to replace null values in the data array with zeros (default is False).
        smoothBox (int, optional): Parameter for smoothing the data array (default is None).
        medianFilter (Any, optional): Parameter for median filtering the data array (default is None).
        as_is (Any, optional): Parameter for handling null values in smoothing (default is None).
        attr (Attr, optional): Attribute to assign to the node (default is None).
        filename (str, optional): The name of the file associated with the array (default is None).
        name (str, optional): Name for the new node (default is None).
        aux (int, optional): If nonzero, treats the file as an auxiliary file (default is 0).
        to_save (bool, optional): Flag to indicate whether to save the data array (default is False).

    Returns:
        None
    """

    if node is None:
        return
    if var is None:
        return

    if no_null:
        ind_null, count_null, ind_void, count_void = dpg.values.count_invalid_values(
            var
        )
        var[ind_null] = 0

    if smoothBox is None:
        smoothBox, _ = get_par(node, "smooth", 0)
    if smoothBox > 0:
        if medianFilter is None:
            medianFilter, _ = get_par(node, "medianFilter", 0)
        var = dpg.prcs.smooth_data(var, smoothBox, opt=medianFilter, no_null=as_is)

    if linear:
        var = dpg.prcs.unlinearizeValues(var, scale=1)

    newNode = node
    if name is not None:
        newNode, _ = dpg.tree.addNode(node, name=name)

    dim = np.shape(var)
    dpg.array.set_array(newNode, pointer=var, aux=aux, no_copy=no_copy, filename=filename)

    if attr is not None and isinstance(attr, dpg.attr__define.Attr):
        put_values(newNode, attr)
    else:
        if main is not None:
            _, calib, out_dict = dpg.calibration.get_values(newNode)
            if calib is None:
                dpg.calibration.copy_calibration(main, newNode)

    if main is not None:
        dpg.navigation.copy_geo_info(fromNode=main, toNode=newNode)
        if np.size(dim) < 3:
            dpg.navigation.remove_el_coords(newNode)

    if to_save:
        dpg.array.save_array(newNode)
    return


# TODO controllare comportamento funzione
def put_values(node, calib, alt_node=None, bottom=None, nullInd=None, voidInd=None):
    """
    Sets calibration values to a node and updates the 'bitplanes' attribute if necessary.

    This function applies calibration settings to a specified node using a set of
    calibration values. It also checks and updates the 'bitplanes' attribute of the node
    based on the calibration settings. The function allows for optional specification of
    an alternative node, bottom limit, and indices for null and void data.

    Args:
        node: The node to which the calibration values will be applied.
        calib: The calibration values to be set to the node.
        alt_node (optional): An alternative node to which the calibration may be applied.
                             Defaults to None.
        bottom (optional): The bottom limit value for the calibration. Defaults to None.
        nullInd (optional): The index or indices representing null values. Defaults to None.
        voidInd (optional): The index or indices representing void values. Defaults to None.

    Returns:
        None: The function does not explicitly return a value.

    Note:
        The function first sets calibration values using 'dpg.calibration.set_values'.
        It then checks the 'bitplanes' attribute in the calibration settings and compares
        it with the 'bitplanes' attribute of the node. If there is a mismatch, the
        'bitplanes' attribute of the node is updated to match the calibration settings.
        This ensures consistency between the node's data representation and its calibration.
    """
    dpg.calibration.set_values(
        node, calib, alt_node=alt_node, bottom=bottom, nullInd=nullInd, voidInd=voidInd
    )

    bitplanes, exists, _ = dpg.attr.getAttrValue(calib, "bitplanes", 0)
    if not exists:
        return

    attr = dpg.array.get_idgeneric(node, standard=True, only_current=True)
    bits, exists, _ = dpg.attr.getAttrValue(attr, "bitplanes", 0)
    if not exists:
        return

    if bits == bitplanes:
        return

    _ = dpg.attr.replaceTags(attr, "bitplanes", bitplanes)


def get_scans(volId, reload=False, continue_on_err=False, min_el=None, max_el=None):
    """
    Retrieves all nodes related to a specified scan of low resolution data.

    This function accesses and returns scan data corresponding to a specified volume
    identifier (volId). It allows for the data to be reloaded, error continuation, and
    filtering of scans based on minimum and maximum elevation angles.

    Args:
        volId (Node): The identifier of the volume from which scans are to be retrieved.
        reload (bool, optional): If True, the data is reloaded. Defaults to False.
        continue_on_err (bool, optional): If True, the function continues execution
                                          even if errors are encountered. Defaults to False.
        min_el (float, optional): The minimum elevation angle for filtering scans.
                                  Defaults to None.
        max_el (float, optional): The maximum elevation angle for filtering scans.
                                  Defaults to None.

    Returns:
        dict: The scan data corresponding to the specified volume identifier. The exact
        format and structure of the return value depend on the implementation of
        'dpg.access.get_scans'.

    Note:
        The function delegates the retrieval and filtering of scan data to
        'dpg.access.get_scans', passing all provided arguments to it. This allows for
        flexible data access and manipulation based on the provided parameters.
    """
    return dpg.access.get_scans(
        volId,
        reload=reload,
        continue_on_err=continue_on_err,
        min_el=min_el,
        max_el=max_el,
    )


def search_data(
        schedule,
        name,
        warpNode=None,
        destNode=None,
        main=None,
        date=None,
        time=None,
        sourceNode=None,
        origin="",
        wait=None,
        guess=None,
):
    """
    Searches for data based on the provided parameters and optionally modifies nodes
    and paths within the data structure.

    Args:
        schedule (str): The schedule information used to perform the search.
        name (str): The name of the node or data to search for.
        warpNode (Node, optional): Node to warp to during the search. Defaults to None.
        destNode (Node, optional): Destination node for copying geo info. Defaults to None.
        main (Node, optional): Main node to update based on the search result. Defaults to None.
        date (str, optional): The date to use for modifying the search path. Defaults to None.
        time (str, optional): The time to use for modifying the search path. Defaults to None.
        sourceNode (Node, optional): The source node to search within. Defaults to None.
        origin (str, optional): The origin to consider in the search. Defaults to ''.
        wait (int, optional): The time to wait for the node to become available. Defaults to None.
        guess (str, optional): A guess for the product path. Defaults to None.

    Returns:
        var (np.ndarray): The data retrieved from the search, or None if main is None.
    """
    node, changed = dpg.times.searchNode(
        name,
        schedule,
        date=date,
        time=time,
        sourceNode=sourceNode,
        origin=origin,
        wait=wait,
        guess=guess,
    )

    if isinstance(warpNode, dpg.node__define.Node):
        print("TODO: to be implemented warpnode is a Node")
    else:
        var = get_data(node, numeric=True)
        if isinstance(destNode, dpg.node__define.Node):
            dpg.navigation.copy_geo_info(node, destNode)

    if main is not None:
        main = node
        return

    if changed:
        dpg.tree.removeTree(node)

    return var


def get_prev_data(
        node, min_step, numeric=None, aux=False, n_times=None, next=False, silent=False
):
    """
    This function obtain information from the previous node.
    Args:
        node (Node): The reference node.
        min_step (int): Factor to multiply the steps with
        numeric (bool, optional): Flag to specify if the data should be numeric. Defaults to False.
        aux (bool, optional): Flag to indicate if auxiliary data should be included. Defaults to False.
        n_times (int, optional): Number of times data is recovered. The default value is None.
        next (bool, optional):  If True, moves forward in time to find the next node; otherwise, moves backward.
                                Defaults to False.
        silent (bool,optional): Flag to suppress output messages. Defaults to False.

    Returns:
        var (np.ndarray): The retrieved data array.
    """
    var = [0]
    if not isinstance(node, dpg.node__define.Node):
        return

    if n_times is None:
        n_times = 1

    for ttt in range(1, n_times + 1):
        step = ttt * min_step
        prevNode, _, _ = dpg.times.getPrevNode(node, step, next=next)
        var = get_data(prevNode, numeric=numeric, aux=aux, silent=silent)
        if len(var) > 0:
            return var

    return var


def put_radar_par(
        node,
        site_coords=None,
        par=None,
        el_coords=None,
        h_off=None,
        h_res=None,
        remove_coords=None,
        range_off=None,
        range_res=None,
        azimut_off=None,
        azimut_res=None,
        elevation_off=None,
):
    """

    This method obtain parameters from a specific radar if some conditions are met, and after sets a portion
    of data for the radar.

    Args:
        node (Node): Node for which set radar parameters.
        site_coords (list of float): Coordinate of the radar site.
        par (list of float): Output of the GET_RADAR_PAR procedure.
        el_coords (np.ndarray, optional): Elevation coordinates. Defaults to None.
        h_off (float): Offset at height.
        h_res (float): Quota resolution.
        remove_coords (bool): If set, removes any files containing the original coordinates in azimuth and elevation.
        range_off (float): Range offset for the radar.
        range_res (float): Resolution range fo the radar.
        azimut_off (float): Azimuth offset for the radar.
        azimut_res (float): Azimuth resolution for the radar.
        elevation_off (float): Elevation offset for the radar.

    Returns:
        None
    """
    if len(par) > 7 and range_res is None:
        par_dict = dpg.navigation.get_radar_par(None, par=par)
        range_res = par_dict["range_res"]
        azimut_res = par_dict["azimut_res"]
        range_off = par_dict["range_off"]
        azimut_off = par_dict["azimut_off"]

    if len(par) > 10:
        elevation_off = par[10]

    dpg.navigation.set_radar_par(
        None,
        node=node,
        site_coords=site_coords,
        elevation_off=elevation_off,
        par=par,
        el_coords=el_coords,
        range_off=range_off,
        range_res=range_res,
        azimut_off=azimut_off,
        azimut_res=azimut_res,
        h_off=h_off,
        h_res=h_res,
        remove_coords=remove_coords,
    )


def get_volumes(
        prodId, moment=None, any=False, raw_tree=None, no_wait=False, measure=None
):
    """
    Retrieves volumes based on the provided product ID and other optional parameters.

    Args:
        prodId (Node): Node where the parameters.txt file is present.
        moment (str, optional): Moment or specific time to consider. Defaults to None.
        any (bool, optional): Flag indicating whether to return any measure if none is found. Defaults to False.
        raw_tree (Node, optional): The data tree to search within. Can be a 'dpg.node__define.Node' or other
                                            types that are then converted into a node. Defaults to None.
        no_wait (bool, optional): Parameter not used. Defaults to False.
        measure (str, optional): Specific parameter name to read. Defaults to None.

    Returns:
        volId (list of Node or Node): List of nodes or a single node depending on the search results.
    """
    if moment is not None:
        measure = moment

    if measure is None:
        measure, _ = get_par(prodId, "measure", "", start_with=True)
        if measure is None or measure == "":
            if not any:
                return
            measure = "CZ"

    if not isinstance(measure, list):
        measure = [measure]

    nM = np.size(measure)
    volId = []
    for mmm in range(nM):
        volId.append(dpg.access.find_volume(raw_tree, measure[mmm], prod_id=prodId))

    if nM == 1:
        volId = volId[0]
        if not isinstance(volId, dpg.node__define.Node):
            if any and raw_tree is not None:
                volId = raw_tree

    return volId


def get_geo_info(node, map=None, scale=None):
    """
    Retrieves geographic information related to a specified node.

    Parameters:
    node (dpg.node__define.Node): The node from which geographic information is retrieved.
    map (optional): Map data to be retrieved or updated if specified.
    scale (optional): Scale information to be retrieved if specified.

    Returns:
    tuple: A tuple containing the map and geographic parameters (par). If the node or map
           is invalid, returns None.
    """
    if not isinstance(node, dpg.node__define.Node):
        return

    _, _, ddd, _, _, _, _, _ = node.getArrayInfo()

    if len(ddd) > 0:
        if sum(ddd) > 0:
            dim = ddd

    box, par, _, prj_lat, prj_lon, projection, _, _ = dpg.navigation.get_geo_info(node)

    if (
            box is None
    ):  # TODO: da controllare. In IDL è ret=0 quando non trova attr dentro get_geo_info
        son = dpg.tree.getSons(node)
        nSons = len(son)  # TODO. da controllare
        if nSons != 1:
            return
        ddd = dpg.array.get_array_info(son)
        if len(ddd) > 0:
            if sum(ddd) > 0:
                dim = ddd
        # TODO: da fare
        dpg.navigation.get_geo_info(
            node,
            mosaic=mosaic,
            projection=projection,
        )

    if map:
        map = dpg.map.findSharedMap(projection=projection, p0lat=prj_lat, p0lon=prj_lon)

    if scale:
        if len(par) > 4:
            scale = np.abs([par[1], par[3]])
            scale = float(scale)

    return map, par


def get_last_node(schedule, name, date, time, path=None, minmax=None):
    """
    Retrieves the last node from a schedule based on the given parameters.

    Args:
        - schedule: The schedule to search within.
        - name: The name of the node to find (optional).
        - date: The date to search for the node.
        - time: The time to search for the node.
        - path: An optional specific path to search within (default is None).
        - minmax: Optional parameter to specify the time range for searching (default is None).

    Returns:
        - lastNode: The last node found based on the criteria or None if not found.
    """
    if path:
        lastPath = dpg.times.search_path(path, date, time, nMin=minmax)
        if lastPath == "":
            return
        lastNode = dpg.tree.createTree(lastPath)
        if name:
            lastNode = dpg.tree.findSon(lastNode, name)
        return lastNode

    lastNode, _ = dpg.times.searchNode(name, schedule=schedule, date=date, time=time)

    return lastNode


def get_last_volume(prodId, moment, reload=False, projected=False, site=None, date_=None, time_=None, linear=False):
    """
    Retrieves the latest sampled volume for a given product and moment, with options for data reloading, projection,
    and linear scaling

    Args:
        prodId: The product node associated with the radar data
        moment (str): The requested variable or data type
        reload (bool, optional): If True, reloads the data from the file system. Defaults to False
        projected (bool, optional): If True, applies ground projection to the beam data using cos(elevation).
        Defaults to False
        site (str, optional): The name of the site (required for Phase 2). Defaults to None
        date_ (str, optional): The date for which to retrieve the volume. Defaults to None
        time_ (str, optional): The time for which to retrieve the volume. Defaults to None
        linear (bool, optional): If True, converts the data to linear scale. Defaults to False

    Returns:
        tuple:
            - var (np.ndarray): The data array retrieved from the volume, or None if no volume is found
            - main (Node): The node associated with the volume, or None if no volume is found

    Notes:
        - Phase 1: Uses a shared tree structure to retrieve the data
        - Phase 2: Creates a new tree structure if the `site` parameter is provided
        - If the `site` parameter is provided, the method searches for the volume based on the schedule and time
        - If `projected` is True, the method applies additional range sampling to ground-project the beams
    """
    if site is not None:
        if site == '':
            return None, None
        schedule = os.path.join("$SCHEDULES", "RADAR_0", "VOL", "LR")
        if date_ is None:
            date_, time_, _ = dpg.times.get_time(prodId)
        raw_tree = dpg.times.searchNode(site, schedule=schedule, date=date_, time=time_)
    else:
        raw_tree = dpg.access.getRawTree(prodId, sampled=True, reload=reload)

    main = dpg.access.find_volume(raw_tree, moment)
    if not isinstance(main, dpg.node__define.Node):
        return None, None

    var = get_data(main, numeric=True, linear=linear)
    if projected:
        var = project_volume(main, var)

    return var, main


def put_var(node, name, var, set_array=False, append=False, transpose=None, aux=None, to_save=False):
    ex = None

    if var is None or np.size(var) <= 1:
        return

    if append:
        ex, _ = get_var(node, name)

    if np.size(ex) > 1:
        if transpose:
            ex = [[ex], [var]]
        else:
            ex = [ex, var]
        pointer = ex

    else:
        pointer = var

    format_ = 'ASCII'

    if set_array:
        dpg.array.set_array(node, pointer, filename=name, format=format_, aux=aux, to_save=to_save)
    else:
        dpg.tree.addAttr(node, name, pointer, format=format_, to_save=to_save)

    return


def get_values(node=None, name=None, main=None):
    """
    Retrieves calibration-related values and metadata from a given node

    Args:
        node (Node, optional): The node from which to retrieve values. Must be an instance of `dpg.node__define.Node`
        name (str, optional): The name of the specific attribute to retrieve. Currently not implemented. Defaults to
        None
        main (Node, optional): The main node to be updated if provided. If not provided, it defaults to `node`.
        Defaults to None

    Returns:
        tuple:
            - values (np.ndarray): The data values retrieved from the node
            - main (Node): The main node associated with the retrieved values
            - scale (float): The scaling factor for the values
            - calib (float): Calibration information for the values
            - nullInd (int): Indicator for null values in the dataset
            - voidInd (int): Indicator for void values in the dataset

    Notes:
        - If the `node` parameter is not an instance of `dpg.node__define.Node`, the function immediately returns `None`
        - If `name` is provided, the function searches for a descendant node matching the name within `node`. If no
          descendant is found, it searches in the parent of `node`. If a match is found, the values are retrieved
        - The `main` parameter defaults to the same node used to retrieve the values if not explicitly provided
        - The function relies on `dpg.calibration.get_array_values` to extract the calibration metadata
    """
    if not isinstance(node, dpg.node__define.Node):
        return

    ret = 1
    son = node

    if name is not None:
        log_message("TODO: Ramo non ancora portato", "ERROR")
    if ret <= 0:
        return

    values, calib, out_dict = dpg.calibration.get_array_values(son)
    simmetric = out_dict["simmetric"]  # is not returned
    scale = out_dict["scale"]
    nullInd = out_dict["nullInd"]
    voidInd = out_dict["nullInd"]
    if not isinstance(main, dpg.node__define.Node):
        main = son

    return values, main, scale, calib, nullInd, voidInd


def put_par(node, name, par, parFile=None, to_save=False):
    dpg.radar.set_par(node, name, par, only_current=True, parFile=parFile, to_save=to_save)
    return


def merge_data(node, var, maxDim, step, name, noTimes=None, retry=None):
    var = dpg.times.mergePrevData(node, var, maxDim, step, noTimes=noTimes, retry=retry)
    if np.size(var) <= 0:
        return None

    if not isinstance(var, list):
        var = [var]

    pointer = np.array(var, dtype=np.float32)
    dpg.array.set_array(node, pointer, filename=name, format='ASCII')

    return var


def put_date(node, date, time_, to_save=False, nominal=False, both=False):

    if not to_save:
        root, level = dpg.tree.getRoot(node)
        if level <= 1:
            to_save = True

    if both:
        dpg.times.set_time(node, date=date, time=time_)
        nominal = True
    dpg.times.set_time(node, date=date, time=time_, to_save=to_save, nominal=nominal)

    return
