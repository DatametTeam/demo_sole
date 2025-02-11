import inspect
import os
from numbers import Number

import numpy as np
import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
- FUNCTION IDL_rv_find_aux_product 
- FUNCTION IDL_rv_find_single_volume 
- FUNCTION IDL_rv_get_brother_volume 
- FUNCTION IDL_rv_get_equivalent_volume  
- FUNCTION IDL_rv_get_single_scan 
- FUNCTION IDL_rv_get_single_volume 
- FUNCTION IDL_rv_is_data_node 
- FUNCTION IDL_rv_select_scan 
- PRO REMOVE_QUALITY 
- FUNCTION GetCurrRawTree               // UNUSED
- FUNCTION IDL_rv_check_aux_schedule    // UNUSED
- FUNCTION IDL_rv_get_regions           // UNUSED
"""


# Original Name : IDL_rv_get_height_beams
def get_height_beams(
        elevSet,
        nBins: int,
        rangeRes: float,
        site_height: float = None,
        range_off: float = None,
        projected: bool = False,
        no_reform: bool = False,
) -> np.ndarray:
    """
    Calculates the height of radar beams for each scan in an elevation set.

    This function computes the height of radar beams based on elevation angles, range
    resolution, and the number of bins. It supports both single elevation values and
    arrays of elevation angles. The function can account for site height, range offset,
    and whether the heights should be projected. An option is also provided to prevent
    reshaping the output array when there's only one scan.

    Args:
        elevSet (Number or np.ndarray): A single elevation angle or an array of elevation
                                        angles for which heights need to be calculated.
        nBins (int): The number of bins in each radar scan.
        rangeRes (float): The range resolution of the radar system.
        site_height (float, optional): The height of the radar site above mean sea level.
                                       Defaults to None.
        range_off (float, optional): The range offset to apply. Defaults to None.
        projected (bool, optional): If True, calculates projected heights. Defaults to False.
        no_reform (bool, optional): If True and there is only one scan, prevents the output
                                    from being reshaped. Defaults to False.

    Returns:
        np.ndarray: An array of heights for each beam in each scan, with shape (nScans, nBins)
                    unless no_reform is True and nScans is 1, in which case the shape is (nBins,).

    Note:
        The function leverages 'dpg.beams.getHeightBeam' for the actual height calculations.
        If 'elevSet' is a single number, it is converted to a one-element numpy array for
        uniform processing. The 'projected' parameter allows for calculating heights either
        in actual vertical distance or projected onto a plane (useful for certain types of
        radar analysis).
    """

    if isinstance(elevSet, Number):
        elevSet = np.array([elevSet])

    nScans = elevSet.size
    heightBeams = np.zeros((nScans, nBins), dtype=np.float32)

    for eee in range(nScans):
        heightBeams[eee, :], _ = dpg.beams.getHeightBeam(
            rangeRes,
            nBins,
            elevSet[eee],
            site_height=site_height,
            range_off=range_off,
            projected=projected,
        )

    if nScans == 1 and not no_reform:
        return np.squeeze(heightBeams)
    # TODO così non ha molto senso, se si restituisce sempre np.squeeze(heightBeams) si può semplificare
    return np.squeeze(heightBeams)


# Original name: IDL_rv_find_volume
def find_volume(
        tree=None,
        measure: str = None,
        sampled: bool = False,
        prod_id: bool = None,
        force: bool = False,
        get_tree: bool = False,
):
    """
    Searches for a volume node in a data tree based on a specified measure, potentially using alternative measures.

    This function attempts to locate a volume node within a given tree that corresponds
    to a specified measure (e.g., 'Z', 'UZ', 'CZ'). If the initial tree is not a valid
    'dpg.node__define.Node', it tries to generate one using various parameters. If the
    desired measure is not found, the function searches for alternative measures
    (e.g., 'CZ' and 'UZ' as alternatives for 'Z').

    Args:
        tree: The data tree to search within. Can be a 'dpg.node__define.Node' or other
              types that are then converted into a node.
        measure (str): The measure to search for in the tree.
        sampled (bool, optional): Flag used when generating a tree if the initial tree is
                                  invalid. Defaults to False.
        prod_id (bool, optional): Product ID used when generating a tree if the initial
                                  tree is invalid. Defaults to False.
        force (bool, optional): Force flag used when generating a tree if the initial tree
                                is invalid. Defaults to False.

    Returns:
        dpg.node__define.Node or None: The found volume node if successful, or None if no
                                       suitable volume node is found.

    Note:
        If the initial 'tree' is not a 'dpg.node__define.Node', the function attempts to
        generate one using 'getRawTree' with the provided parameters. If the desired
        measure is not found directly, the function searches for predefined alternative
        measures.
    """
    if not isinstance(tree, dpg.node__define.Node):
        tree = getRawTree(sampled=sampled, prod_id=prod_id, force=force)
    if not isinstance(tree, dpg.node__define.Node):
        log_message("Find Volume: Cannot find Raw Data", all_logs=True, level="ERROR")
        # IDL_rv_add_log, 'Cannot Find Raw Data', NODE=prod_id, /FATAL
        if get_tree:
            return None, None
        else:
            return None

    volId = dpg.tree.findSon(tree, measure)
    if isinstance(volId, dpg.node__define.Node):
        if get_tree:
            return volId, tree
        else:
            return volId

    if measure == "Z":
        eqMeas = ["CZ", "UZ"]
    elif measure == "UZ":
        eqMeas = ["Z", "CZ"]
    elif measure == "CZ":
        eqMeas = ["Z", "UZ"]
    else:
        if get_tree:
            return None, tree
        else:
            return None

    # log_message(f"Cannot find {measure} @ node {tree.path}")

    for zzz in range(2):
        volId = dpg.tree.findSon(tree, eqMeas[zzz])
        if isinstance(volId, dpg.node__define.Node):
            log_message(f"Using {eqMeas[zzz]}")
            if get_tree:
                return volId, tree
            else:
                return volId

    return None


# la funzione restituisce rawTree, non lo riceve più in input (che poi era output)
def getRawTree(
        prod_id=None,
        sampled: bool = False,
        reload: bool = False,
        remove: bool = False,
        raw_path=None,
        force: bool = False,
):
    """
    Generates a raw data tree from a specified path or based on production ID and global parameters.

    This function creates a raw data tree, used for radar data processing, based on a raw
    path or production ID. It handles different scenarios including sampled data, forced
    data retrieval, and tree reloading or removal. The function also supports shared tree
    creation and management based on global parameters.

    Args:
        prod_id (optional): The production ID used to determine the raw path if not
                            provided. Defaults to None.
        sampled (bool, optional): Indicates whether to retrieve a sampled volume path.
                                  Defaults to False.
        reload (bool, optional): If True, forces the reloading of the raw tree. Defaults
                                 to False.
        remove (bool, optional): If True, removes the existing tree before creating a new
                                 one. Defaults to False.
        raw_path (str or list, optional): The path to the raw data. If a list is provided,
                                          only the first string element is used. Defaults
                                          to an empty string.
        force (bool, optional): Forces the retrieval of the raw volume path if necessary.
                                Defaults to False.

    Returns:
        dpg.node__define.Node or None: The created raw data tree, or None if the tree
                                       cannot be created.

    Note:
        The function prioritizes the provided 'raw_path' for tree creation. If 'raw_path'
        is not specified, it is derived from the provided 'prod_id', 'sampled', and
        'force' arguments using helper functions like 'getSampledVolumePath' and
        'getRawVolumePath'. The function also handles shared nodes and tree management
        based on 'reload' and 'remove' flags.
        This approach ensures efficient memory management and tree reuse.
    """
    site = dpg.navigation.get_site_name(prod_id)
    if raw_path is None:
        raw_path = ""
    if isinstance(raw_path, list) and isinstance(raw_path[0], str):
        raw_path = raw_path[0]
    if not isinstance(raw_path, str):
        raw_path = ""

    if raw_path == "":
        if sampled:
            raw_path = getSampledVolumePath(prod_id)
        else:
            raw_path = getRawVolumePath(prod_id, force=force)

    if raw_path == "":
        raw_path, prefix, _ = dpg.radar.get_par(prod_id, "rawPath", "")
        if raw_path == "" or raw_path is None or raw_path == 'None':
            if site in dpg.globalVar.GlobalState.LAST_RAW.keys() and remove:
                node = dpg.tree.findSharedNode(dpg.globalVar.GlobalState.LAST_RAW)
                dpg.tree.removeTree(node, shared=True)
            return None

    if sampled:
        if (
                site in dpg.globalVar.GlobalState.LAST_SAMPLED.keys() and
                (isinstance(dpg.globalVar.GlobalState.LAST_SAMPLED[site], str)
                 and dpg.globalVar.GlobalState.LAST_SAMPLED[site] != "")
        ):
            if (
                    dpg.globalVar.GlobalState.LAST_SAMPLED[site] != raw_path
                    or reload
                    or remove
            ):
                node = dpg.tree.findSharedNode(dpg.globalVar.GlobalState.LAST_SAMPLED[site])
                dpg.tree.removeTree(node, shared=True)

        dpg.globalVar.GlobalState.update("LAST_SAMPLED", raw_path, site)

        raw_tree = dpg.tree.createTree(raw_path, shared=True)
        if isinstance(raw_tree, dpg.node__define.Node):
            return raw_tree
        else:
            return 0

    if site in dpg.globalVar.GlobalState.LAST_RAW.keys():
        if dpg.globalVar.GlobalState.LAST_RAW[site] != raw_path or reload or remove:
            node = dpg.tree.findSharedNode(dpg.globalVar.GlobalState.LAST_RAW[site])
            dpg.tree.removeTree(node, shared=True)

    dpg.globalVar.GlobalState.update("LAST_RAW", raw_path, key=site)
    raw_tree = dpg.tree.createTree(raw_path, shared=True)

    return raw_tree


# Orinigal place dpg.pro
def getSampledVolumePath(node) -> str:
    """
    Retrieves the path for sampled volume data based on node information and global parameters.

    This function determines the path to sampled radar volume data. It uses the provided
    node and global parameters to ascertain the site, date, and time associated with the
    node. The function then compares this information with existing global parameters to
    decide whether to return a predefined path or an empty string.

    Args:
        node: The node associated with the desired sampled volume data. This can be an
              instance of 'dpg.node__define.Node' or other types.

    Returns:
        str: The path to the sampled volume data. Returns an empty string if the path
             cannot be determined or if the node and global parameters do not align.

    Note:
        The function checks if 'SAMPLED_PATH' in 'globalVar' is None initially. If the
        provided 'node' is not a 'dpg.node__define.Node' instance, it returns the
        'SAMPLED_PATH'. Otherwise, it fetches the site name, date, and time from the node
        and compares these with the corresponding global parameters. If all parameters
        match, it returns the 'SAMPLED_PATH', potentially updating it based on the
        'CHECK_PATH' parameter. If there's a mismatch or if insufficient information
        is available, the function returns an empty string.
    """

    if not dpg.globalVar.GlobalState.SAMPLED_PATH:
        log_message(f"dpg.globalVar.GlobalState.SAMPLED_PATH was not Initialized", 'ERROR')
        return ""
    if not isinstance(node, dpg.node__define.Node):
        return ""

    site = dpg.navigation.get_site_name(node)
    if site == "":
        return ""

    date, time, flag = dpg.times.get_time(node, nominal=True)
    if not flag:
        return ""

    if (dpg.globalVar.GlobalState.SAMPLED_SITE[site] == site
            and dpg.globalVar.GlobalState.SAMPLED_DATE[site] == date
            and dpg.globalVar.GlobalState.SAMPLED_TIME[site] == time):
        if (dpg.globalVar.GlobalState.SAMPLED_PATH[site] == ""
                and dpg.globalVar.GlobalState.CHECK_PATH[site] != ""):
            dpg.globalVar.GlobalState.update("SAMPLED_PATH",
                                             dpg.utility.getLastPath(dpg.globalVar.GlobalState.CHECK_PATH[site],
                                                                     dpg.globalVar.GlobalState.SAMPLED_DATE[site],
                                                                     dpg.globalVar.GlobalState.SAMPLED_TIME[site], 0),
                                             site
                                             )
        # log_message(f"Found Sampled Path {dpg.globalVar.GlobalState.SAMPLED_PATH[site]}",
        #             'INFO')
        if dpg.globalVar.GlobalState.SAMPLED_PATH[site] == '' or dpg.globalVar.GlobalState.SAMPLED_PATH[site] is None:
            log_message(f"Found Empty Sampled Path for site {site}", 'WARNING+')
        return dpg.globalVar.GlobalState.SAMPLED_PATH[site]

    log_message(f"Cannot find Sampled Path for site {site}! in {dpg.globalVar.GlobalState.SAMPLED_PATH}", 'WARNING+')
    return ""


def setSampledVolumePath(
        path: str,
        site: str,
        date: str,
        time: str,
        maxMin: int = 0,
        sub: str = None,
        lastPath: str = None,
        sampledCheck: bool = False,
) -> str:
    """
    Sets the path for sampled volume data based on various parameters and updates global parameters.

    This function constructs and sets a path for sampled volume data. It modifies the
    path based on the provided date, time, and other parameters. The function also updates
    global parameters with the new path and related information.

    Args:
        path (str): The base path for the sampled volume data.
        site (str): The site identifier associated with the volume data.
        date (str): The date associated with the volume data.
        time (str): The time associated with the volume data.
        maxMin (int): A parameter influencing the path determination.
        sub (str, optional): An optional sub-path to append to the base path. Defaults to None.
        lastPath (str, optional): The last used path, influencing the new path creation.
                                  Defaults to None.
        sampledCheck (bool, optional): A flag indicating whether to perform a sampled check
                                       operation. Defaults to False.

    Returns:
        str: The constructed and updated path for sampled volume data.

    Note:
        The function constructs the path by ensuring correct formatting and appending the
        year if needed. It uses 'dpg.times.path2Date' to extract date and time from
        'lastPath' if provided. The 'dpg.times.changePath' method is used to modify the
        path based on date and time. If 'sub' is provided, it is appended to the path.
        The function updates 'global_par' with the new path, site, date, and time, and
        uses 'dpg.utility.getLastPath' to finalize the path construction.
    """

    sampledPath = dpg.path.checkPathname(path, with_separator=False)
    if sampledPath.find("20") == -1:
        sampledPath = os.path.join(sampledPath, '2000')

    ddd = date
    ttt = time
    if lastPath is not None and lastPath != "":
        ddd, ttt = dpg.times.path2Date(lastPath)

    sampledPath = dpg.times.changePath(sampledPath, ddd, ttt)
    if sub:
        sampledPath = dpg.path.checkPathname(sampledPath, with_separator=False)
        sampledPath = os.path.join(sampledPath, sub)

    check_path = dpg.path.getFullPathName(sampledPath, site)
    dpg.globalVar.GlobalState.update("CHECK_PATH", check_path, site)
    sampledPath = dpg.utility.getLastPath(check_path, ddd, ttt, maxMin)

    if not sampledCheck:
        dpg.globalVar.GlobalState.update("CHECK_PATH", "", site)

    dpg.globalVar.GlobalState.update("SAMPLED_PATH", sampledPath, site)
    dpg.globalVar.GlobalState.update("SAMPLED_SITE", site, site)
    dpg.globalVar.GlobalState.update("SAMPLED_DATE", date, site)
    dpg.globalVar.GlobalState.update("SAMPLED_TIME", time, site)
    return sampledPath


# Orinigal place dpg.pro
def getRawVolumePath(node, force: bool = False) -> str:
    """
    Retrieves the raw volume path based on node information and global parameters.

    This function determines the path to raw radar volume data using the provided node
    and global parameters. If the path is not set or if the 'force' parameter is true,
    it attempts to forcefully update or retrieve the raw path. The function compares
    node-specific information like site, date, and time with the corresponding global
    parameters to decide whether to return the predefined raw path or an empty string.

    Args:
        node: The node associated with the desired raw volume data. This can be an instance
              of 'dpg.node__define.Node' or other types.
        force (bool, optional): Forces the retrieval of the raw volume path if necessary.
                                Defaults to False.

    Returns:
        str: The path to the raw volume data. Returns an empty string if the path cannot
             be determined or if the node and global parameters do not align.

    Note:
        The function initially checks if 'RAW_PATH' in 'global_par' is a string and sets
        it if not. If 'RAW_PATH' is empty and 'force' is true, 'forceRawVolumePath' is
        called to update it. The function then checks if the node's site, date, and time
        match the corresponding global parameters. If there is a mismatch or if the
        necessary information is not available, an empty string is returned.
    """

    if not dpg.globalVar.GlobalState.RAW_PATH:
        setRawVolumePath("", "", "", "")

    # if dpg.globalVar.GlobalState.RAW_PATH[site] == "":
    #     if force:
    #         forceRawVolumePath()
    #     # return global_par['RAW_PATH']
    #     return dpg.globalVar.GlobalState.RAW_PATH[site]

    site = dpg.navigation.get_site_name(node)
    if site == "":
        return ""

    if not isinstance(node, dpg.node__define.Node):
        return dpg.globalVar.GlobalState.RAW_PATH[site]

    if site == "":
        return dpg.globalVar.GlobalState.RAW_PATH[site]

    date, time, flag = dpg.times.get_time(node, nominal=True, only_current=False)
    if not (flag):
        return dpg.globalVar.GlobalState.RAW_PATH[site]

    if (dpg.globalVar.GlobalState.RAW_DATE == date
            and dpg.globalVar.GlobalState.RAW_TIME == time):
        return dpg.globalVar.GlobalState.RAW_PATH[site]
    return ""


# Orinigal place dpg.pro
def setRawVolumePath(rawPath, site, date, time):
    """
    Updates global state variables with the provided raw path, site, date, and time.

    Args:
        rawPath: The path to the raw data.
        site: The name of the site.
        date: The date associated with the raw data.
        time: The time associated with the raw data.

    Returns:
        - **None**
    """

    dpg.globalVar.GlobalState.update("RAW_PATH", rawPath, site)
    dpg.globalVar.GlobalState.update("RAW_DATE", date)
    dpg.globalVar.GlobalState.update("RAW_TIME", time)


# Orinigal place dpg.pro
def forceRawVolumePath():
    """
    Forces an update of the raw volume path based on the sampled path and configuration attributes.

    Returns:
        - **None**
    """

    """TBD"""
    # if global_par['SAMPLED_PATH'] is None or global_par['SAMPLED_PATH'] == '':
    log_message("Code never called! TO BE CHECKED", level="WARNING+")
    if (
            dpg.globalVar.GlobalState.SAMPLED_PATH is None
            or dpg.globalVar.GlobalState.SAMPLED_PATH == ""
    ):
        return

    prefix = "phase_1"
    system = "RADAR"
    path = dpg.path.getDir("CFG_SCHED", with_separator=True)
    attr = dpg.attr.loadAttr(path, system + ".txt")
    searchPath, _, _ = dpg.attr.getAttrValue(
        attr, "rawPath", "/data1/RPG/RAW/RDR/VOL", prefix=prefix
    )
    rawSub, _, _ = dpg.attr.getAttrValue(attr, "rawSub", "$site/H", prefix=prefix)
    attr = None

    rawPath = dpg.path.checkPathname(searchPath, with_separator=False)
    if "20" not in rawPath:
        rawPath += "2000"

    # ddd, ttt = dpg.times.path2Date(global_par['SAMPLED_PATH'])
    ddd, ttt = dpg.times.path2Date(dpg.globalVar.GlobalState.SAMPLED_PATH)
    rawPath = dpg.times.changePath(rawPath, ddd, ttt)

    pos = rawSub.find("$site")  # TODO controllare il comportamento di $ in IDL
    if pos >= 0:
        # sss = rawSub[:pos] + global_par['SAMPLED_SITE'] + rawSub[pos + 5:]
        sss = rawSub[:pos] + dpg.globalVar.GlobalState.SAMPLED_SITE + rawSub[pos + 5:]
    else:
        sss = rawSub
    rawPath = dpg.path.checkPathname(rawPath + sss, with_separator=False)
    if not (os.path.exists(rawPath) and os.path.isfile(rawPath)):
        return

    # global_par['RAW_PATH'] = rawPath
    # global_par['RAW_SITE'] = global_par['SAMPLED_SITE']
    # global_par['RAW_DATE'] = global_par['SAMPLED_DATE']
    # global_par['RAW_TIME'] = global_par['SAMPLED_TIME']

    dpg.globalVar.GlobalState.update("RAW_PATH", rawPath)
    dpg.globalVar.GlobalState.update("RAW_SITE", dpg.globalVar.GlobalState.SAMPLED_SITE)
    dpg.globalVar.GlobalState.update("RAW_DATE", dpg.globalVar.GlobalState.SAMPLED_DATE)
    dpg.globalVar.GlobalState.update("RAW_TIME", dpg.globalVar.GlobalState.SAMPLED_TIME)


# end func


# Original name FUNCTION IDL_rv_get_aux_products
def get_aux_products(
        prodId,
        aux_path=None,
        site_name=False,
        mosaic: bool = False,
        local: bool = False,
        schedule=None,
        interactive: bool = False,
        origin: bool = False,
        search_path=None,
        oldest_first: bool = False,
        last_date=False,
        last_time: str = None,
        n_hours: int = None,
        min_step: int = None,
        min_tol: int = None,
        dates=None,
        times: bool = False,
        seconds=None,
):
    """
    Retrieves and processes auxiliary product paths based on various criteria.

    Args:
        prodId: The product identifier used to obtain the schedule.
        aux_path: Optional; A list of auxiliary product paths.
        site_name: Optional; Boolean indicating if site names should be included in paths.
        mosaic: Optional; Boolean indicating if mosaic processing is required.
        local: Optional; Boolean indicating if local processing is required.
        schedule: Optional; The schedule to be used. Defaults to `None`, in which case it is fetched using `prodId`.
        interactive: Optional; Boolean indicating if interactive mode is enabled.
        origin: Optional; Boolean indicating if origin data is to be included.
        search_path: Optional; A list of search paths for locating products.
        oldest_first: Optional; Boolean indicating if older products should be prioritized.
        last_date: Optional; The last date to consider for filtering products.
        last_time: Optional; The last time to consider for filtering products.
        n_hours: Optional; The number of hours for filtering recent products.
        min_step: Optional; The minimum step interval for filtering.
        min_tol: Optional; The minimum tolerance for filtering.
        dates: Optional; A list of dates associated with the products.
        times: Optional; Boolean indicating if times should be included.
        seconds: Optional; A list of seconds associated with the products.

    Returns:
        - **count**: The number of products found and processed.
        - **aux_path**: A list of paths to the auxiliary products.
    """

    """TBD"""
    if schedule is None:
        schedule, _, _ = dpg.radar.get_par(prodId, "schedule", "")
    if not search_path and schedule is None:
        # LOG
        return None, False

    aux_path = get_prod_list(schedule)
    count = len(aux_path)
    if last_date is None:
        return count

    newest_time = dpg.times.convertDate(last_date, time=last_time)

    if not search_path and interactive:
        if count > 0:
            count, dates, times = dpg.times.get_times(aux_path + site_name)
            count, ind, seconds = dpg.times.sortDates(
                dates,
                times,
                newest_time=newest_time,
                n_hours=n_hours,
                unique=True,
                seconds=seconds,
                oldest_first=oldest_first,
            )
            if count > 0:
                aux_path = aux_path[ind]
                dates = dates[ind]
                times = times[ind]
                return count
        count, tmp = get_prod_list(schedule)
        if count > 0:
            aux_path = tmp

    if search_path is not None:
        aux_path = search_path[0]
        count = 1

    if count < 1:
        # LOG
        return count, aux_path

    if min_step is not None:
        minStep = -min_step
    else:
        minStep = -5

    aux_path = aux_path[0]

    nh = 0
    if n_hours is not None:
        nh = n_hours

    aux_path = dpg.times.create_path_list(
        aux_path, last_date, last_time, minStep, nh, min_tol=min_tol
    )

    if origin:
        count, dates, times = dpg.times.get_times(
            aux_path + origin, dates=dates, times=times
        )
        pass
    else:
        if site_name:
            new_paths = [os.path.join(path, site_name) for path in aux_path]
            dates, times = dpg.times.get_times(new_paths)
            pass
        else:
            count, dates, times = dpg.times.get_times(aux_path)
            pass

    if seconds:
        tmp_seconds = seconds.copy

    count, ind, seconds = dpg.times.sortDates(
        dates,
        times,
        newest_time=newest_time,
        n_hours=nh,
        unique=True,
        seconds=seconds,
        oldest_first=oldest_first,
    )

    if count < 1:
        log_message(
            f"No Products found in {schedule}",
            general_log=True,
            level="WARNING",
        )
        return count, aux_path

    aux_path = [aux_path[i] for i in ind]
    dates = [dates[i] for i in ind]
    times = [times[i] for i in ind]

    return count, aux_path


def get_prod_list(
        path: str,
        interactive: bool = False,
        pathname: str = "",
        get_list_name: bool = False,
):
    """
    Retrieves a list of products from a specified path based on configuration and attribute settings.

    This function fetches a product list from a given path. It first determines the name of
    the product list to be retrieved, which can depend on whether the operation is interactive.
    It then loads the attribute containing the product list and checks for its existence.
    Optionally, the name of the product list can also be returned.

    Args:
        path (str): The path from which to load the product list.
        interactive (bool, optional): Determines whether the operation is interactive,
                                      influencing the product list name. Defaults to False.
        pathname (str, optional): The pathname to use when loading the attribute.
                                  Defaults to an empty string.
        get_list_name (bool, optional): If True, the function also returns the name of
                                        the product list. Defaults to False.

    Returns:
        list or tuple: If 'get_list_name' is False, returns the product list. If True,
                       returns a tuple containing the product list and its name. If the
                       product list does not exist, returns None or (None, list_name).

    Note:
        The function uses 'dpg.cfg.getProdListName' to determine the name of the product
        list and 'dpg.attr.loadAttr' to load the attribute containing the product list.
        'dpg.attr.getAttrValue' is then used to fetch the product list from the attribute.
        The product list's existence is checked, and it is returned if available. The
        function's behavior changes slightly based on the 'get_list_name' flag.
    """
    prodList = None
    list_name = dpg.cfg.getProdListName(interactive=interactive)
    attr = dpg.attr.loadAttr(path, list_name, pathname=pathname)
    list_prod, exists, _ = dpg.attr.getAttrValue(attr, "path", default="")

    if exists:
        prodList = list_prod

    if prodList is None:
        return []

    if not isinstance(prodList, list):
        prodList = [prodList]

    if get_list_name:
        return prodList, list_name

    return prodList


# TODO controllare sia usata
def checkCurrRawTree(raw_path, remove: bool = False):
    """
    Checks and retrieves the current raw data tree from the specified path.

    This function retrieves a raw data tree from a provided file path and optionally removes
    the existing tree before creating or accessing the current tree.

    Parameters:
        raw_path (str): The path to the raw data tree to be checked and retrieved.
        remove (bool, optional): If True, removes the existing tree before retrieval. Defaults to False.

    Returns:
        rawTree (Node): The raw data tree retrieved or created from the specified path.
    """
    rawTree = getRawTree(raw_path=raw_path, remove=remove)
    return rawTree


def getLowResName(H: bool = False) -> str:
    """
    Retrieves the name for low or high-resolution data based on environmental variables and an input flag.

    This function fetches names for low and high-resolution data, which are set based on
    specific environment variables ('RV_LOW_RES_NAME' and 'RV_HIGH_RES_NAME'). If these
    variables are not set, default values are used. The function returns either the low
    or high-resolution name based on the provided flag.

    Args:
        H (bool, optional): A flag that determines whether to return the high-resolution
                            name. If False, the low-resolution name is returned. Defaults
                            to False.

    Returns:
        str: The name of the low or high-resolution data, depending on the 'H' flag.

    Note:
        The function checks for the existence of environment variables 'RV_LOW_RES_NAME'
        and 'RV_HIGH_RES_NAME' to set global variables for low and high-resolution names
        respectively. If these environment variables are not set, it defaults to 'L' for
        low and 'H' for high resolution. This approach allows for dynamic configuration
        based on the environment settings.
    """
    dpg.globalVar.GlobalState.update("LOW", os.getenv("RV_LOW_RES_NAME"))
    if dpg.globalVar.GlobalState.LOW is None:
        dpg.globalVar.GlobalState.update("LOW", "L")
    dpg.globalVar.GlobalState.update("HIGH", os.getenv("RV_HIGH_RES_NAME"))
    if dpg.globalVar.GlobalState.HIGH is None:
        dpg.globalVar.GlobalState.update("HIGH", "H")

    if H:
        return dpg.globalVar.GlobalState.HIGH
    return dpg.globalVar.GlobalState.LOW


def getLowResPath(path: str, H: bool = False, ROOT: bool = False) -> str:
    """
    Modifies a given file path to point to either low or high-resolution data based on specified parameters.

    This function alters a provided file path to switch between low and high-resolution
    data directories. It uses the presence of resolution-specific directory names within
    the path to determine the current resolution and then switches to the other resolution
    based on the input flag. Additionally, there is an option to truncate the path to the
    root of the resolution directory.

    Args:
        path (str): The original file path that needs to be modified.
        H (bool, optional): A flag indicating whether to switch to high-resolution (True)
                            or low-resolution (False) data path. Defaults to False.
        ROOT (bool, optional): If True, truncates the path to the root of the resolution
                               directory instead of the full path. Defaults to False.

    Returns:
        str: The modified file path pointing to the specified resolution's data.

    Note:
        The function identifies the current resolution in the path using directory names
        obtained from 'getLowResName'. It then switches to the other resolution by
        replacing or appending the appropriate directory name. The 'ROOT' parameter allows
        for returning just the root directory path of the specified resolution, which can
        be useful for operations that require working at the root level of data directories.
    """
    sep = os.path.sep
    result = path.split(sep)
    hhh = not H
    name = getLowResName(H=hhh)
    ind = [i for i, x in enumerate(result) if x == name]
    if len(ind) == 1:
        result[ind[0]] = getLowResName(H=not hhh)
        if ROOT:
            result = result[: ind[0]]
    else:
        if ROOT:
            result.append(getLowResName(H=not hhh))
    new_data_path = sep.join(result)
    if path.startswith(sep):
        new_data_path = sep + new_data_path
    return new_data_path


# TODO controllare sia usata
def setInteractiveSession(interactive=False):
    """
    Set the interactive session flag
    Args:
        interactive: parameter passed to update the flag

    Returns:
        None
    """
    dpg.globalVar.GlobalState.update("interactive", interactive)


# TODO controllare sia usata
def isInteractiveSession():
    """
    Check the interactive session flag.

    Returns:
        boolean: the value of interactive parameter
    """
    if dpg.globalVar.GlobalState.isInteractive is None:
        dpg.globalVar.GlobalState.update("interactive", 0)

    return dpg.globalVar.GlobalState.isInteractive


# HACK inutile!
def isLowResPath(path: str) -> bool:
    """Not used, TO BE REMOVED"""
    sep = os.path.sep
    result = path.split(sep)
    return getLowResName() in result


def getClutterMapRoot(site_name: str, class_name: str, sep=os.sep):
    """
    Constructs a clutter map root path for a given site and class, and creates a root node.

    This function builds a path to the root of a clutter map directory based on the
    specified site name and class name. It then uses this path to create and return a
    root node. The path construction is cross-platform, as it utilizes 'os.path.join'
    for concatenating directory names.

    Args:
        site_name (str): The name of the site for which the clutter map root is to be obtained.
        class_name (str): The class name associated with the clutter map.
        sep (str, optional): The path separator to be used. Defaults to 'os.sep', which
                             is the default path separator for the operating system.

    Returns:
        A root node created based on the constructed clutter map root path.

    Note:
        The clutter map root path is determined using 'dpg.cfg.getClutterHome', which
        provides the base path for clutter maps. The function then appends the site
        name and class name to this base path to form the complete root path. The root
        node is created using 'dpg.tree.createRoot', specifying this path.
    """
    root_path = dpg.cfg.getClutterHome(sep=sep)

    # Construct the path using os.path.join for cross-platform compatibility
    root_path = os.path.join(root_path, site_name, class_name)

    root = dpg.tree.createRoot(root_path)  # /ONLY_ROOT

    return root


def get_quality_volume(prod_id, raw_tree=None):
    """
    Retrieves the quality volume node for a specified product ID from a raw data tree.

    This function locates and returns a node representing the quality volume associated
    with a given product ID. It first determines the name of the quality volume using
    product-specific parameters and then searches for this volume in the provided raw
    data tree.

    Args:
        prod_id: The identifier of the product for which the quality volume is to be retrieved.
        raw_tree (dpg.node__define.Node, optional): The raw data tree to search for the
                                                    quality volume. Defaults to None.

    Returns:
        dpg.node__define.Node or None: The node corresponding to the quality volume if
                                       found, otherwise None.

    Note:
        The function uses 'dpg.radar.get_par' to fetch the name of the quality volume,
        which defaults to 'Quality' if not explicitly set for the product ID. It then
        calls 'find_volume' with this name and the provided raw data tree to locate the
        quality volume node. If such a node is found within the tree and is of type
        'dpg.node__define.Node', it is returned; otherwise, None is returned.
    """
    name, _, _ = dpg.radar.get_par(prod_id, "quality_name", "Quality")
    volume = find_volume(raw_tree, name, prod_id=prod_id)
    if isinstance(volume, dpg.node__define.Node):
        return volume
    return None


def get_scans(
        volId,
        reload: bool = False,
        continue_on_err: bool = False,
        min_el: float = None,
        max_el: float = None,
        coord_set=None,
) -> dict:
    """
    Retrieves scan data and related information for a given volume ID with optional filtering.

    This function extracts scan data and various related parameters from a volume
    identified by 'volId'. It handles scenarios with no scans, single scans, or multiple
    scans. The function also applies optional filters based on minimum and maximum
    elevation angles and provides a comprehensive dictionary of scan data and metadata.

    Args:
        volId: The volume ID for which scans are to be retrieved.
        reload (bool, optional): If True, forces the reloading of scan data. Defaults to False.
        continue_on_err (bool, optional): If True, continues execution even if errors are
                                          encountered. Defaults to False.
        min_el (float, optional): Minimum elevation angle filter. Defaults to None.
        max_el (float, optional): Maximum elevation angle filter. Defaults to None.

    Returns:
        dict: A dictionary containing various pieces of scan data and metadata, including
              'scans', 'best_scan_ind', 'mode', 'azimut_res', 'scan_dim', 'range_res',
              'simmetric', 'scale', 'unit', 'values', 'site_coords', 'beam_width', and
              'coord_set'.

    Note:
        The function iterates through all scans, applying filters and aggregating
        relevant data. It calculates the azimuth resolution, range resolution, and
        dimensions of scans. Calibration data such as scale, symmetry, and unit of
        measure are also retrieved. The function ensures that the retrieved data is
        consistent with the specified mode (elevation or azimuth) and filters.
    """

    best_scan_ind = 0
    scans = dpg.tree.getSons(volId)
    # nScans = len(scans)
    if scans is None or not scans:
        if not isinstance(volId, dpg.node__define.Node):
            return None
        nScans = 1
        scans = [volId]
    scan_dim = [0, 0]
    range_res = 0.0
    azimut_res = 0.0
    coord = 0.0
    ret = 0
    valids = []
    mode = 0

    if coord_set is not None:
        if not isinstance(coord_set, list):
            coord_set = [coord_set]

    for scan in scans:
        ok = 1
        data, tmp_dict = dpg.array.get_array(scan, reload=reload)
        dim = tmp_dict["dim"]
        if np.size(dim) != 2:
            data = None
        if data is not None:
            if dim[0] <= 1:
                ret = 0
            elif dim[1] <= 1:
                ret = 0
            else:
                ret = 1
        if continue_on_err:
            ret = 1
        radar_dict = dpg.navigation.get_radar_par(scan, reload=reload)

        mode = radar_dict["mode"]
        elevation_off = radar_dict["elevation_off"]
        azimut_off = radar_dict["azimut_off"]
        res = radar_dict["range_res"]
        az_res = radar_dict["azimut_res"]
        site_coords = radar_dict["site_coords"]
        beam_width = radar_dict["beam_width"]

        if mode == 1:
            coord = elevation_off
        elif mode == 2:
            coord = azimut_off
        else:
            ret = 0

        if coord >= 360.0:
            coord -= 360.0

        if ret > 0 and max_el is not None:
            if max_el > 0.0 and mode == 1:
                if coord > max_el:
                    ok = 0

        if ret > 0 and min_el is not None:
            if min_el >= 0.0 and mode == 1:
                if coord < min_el:
                    ok = 0

        if ok > 0:
            if coord_set is None:
                coord_set = [coord]
                valids = [scan]
            else:
                coord_set.append(coord)
                valids.append(scan)

        if ret > 0 and ok > 0:
            for ddd in range(len(dim)):
                if scan_dim[ddd] < dim[ddd]:
                    scan_dim[ddd] = dim[ddd]
            if res < range_res or range_res == 0.0:
                range_res = res
                best_scan = scan
            if az_res < azimut_res or azimut_res == 0.0:
                azimut_res = az_res

    if mode == 1:
        nAz = int(360.0 / abs(azimut_res))
        if scan_dim[0] > nAz:
            scan_dim[0] = nAz
    if coord_set is None:
        tmp = scans

    if mode == 1:
        if np.all(coord_set == coord_set[0]):
            ind = range(len(coord_set))
        else:
            ind = np.argsort(coord_set)
        coord_set = [coord_set[i] for i in ind]
        valids = [valids[i] for i in ind]

    scans = valids
    values, calib, out_dict = dpg.calibration.get_array_values(scans[best_scan_ind])
    simmetric = out_dict["simmetric"]
    scale = out_dict["scale"]
    unit = out_dict["unit"]

    # if nScans == 1:
    #     scans = scans[0]

    # TODO: non ha senso ricreare un nuovo dizionario mescolando tutto.
    #  sarebbe meglio tenere il navigation e il calibration separati e ritornare 2 dizionari!!!!!!!!!!!
    ret_dict = {}
    ret_dict["scans"] = scans
    ret_dict["best_scan_ind"] = best_scan_ind
    ret_dict["mode"] = mode
    ret_dict["azimut_res"] = azimut_res
    ret_dict["scan_dim"] = scan_dim
    ret_dict["range_res"] = range_res
    ret_dict["simmetric"] = simmetric
    ret_dict["scale"] = scale
    ret_dict["unit"] = unit
    ret_dict["values"] = values
    ret_dict["site_coords"] = site_coords
    ret_dict["beam_width"] = beam_width
    ret_dict["coord_set"] = coord_set

    return ret_dict


def check_coord_set(
        volId1,
        volId2,
        reload: bool = False,
        reverse_order: bool = False,
        coordSet=None,
):
    """
    Compares and aligns the coordinate sets of two volumes, returning aligned scan data.

    This function compares the coordinate sets of two volumes, identified by 'volId1' and
    'volId2', and aligns their scans based on the closest matching coordinates. It handles
    optional reloading of data and can reverse the order of the coordinate set. The function
    returns a tuple containing mode, values, quality scans, scans, and the coordinate set.

    Args:
        volId1: The ID of the first volume.
        volId2: The ID of the second volume.
        reload (bool, optional): If True, forces the reloading of scan data. Defaults to False.
        reverse_order (bool, optional): If True, reverses the order of the coordinate set.
                                        Defaults to False.
        coordSet (list or None, optional): A list of coordinates to use instead of extracting
                                           from 'volId1'. Defaults to None.

    Returns:
        tuple: A tuple containing:
               - mode (int): The mode of the scan.
               - values: The values associated with 'volId2'.
               - qScans: Quality scans aligned with 'volId1'.
               - scans: The scans from 'volId1'.
               - coordSet: The coordinate set used for alignment.

    Note:
        The function retrieves scans and coordinate sets from both volumes using 'get_scans'.
        It then aligns the scans from 'volId2' (quality scans) with those from 'volId1'
        based on the closest matching coordinates. The alignment considers a threshold for
        matching coordinates and handles cases where there is no close match. Optionally,
        the order of the coordinates and scans can be reversed.
    """
    """
    input: volId1, volId2, reload, reverse_order, coordSet
    output: mode, values, qScans, scans, coordSet(?)
    """
    scans = None
    mode = None
    qNodes = None
    values = None

    scan_dict_volId2 = get_scans(volId2, reload=reload)
    if scan_dict_volId2 is not None:
        qualSet = scan_dict_volId2["coord_set"]
        values = scan_dict_volId2["values"]
        qNodes = scan_dict_volId2["scans"]

    if isinstance(qNodes, list):
        qScans = len(qNodes)
    else:
        qScans = 0
    if isinstance(coordSet, list) or isinstance(coordSet, np.ndarray):
        nScans = np.size(coordSet)
    elif isinstance(coordSet, Number):
        coordSet = [coordSet]
        nScans = np.size(coordSet)
    else:
        nScans = 0

    if nScans == 0:
        scan_dict_volId1 = get_scans(volId1)
        scans = np.array(scan_dict_volId1["scans"])
        coordSet = np.array(scan_dict_volId1["coord_set"])
        mode = scan_dict_volId1["mode"]
        if scans is not None:
            nScans = len(scans)
        else:
            nScans = 0
    if nScans == 0 or qScans == 0:
        return mode, values, [None] * nScans, scans, coordSet

    ind = np.argsort(coordSet)
    if reverse_order:
        ind = ind[::-1]
    # coordSet = coordSet[ind]
    coordSet = [coordSet[i] for i in ind]
    if scans is not None and nScans > 1:
        scans = scans[ind].tolist()

    qScans = [None] * nScans

    for sss in range(nScans):
        diff = qualSet - coordSet[sss]
        abs_diff = np.abs(diff)  # Take the absolute value of each element in diff
        mmm = np.nanmin(abs_diff)  # Find the minimum value in abs_diff, ignoring NaNs
        ind = np.argmin(
            np.where(np.isnan(diff), np.inf, abs_diff)
        )  # Get the index of the min value
        if mmm < 0.3:
            qScans[sss] = qNodes[ind]
        else:
            log_message(f"Cannot find {coordSet[sss]} in {qualSet}", level="WARNING")

    # if nScans == 1:
    #     qScans = qScans[0]

    return mode, values, qScans, scans, coordSet


def get_volumes(tree=None, measure=None, corrected=None, sampled=None, path: str = ""):
    """
    Returns a volume node in a data tree based on a specified measure.
    If a tree is not provided, a new one is created.

    Args:
               - tree:        tree for which search volumes of data
               - measure:     number of mesures
               - corrected:   --
               - sampled:     --
               - path:        --

    Returns:
               - volumes:     volume of data for the specified input tree
               - flag:        a flag integer number for specify if the number of volumes for the tree is 1 or less
               then 1
    """

    if tree is None:
        tree = dpg.tree.createTree(path=path)

    nMeas = np.size(measure)
    if nMeas == 1:
        volumes = find_volume(tree=tree, measure=measure)
        if isinstance(volumes, dpg.node__define.Node):
            return volumes, 1

    name = dpg.cfg.getValueDescName()
    nVolumes, sons = dpg.tree.findAttr(tree, name=name, all=True, down=True)

    if len(nVolumes) <= 0:
        log_message(f"No data found in {dpg.tree.getNodePath(tree)}", level="ERROR")
        return None, 0

    log_message(
        "PARTE ANCORA DA IMPLEMENTARE DENTRO get_volumes()",
        level="ERROR",
        all_logs=True,
    )
    return


def get_needed_volumes(
        prod_id: str,
        measure: str = None,
        remove_if_not_exists=False,
        raw_tree=None,
        sampled: bool = None,
        corrected: bool = None,
):
    """
    Retrieves the necessary volumes for a given product based on specified measures,
    correcting and sampling flags, and handles cases where volumes are not found.

    This function attempts to retrieve the necessary radar data volumes from a raw data tree
    for a specified product. It checks for the existence of volumes based on the given
    measure, applies sampling and correction settings, and manages the scenario where the
    volumes are not found, potentially removing nodes if specified.

    Args:
        prod_id (str):                              The product ID for which volumes are to be retrieved.
        measure (str, optional):                    The measure to search for in the volumes. Defaults to None
        remove_if_not_exists (bool, optional):      If True, removes nodes if the volumes do not
                                                    exist. Defaults to False
        raw_tree (dpg.node__define.Node, optional): The raw data tree to search within. Defaults to None
        sampled (bool, optional):                   Flag indicating whether the volume should be sampled. Defaults to
        None
        corrected (bool, optional):                 Flag indicating whether the volume should be corrected. Defaults
        to None

    Returns:
        tuple:                                      A tuple containing the volumes found and the number of volumes
        found.

    Note:
        The function initializes the raw data tree if it is not provided. If the measure is
        not specified, it logs an error. It then attempts to retrieve the volumes using
        `get_volumes`. If no volumes are found and the 'any' flag is used (not implemented),
        it tries to retrieve moments and reattempts to get volumes. Depending on the removal
        flag, it may remove nodes if volumes are still not found. The function ensures that
        the product is marked as corrected and sampled if volumes are found.
    """
    if raw_tree is None:
        raw_tree = getRawTree(prod_id)

    if measure is None:
        log_message(
            "Measure is NONE. Caso non ancora implementato",
            level="ERROR",
            all_logs=True,
        )
        return
        # measure = dpg.radar.get_par(prodId, 'measure', default='', start_with=True)
        # if measure == '':
        #     if any is None:
        #         return
        #     measure = 'CZ'

    if sampled is None:
        sampled, _, _ = dpg.radar.get_par(prod_id, "sampled", default=0)

    volumes, nVol = get_volumes(
        tree=raw_tree, measure=measure, corrected=corrected, sampled=sampled
    )

    if nVol == 0 and any:
        log_message(
            "nVOL == 0: caso non ancora implementato",
            all_logs=True,
            level="ERROR",
        )
        moments, nm = get_moments(raw_tree=raw_tree, optim=True)
        if nm > 0:
            measure = moments[0]
        volumes, nVol = get_volumes(
            tree=raw_tree, measure=measure, corrected=corrected, sampled=sampled
        )

    rem = 0
    if nVol == 0:
        if remove_if_not_exists:
            rem = 3
        else:
            rem, _, _ = dpg.radar.get_par(
                prod_id, "toRemove", 0, parFile=dpg.cfg.getProcDescName()
            )

    if rem > 2:
        id = prod_id
        if rem > 3:
            log_message(
                f"rem > 3: caso non ancora implementato. Line {inspect.currentframe().f_lineno} in "
                f"{inspect.getframeinfo(inspect.currentframe()).filename}",
                all_logs=True,
                level="ERROR",
            )
            id = dpg.tree.getParent(id)
    else:
        valid = 1
        dpg.radar.set_par(prod_id, "sampled", 0)

    return volumes, nVol


def roundClutterElev(elevation: str) -> (float, str):
    """
    Rounds the elevation to the nearest half unit and formats it as a string with one decimal place.

    Args:
        elevation (float): The elen to round.

    Returns:
        tuple:
            - newEl (float): The elevation rounded to the nearest half unit.
            - strEl (str): The rounded elevation as a string with one decimal place.
    """

    newEl = round(elevation * 2) / 2
    strEl = "{:.1f}".format(newEl)

    return newEl, strEl


def getClutterMapNode(scanId, clutterRoot):
    """
    Method that returns the last node added to the tree data structure, calculated based on radar coverage.

    Args:
        - scanId (Node):  Node representing the radar scan, used to retrieve array information and radar parameters.
        - clutterRoot (Node): Node to which add the new calculated node.

    Returns:
        - node (Node): Final node added to the tree structure.
    """

    _, _, dim, _, _, _, _, _ = scanId.getArrayInfo()

    if dim is None:
        log_message("Cannot evaluate coverage!", level="WARNING", all_logs=True)
        return
    if len(dim) <= 1:
        log_message("Cannot evaluate coverage!", level="WARNING", all_logs=True)
        return

    out_dict = dpg.navigation.get_radar_par(scanId)
    par = out_dict["par"]
    range_res = out_dict["range_res"]
    elevation = out_dict["elevation_off"]

    newEl, strEl = roundClutterElev(elevation)

    cov = dim[1] * range_res
    covName = str(round(cov / 1000.0))
    node, _ = dpg.tree.addNode(clutterRoot, covName)
    node, _ = dpg.tree.addNode(node, strEl)

    if node is None:
        log_message(
            "Cannot find ClutterMap with Coverage "
            + covName
            + " and Elevation "
            + strEl,
            level="WARNING",
            all_logs=True,
        )

    return node


def get_single_scan(volId, elev):
    """

    Get the single scan from volId

    Args:
        volId1: The ID of the first volume.
        elev: A list of coordinates.

    Returns:
        scans: The scans from 'volId1'.
    """
    _, _, scans, _, coordSet = check_coord_set(None, volId, coordSet=elev)

    if np.size(scans) <= 0 or scans is None:
        return None

    if isinstance(scans, list):
        return scans[0]
    elif isinstance(scans, np.ndarray):
        log_message("QUALCOSA NON VA, DA CONTROLLARE", level="ERROR")

    else:
        return scans


def search_raw_path(prod_id, sep=os.sep, searchFile=None, parFile=None):
    """
    Searches for the path of raw data based on product ID and various parameters.

    Parameters:
        - prod_id: Node used to retrieve radar parameters.
        - sep: The separator used in paths (default is the operating system's default separator).
        - searchFile: Optional file to check against (default is None).
        - parFile: Optional parameter file for configuration (default is None).

    Returns:
        - raw_path: The generated search path based on provided parameters.
        - searchFile: The file name used in the search process.
    """
    if parFile is None:
        parFile = dpg.cfg.getScheduleDescName()

    searchStep, site, _ = dpg.radar.get_par(
        prod_id, "searchStep", -5, parFile=parFile
    )
    searchHours, _, _ = dpg.radar.get_par(
        prod_id, "searchHours", 0.5, prefix=site, parFile=parFile
    )
    searchPath, _, _ = dpg.radar.get_par(
        prod_id, "searchPath", "", prefix=site, parFile=parFile
    )
    searchFile, _, _ = dpg.radar.get_par(
        prod_id, "searchFile", "", prefix=site, parFile=parFile
    )
    bilateral, _, _ = dpg.radar.get_par(prod_id, "bilateral", 0, prefix=site, parFile=parFile)
    tmp, _, _ = dpg.radar.get_par(prod_id, "exclude", "", prefix=site, parFile=parFile)

    exclude = tmp if tmp != "" else None
    raw_path = dpg.path.checkPathname(searchPath, with_separator=False)
    date, time, _ = dpg.times.get_time(prod_id)
    strDate = dpg.times.checkDate(date, sep=sep, year_first=True)
    strDate += sep
    strDate += dpg.times.checkTime(time=time, sep="")[0]
    raw_path += sep
    raw_path += strDate

    raw_path = dpg.times.search_path(
        raw_path,
        date,
        time,
        searchStep,
        searchHours,
        bilateral=bilateral,
        check_files=searchFile,
        exclude=exclude,
    )

    return raw_path, searchFile


def getCurrRawTree(moment, sampled=False):
    """
    Retrieves the current raw tree node, optionally filtering by a specific node.

    Parameters:
        - moment: The name of the descendant node to find.
        - sampled: Flag to specify if the tree is sampled (default is False).

    Returns:
        - raw_tree: The raw tree node for the given moment or the entire raw tree if no moment is specified.
    """
    raw_tree = getRawTree(sampled=sampled)
    if np.size(moment) < 1 or moment == "":
        return raw_tree

    return dpg.tree.findSon(node=raw_tree, name=moment)


def remove_old_samples(samplesNode, max_samples):
    dpg.tree.updateTree(samplesNode, only_current=True)
    scans = dpg.tree.getSons(samplesNode)
    if len(scans) < max_samples:
        return

    dates = []
    times = []
    nodes = []

    for sss in range(len(scans)):
        date, time, ret = dpg.times.get_time(scans[sss])
        if ret > 0:
            if np.size(dates) == 0:
                dates = [date]
                times = [time]
                nodes = [scans[sss]]
            else:
                dates = dates + [date]
                times = times + [time]
                nodes = nodes + [scans[sss]]

    nScans, ind, _ = dpg.times.sortDates(dates, times, oldest_first=True)
    nScans -= max_samples
    for sss in range(nScans):
        dpg.tree.removeNode(nodes[ind[sss]], directory=True)

    return


def remove_old_nodes(samplesNode, currDate, currTime, max_hours):
    dpg.tree.updateTree(samplesNode, only_current=True)
    scans = dpg.tree.getSons(samplesNode)
    for sss in scans:
        date, time, ret = dpg.times.get_time(sss)
        if ret:
            nh, _ = dpg.times.getNHoursBetweenDates(currDate, currTime, date, time, double=True)
            if nh > max_hours:
                dpg.tree.removeNode(sss, directory=True)
