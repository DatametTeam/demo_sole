import os

import numpy as np

import sou_py.dpg as dpg
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION GetAllFile 
FUNCTION GetArgs 
FUNCTION GetAvailableFontList 
FUNCTION GetCombinationName 
FUNCTION GetCurrSymbol 
FUNCTION GetExecuteName 
FUNCTION GetFormatExport 
FUNCTION GetInfoName 
FUNCTION GetLayoutDescName 
FUNCTION GetLevels 
FUNCTION GetLocalSites 
FUNCTION GetMMICfg 
FUNCTION GetNearestSensor 
FUNCTION GetOldSensorCode 
FUNCTION GetOptions 
FUNCTION GetProducts 
FUNCTION GetSensorName 
FUNCTION GetSites  
FUNCTION GetUsersListName 
FUNCTION LoadSummary 
FUNCTION OS_IsUnix 
PRO SetCurrSymbol 
PRO UPDATE_PATH 
FUNCTION GetArg                     // UNUSED
FUNCTION GetCommandFileName         // UNUSED
FUNCTION GetCorrPath                // UNUSED
FUNCTION GetDefaultAshStatusFile    // UNUSED
FUNCTION GetLinkName                // UNUSED
FUNCTION GetMosaicNodeName          // UNUSED
FUNCTION GetOutConvName             // UNUSED
FUNCTION GetPreprocessing           // UNUSED
FUNCTION GetProdPath                // UNUSED
FUNCTION GetRawListName             // UNUSED
FUNCTION GetRpgPath                 // UNUSED
FUNCTION GetSchedules               // UNUSED
FUNCTION GetStatisticsName          // UNUSED
FUNCTION IsTrueTypeFont             // UNUSED
"""


def getSubDir(path: str, basename=None):
    """
    Retrieves a list of subdirectories from a given path.

    This function scans the provided directory path and compiles a list of all subdirectories within it.
    It ensures that the path is valid and handles cases where the path is a directory or a file.

    Args:
        path (str):          The directory path from which subdirectories are to be listed.
        basename (optional): An optional parameter currently not implemented in the function.
                             Intended for future use to filter or process directory names.
                             Defaults to None.

    Returns:
        list:                A list of paths for each subdirectory within the specified directory.
                             If the path is a file, the list contains only that file.

    Note:
        The function first checks if the provided path is a string and not empty. If the path is
        a directory, it lists all subdirectories within. If the path is a file, it returns a list
        containing only the file path. The 'basename' parameter is present for potential future
        enhancements but is not currently used. This function is useful for directory traversal,
        particularly in file management and organization tasks.
    """
    dirs = []
    if not isinstance(path, str):
        print("Error: path is not string")
        return dirs
    if path == "":
        return dirs

    curr = path  # CheckPathname(path, /WITHSEPARATOR)
    if curr[-1] != os.path.sep:
        curr += os.path.sep

    if os.path.isdir(curr):
        content = os.listdir(curr)
        for obj in content:
            if os.path.isdir(curr + obj):
                dirs.append(curr + obj)
            # end
        # end
    elif os.path.isfile(curr):
        print("Warning: e' corretto dirs = [curr] quando curr e' file?")
        dirs = [curr]
    if basename:
        # dir=file_basename(dir)
        pass
    return dirs


def getArrayDescName() -> str:
    """
    Retrieves the default name for an array descriptor file.

    This function returns a predefined string that represents the default name for an array
    descriptor file. It's a simple function with no parameters and a fixed return value.

    Returns:
        str: The default name of the array descriptor file.

    Note:
        The function is designed to provide a consistent file name for array descriptor files
        across different parts of an application or system. By using this function, different
        components of the system can refer to the same descriptor file name, ensuring uniformity
        and reducing the likelihood of errors from hard-coded file names.
    """
    return "generic.txt"


def getGeoDescName() -> str:
    """
    Retrieves the default name for a geographic descriptor file.

    This function returns a fixed string that represents the default name for a geographic
    descriptor file. It's a straightforward function without any parameters and returns a
    constant value.

    Returns:
        str: The default name of the geographic descriptor file.

    Note:
        The function provides a standard file name for geographic descriptor files across
        different modules or parts of a system. The use of this function ensures consistency
        in file naming conventions for geographic descriptors, aiding in system-wide uniformity
        and potentially reducing errors related to hardcoded file names in various parts of the system.
    """
    return "navigation.txt"


def getSitesListName() -> str:
    """
    Retrieves the default name for the sites list file.

    This function returns a consistent string identifying the default file name used for
    storing or referencing a list of sites. It is a simple function that requires no parameters
    and provides a fixed return value.

    Returns:
        str: The default name of the sites list file.

    Note:
        Utilizing this function promotes uniformity in file naming conventions for the sites
        list across various parts of an application or system. This approach helps maintain
        consistency and can reduce errors associated with hardcoded file names, especially in
        larger systems where such lists are commonly used for referencing multiple sites or locations.
    """
    return "sites.txt"


def getTimesListName() -> str:
    """
    Retrieves the default name for the times list file.

    This function returns a predefined string that represents the default name for a times list file.
    It is a straightforward function with no parameters and a constant return value.

    Returns:
        str: The default name of the times list file.

    Note:
        By providing a consistent file name for times list files, this function ensures uniformity
        across various parts of an application or system. The use of this function is particularly
        beneficial in systems that manage or reference time-related data, helping to prevent
        discrepancies and errors that could arise from hard-coded or inconsistent file naming.
    """
    return "times.txt"


def getParDescName(gui: bool = False) -> str:
    """
    Retrieves the default name for the parameters descriptor file, with an option for GUI-specific naming.

    This function returns a standard file name for a parameters descriptor file. It provides an option
    to get a GUI-specific file name, depending on whether the file is intended for use in a graphical
    user interface (GUI) environment.

    Args:
        gui (bool, optional): If True, returns the name of the GUI-specific parameters descriptor file.
                              Defaults to False.

    Returns:
        str: The default name of the parameters descriptor file. Returns a GUI-specific name if 'gui' is True.

    Note:
        The function facilitates the use of different descriptor files for GUI and non-GUI contexts within
        an application or system. By using this function, it becomes easier to manage and reference
        appropriate parameter descriptor files based on the application's context, ensuring better
        organization and reducing potential errors from manually specifying file names.
    """
    if gui:
        return "parameters.gui"
    return "parameters.txt"


def getMosaicDescName(advanced: bool = False) -> str:
    """
    Retrieves the default name for the mosaic descriptor file, with an option for advanced naming.

    This function returns a standard file name for a mosaic descriptor file and provides an option
    to use an advanced naming convention. The advanced option allows for differentiation between
    basic and more complex mosaic descriptor files.

    Args:
        advanced (bool, optional): If True, returns the name of the advanced mosaic descriptor file.
                                   Defaults to False.

    Returns:
        str: The default name of the mosaic descriptor file. Returns an advanced file name if 'advanced' is True.

    Note:
        Utilizing this function helps maintain consistent naming conventions for mosaic descriptor files
        within an application or system. The option to choose between basic and advanced naming schemes
        allows for flexibility and better organization, especially in systems that manage different
        levels of data complexity or details within mosaic files.
    """
    if advanced:
        return "mosaic_adv.txt"
    return "mosaic.txt"


def getArchiveDescName(mk: bool = False) -> str:
    """
    Retrieves the default name for the archive descriptor file, with an option for a specific naming format.

    This function returns a predefined file name for an archive descriptor file. It includes an option
    to return a different file name format based on the 'mk' parameter, which can be useful in specific
    archival or data management contexts.

    Args:
        mk (bool, optional): If True, the function returns a different format for the archive descriptor
                             file name (typically used in makefile contexts). Defaults to False.

    Returns:
        str: The default name of the archive descriptor file. Returns a different format if 'mk' is True.

    Note:
        The function's ability to switch between standard and alternate file name formats (based on the
        'mk' parameter) provides flexibility in addressing different file naming requirements. This is
        particularly useful in systems where archival processes may vary or require specific file naming
        conventions, such as in automated building or deployment scenarios.
    """
    if mk:
        return "archiviation.mk"
    return "archiviation.txt"


def getScheduleDescName() -> str:
    """
    Retrieves the default name for the schedule descriptor file.

    This function returns a fixed string that represents the default name for a schedule
    descriptor file. It is a straightforward function without any parameters and provides
    a constant return value.

    Returns:
        str: The default name of the schedule descriptor file.

    Note:
        The function provides a standardized file name for schedule descriptor files across
        various parts of an application or system. Using this function ensures consistency
        in file naming conventions for schedule-related information, which is particularly
        beneficial in systems involving task scheduling, time management, or event planning.
    """
    return "schedule.txt"


def getProdListName(interactive: bool = False, progress: bool = False) -> str:
    """
    Retrieves the default name for the product list file, with options for interactive and progress-specific naming.

    This function returns a standard file name for a product list file. It provides options to get names for
    interactive mode or progress tracking, depending on the intended use of the product list.

    Args:
        interactive (bool, optional): If True, returns the name of the product list file specific to interactive mode.
                                      Defaults to False.
        progress (bool, optional): If True, returns the name of the product list file specific to progress tracking.
                                   Defaults to False.

    Returns:
        str: The default name of the product list file. Returns a name specific to interactive mode or progress
             tracking if corresponding flags are set to True.

    Note:
        The function facilitates the use of different product list files for various operational contexts within
        an application or system, such as interactive sessions or progress monitoring. By using this function,
        it becomes easier to manage and reference appropriate product list files based on the specific needs of
        different parts of the system, enhancing organization and reducing potential errors from manually
        specifying file names.
    """
    if progress:
        return "progress.txt"
    if interactive:
        return "int_products.txt"
    return "products.txt"


def getProcDescName() -> str:
    """
    Retrieves the default name for the process descriptor file.

    This function returns a predetermined string that represents the default name for a process
    descriptor file. It's a simple function without parameters and provides a fixed return value.

    Returns:
        str: The default name of the process descriptor file.

    Note:
        The function offers a standardized file name for process descriptor files across
        different modules or parts of a system. The use of this function promotes consistency
        in file naming conventions for process-related information, aiding in system-wide
        uniformity and potentially reducing errors related to hardcoded file names in various
        parts of the system.
    """
    return "process.txt"


def getValueDescName() -> str:
    """
    Retrieves the default name for the value descriptor file.

    This function returns a fixed string that represents the default name for a value descriptor file,
    typically used in calibration contexts. It is a straightforward function with no parameters and a
    constant return value.

    Returns:
        str: The default name of the value descriptor file.

    Note:
        The function provides a standard file name for value descriptor files, particularly for calibration
        purposes, across various parts of an application or system. Using this function ensures consistency
        in file naming for calibration descriptors, which is crucial for maintaining uniformity in calibration
        processes and reducing potential errors from manually specified or inconsistent file names.
    """
    return "calibration.txt"


def getItemDescName() -> str:
    """
    Retrieves the default name for the item descriptor file.

    This function returns a fixed string indicating the default name for an item descriptor file.
    It's a straightforward function with no parameters and a constant return value.

    Returns:
        str: The default name of the item descriptor file.

    Note:
        The function serves to provide a consistent file name for item descriptor files across
        various parts of an application or system. By using this function, different components
        of the system can refer to the same descriptor file name, ensuring uniformity and reducing
        the likelihood of errors from hard-coded file names.
    """
    return "item.txt"


def getModelDescName() -> str:
    """
    Retrieves the default name for the model descriptor file.

    This function returns a predetermined string that represents the default name for a model
    descriptor file. It is a simple function without parameters and offers a constant return value.

    Returns:
        str: The default name of the model descriptor file.

    Note:
        The function is designed to provide a standardized file name for model descriptor files
        across different modules or parts of a system. Using this function ensures consistency
        in file naming conventions for model-related information, which is particularly beneficial
        in systems involving data modeling, simulation, or analysis, helping to prevent discrepancies
        and errors that could arise from hard-coded or inconsistent file names.
    """
    return "model.txt"


def getProductDescName() -> str:
    """
    Retrieves the default name for the product descriptor file.

    This function returns a standard string that denotes the default name for a product descriptor
    file. It is a straightforward function with no parameters and provides a consistent return value.

    Returns:
        str: The default name of the product descriptor file.

    Note:
        The function serves to establish a uniform file name for product descriptor files across
        various parts of an application or system. By utilizing this function, consistent naming
        is maintained for files that describe products, which is crucial in systems where multiple
        products or output types are managed, ensuring clarity and reducing the risk of errors
        associated with manually specified or inconsistent file names.
    """
    return "definition.txt"


def getDefaultProjection() -> str:
    """
    Retrieves the default projection type from the configuration settings.

    This function returns the default projection type as specified in the system's configuration settings.
    If no specific projection type is set in the configuration, it defaults to 'Gnomonic'.

    Returns:
        str: The default projection type as specified in the configuration, or 'Gnomonic' if not specified.

    Note:
        The function is essential for systems that require geographical data projection. By fetching the
        default projection type from the configuration, it ensures consistent use of projection methods
        across different parts of the system. This approach aids in maintaining uniformity in geographical
        data handling and reduces the risk of inconsistencies that might arise from manually setting or
        changing projection types in different modules or components of the system.
    """
    return getCfg("Default_Projection", default="Gnomonic")


def getCfg(key: str, default=0, prefix: bool = False):
    """
    Retrieves a configuration value based on a specified key from the system's configuration settings.

    This function fetches the value associated with a given key from the system's configuration file.
    If the key is not found, it returns a default value. The function also has an option to consider
    a prefix in the configuration retrieval process.

    Args:
        key (str): The key for which the configuration value is to be retrieved.
        default (optional): The default value to return if the key is not found in the configuration.
                            Defaults to 0.
        prefix (bool, optional): If True, considers a prefix in the configuration retrieval.
                                 Defaults to False.

    Returns:
        The value associated with the specified key in the configuration. Returns the default value if
        the key is not found.

    Note:
        The function is crucial for accessing various configuration settings within an application or system.
        By specifying the key and optionally a default value, it allows for flexible retrieval of configuration
        parameters. This is particularly useful in systems with modular or dynamic configurations, where
        settings might vary or need to be adjusted based on different operational contexts.
    """
    CFG_PTR = dpg.attr.loadAttr(
        path=dpg.path.getDir("CFG", with_separator=True), name="IDL.ini"
    )

    val, _, _ = dpg.attr.getAttrValue(CFG_PTR, key, default)

    return val


def getClutterHome(sep: str = None) -> str:
    """
    Retrieves the path to the Clutter home directory.

    This function returns the path to the Clutter home directory by reading the 'CLUTTER' environment variable.
    Optionally, a separator (`sep`) can be appended to the path if provided.

    Args:
        sep (str, optional): An optional separator to append to the Clutter home directory path.
                            Defaults to None.

    Returns:
        str: The path to the Clutter home directory, or an empty string if the 'CLUTTER' environment variable
             is not set.

    Note:
        The Clutter home directory is typically used to store user-specific configuration, data, and settings
        related to the Clutter application or framework. This function is useful for locating and accessing
        user-specific Clutter-related files and directories.
    """
    # Retrieve the path from an environment variable named 'CLUTTER'
    # clutter_path = os.environ.get('CLUTTER')
    clutter_path = dpg.path.getDir("CLUTTER", with_separator=True, sep=sep)

    # If separator is specified and the path doesn't end with it, append it
    if sep and clutter_path and not clutter_path.endswith(sep):
        clutter_path += sep

    return clutter_path


def getDEMPath(hr: bool = False) -> str:
    """
    Retrieves the path to the DEM directory.

    This function fetches the path to the DEM directory within the 'backgrounds' environment variable directory.
    It allows for selection between high-resolution and standard-resolution DEMs based on the hr.
    If the high-resolution DEM is requested (hr set to True) but if not found, it logs a warning and defaults to the standard-resolution DEM directory.

    Args:
        hr (bool, optional): to indicate whether to fetch the high-resolution DEM directory (True) or the standard-resolution (False). Defaults to False.

    Returns:
        str: The path to the selected DEM directory, constructed by joining the 'backgrounds' directory with the appropriate DEM subdirectory ('HRDEM', 'DEMHR', or 'DEM').

    Note:
        The function checks for two possible names for the high-resolution DEM directory ('HRDEM' and 'DEMHR').
    """

    dir = dpg.path.getDir("BACKGROUNDS", with_separator=True)
    if hr:
        if os.path.isdir(dir + "HRDEM"):
            path = os.path.join(dir, "HRDEM")
        else:
            if os.path.isdir(dir + "DEMHR"):
                path = os.path.join(dir, "DEMHR")
            else:
                log_message(
                    "Cannot find High Res DEM. Using Low Res.",
                    level="WARNING",
                    all_logs=True,
                )
                path = os.path.join(dir, "DEM")

    else:
        path = os.path.join(dir, "DEM")

    return path


def getSystem(all_par=None):
    """
    Retrieves the system's home directory path and reorders the list of parameters so that the 'RADAR' parameter is
    placed first.

    The function first obtains the home directory path for the application and a list of subdirectories.
    If the list `all_par` is provided and contains a subdirectory named 'RADAR' but it's not in the first position,
    the list is reordered to move 'RADAR' to the front. If `all_par` is `None`, no reordering occurs.

    Args:
        all_par (list, optional): A list of directory paths, possibly including a directory named 'RADAR'.
                                  Defaults to None.

    Returns:
        tuple:
            - str: The home directory path of the system.
            - list: A reordered copy of the input list with 'RADAR' in the first position, if applicable.
                    Returns None if `all_par` is None.
    """
    new_all_par = None
    if all_par is not None:
        log_message("getSystem da controllare", level='ERROR')
        path = dpg.path.getDir('SCHEDULES')
        all_par = dpg.cfg.getSubDir(path, basename=True)
        new_all_par = all_par.copy()
        all_par_base = [os.path.basename(elem) for elem in all_par]
        ind = [index for index, element in enumerate(all_par_base) if element == 'RADAR']
        if ind[0] > 0:
            new_all_par[ind[0]] = new_all_par[0]
            new_all_par[0] = all_par[ind[0]]

    h, _, system = dpg.path.getHome()

    return system, new_all_par


def getDefaultStatusFile(siteName):

    path = dpg.path.getDir('STATUS', with_separator=True)
    return path + siteName + '.txt'


def getVPRHome(site_name=None):

    rootpath = dpg.path.getDir('VPR', with_separator=True)
    if site_name is None:
        return rootpath

    rootpath += site_name
    return rootpath
