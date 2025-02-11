import os
import sys

import sou_py.dpg as dpg

from sys import platform

from sou_py.dpg.log import log_message
from sou_py.paths import DATAMET_RADVIEW_PATH, DATAMET_DATA_PATH

"""
Funzioni ancora da portare
FUNCTION CheckEnv
FUNCTION CheckExtensionPresent
PRO SetHost
FUNCTION CheckLastBranch        // UNUSED
FUNCTION GetRawPath             // UNUSED
PRO SetRawPath                  // UNUSED
"""


def checkPathname(pathname, with_separator=False):
    """
    Processes a pathname by resolving environment variables and ensuring proper path handling.

    Args:
        pathname (str): The input pathname to process.
        with_separator (bool, optional): If True, ensures the path ends with a separator. Defaults to False.

    Returns:
        str: The processed pathname.
    """

    path = checkAt(pathname)
    path, contain_sep, sep = checkSep(path)
    pos = path.find("$")

    if pos == 0:
        if contain_sep == 1:
            posSep = path.find(sep)
            rest = path[posSep:]

            env = path[1:posSep]
        else:
            env = path[1:]

        path = getDir(env)

        if contain_sep == 1:
            path += rest

    if not with_separator:
        return path

    length = len(path) - 1
    pos = path.rfind(sep)

    if pos != length:
        path += sep

    return path, sep


def getNewPath(rif, sep="", alt_path=None):
    """
    This function maps a given identifier (rif) to a corresponding file path within a
    complex directory structure. It covers a wide range of use cases such as configuration
    files, schedules, overlays, alerts, and other system paths. The function dynamically
    builds paths using the os.path.join.

    Args:
        rif (str): The reference identifier for the path to retrieve.
        sep (str, optional): Unused parameter.
        alt_path (str, optional): Unused parameter.

    Returns:
        str: The constructed file path based on the 'rif' value.
    """
    # cfgPath = dpg.globalVar.GlobalState.RV_HOME + 'cfg' + sep
    cfgPath = os.path.join(dpg.globalVar.GlobalState.RV_HOME, "cfg")

    rif = rif.upper()

    if rif == "DATA":
        value = dpg.globalVar.GlobalState.dataPath
    elif rif == "SCHEDULES":
        # value = dpg.globalVar.GlobalState.dataPath + 'schedules'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "schedules")
    elif rif == "TEMP_PROCESSING":
        # value = dpg.globalVar.GlobalState.dataPath + 'schedules' + sep + dpg.globalVar.GlobalState.RV_HOST + sep + 'session'
        value = os.path.join(
            dpg.globalVar.GlobalState.dataPath,
            "schedules",
            dpg.globalVar.GlobalState.RV_HOST,
            "session",
        )
    elif rif == "OVERLAYS":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "overlays")
    elif rif == "ALL_ALERTS":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'shapes' + sep + 'ZoneAllerta'
        value = os.path.join(
            dpg.globalVar.GlobalState.dataPath,
            "overlays",
            "shapes",
            "ZoneAllerta",
        )
    elif rif == "ALERTS":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'alerts'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "overlays", "alerts")
    elif rif == "GRIDS":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'grids'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "overlays", "grids")
    elif rif == "OBJECTS":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'objects'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "overlays", "objects")
    elif rif == "SHAPES":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'shapes'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "overlays", "shapes")
    elif rif == "BASINS":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'basins'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "overlays", "basins")
    elif rif == "RAINGAUGES_Z":
        # value = dpg.globalVar.GlobalState.dataPath + 'overlays' + sep + 'radargauges'
        value = os.path.join(
            dpg.globalVar.GlobalState.dataPath, "overlays", "radargauges"
        )
    elif rif == "USER":
        value = dpg.globalVar.GlobalState.dataPath + "user"
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "user")
    elif rif == "UNDERLAYS":
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "underlays")
        if value != "C:\\datamet\\data\\underlays" and os.path.isdir(
                "C:\\datamet\\data\\underlays"
        ):
            alt_path = "C:\\datamet\\data\\underlays\\"
    elif rif == "BACKGROUNDS":
        # value = dpg.globalVar.GlobalState.dataPath + 'underlays'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "underlays")
    elif rif == "LEGENDS":
        # value = dpg.globalVar.GlobalState.dataPath + 'user' + sep + 'legends'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "user", "legends")
    elif rif == "TEXTS":
        # value = dpg.globalVar.GlobalState.dataPath + 'user' + sep + 'texts'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "user", "texts")
    elif rif == "LOGOS":
        # value = dpg.globalVar.GlobalState.dataPath + 'user' + sep + 'logos'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "user", "logos")
    elif rif == "PRODUCTS":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp' + sep + 'products'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "tmp", "products")
    elif rif == "TEMP":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "tmp")
    elif rif == "EXPORT":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp' + sep + 'export'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "tmp", "export")
    elif rif == "PRODCLIP":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp' + sep + 'clipboard' + sep + 'products'
        value = os.path.join(
            dpg.globalVar.GlobalState.dataPath, "tmp", "clipboard", "products"
        )
    elif rif == "TEMPLCLIP":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp' + sep + 'clipboard' + sep + 'templates'
        value = os.path.join(
            dpg.globalVar.GlobalState.dataPath, "tmp", "clipboard", "templates"
        )
    elif rif == "TEMP_BUFR":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp' + sep + 'export' + sep + 'BUFR'
        value = os.path.join(
            dpg.globalVar.GlobalState.dataPath, "tmp", "export", "BUFR"
        )
    elif rif == "TEMP_RDP":
        # value = dpg.globalVar.GlobalState.dataPath + 'tmp' + sep + 'rdp'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "tmp", "rdp")
    elif rif == "CLUTTER":
        # value = dpg.globalVar.GlobalState.dataPath + clutter
        if dpg.globalVar.GlobalState.clutterRoot is not None:
            log_message(f"ClutterRoot path has been set externally using: {dpg.globalVar.GlobalState.clutterRoot}",
                        "INFO")
            value = str(dpg.globalVar.GlobalState.clutterRoot)
        else:
            value = os.path.join(dpg.globalVar.GlobalState.dataPath, "clutter")
    elif rif == "INTERCALIB":
        # value = dpg.globalVar.GlobalState.dataPath + 'intercalib'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "intercalib")
    elif rif == "VPR":
        if dpg.globalVar.GlobalState.vprRoot is not None:
            log_message(f"vprRoot path has been set externally using: {dpg.globalVar.GlobalState.vprRoot}", "INFO")
            value = str(dpg.globalVar.GlobalState.vprRoot)
        else:
            value = os.path.join(dpg.globalVar.GlobalState.dataPath, "vpr")
    elif rif == "VARR":
        # value = dpg.globalVar.GlobalState.dataPath + 'varr'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "varr")
    elif rif == "ASSESMENT":
        # value = dpg.globalVar.GlobalState.dataPath + 'assesment'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "assesment")
    elif rif == "STATUS":
        # value = dpg.globalVar.GlobalState.dataPath + 'status'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "status")
    elif rif == "MP_COEFFS":
        # value = dpg.globalVar.GlobalState.dataPath + 'mp'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "mp")
    elif rif == "RAINGAUGES_SYNC":
        # value = dpg.globalVar.GlobalState.dataPath + 'raingauges'
        value = os.path.join(dpg.globalVar.GlobalState.dataPath, "raingauges")
    elif rif == "TEMPLATES":
        if os.path.isdir(
                os.path.join(cfgPath, "templates", dpg.globalVar.GlobalState.RV_HOST)
        ):
            value = os.path.join(
                cfgPath, "templates", dpg.globalVar.GlobalState.RV_HOST
            )
        else:
            value = os.path.join(cfgPath, "templates", "RADAR")
    elif rif == "SITES":
        # value = cfgPath + 'sites'
        value = os.path.join(cfgPath, "sites")
    elif rif == "CFG_SCHED":
        value = os.path.join(cfgPath, "schedules")
    elif rif == "SENSORS":
        # value = cfgPath + 'sensors'
        value = os.path.join(cfgPath, "sensors")
    elif rif == "USERS":
        # value = cfgPath + 'users'
        value = os.path.join(cfgPath, "users")
    elif rif == "BUFR":
        # value = cfgPath + 'calibration' + sep + 'BUFR'
        value = os.path.join(cfgPath, "calibration", "BUFR")
    elif rif == "PROPERTIES":
        # value = cfgPath + 'properties'
        value = os.path.join(cfgPath, "properties")
    elif rif == "HELP":
        # value = cfgPath + 'help'
        value = os.path.join(cfgPath, "help")
    elif rif == "INIT":
        # value = cfgPath + 'init'
        value = os.path.join(cfgPath, "init")
    elif rif == "I18N":
        # value = cfgPath + 'initinit'
        value = os.path.join(cfgPath, "init")
    elif rif == "HDF":
        # value = cfgPath + 'hdf'
        value = os.path.join(cfgPath, "hdf")
    elif rif == "CFG":
        # value = cfgPath + 'init'
        value = os.path.join(cfgPath, "init")
    elif rif == "BITMAPS":
        # value = cfgPath + 'resources'
        value = os.path.join(cfgPath, "resources")
    elif rif == "MISC":
        value = dpg.globalVar.GlobalState.miscPath
    elif rif == "LOG":
        # value = dpg.globalVar.GlobalState.miscPath + 'log'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "log")
    elif rif == "QUERY":
        # value = dpg.globalVar.GlobalState.miscPath + 'query'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "query")
    elif rif == "LOCAL_QUERY":
        value = "datamet_data/data/misc/query"
        if os.path.isdir(value):
            # value = dpg.globalVar.GlobalState.miscPath + 'query'
            value = os.path.join(dpg.globalVar.GlobalState.miscPath, "query")
    elif rif == "NAVIGATIONS":
        # value = dpg.globalVar.GlobalState.miscPath + 'navigations'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "navigations")
    elif rif == "CALIBRATION":
        # value = dpg.globalVar.GlobalState.miscPath + 'calibrations' + sep + 'RADAR'
        value = os.path.join(
            dpg.globalVar.GlobalState.miscPath, "calibrations", "RADAR"
        )
    elif rif == "CALIBRATIONS":
        # value = dpg.globalVar.GlobalState.miscPath + 'calibrations'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "calibrations")
    elif rif == "PALETTES":
        # value = dpg.globalVar.GlobalState.miscPath + 'palettes'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "palettes")
    elif rif == "COLOURS":
        # value = dpg.globalVar.GlobalState.miscPath + 'palettes'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "palettes")
    elif rif == "SYMBOLS":
        # value = dpg.globalVar.GlobalState.miscPath + 'symbols'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "symbols")
    elif rif == "FLY":
        # value = dpg.globalVar.GlobalState.miscPath + 'flights'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "flights")
    elif rif == "FILTERS":
        # value = dpg.globalVar.GlobalState.miscPath + 'filters'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "filters")
    elif rif == "MOSAIC":
        # value = dpg.globalVar.GlobalState.miscPath + 'visibility'
        value = os.path.join(dpg.globalVar.GlobalState.miscPath, "visibility")
    elif rif == "RPG":
        value = dpg.globalVar.GlobalState.rpgPath
    elif rif == "SUMMARY":
        value = checkPathname("\\data1\\SENSORS\\SUMMARY\\")
    elif rif == "EXTERN":
        value = dpg.globalVar.GlobalState.externPath
    elif rif == "RAINGAUGES_R":
        value = "\\data1\\SENSORS\\RAW\\RAINGAUGES\\"
    elif rif == "RAINGAUGES":
        # value = dpg.globalVar.GlobalState.externPath + 'RAIN'
        value = os.path.join(dpg.globalVar.GlobalState.externPath, "RAIN")
    elif rif == "TEMPERATURES":
        # value = dpg.globalVar.GlobalState.externPath + 'TEMP'
        value = os.path.join(dpg.globalVar.GlobalState.externPath, "TEMP")
    elif rif == "TERMOMETERS":
        # value = dpg.globalVar.GlobalState.externPath + 'TERMO'
        value = os.path.join(dpg.globalVar.GlobalState.externPath, "TERMO")
    elif rif == "TEMP_METAR":
        # value = dpg.globalVar.GlobalState.externPath + 'METAR'
        value = os.path.join(dpg.globalVar.GlobalState.externPath, "METAR")
    elif rif == "RADIOSONDE":
        # value = dpg.globalVar.GlobalState.externPath + 'RADIO'
        value = os.path.join(dpg.globalVar.GlobalState.externPath, "RADIO")
    elif rif == "RPGDATA":
        value = dpg.globalVar.GlobalState.dataPath
    elif rif == "RPVDATA":
        value = dpg.globalVar.GlobalState.rpvPath
    elif rif == "IMAGES":
        # value = dpg.globalVar.GlobalState.rpvPath + 'images'
        value = os.path.join(dpg.globalVar.GlobalState.rpvPath, "images")
    elif rif == "SAT":
        # value = dpg.globalVar.GlobalState.rawPath + 'SAT'
        value = os.path.join(dpg.globalVar.GlobalState.rawPath, "SAT")
    elif rif == "RADAR":
        value = dpg.globalVar.GlobalState.rawPath
    elif rif == "PREPROCESSING":
        # value = dpg.globalVar.GlobalState.rpgPath + 'SAT'
        value = os.path.join(dpg.globalVar.GlobalState.rpgPath, "preprocessing")
    elif rif == "TARGET":
        value = os.environ.get("RV_TARGET_PATH", "")
        if not value:
            value = os.path.join(dpg.globalVar.GlobalState.RV_HOME, "target")
    elif rif == "PRODUCTS_SOU":
        # value = dpg.globalVar.GlobalState.souPath + 'SAT'
        value = os.path.join(dpg.globalVar.GlobalState.souPath, "products")
    elif rif == "PRE_SOU":
        # value = dpg.globalVar.GlobalState.souPath + 'SAT'
        value = os.path.join(dpg.globalVar.GlobalState.souPath, "preprocessing")
    elif rif == "DPB_SOU":
        # value = dpg.globalVar.GlobalState.souPath + 'SAT'
        value = os.path.join(dpg.globalVar.GlobalState.souPath, "dpb")
    elif rif == "MODELS":
        value = dpg.globalVar.GlobalState.modelsPath
    elif rif == "HOME":
        value = dpg.globalVar.GlobalState.RV_HOME
    else:
        value = ""

    # TODO andrebbe restituito anche alt_path
    # TODO bene così o serve aggiungere controlli di esistenza?
    return value
    """
    #TODO scritto da Monica, da riguardare
    DMT_DATA_PATH = os.getenv('DMT_DATA_PATH')
    DMT_HOME = os.getenv('DMT_HOME')
    cfgPath = os.path.join(DMT_HOME,'cfg')
    if rif.upper() == 'DATA':
        value = DMT_DATA_PATH
    elif rif.upper() == 'VPR':
        value = os.path.join(DMT_DATA_PATH,rif.lower())
    elif rif.upper() == 'CFG_SCHED': 
        value = os.path.join(cfgPath, 'schedules')
    else:
        value = 'undefined'
    exists = os.path.exists(value)
    return value
    """


def getDir(rif, sep="", with_separator=False, alt_path=False):
    """
    Retrieves a directory path based on the provided reference and optional parameters.

    Args:
        rif (_type_): The reference used to retrieve the directory path.
        sep (str, optional): The path separator to use. Defaults to "".
        with_separator (bool, optional): If True, includes the separator at the end of the path. Defaults to False.
        alt_path (bool, optional): If True, uses an alternative path retrieval method. Defaults to False.

    Returns:
        _type_: The retrieved directory path.
    """

    if dpg.globalVar.GlobalState.RV_HOME is None:

        rv_home, rvsep, host = getHome(with_separator=True)
        dpg.globalVar.GlobalState.update("RV_HOME", rv_home)
        dpg.globalVar.GlobalState.update("RV_HOST", host)
        value = os.getenv("RV_DATA_PATH")
        if not value:
            value = str(DATAMET_DATA_PATH)
            log_message(f"Cannot find $RV_DATA_PATH: using {value}", level='WARNING')

        dpg.globalVar.GlobalState.update("rpgPath", os.path.join(value, "rpg"))
        dpg.globalVar.GlobalState.update("rdpPath", os.path.join(value, "rdp"))
        dpg.globalVar.GlobalState.update("miscPath", os.path.join(value, "misc"))
        dpg.globalVar.GlobalState.update("externPath", os.path.join(value, "extern"))
        dpg.globalVar.GlobalState.update("modelsPath", checkPathname("/data1/MODELS/RAW/PRE"))
        dpg.globalVar.GlobalState.update("rawPath", checkPathname("/data1/RADAR/RAW"))
        dpg.globalVar.GlobalState.update("dataPath", value + rvsep)
        dpg.globalVar.GlobalState.update(
            "rpvPath",
            dpg.globalVar.GlobalState.RV_HOME + "data" + rvsep + "rpv" + rvsep,
        )
        value = os.getenv("RV_EXTERN_PATH")
        if value:
            dpg.globalVar.GlobalState.update("externPath", value)

        value = os.getenv("RV_MODELS_PATH")
        if value:
            dpg.globalVar.GlobalState.update("modelsPath", value)

        value = os.getenv("RV_RAW_PATH")
        if value:
            dpg.globalVar.GlobalState.update("rawPath", value)

        value = os.getenv("RV_MISC_PATH")
        if value:
            dpg.globalVar.GlobalState.update("miscPath", value + rvsep)
        value = os.getenv("RPG_DATA_PATH")
        if value:
            dpg.globalVar.GlobalState.update("rpgPath", value + rvsep)
        value = os.getenv("RPV_DATA_PATH")
        if value:
            dpg.globalVar.GlobalState.update("rdpPath", value + rvsep)
        value = os.getenv("RDP_DATA_PATH")
        if value:
            dpg.globalVar.GlobalState.update("rdpPath", value + rvsep)
        dpg.globalVar.GlobalState.update(
            "souPath", rvsep + "datamet" + rvsep + "sou" + rvsep
        )

    sep = dpg.globalVar.GlobalState.RV_SEP
    if len(rif) < 1:
        return ""

    value = getNewPath(rif, sep, alt_path=alt_path)
    if value == "":
        return value

    pos = value.rfind(sep)
    length = len(value) - 1
    if pos == length:
        value = value[:length]

    if with_separator:
        return value + sep

    return value


def getFullPathName(path, name):
    """
    Constructs a full path by combining the given path and name, ensuring proper handling
    of path separators and special conditions.

    Args:
        path (str): The base path to which the name will be appended.
        name (str): The name to be appended to the path.

    Returns:
        str: The combined full path.
    """

    # TODO: implementazione di Monica, da guardare
    """
    if not isinstance(path, str) or not isinstance(name, str):
        print('Error: either path or name is not str')
        return
    if path == '':
        return name
    if name == '':
        return path
    if path[-1] != '/' and name[0] != '/':
        path +='/'
    currName = name
    currPath = path
    return currPath + currName
    """
    path = str(path)
    name = str(name)
    currName, contain_sep, _ = checkSep(name)
    length = 0

    if len(path) > 0:
        length = len(path) - 1
    if length <= 0:
        return currName
    if contain_sep == 1 and currName.find(".") != 0:
        pos = currName.find("$")
        if pos == 0:
            return checkPathname(currName)
        return currName

    currPath = checkPathname(path, with_separator=False)
    return os.path.join(currPath, currName)


def getPrefixToCheck():
    """
    This method retrieves and sets up prefix mappings for paths based on environment variables and platform-specific conditions.
    It ensures that the global state is updated with the correct prefix lists for both input and output paths.

    Returns:
        tuple: A tuple containing two lists:
            - in_prefix: The list of input path prefixes.
            - out_prefix: The list of output path prefixes.
    """

    """
    os.getenv() is used to get environment variables
    str.split() is used to split strings into lists based on a delimiter
    os.uname().sysname is used to get the system name (OS name)
    """

    if dpg.globalVar.GlobalState.pref_in is None:
        pref_in = os.getenv("RV_HOST_PREFIX") or ""
        pref_out = os.getenv("RV_LOCAL_PREFIX") or ""

        dpg.globalVar.GlobalState.update("pref_in", pref_in.split(";"))
        dpg.globalVar.GlobalState.update("pref_out", pref_out.split(";"))

        if dpg.globalVar.GlobalState.pref_in[0] == "":
            if platform == "win32":
                dpg.globalVar.GlobalState.update("pref_in", ["\\data1\\"])
                dpg.globalVar.GlobalState.update("pref_out", ["R:\\"])

                if not os.path.isdir(dpg.globalVar.GlobalState.pref_out[0]):
                    dpg.globalVar.GlobalState.update("pref_out", ["C:\\"])

            elif platform == "linux":
                # TODO da controllare il caso linux ed eventualmente gestire il caso MAC
                print("Da gestire caso linux")
                dpg.globalVar.GlobalState.update("pref_in", ["\\data1\\"])
                dpg.globalVar.GlobalState.update("pref_out", ["R:\\"])

                if not os.path.isdir(dpg.globalVar.GlobalState.pref_out[0]):
                    dpg.globalVar.GlobalState.update("pref_out", ["C:\\"])

    if dpg.globalVar.GlobalState.pref_in[0] == "":
        return 0

    in_prefix = dpg.globalVar.GlobalState.pref_in
    out_prefix = dpg.globalVar.GlobalState.pref_out
    return in_prefix, out_prefix


def checkSep(name: str):
    """
    This method processes the given path string by ensuring it uses the correct platform-specific
    path separator and handles potential prefix replacements.

    Args:
        name (str): The input path string to process.

    Returns:
        tuple: A tuple containing the processed path string, a flag indicating if a separator was found,
               and the platform-specific path separator.
    """

    """
    os.path.sep is used to get the platform-specific path separator.
    the := operator is used for assignment within expressions in Python.
    the len() function is used to get the length of a string.
    the str.find() method is used to find the position of a substring.
    the str[:pos] and str[pos + len(substring):] slicing is used to modify strings.
    """
    sep = os.path.sep
    sepInName = "/"
    contain_sep = 0

    if len(name) == 0:
        return "", contain_sep, sep

    pos = name.find(sepInName)
    if pos < 0:
        sepInName = "\\"
        pos = name.find(sepInName)  # TODO: no decision are made if pos is -1 or >= 0
        return name, contain_sep, sep

    contain_sep = 1
    if sepInName == sep:
        return name, contain_sep, sep

    # here sepInName is not '/' but '\'
    currName = name
    while (
            pos := currName.find(sepInName)
    ) >= 0:  # The loop continues as long as sepInName is found in currName
        # This line replaces the first occurrence of sepInName in currName with sep. It does this by slicing currName
        # into two parts at the position pos, and then concatenating the first part, sep, and the second part (excluding sepInName) together.
        currName = currName[:pos] + sep + currName[pos + len(sepInName):]

    in_prefix, out_prefix = getPrefixToCheck()

    for ppp in range(len(in_prefix)):
        pos = currName.find(in_prefix[ppp])
        if pos == 0:
            currName = currName[len(in_prefix[ppp]):]
            return out_prefix[ppp] + currName, contain_sep, sep

    return currName, contain_sep, sep


def checkAt(path: str) -> str:
    """
    This method processes the given path string by removing leading and trailing whitespaces
    and handling the '@' character at the beginning of the string if present.

    Args:
        path (str): The input path string to process.

    Returns:
        str: The processed path string with any leading '@' removed.
    """

    """
    .strip() is used to remove leading and trailing whitespaces from the path.
    .find('@') is used to find the position of @ in the currPath.
    if pos is 0, currPath[1:] is used to remove the @ at the beginning.
    """
    if isinstance(path, str):
        currPath = path.strip()
    elif isinstance(path, dict):
        path = path[list(path.keys())[0]]
        currPath = path.strip()
    else:
        currPath = ""

    pos = currPath.find("@")
    if pos == 0:
        currPath = currPath[1:]
    return currPath


def getHome(with_separator: bool = False):
    """
    This method retrieves the home directory path for the application, ensuring necessary environment
    variables are set and updating the global state accordingly. It also returns the path separator
    and the host name used by the application.

    Args:
        with_separator (bool): If True, appends the path separator to the home directory path.

    Returns:
        tuple: A tuple containing the home directory path (optionally with separator), the path separator, and the host name.
    """

    """
    os.getenv() is used to get environment variables.
    os.path.sep is used to get the platform-specific path separator.
    os.uname().sysname is used to get the system name (OS name).
    if not variable: is used to check if a list (or any iterable) is empty.
    """
    if not dpg.globalVar.GlobalState.RV_HOME:
        dpg.globalVar.GlobalState.update("RV_HOME", os.getenv("RV_HOME"))
        if not dpg.globalVar.GlobalState.RV_HOME:
            dpg.globalVar.GlobalState.update("RV_HOME", os.getenv("RV_RRPC_HOME"))

        if not dpg.globalVar.GlobalState.RV_HOME:
            if platform == "win32":
                dpg.globalVar.GlobalState.update("RV_HOME", str(DATAMET_RADVIEW_PATH))
            else:
                dpg.globalVar.GlobalState.update("RV_HOME", str(DATAMET_RADVIEW_PATH))

        dpg.globalVar.GlobalState.update("RV_SEP", os.path.sep)
        dpg.globalVar.GlobalState.update("RV_HOST", os.getenv("RV_HOST"))
        if not dpg.globalVar.GlobalState.RV_HOST:
            dpg.globalVar.GlobalState.update("RV_HOST", "RADAR")

    dir = dpg.globalVar.GlobalState.RV_HOME
    host = dpg.globalVar.GlobalState.RV_HOST
    sep = dpg.globalVar.GlobalState.RV_SEP

    if with_separator:
        dir += sep

    return dir, sep, host


def isDemoMode() -> bool:
    """
    Check if is in demo mode

    Returns:
        bool: check if is in demo mode
    """
    return False


def openWriteFile(pathname: str, append: bool = False, silent: bool = False):
    """
    This method attempts to open a file for writing based on the provided pathname.
    It includes options to append to the file and to suppress error messages.
    If the pathname is invalid or the application is in demo mode, the method returns None.

    Args:
        pathname (str): The path of the file to open.
        append (bool): If True, the file is opened in append mode; otherwise, it is opened in write mode.
        silent (bool): If True, suppresses error messages when file opening fails.

    Returns:
        file: The file handle if the file is successfully opened; otherwise, returns None.
    """
    if not pathname or len(pathname) <= 1:
        return None

    if isDemoMode():
        return None

    try:
        mode = "a" if append else "w"
        file_handle = open(pathname, mode)
        return file_handle
    except IOError as e:
        if not silent:
            print(f"Cannot open file {pathname}: {e}")
        return None


def checkEnv(path: str) -> str:
    # TODO controllare il comportamento e aggiornare di conseguenza i test
    """
    This method takes a string `path` as input and returns a modified string
    based on specific conditions. If `path` starts with the `$` symbol, it is
    returned unchanged. Otherwise, it checks if `path` starts with one of the
    prefixes defined in the `env` list. If a match is found, it returns `path`
    with the corresponding prefix preceded by the `$` symbol. If `path` is empty,
    it returns `path` as is. In all other cases, it returns `path` preceded by the `@` symbol.

    Args:
        path (str): The path string to check and modify.

    Returns:
        str: The string modified according to the described rules.
    """
    if path.startswith("$"):
        return path

    env = [
        "OVERLAYS",
        "UNDERLAYS",
        "USER",
        "SYMBOLS",
        "LEGENDS",
        "PALETTES",
        "EXTERN",
        "SCHEDULES",
    ]

    for directory in env:
        if path.startswith(directory):
            rest = path[len(directory):]
            return f"${directory}{rest}"

    if not path:
        return path

    return f"@{path}"
