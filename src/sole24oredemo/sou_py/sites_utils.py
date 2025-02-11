import os
import platform
import sys
from pathlib import Path

from sou_py.dpg.log import log_message
# from sou_py.dpg.log import logger
from sou_py.paths import DATAMET_RADVIEW_PATH


def get_sites(rv_center: str):
    """
    Retrieves a list of radar stations based on the specified RV center or the current site.

    This function determines which radar stations to use based on the `rv_center`
    argument or the current site. If `rv_center` is provided, it splits the value
    into a list of station names and returns them. If `rv_center` is not provided,
    the function checks whether the current site is in the list of known radar
    stations and reads the corresponding site names from a file.

    Args:
        rv_center (str): A comma-separated string of radar station names. If provided,
                         these stations are returned directly. If not, the function
                         falls back to using the current site.

    Returns:
        List[str]: A list of radar stations determined by either the `rv_center`
                   argument or the current site.
    """

    if isinstance(rv_center, list):
        if isinstance(rv_center, str):
            stations = [x.strip() for x in rv_center.split(",")]
        elif isinstance(rv_center, list):
            stations = rv_center
        else:
            sys.exit("Center is not a list or a string!")
        log_message(
            f"RV_CENTER is set to {rv_center}. Using stations: {stations}"
        )
        return stations

    if _is_current_site_in_list(rv_center):
        rv_file = rv_center + '.txt'
        sites_filename = DATAMET_RADVIEW_PATH / "cfg" / "sites" / rv_file
        sites = _read_sites_from_file(sites_filename)
        if len(sites) == 0:
            log_message(
                f"Current site {rv_center} is in cfg/sites, but the file is empty. Using stations from all.txt ."
            )
            all_filename = DATAMET_RADVIEW_PATH / "cfg" / "sites" / 'all.txt'
            sites = _read_sites_from_file(all_filename)
        else:
            log_message(
                f"Current site {rv_center} found in RadView/cfg/sites. Using stations from {sites_filename} ."
            )
    else:
        sites_filename = _get_current_machine() + ".txt"
        sites = _read_sites_from_file(sites_filename)
        if len(sites) == 0:
            log_message(
                f"Machine name {_get_current_machine()} is in cfg/sites, but the file is empty. Using stations from all.txt ."
            )
            all_filename = DATAMET_RADVIEW_PATH / "cfg" / "sites" / 'all.txt'
            sites = _read_sites_from_file(all_filename)

        else:
            log_message(
                f"Current site {_get_current_machine()} is in the list of stations. Using stations from {sites_filename} ."
            )

    return sites


def _get_current_machine():
    """
    Retrieves the current computer's network name.

    This function returns the computer's network name using `platform.node()`,
    which serves as a proxy for the machine's hostname. If the network name
    cannot be determined, an empty string is returned.

    Returns:
        str: The network name of the computer, or an empty string if not available.
    """
    return platform.node()


def _is_current_site_in_list(site):
    """
    Checks if the current site is present in the list of radar stations.

    This function generates a list of radar station names by reading filenames
    from a directory. It then checks if the current site, obtained using `_get_current_site()`,
    is present in that list.

    Returns:
        bool: True if the current site is in the list of radar stations, False otherwise.
    """
    radar_stations_list = [
        name.replace(".txt", "")
        for name in os.listdir(DATAMET_RADVIEW_PATH / "cfg" / "sites")
    ]

    if site in radar_stations_list:
        return site
    else:
        return False


def _parse_site_name(line: str):
    """
    Extracts and returns the site name from a given line of text.

    The function splits the string at the "=" character and returns the
    part after the "=" symbol, with any leading or trailing whitespace removed.

    Args:
        line (str): A string from which to extract site name.

    Returns:
        str: The site name extracted from the input line, with surrounding whitespace
        removed.
    """
    return line.split("=")[1].strip()


def _read_sites_from_file(sites_filepath: str | Path):
    """
    Reads a list of site names from a specified file.

    This function opens a file, reads each line, and extracts
    site names that start with the keyword "site". The extracted site names are
    processed using the `_parse_site_name` function and stored in a list.

    Args:
        sites_filename (str): The name of the file from which the site names will be read.

    Returns:
        List[str]: A list of site names extracted from the file.
    """
    if not os.path.isfile(sites_filepath):
        return []
    with open(sites_filepath, encoding="UTF-8") as f:
        lines = f.readlines()
    stations = []
    for line in lines:
        if line.startswith("site"):
            stations.append(_parse_site_name(line))
    return stations
