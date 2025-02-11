"""
Funzioni ancora da portare
PRO StartProgress 
PRO StartSubProgress 
PRO StopProgress 
PRO UpdateProgress 
FUNCTION FilterSite         // UNUSED
FUNCTION GetSchedulePaths   // UNUSED
PRO RPK                     // UNUSED
"""

import os

import sou_py.dpg as dpg
from sou_py.dpg.attr__define import Attr


def checkSiteList(schedule: str, sites: list, sc_attr: Attr = None) -> int:
    """
    Checks if any of the specified sites are included in a schedule, based on certain conditions.

    This function evaluates whether any of the sites listed in 'sites' are included in a 'schedule'.
    It considers the 'allsites' attribute and a site filter defined in the schedule. If 'allsites' is
    set, or if any of the specified sites are in the schedule's site filter, the function returns 1,
    indicating inclusion. Otherwise, it returns 0.

    Args:
        schedule (str): The schedule to check against.
        sites (list[str]): A list of site names to check for in the schedule.
        sc_attr (Attr object, optional): An attribute object associated with the schedule. Defaults to None.

    Returns:
        int: Returns 1 if any of the sites are included in the schedule based on the conditions, otherwise returns 0.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'getAttrValue' and 'loadAttr'
        methods from 'dpg.attr'.
    """
    if not schedule:
        return 1
    if len(sites) == 0:
        return 1

    all, _, _ = dpg.attr.getAttrValue(sc_attr, "allsites", 0)

    if all > 0:
        return 1

    attr = dpg.attr.loadAttr(schedule, dpg.cfg.getSitesListName())
    siteFilter, exists, _ = dpg.attr.getAttrValue(attr, "site", "")

    if exists <= 0:
        return 1

    for sss in sites:
        if sss in siteFilter:
            return 1

    return 0


def checkTimeList(schedule: str, nominalTime, sc_attr: Attr = None) -> int:
    """
    Checks if the nominal time is included in a schedule's time list, with consideration for an 'always' attribute.

    This function evaluates whether a specified nominal time ('nominalTime') is included in a time list
    associated with a 'schedule'. It first checks if the 'always' attribute is set, which would automatically
    include the time. If not, it then checks the time list defined in the schedule. If the time is included,
    or if no specific times are defined, the function returns 1, indicating inclusion.

    Args:
        schedule (str): The schedule to check against.
        nominalTime (str or datetime): The nominal time to check for in the schedule.
        sc_attr (Attr object, optional): An attribute object associated with the schedule. Defaults to None.

    Returns:
        int: Returns 1 if the nominal time is included in the schedule based on the conditions, otherwise,
        the logic for determining the return value is not fully implemented.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'getAttrValue' and 'loadAttr'
        methods from 'dpg.attr'.
    """
    if not nominalTime:
        return 1

    always, _, _ = dpg.attr.getAttrValue(sc_attr, "always", 0)

    if always > 0:
        return 1

    attr = dpg.attr.loadAttr(schedule, dpg.cfg.getTimesListName())
    schTimes, exists, _ = dpg.attr.getAttrValue(attr, "time", "")

    if exists <= 0:
        return 1

    ret = dpg.times.isTime(nominalTime, schTimes)

    return ret


def getSchedulePath(system: str) -> str:
    """
    Retrieves the full path to the schedules directory for the specified system.

    If the `system` argument is not provided (None or empty), the function fetches the system's default value.
    It then constructs and returns the path to the schedules directory within the DATA directory for the given system.

    Args:
        system (str): The system identifier. If None or empty, a default system value is retrieved.

    Returns:
        str: The full path to the 'schedules' directory for the specified (or default) system.
    """
    path = dpg.path.getDir(
        "DATA", with_separator=True
    )
    if system is None or system == "":
        system, _ = (
            dpg.cfg.getSystem()
        )

    return os.path.join(path, "schedules", system)
