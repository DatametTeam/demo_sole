# from asyncio.windows_events import NULL
import copy
import os
import re
import sou_py.dpg as dpg
import numpy as np
from datetime import datetime, timedelta

from sou_py.dpg.log import log_message
from sou_py.dpg.schedule__define import Schedule

"""
Funzioni ancora da portare
FUNCTION IDL_rv_guess_prod_path 
FUNCTION WaitDataFile 
PRO IDL_rv_save_time 
PRO IDL_rv_save_times 
PRO MergePrevData 
PRO WAIT_LAST_PROD 
FUNCTION IDL_rv_check_path_list     // UNUSED
PRO IDL_rv_copy_times               // UNUSED
PRO IDL_rv_update_dates             // UNUSED
PRO WAIT_SCHEDULE                   // UNUSED
"""


def checkDate(date: str, sep: str = "-", year_first: bool = False) -> str:
    """
    Validates and formats a date string according to specified conventions.

    This function takes a date string ('date') and checks it against various common date formats. If the
    date is valid, it is reformatted according to the 'year_first' flag and the specified separator 'sep'.
    If no valid format is found, a default date '1970-01-01' is returned.

    Args:
        date (str): The date string to validate and format.
        sep (str, optional): The separator to use in the formatted date. Defaults to '-'.
        year_first (bool, optional): If True, the year is placed first in the formatted date. Defaults to False.

    Returns:
        datetime: The validated and formatted date string, or '1970-01-01' if the input is invalid.

    Raises:
        ValueError: Catches and ignores ValueError if the input date string does not match any known format.
    """
    date_str = date.strip()  # Remove any leading/trailing spaces

    if sep is None:
        sep = "-"

    if not date_str:
        return "1970-01-01"

    date_formats = [
        "%Y\\%m\\%d",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d\\%m\\%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m\\%d\\%Y",
        "%Y/%m/%d",

    ]

    for fmt in date_formats:
        try:
            dt_obj = datetime.strptime(date_str, fmt)
            if year_first:
                formatted_date = dt_obj.strftime("%Y" + sep + "%m" + sep + "%d")
            else:
                formatted_date = dt_obj.strftime("%d" + sep + "%m" + sep + "%Y")
            return formatted_date
        except ValueError:
            pass

    return "1970-01-01"


def checkTime(time: str, tt: str = None, sep: str = ":"):
    """
    Validates and formats a time string, with options to provide a fallback time.

    This function takes a time string ('time') and formats it into 'HH:MM' format, adjusting for various separators
    and handling cases where the input is not a string. If 'time' is not a valid string and a fallback time ('tt') is
    provided, 'tt' is used instead. It returns the formatted time along with the hours and minutes as separate integers.

    Args:
        time (str): The time string to validate and format.
        tt (str, optional): A fallback time string to use if 'time' is invalid. Defaults to None.
        sep (str, optional): The separator to use in the formatted time. Defaults to ':'.

    Returns:
        tuple: A tuple containing three elements:
            - time (str): The validated and formatted time string.
            - hh (int): The hour component of the time.
            - mm (int): The minute component of the time.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    hh = 0
    mm = 0

    if not isinstance(time, str):
        if tt is None:
            return "00:00", 0, 0
        time = tt

    hh = int(time[:2])
    pos = time.find(":") + 1

    if pos <= 0:
        pos = time.find(".") + 1
        if pos <= 0:
            pos = time.find("-") + 1
            if pos <= 0:
                pos = 2

    mm = int(time[pos: pos + 2])
    ore = str(hh).rjust(2, "0")
    minuti = str(mm).rjust(2, "0")

    if len(ore) == 1:
        ore = "0" + ore
    if len(minuti) == 1:
        minuti = "0" + minuti

    if np.size(sep):
        time = ore + sep + minuti
    else:
        time = ore + ":" + minuti

    tt = time

    return time, hh, mm


def date2dtAcq(date: list, time: list):
    """
    Combines date and time strings into a unified datetime format.

    This function takes lists of date and time strings and combines them into a single datetime format for each pair.
    If the number of time entries is less than the number of date entries, '00:00' is used as the default time.
    It also performs date and time validation using the 'checkDate' and 'checkTime' functions.

    Args:
        date (list[str]): A list of date strings in various possible formats.
        time (list[str]): A list of time strings, corresponding to the dates.

    Returns:
        tuple: A tuple containing two elements:
            - nD (int): The number of date-time combinations processed.
            - dt_acq (list[str]): A list of combined date and time strings in 'YYYY-MM-DD_HH:MM' format.
            - time (list[str]): The list of processed time strings, updated during the function execution.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    nD = len(date)
    if nD <= 0:
        return 0, []

    dt_acq = []
    for ppp in range(nD):
        dt_acq.append(checkDate(date[ppp]))
        cur_time = "00:00"
        if ppp < len(time):
            ttt = time[ppp]
            cur_time, _, _ = checkTime(ttt)
            time[ppp] = cur_time
        dt_acq[ppp] += "_" + cur_time

    return nD, dt_acq, time


def dtAcq2Date(dt_acq: list):
    """
    Converts a list of datetime strings into separate date and time lists.

    This function takes a list of datetime strings in the format 'YYYY-MM-DD HH:MM:SS' and splits them into separate
    date and time strings. It returns these as separate lists along with the count of datetime entries processed.

    Args:
        dt_acq (list[str]): A list of datetime strings to be split into date and time components.

    Returns:
        tuple: A tuple containing three elements:
            - nD (int): The number of datetime entries processed.
            - date (list[str]): A list of date strings in the format 'YYYY-MM-DD'.
            - time (list[str]): A list of time strings in the format 'HH:MM:SS'.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    nD = len(dt_acq)
    if nD <= 0:
        return 0, None, None

    date = []
    time = []

    for dt in dt_acq:
        dt = dt.strip()  # Remove any leading/trailing spaces
        dt_obj = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

        date_str = dt_obj.strftime("%Y-%m-%d")
        time_str = dt_obj.strftime("%H:%M:%S")

        date.append(date_str)
        time.append(time_str)

        # TODO aggiungere checkTime()

    return nD, date, time


def get_time(
        node,
        silent: bool = False,
        only_current: bool = False,
        down: bool = False,
        check_date: bool = False,
        load_if_changed: bool = False,
        nominal: bool = False,
        path: bool = False,
):
    """
    Retrieves the date and time associated with a node based on various conditions.

    This function fetches the date and time for a given node ('node'), considering different attributes
    and conditions such as 'nominal', 'path', and 'down'. It checks various attributes like 'nominal_date',
    'date', 'nominal_time', 'time', and optionally looks for attributes in parent nodes or based on the node's path.

    Args:
        node (Node object): The node from which to retrieve the date and time.
        silent (bool, optional): If True, suppresses print statements. Defaults to False.
        only_current (bool, optional): If True, considers only the current attribute. Defaults to True.
        down (bool, optional): If True, searches attributes in child nodes. Defaults to False.
        check_date (bool, optional): If True, validates the date. Defaults to False.
        load_if_changed (bool, optional): If True, reloads the attribute if it has changed. Defaults to False.
        nominal (bool, optional): If True, prefers 'nominal_date' and 'nominal_time' attributes. Defaults to False.
        path (bool, optional): If True, retrieves date and time based on the node's path. Defaults to False.

    Returns:
        tuple: A tuple containing three elements:
            - date (str or list[str]): The retrieved date or dates.
            - time (str or list[str]): The retrieved time or times.
            - exists (bool): True if the date and time were successfully retrieved, False otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    if not isinstance(node, dpg.node__define.Node):
        return None, None, False
    name = dpg.cfg.getArrayDescName()
    attr = node.getAttr(
        name,
        only_current=only_current,
        check_date=check_date,
        load_if_changed=load_if_changed,
    )
    exists = 0
    date = 0
    time = 0

    if nominal:
        tmp, exists, _ = dpg.attr.getAttrValue(attr, "nominal_date", "")
    if exists == 0:
        tmp, exists, _ = dpg.attr.getAttrValue(attr, "date", "")
    if exists == 0 and (not nominal):
        tmp, exists, _ = dpg.attr.getAttrValue(attr, "nominal_date", "")

    if exists == 0 and (down):
        attrSet = dpg.tree.findAttr(
            node, name, all=True, down=True
        )  # TODO ramo non controllato
        if len(attrSet) > 0:
            attr = attrSet[0]
        tmp, exists, _ = dpg.attr.getAttrValue(attr, "date", "")
    # endif

    if exists:
        exists = 0
        date = tmp
        time = ""
        if isinstance(date, list):
            for ddd in range(date):
                date[ddd] = checkDate(date[ddd])
            # endfor
        elif isinstance(date, str):
            date = checkDate(date)
        # endif
        if nominal:
            time, exists, _ = dpg.attr.getAttrValue(
                attr, "nominal_time", ""
            )
        if exists == 0:
            time, exists, _ = dpg.attr.getAttrValue(attr, "time", "")
        if exists == 0 and (not nominal):
            time, exists, _ = dpg.attr.getAttrValue(
                attr, "nominal_time", ""
            )
        if isinstance(time, list):
            date, time = sortDates(date, time)  # TODO ramo non controllato
        # endif
        time, _, _ = checkTime(time)
        return date, time, exists
    # endif

    if path:
        tmp, ttt = path2Date(node.path)  # TODO ramo non controllato
        if tmp != "":
            date = tmp
            time = ttt
            return date, time, True
        ##endif
    # endif

    name = dpg.cfg.getArchiveDescName()
    attr = node.getSingleAttr(name, only_current=only_current)
    dt_acq, exists, _ = dpg.attr.getAttrValue(attr, "dt_acq", "")
    if not exists:
        if not silent:
            print("Cannot find date in " + node.path)
        # endif
        return date, time, False
    # endif
    _, date, time = dtAcq2Date(
        dt_acq, date=date, time=time
    )  # TODO ramo non controllato

    return date, time, True


def set_time(
        node,
        date: str = None,
        time: str = None,
        site_name: bool = False,
        to_save: bool = False,
        nominal: bool = False,
):
    """
    Sets date and time attributes for a given node.

    This function updates the date and time attributes of a node ('node'). It allows setting either 'nominal'
    or regular date and time attributes. Additionally, it can set the 'origin' attribute to a site name.

    Args:
        node (Node object): The node for which date and time are to be set.
        date (str, optional): The date to set for the node. Defaults to None.
        time (str, optional): The time to set for the node. Defaults to None.
        site_name (str, optional): The site name to set as the origin attribute. If False, the origin is not set.
        Defaults to False.
        to_save (bool, optional): If True, saves the updated attributes. Defaults to False.
        nominal (bool, optional): If True, sets nominal date and time attributes. Defaults to False.

    Returns:
        None: This function does not return anything.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    varnames = []
    values = []

    if time is not None:
        if nominal:
            varnames.append("nominal_time")
        else:
            varnames.append("time")
        values.append(str(time))

    if date is not None:
        if varnames:
            if nominal:
                varnames.append("nominal_date")
            else:
                varnames.append("date")
            values.append(date)
        else:
            if nominal:
                varnames.append("nominal_date")
            else:
                varnames.append("date")
            values = date

    if site_name:
        varnames.append("origin")
        values.append(site_name)

    dpg.tree.replaceAttrValues(
        node,
        dpg.cfg.getArrayDescName(),
        varnames=varnames,
        values=values,
        to_save=to_save,
    )
    return


def changePath(path: bool, date: bool, time: bool) -> bool:
    """
    Modifies a file path by inserting a specific date and time.

    This function takes a file path ('path') and replaces a part of it with a formatted string based on the
    provided 'date' and 'time'. The function expects the path to contain a specific pattern that matches the
    date and time format, which it then replaces. If the pattern is not found, it appends the date and time at
    a specific location in the path.

    Args:
        path (str): The original file path to be modified.
        date (str): The date string in 'YYYY-MM-DD' format.
        time (str): The time string in 'HH:MM' format.

    Returns:
        str: The modified file path with the date and time inserted or appended.

    Raises:
        None: This function does not explicitly raise any exceptions but prints a message if input types are incorrect.
    """
    if (
            not isinstance(path, str)
            or not isinstance(date, str)
            or not isinstance(time, str)
    ):
        print("Wrong type!")
        return ""

    path = path.replace("\\", "/")
    DD, MM, YYYY = date.split("-")
    hh, mm = time.split(":")

    replacement = YYYY + "/" + MM + "/" + DD + "/" + hh + mm
    # Match pattern 0000/00/00/0000
    regex_pattern = "\d{4}\/\d{2}\/\d{2}\/\d{4}"

    if re.search(regex_pattern, path) is not None:
        path = re.sub(regex_pattern, replacement, path)
    else:
        regex_pattern = "\/\d{4}"
        path = re.sub(regex_pattern, "/" + replacement, path)

    if os.path.sep == "\\":
        path = path.replace("/", "\\")

    return path


def addMinutesToDate(date: str, time: str, minutes: int):
    """
    Adds a specified number of minutes to a given date and time.

    This function takes a date and time, adds a specified number of minutes to it, and returns the new date
    and time. The function first checks the format of the date (whether the year comes first) and then
    performs the date-time calculation.

    Args:
        date (str): The initial date in either 'YYYY-MM-DD' or 'DD-MM-YYYY' format.
        time (str): The initial time in 'HH:MM' format.
        minutes (int): The number of minutes to add to the date and time. Can be negative.

    Returns:
        tuple: A tuple containing two elements:
            - new_date_str (str): The new date string after adding the minutes.
            - new_time_str (str): The new time string after adding the minutes.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    year_first = getDateFormat(date)

    datetime_str = f"{date} {time}"

    if year_first:
        date_format = "%Y-%m-%d %H:%M"
    else:
        date_format = "%d-%m-%Y %H:%M"
    original_datetime = datetime.strptime(datetime_str, date_format)

    # Calculate the new datetime by adding (or subtracting) minutes
    new_datetime = original_datetime + timedelta(minutes=minutes)

    # Extract the new date and time strings
    if year_first:
        new_date_str = new_datetime.strftime("%Y-%m-%d")
    else:
        new_date_str = new_datetime.strftime("%d-%m-%Y")

    new_time_str = new_datetime.strftime("%H:%M")

    return new_date_str, new_time_str


def path2Date(path: str):
    """
    Extracts date and time information from a file path.

    This function parses a file path ('path') to extract date and time information. The expected format in
    the path is '/YYYY/MM/DD/HHMM'. If the date and time are not found in the expected format, empty strings
    are returned.

    Args:
        path (str): The file path from which to extract date and time.

    Returns:
        tuple: A tuple containing two elements:
            - date (str): The extracted date in 'YYYY-MM-DD' format, or an empty string if not found.
            - time (str): The extracted time in 'HH:MM' format, or an empty string if not found.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    # sep = "/"  # TODO modificare checkPathname per farglielo restituire
    fpath, sep = dpg.path.checkPathname(path, with_separator=True)

    pos = fpath.find(sep + "20")
    if pos < 3:
        return "", ""

    if fpath.find(sep, pos + 1) != pos + 5:
        return "", ""

    date = fpath[pos + 1: pos + 11]
    time = fpath[pos + 12: pos + 16]
    date = checkDate(date, sep="-")
    time, _, _ = checkTime(time, sep=":")

    return date, time


def getJulDay(date: list, time: list = None) -> list:
    """
    Calculates the Julian day for each date in the provided list, optionally considering time.

    This function takes a list of dates (and optionally, a corresponding list of times) and calculates the
    Julian day for each date. The Julian day is the day of the year (1-365 or 1-366 for leap years).

    Args:
        date (list[str]): A list of date strings in the format 'YYYY-MM-DD'.
        time (list[str], optional): An optional list of corresponding time strings in the format 'HH:MM:SS'. Defaults
        to None.

    Returns:
        list[int]: A list of Julian days corresponding to each date in the 'date' list.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    n_times = len(date)
    if n_times <= 0:
        return 0

    jul_days = []

    for i in range(n_times):
        date_str = date[i].strip()  # Remove any leading/trailing spaces
        dt_str = date_str

        if i < len(time):
            time_str = time[i].strip()
            if time_str:
                dt_str += " " + time_str
                dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                jul_day = dt_obj.timetuple().tm_yday
                jul_days.append(jul_day)
        else:
            dt_obj = datetime.strptime(dt_str, "%Y-%m-%d")
            jul_day = dt_obj.timetuple().tm_yday
            jul_days.append(jul_day)

    return jul_days


def getDateFormat(date: str) -> bool:
    """
    Determines whether the year is first in a given date string.

    This function checks the format of a date string to determine if the year appears first. It supports
    different separators ('-', '/', '\\'). If no standard separator is found, it checks the position of '20'
    to guess the format.

    Args:
        date (str): The date string to be checked.

    Returns:
        bool: True if the year is first in the date format, False otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    sep = "-"
    year_first = False

    ddmmyyyy = date.split(sep)

    if len(ddmmyyyy) <= 2:
        sep = "/"
        ddmmyyyy = date.split(sep)

        if len(ddmmyyyy) <= 2:
            sep = "\\"
            ddmmyyyy = date.split(sep)

    if len(ddmmyyyy) != 3:
        sep = ""
        if date.find("20") != 0:
            return False
        return True

    if len(ddmmyyyy[2]) < len(ddmmyyyy[0]):
        year_first = True

    return year_first


def getMinuteString(minutes: float, no_zero: bool = False) -> str:
    """
    Converts a time in minutes to a formatted string of minutes and seconds.

    This function takes a numeric value representing time in minutes and converts it into a string
    format showing minutes and seconds. If 'no_zero' is True, it returns 'ND' for zero or negative
    minutes. The function also handles non-finite values by returning 'ND'.

    Args:
        minutes (float): The time value in minutes to be converted.
        no_zero (bool, optional): If True, returns 'ND' for zero or negative minutes. Defaults to False.

    Returns:
        str: A formatted string representing the time in minutes and seconds, or 'ND' for non-finite or zero/negative
        values (if 'no_zero' is True).

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if not np.isfinite(minutes):
        return "ND"
    if no_zero and minutes <= 0:
        return "ND"

    mins = int(minutes)
    seconds = int((minutes - mins) * 60)
    seconds = str(seconds).rjust(2, "0")

    minute_string = f"{mins}' {seconds}\""
    return minute_string


def seconds2Date(seconds):
    """
    Converts a time in seconds since the epoch to a formatted date and time string.

    This function takes a numeric value representing time in seconds since the Unix epoch (1 January 1970)
    and converts it into formatted date and time strings. The date is formatted as 'DD-MM-YYYY', and the time
    is formatted as 'HH:MM'.

    Args:
        seconds (int or float): The time in seconds since the Unix epoch.

    Returns:
        tuple: A tuple containing two elements:
            - date (str): The formatted date string in 'DD-MM-YYYY' format.
            - time (str): The formatted time string in 'HH:MM' format.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'utcfromtimestamp' from datetime.
    """
    fulldate = datetime.utcfromtimestamp(seconds)
    day = fulldate.day
    month = fulldate.strftime("%m")  # Use "%m" for zero-padded month
    year = fulldate.year
    time = fulldate.strftime("%H:%M")
    date = f"{day:02d}-{month}-{year}"  # Ensure day is zero-padded

    return checkDate(date), time


def getCurrDate():
    """
    Retrieves the current date and time in UTC.

    This function fetches the current date and time in Coordinated Universal Time (UTC) and formats them
    into separate date and time strings. The date is formatted as 'DD-MM-YYYY', and the time is formatted
    as 'HH:MM'.

    Returns:
        tuple: A tuple containing two elements:
            - date_str (str): The current date in 'DD-MM-YYYY' format.
            - time_str (str): The current time in 'HH:MM' format.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    curr_time = datetime.utcnow()
    date_str = curr_time.strftime("%d-%m-%Y")
    time_str = curr_time.strftime("%H:%M")

    return date_str, time_str


def convertDate(date: list, time: list = None) -> list:
    """
    Converts a list of date strings (with optional time) to seconds since the Unix epoch.

    This function takes a list of date strings, and optionally corresponding time strings, and converts
    each date-time combination into the number of seconds since the Unix epoch (1 January 1970). If the
    conversion fails for any date-time combination, 0 is added to the list for that entry.

    Args:
        date (list[str]): A list of date strings in the format 'YYYY-MM-DD'.
        time (list[str], optional): An optional list of corresponding time strings in the format 'HH:MM:SS'. Defaults
        to None.

    Returns:
        list[int]: A list of integers representing the number of seconds since the Unix epoch for each date-time
        combination.

    Raises:
        ValueError: Handles and ignores ValueError if the date-time string does not match the expected format.
    """

    if date is None:
        return []
    n_times = np.size(date)
    if n_times <= 0 or date is None or date == 0:
        return [0]

    seconds_list = []

    if not isinstance(date, list):
        date = [date]
    if not isinstance(time, list):
        time = [time]

    for i, dt in enumerate(date):
        try:
            date_time_str = dt
            # TODO va bene gestire solo questi due casi di formato?
            if time:
                date_time_str += " " + time[i]
                format_string = (
                    "%d-%m-%Y %H:%M:%S"
                    if len(time[i].split(":")) == 3
                    else "%d-%m-%Y %H:%M"
                )
                dt_obj = datetime.strptime(date_time_str, format_string)
            else:
                dt_obj = datetime.strptime(date_time_str, "%Y-%m-%d")

            ref_day = datetime(1970, 1, 1)
            seconds = (dt_obj - ref_day).total_seconds()
            seconds_list.append(int(seconds))
        except ValueError:
            seconds_list.append(0)

    return seconds_list


def checkCurrDate(date: list, time: list = None, curr: int = None) -> list:
    """
    Checks if a list of dates (and optional times) is earlier than the current time and adjusts if necessary.

    This function processes a list of date strings, and optionally corresponding time strings, by converting
    them into seconds since the Unix epoch. If a date-time combination is later than the current time, it
    adjusts the date to one day earlier. The current time can be provided, or the function will use the
    current system time by default.

    Args:
        date (list[str]): A list of date strings in the format 'YYYY-MM-DD'.
        time (list[str], optional): An optional list of corresponding time strings in the format 'HH:MM:SS'. Defaults
        to None.
        curr (int, optional): The current time in seconds since the Unix epoch. Defaults to the system's current time
        if not provided.

    Returns:
        list[str]: A list of adjusted date strings, where each date is one day earlier than the original if it was
        later than the current time.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    # INFO usato sempre con una sola data
    seconds_list = convertDate(date, time)

    if curr is None:
        curr = int(datetime.now().timestamp())

    new_dates = []
    diff = 24 * 60 * 60  # INFO delta calcolato con funzione di libreria

    for seconds in seconds_list:
        if seconds <= curr:
            return date
        else:
            new_date_time = datetime.fromtimestamp(seconds) - timedelta(days=1)
            new_dates.append(new_date_time.strftime("%Y-%m-%d"))

    return new_dates


def getPrevDay(date):
    """
    Calculates the previous day.

    This function takes a date string and calculates the date for one day earlier.
    The dates are first converted to datetime format, and then the timedelta of one day is
    subtracted to find the previous day.

    Args:
        date (str): A date string in the format '%d-%m-%Y'.

    Returns:
        str: A date string representing the day before the input date.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    date = datetime.strptime(date, "%d-%m-%Y")
    prev_day = (date - timedelta(days=1)).strftime("%d-%m-%Y")

    return prev_day


def getNextDay(date: list) -> list:
    """
    Calculates the next day for each date in a given list.

    This function takes a list of date strings and calculates the date for one day later for each entry.
    The dates are first converted to seconds since the Unix epoch, and then the timedelta of one day is
    added to find the next day.

    Args:
        date (list[str]): A list of date strings in the format 'YYYY-MM-DD'.

    Returns:
        list[str]: A list of date strings, each representing the day after the corresponding date in the input list.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    seconds_list = convertDate(date)
    diff = 24 * 60 * 60  # INFO delta calcolato con funzione di libreria

    next_days = []
    for seconds in seconds_list:
        next_time = datetime.fromtimestamp(seconds) + timedelta(days=1)
        next_days.append(next_time.strftime("%Y-%m-%d"))

    return next_days


def getNHoursBetweenDates(
        date1: str, time1: str, date2: str, time2: str, double: bool = False
):
    """
    Calculates the difference in hours and minutes between two dates and times.

    This function computes the difference in hours and minutes between two given dates and times.
    The dates and times are first converted to seconds since the Unix epoch, and then the difference
    is calculated. The function can return the result as a floating-point number (if 'double' is True)
    or as an integer.

    Args:
        date1 (str): The first date in the format 'YYYY-MM-DD'.
        time1 (str): The time corresponding to the first date in 'HH:MM:SS' format.
        date2 (str): The second date in the format 'YYYY-MM-DD'.
        time2 (str): The time corresponding to the second date in 'HH:MM:SS' format.
        double (bool, optional): If True, returns the hour difference as a floating-point number. Defaults to False.

    Returns:
        tuple: A tuple containing two elements:
            - The difference in hours between the two dates and times (float or int based on 'double').
            - The difference in minutes between the two dates and times (float).

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    # TODO da ottimizzare
    seconds1 = convertDate(date1, time1)[0]
    seconds2 = convertDate(date2, time2)[0]

    mins_diff = (seconds1 - seconds2) / 60
    seconds1 = seconds1 / 3600
    seconds2 = seconds2 / 3600

    if not double:
        seconds1 = int(seconds1)
        seconds2 = int(seconds2)

    return (seconds1 - seconds2), mins_diff


def isTime(time: str, check: list) -> int:
    """
    Checks if a given time matches any of the specified criteria.

    This function assesses whether a provided time string matches any of the criteria in the 'check' list.
    The criteria can include specific times, wildcards, or empty strings. The function validates the time
    using 'checkTime' and then compares it against each criterion.

    Args:
        time (str): The time string to be checked, in 'HH:MM' format.
        check (list[str]): A list of criteria to check against. Each criterion can be a specific time, a wildcard (
        '*:*'), or an empty string.

    Returns:
        int: 1 if the time matches any of the criteria, 0 otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if time is None:
        return 0

    if time == "":
        return 1

    time, hh, mm = checkTime(time)

    if not isinstance(check, list):
        check = [check]

    for chk in check:
        if chk == "":
            return 1
        if chk == "*:*":
            return 1
        if time == chk:
            return 1
        pos = chk.find("*")
        pos_sep = chk.find(":")

        if pos >= 0 and pos_sep > 0:
            if pos == 0:
                cmm = int(chk[pos_sep + 1: pos_sep + 3])
                if mm == cmm:
                    return 1
            else:
                chh = int(chk[0:2])
                if hh == chh:
                    return 1

    return 0


def getHourlyWeights(
        seconds: list,
        oldest_time: int = None,
        newest_time: int = None,
        max_weight: float = None,
        cum: bool = None,
) -> np.ndarray:
    """
    Calculates hourly weights for a series of timestamps.

    This function computes hourly weights for a list of timestamps in seconds. The weights represent the amount
    of time in hours between each timestamp and its successor. Additional parameters allow for adjustments based
    on the oldest and newest times, a maximum weight, and cumulative calculation.

    Args:
        seconds (list[int]): A list of timestamps in seconds.
        oldest_time (int, optional): The timestamp to consider as the oldest time. Defaults to None.
        newest_time (int, optional): The timestamp to consider as the newest time. Defaults to None.
        max_weight (float, optional): The maximum weight to assign. Defaults to None.
        cum (bool, optional): If True, performs cumulative calculation. Defaults to None.

    Returns:
        np.ndarray: An array of hourly weights corresponding to the time differences between the timestamps.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    n_times = len(seconds)
    if n_times <= 0:
        return np.array([0])
    if n_times == 1:
        www = 1.0
        if oldest_time is not None and newest_time is not None:
            www = (newest_time - oldest_time) / 3600.0
        if max_weight is None:
            return np.array([www])
        if max_weight <= 0:
            return np.array([www])
        if www > max_weight:
            return np.array([max_weight])
        return np.array([www])

    if oldest_time is None:
        oldest_time = 0
    if newest_time is None:
        newest_time = 0
    if seconds[n_times - 1] <= seconds[0]:
        sec = seconds
    else:
        sec = np.flip(seconds)

    if oldest_time == 0:
        oldest_time = sec[n_times - 1]
    if newest_time == 0:
        newest_time = sec[0]

    S1 = np.roll(sec, -1)
    if cum is None:
        S1[n_times - 1] = oldest_time
        avg = (sec + S1) / 2
        S1 = avg
        avg = np.concatenate(([newest_time], avg[0: n_times - 2]))
    else:
        avg = sec
    S1[n_times - 1] = oldest_time
    www = (avg - S1) / 3600.0
    www[www < 0] = 0
    if seconds[n_times - 1] > seconds[0]:
        www = np.flip(www)

    if max_weight is None:
        return www
    if max_weight <= 0:
        return www
    www[www > max_weight] = max_weight
    return www


def last_uniq(array: np.ndarray, idx: np.ndarray, last: bool = False) -> np.ndarray:
    """
    Identifies the last unique elements in a subarray of 'array' based on indices 'idx'.

    This function processes a subset of 'array' specified by 'idx' and identifies the indices of elements
    that are different from their next element. If 'last' is True, it modifies the indices to include
    the last occurrences of unique elements. If no unique elements are found, it returns the maximum index.

    Args:
        array (np.ndarray): The input array from which unique elements are identified.
        idx (np.ndarray): An array of indices representing a subset of 'array'.
        last (bool, optional): If True, returns indices of the last occurrences of unique elements. Defaults to False.

    Returns:
        np.ndarray: An array of indices representing the unique elements in the specified subset of 'array'.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    """
    # Altra possibile implementazione
    q = array[idx]
    
    # Indices where q is not equal to its shifted version
    indices = np.where(q[:-1] != q[1:])[0]
    discarted = np.where(q[:-1] == q[1:])[0]
    
    # If there are no unique values, return the maximum of idx
    if len(indices) == 0:
        return np.max(idx)
    
    ind = idx[indices]
    
    # If there are no repeated values or the 'last' keyword is not set, return ind
    if len(discarted) == 0 or not last:
        return ind
    
    inc = discarted - np.arange(len(discarted))
    selected = indices[inc]
    ind[inc] = idx[selected] > idx[discarted]
    return ind
    """
    q = np.array(array)[idx]
    indices = np.where(q != np.roll(q, -1))[0]

    if len(indices) <= 0:
        return max(idx)

    discarded_values = np.where(q == np.roll(q, -1))[0]

    ind = idx[indices]
    nd = len(discarded_values)

    if nd <= 0 or last is False:
        return ind

    inc = np.setdiff1d(np.arange(nd), ind)
    selected = indices[inc]
    ind[inc] = idx[selected][idx[selected] > idx[indices[-1]]]

    return ind


def get_julian_day(seconds: int):
    """
    Calculates the Julian day number and hour for a given timestamp in seconds.

    This function converts a timestamp (in seconds since the Unix epoch) to the Julian day number, which
    represents the day of the year. It also computes the hour of the day for the given timestamp.

    Args:
        seconds (int): The timestamp in seconds since the Unix epoch.

    Returns:
        tuple: A tuple containing two elements:
            - The Julian day number (int) representing the day of the year.
            - The hour of the day (int) corresponding to the timestamp.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    dt = datetime.utcfromtimestamp(seconds)
    curr_day = dt.timetuple().tm_yday
    first_day = datetime(dt.year, 1, 1).timetuple().tm_yday
    hour = dt.hour
    return curr_day - first_day, hour


def search_path(
        fromPath: str,
        date: str,
        time: str,
        minStep: int = None,
        nHours: int = None,
        nMin: int = None,
        bilateral: bool = False,
        check_files: bool = False,
        exclude: list = None,
) -> str:
    """
    Searches for a valid path by incrementing/decrementing time from a given start date and time.

    This function searches for a valid path by modifying the provided 'fromPath' based on a given date
    and time, and incrementing or decrementing the time by 'minStep' minutes. It checks for file existence
    at each step and returns the first path where files are found. The search can be bilateral, meaning it
    goes both forward and backward in time. It can also exclude certain times and stop after a specified
    duration ('nHours' or 'nMin').

    Args:
        fromPath (str): The base path to be modified in the search.
        date (str): The starting date for the search.
        time (str): The starting time for the search.
        minStep (int): The minute increment/decrement for each search step.
        nHours (int, optional): The total duration of the search in hours. Defaults to None.
        nMin (int, optional): The total duration of the search in minutes. Defaults to 15 if not provided.
        bilateral (bool, optional): If True, searches both forward and backward in time. Defaults to False.
        check_files (bool, optional): If True, checks for file existence at each path. Defaults to False.
        exclude (list[str], optional): A list of times to exclude from the search. Defaults to None.

    Returns:
        str: The first valid path found or an empty string if none is found within the specified duration.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    if date is None:
        return fromPath
    if time is None:
        return fromPath
    if date == "" and time == "":
        return fromPath

    if minStep is None:
        minStep = -1
    if minStep == 0:
        minStep = -1

    if nMin is None:
        nMin = 15
        if nHours is not None and np.size(nHours) == 1:
            nMin = int(nHours * 60)
    else:
        nMin = nMin

    currDate = date
    currTime = time
    currDate2 = date
    currTime2 = time
    ok = 1
    nTimes = int(nMin / abs(minStep))
    if nTimes < 1:
        nTimes = 1
    for lll in range(nTimes):
        if exclude is not None:
            ok = 1 - isTime(currTime, exclude)
        if ok > 0:
            path = changePath(fromPath, currDate, currTime)
            if dpg.utility.checkAllFiles(path, check_files):
                return path
        currDate, currTime = addMinutesToDate(currDate, currTime, minStep)
        if bilateral:
            currDate2, currTime2 = addMinutesToDate(currDate2, currTime2, -minStep)
            if exclude is not None and len(exclude) > 0:
                ok = 1 - isTime(currTime2, exclude)
            if ok > 0:
                path = changePath(fromPath, currDate2, currTime2)
                if dpg.utility.checkAllFiles(path, check_files):
                    return path
    log_message(
        f"Cannot find data for {nMin} minutes in {fromPath}",
        general_log=True,
        level="WARNING",
    )

    return ""


def getPrevNode(
        node,
        min_step: int,
        date: str = None,
        time: str = None,
        remove: bool = False,
        next: bool = False,
):
    """
    Retrieves the previous or next node based on time increments from a given node.

    This function identifies a node before or after the given 'node' by altering the date and time based on
    'min_step'. The function can either move backward (previous node) or forward (next node) in time. If the
    path derived from the new date and time does not exist, or if the 'remove' flag is set, the original node
    is affected as specified.

    Args:
        node (Node object): The reference node.
        min_step (int): The minute increment/decrement to determine the previous/next node.
        date (str, optional): The date to start from. If not provided, it's derived from the node's path. Defaults to
        None.
        time (str, optional): The time to start from. If not provided, it's derived from the node's path or
        'get_time'. Defaults to None.
        remove (bool, optional): If True, removes the original node from the tree. Defaults to False.
        next (bool, optional): If True, moves forward in time to find the next node; otherwise, moves backward.
        Defaults to False.

    Returns:
        Node object or None: The node found at the new path, or None if no valid path is found.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    path = dpg.tree.getNodePath(node)

    if path == "":
        return None, None, None

    if date is None:
        date, time = path2Date(path)
        if date == "":
            date, time, _ = get_time(node, nominal=True)

    if min_step != 0:
        ttt = time
        ddd = date
        step = -abs(min_step)
        if next:
            step = abs(step)
        ddd, ttt = addMinutesToDate(ddd, ttt, step)
        path = changePath(path, ddd, ttt)
    else:
        path = changePath(path, date, time)

    if not os.path.exists(path):
        log_message("Cannot find " + path, general_log=True, level="WARNING")
        if remove:
            dpg.tree.removeTree(node)
        return None, date, time

    if remove:
        dpg.tree.removeTree(node)

    return dpg.tree.createTree(path), date, time


def sortDates(
        dates: list,
        times: list,
        unique: bool = False,
        last: bool = False,
        n_hours: int = None,
        newest_time: int = None,
        strictly: bool = False,
        seconds: list = None,
        oldest_first: bool = False,
):
    """
    Sorts date and time combinations, optionally applying filters like uniqueness and time range.

    This function sorts a list of date and time combinations. It supports filtering for unique times,
    selecting the last occurrence, limiting to a specific time range, and reversing the sort order.
    It can also handle the conversion of dates and times to seconds since the Unix epoch.

    Args:
        dates (list[str]): A list of date strings in 'YYYY-MM-DD' format.
        times (list[str]): A list of corresponding time strings in 'HH:MM:SS' format.
        unique (bool, optional): If True, filters for unique times. Defaults to False.
        last (bool, optional): If True, selects the last occurrence in case of duplicates. Defaults to False.
        n_hours (int, optional): Limits the range to the last 'n_hours' hours. Defaults to None.
        newest_time (int, optional): The most recent time in seconds since the Unix epoch. Defaults to None.
        strictly (bool, optional): If True, adjusts the time range to be strictly within limits. Defaults to False.
        seconds (list[int], optional): Precomputed seconds since the Unix epoch for each date-time combination.
        Defaults to None.
        oldest_first (bool, optional): If True, sorts with the oldest times first. Defaults to False.

    Returns:
        tuple: A tuple containing three elements:
            - The number of date-time combinations after sorting and applying filters (int).
            - A list of indices indicating the order of the sorted and filtered date-time combinations (list[int]).
            - A list of seconds after sorting and filtering (list[int]).

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if len(dates) > 0:
        if seconds is None or len(seconds) != len(dates):
            seconds = convertDate(dates, times)

    else:
        seconds = np.array(seconds)

    if not seconds or len(seconds) == 0:
        return 0, [], []

    ind = np.argsort(seconds)
    if newest_time is None:
        newest_time = [0]

    if newest_time[0] <= 0:
        newest_time = seconds[ind[-1]]
    if unique:
        ind = last_uniq(seconds, ind, last=last)
    if n_hours is not None:
        n_sec = int(n_hours * 3600)
        oldest_time = newest_time[0] - n_sec
        if strictly:
            oldest_time += 1
        sss = np.array(seconds)[ind]
        valids = np.where((sss >= oldest_time) & (sss <= newest_time[0]))[0]
        if len(valids) <= 0:
            return 0, [], []
        ind = ind[valids]
    if not oldest_first:
        ind = ind[::-1]
    seconds = np.array(seconds)[ind]
    return len(ind), ind, seconds


def get_times(paths: str = None, node=None, only_seconds: bool = False):
    """
    Retrieves the dates, times, and seconds from the provided paths or node. If `only_seconds` is set to True, only
    seconds will be returned. Either node or paths must be valid input of the function.

    Args:
        paths (str, list): One or more paths from which to load the attributes. Defaults to an empty string.
        node (Node, optional): A node object from which to retrieve the path and associated attributes. Defaults to
        None.
        only_seconds (bool): If True, the function will return only the seconds. Defaults to False.

    Returns:
        tuple: A tuple containing two lists (dates and times) or a single list of seconds if `only_seconds` is True.
    """

    first_valid = -1
    attr = None
    nTimes = 0
    dates = None
    times = None
    seconds = None

    if isinstance(node, dpg.node__define.Node):
        attr = dpg.tree.getSingleAttr(node, "times.txt")
        paths = dpg.tree.getNodePath(node)

    if not isinstance(paths, str) and not isinstance(paths, list) and paths != "":
        if only_seconds:
            return seconds
        return dates, times

    if isinstance(paths, str):
        if not isinstance(attr, dpg.attr__define.Attr):
            attr = dpg.attr.loadAttr(paths, "times.txt")
            if not isinstance(attr, dict):  # loadAttr restituisce un dizionario
                attr = dpg.attr.loadAttr(paths, dpg.cfg.getArrayDescName())

        tmp, exists_seconds, _ = dpg.attr.getAttrValue(
            attr, "seconds", default=0.0
        )
        if exists_seconds:
            seconds = tmp
        if only_seconds:
            attr = None
            return seconds

        tmp, exists_date, _ = dpg.attr.getAttrValue(
            attr, "date", default=""
        )
        if exists_date:
            dates = tmp
            times, _, _ = dpg.attr.getAttrValue(attr, "time", default="")
            first_valid = 0
        else:
            dt_acq, exists_dt_acq, _ = dpg.attr.getAttrValue(
                attr, "dt_acq", default=""
            )
            if exists_dt_acq:
                _, date, time = dtAcq2Date(dt_acq)

        return dates, times

    if isinstance(paths, list):
        nTimes = len(paths)

    dates = [""] * nTimes
    times = [""] * nTimes

    for ppp in range(nTimes):
        attr = dpg.attr.loadAttr(paths[ppp], dpg.cfg.getArrayDescName())
        times[ppp], _, _ = dpg.attr.getAttrValue(attr, "time", default="")
        dates[ppp], exists, _ = dpg.attr.getAttrValue(
            attr, "date", default=""
        )

        if not exists:
            attr = None
            # attr = dpg.attr.loadAttr(paths[ppp], dpg.cfg.getArchiveDescName())
            # dt_acq, exists = dpg.attr.getAttrValue(attr, 'dt_acq', default='', get_exists=True)
            # if exists:
            #     _, date, time = dtAcq2Date(dt_acq[0])  # TODO dt_acq è una lista?
            #     dates[ppp] = date
            #     times[ppp] = time
        if first_valid < 0 and exists:
            first_valid = ppp

    if first_valid < 0:
        dates = []
        times = []
        nTimes = 0

    return dates, times


def create_path_list(
        fromPath: str, date: str, time: str, minStep: int, nHours: int, min_tol: int = None
) -> list:
    """
    Creates a list of paths by incrementing time at regular intervals from a start date and time.

    This function generates a list of paths starting from a given date and time, incrementing (or decrementing)
    the time by 'minStep' for a duration of 'nHours'. The function allows for a tolerance 'min_tol' to adjust
    the start time before beginning the sequence.

    Args:
        fromPath (str): The base path to be modified in the sequence.
        date (str): The start date in 'YYYY-MM-DD' format.
        time (str): The start time in 'HH:MM:SS' format.
        minStep (int): The time increment/decrement in minutes for each step.
        nHours (int): The total duration in hours for generating the path list.
        min_tol (int, optional): The tolerance in minutes to adjust the start time. Defaults to 0.

    Returns:
        list[str]: A list of paths created by incrementing or decrementing the time at regular intervals from the
        start date and time.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    tol = 0
    if min_tol is not None:
        tol = min_tol
    nItems = 1
    if minStep != 0:
        nItems += int((nHours * 60 + 2 * tol) / abs(minStep))

    datetime_string = date + " " + time
    format_string = (
        "%d-%m-%Y %H:%M:%S"
        if len(datetime_string.split(":")) == 3
        else "%d-%m-%Y %H:%M"
    )
    currDateTime = datetime.strptime(datetime_string, format_string)
    if tol > 0:
        currDateTime += timedelta(minutes=tol)

    path_list = []
    for _ in range(nItems):
        date_str = currDateTime.strftime("%d-%m-%Y")
        time_str = currDateTime.strftime("%H:%M")
        path_list.append(
            changePath(fromPath, date=date_str, time=time_str)
        )  # TODO correggere, bisogna dare in input un date e un time
        currDateTime += timedelta(minutes=float(minStep))

    return path_list


def searchNode(
        name: str,
        schedule: Schedule,
        guess: str = None,
        sourceNode=None,
        date: str = None,
        time: str = None,
        wait: int = None,
        origin: str = "",
) -> object:
    """
    Searches for a node in a schedule or a source tree based on various criteria.

    This function searches for a node with the specified 'name' in a given schedule or within the tree of a
    'sourceNode'. It can modify the search path based on provided 'date' and 'time', and optionally wait for
    the node to become available. The function can handle multiple scenarios, including guessing the product path
    or finding nodes based on a specific origin.

    Args:
        name (str): The name of the node to search for.
        schedule (Schedule object, optional): The schedule to search in. Defaults to None.
        guess (str, optional): A guess for the product path. Defaults to None.
        sourceNode (Node object, optional): The source node to search within. Defaults to None.
        date (str, optional): The date to use for modifying the search path. Defaults to None.
        time (str, optional): The time to use for modifying the search path. Defaults to None.
        wait (int, optional): The time to wait for the node to become available. Defaults to None.
        origin (str, optional): The origin to consider in the search. Defaults to ''.

    Returns:
        tuple: A tuple containing two elements:
            - The node found based on the search criteria, or None if not found.
            - A flag (int) indicating if the path or tree has changed during the search.

    Raises:
        None: This function does not explicitly raise any exceptions but prints messages for various scenarios.
    """
    changed = 0
    if schedule is not None and schedule != "":
        if guess is None:
            prod_list = dpg.access.get_prod_list(schedule)
            count = len(prod_list)
            if count <= 0:
                print("No products found in schedule.")
                return None, changed
            prod_path = prod_list[-1]
        else:
            print("TODO: guess prod path")
            # prod_path = guess_prod_path()

        new_path = prod_path
        currTree = None
        changed = 1
    else:
        if not isinstance(sourceNode, dpg.node__define.Node):
            log_message("Invalid Tree", level="ERROR")
            return None, changed
        prod_path = dpg.tree.getNodePath(sourceNode)
        currTree = copy.deepcopy(sourceNode)

    if date is not None:
        prod_path = dpg.path.checkPathname(prod_path, with_separator=False)
        new_path = changePath(prod_path, date, time)
        changed = 1 if new_path != prod_path else 0

    if changed > 0 or currTree is None:
        if wait is not None:
            print("TODO: wait to be implemented")
        if not os.path.isdir(new_path):
            # print(f"Cannot find new_path {new_path}")
            return None, changed
        else:
            print(f"Found data at path {new_path}")

        currTree = dpg.tree.createTree(new_path)
        changed = 1

    if name is None:
        return currTree, changed

    nodes = dpg.tree.findAllDescendant(currTree, name)
    if not isinstance(nodes, list):
        nodes = [nodes]
    count = len(nodes)

    if count <= 0:
        print(f"Cannot find Node {name}")
        return None, changed
    else:
        if origin is not None:
            if origin != "":
                print("TODO: Origin is not empty string")

        if count != 1:
            print(f"Found multiple names {name}")
            return None, changed

    return nodes[0], changed


def set_date(node, inId):
    """
    Sets the date and time for a specified node based on the date and time retrieved from another node.

    This function retrieves the date and time associated with the input node (`inId`) and applies them
    to the specified `node` by updating its `date` and `time` attributes.

    Args:
        node (Node): The node for which the date and time will be set.
        inId (Node): The node from which the date and time are retrieved.

    Returns:
        None: This function doesn't return a value.
    """
    date, time, _ = get_time(inId)
    set_time(node, date=date, time=time)
    return


def mergePrevData(prodId, newValue, max_samples, min_step, retry=False, nonull=False,
                  noTimes=None, delay=None):
    nT = None
    fromNode, date, time = getPrevNode(prodId, min_step)
    pointer, out_dict = dpg.array.get_array(fromNode)
    if out_dict is not None:
        dim = out_dict['dim']

    if pointer is None and retry:
        dpg.tree.removeTree(fromNode)
        fromNode, date, time = getPrevNode(prodId, retry)
        pointer, out_dict = dpg.array.get_array(fromNode)
        if out_dict is not None:
            dim = out_dict['dim']

    if pointer is not None:
        if np.size(newValue) == 1:
            newValue = pointer + [newValue]
        else:
            log_message("Da controllare cosa succede nel caso in cui newValue > 1", level='ERROR')
            if np.size(newValue) > 1:
                newValue = [[pointer], [newValue]]
                dim = np.shape(newValue)
                nT = dim[1]  # TODO: controllare dimensione
            else:
                if nonull:
                    newValue = pointer
                else:
                    newValue = [pointer, np.nan]

    else:
        if np.size(newValue) == 0:
            newValue = np.nan

    if np.size(nT) != 1 or nT is None:
        nT = np.size(newValue)

    if nT > max_samples:
        if np.size(dim) > 1:
            newValue = newValue[:, nT - max_samples:]
        else:
            newValue = newValue[nT - max_samples:]
        nT = max_samples

    dpg.tree.removeTree(fromNode)

    if noTimes:
        return  # TODO: controllare

    if delay is not None:
        dpg.times.addMinutesToDate(date, time, -delay)

    seconds = dpg.times.convertDate(date, time)
    if isinstance(seconds, list) and len(seconds) > 0:
        seconds = seconds[0]
    dates = [None] * nT
    times = [None] * nT
    step = -np.abs(min_step)
    ms = step * 60

    for ttt in range(nT):
        date, time = dpg.times.seconds2Date(seconds)
        dates[nT - ttt - 1] = date
        times[nT - ttt - 1] = time
        seconds += ms

    dpg.times.save_times(prodId, dates, times)

    return newValue


def save_times(node, dates, times, dt_format=None, path=None, seconds=None):
    if dt_format == 1:
        _, dt_acq, _ = date2dtAcq(dates, times)
        nTimes = np.size(dt_acq)
        if nTimes <= 0:
            return  # TODO: fixare valore di ritorno
        varnames = ['dt_acq'] * nTimes
        values = dt_acq
    else:
        nTimes = np.size(dates)
        if nTimes <= 0:
            return  # TODO: valore di tirotno?
        varnames = ['date'] * nTimes
        values = dates
        if np.size(times) == nTimes:
            varnames = varnames + ['time'] * nTimes
            values = values + times
    if np.size(seconds) == nTimes and seconds is not None:
        varnames = varnames + ['seconds'] * nTimes
        values = values + [str(seconds)]
    name = 'times.txt'
    if path is None:
        path = dpg.tree.getNodePath(node)
    dpg.attr.saveAttr(path, name, varnames, values)

    return
