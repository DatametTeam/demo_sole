import numpy as np

import sou_py.dpg as dpg


def replaceStormId(field0, strtime):
    """
    Generates a unique storm ID based on a timestamp and modifies it with a padded string.

    This function constructs a storm ID by combining the year and the rest of the timestamp
    with the first part of the provided field. It also ensures the ID is properly padded with spaces to a fixed length.

    Args:
        field0 (str): The initial string that contains storm-related information (typically an ID prefix).
        strtime (str): The timestamp string in the format 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        str: The formatted storm ID with a padded string followed by the timestamp.
    """
    ttt = field0[0:4]
    date = strtime[0:10]
    time = strtime[11:13] + strtime[14:16]
    if int(ttt) > int(time):
        date = dpg.times.getPrevDay(date)
    date = dpg.times.checkDate(date, sep="", year_first=True)
    id = date[2:] + field0
    space = " " * 46
    id = id + space[len(id) :]
    return id + strtime


def shp2txt(prodId, txtPath):
    """
    Converts shape data from a product into a text file format.

    This function extracts shape-related information, including latitude, longitude, and other parameters,
    and then writes them to a text file. It uses a product ID to retrieve the data and ensure it's formatted correctly.

    Args:
        prodId (Node): The product ID from which shape data is accessed.
        txtPath (str): The path to the text file where the data will be saved.

    Returns:
        None: The function saves the records into a text file at the specified location.
    """
    if np.size(txtPath) != 1:
        return

    attr, _, parnames = dpg.coords.get_shape_info(prodId, names=True)
    if len(attr) <= 0 or attr is None:
        return

    count, points = dpg.coords.get_points(prodId, center=True)
    if count <= 0:
        return

    records = [""] * count
    nPar = len(parnames)
    lats = [f"{row[1]:8.2f}" for row in points]
    lons = [f"{row[0]:8.2f}" for row in points]

    for ccc in range(count):
        records[ccc] = replaceStormId(attr.iloc[ccc]["Name"], attr.iloc[ccc]["Time"])
        records[ccc] = records[ccc] + lats[ccc] + lons[ccc]
        for ppp in range(2, nPar):
            if attr.iloc[ccc].iloc[ppp] <= -9999:
                attr.iloc[ccc, ppp] = np.nan
            records[ccc] = records[ccc] + f"{attr.iloc[ccc].iloc[ppp]:8.1f}"

    date, _, _ = dpg.times.get_time(prodId)
    name = date[3:] + ".txt"
    dpg.io.save_values(txtPath, name, records, append=True)
    return


def shape2txt(prodId, txtPath, date=None, nTime=None):
    """
    Wrapper function that handles saving shape data to a text file.

    This function checks the type of the provided product ID and delegates the task to `shp2txt`
    for converting shape data into a text file format.

    Args:
        prodId (str or Node): The product ID from which shape data is accessed.
        txtPath (str): The path to the text file where the data will be saved.
        date (optional): The date parameter (not used in the current implementation).
        nTime (optional): The time parameter (not used in the current implementation).

    Returns:
        None: The function calls `shp2txt` to perform the file writing operation.
    """
    if isinstance(prodId, dpg.node__define.Node):
        shp2txt(prodId, txtPath)
        return
