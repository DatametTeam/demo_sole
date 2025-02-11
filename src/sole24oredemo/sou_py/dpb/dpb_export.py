import os
import sou_py.dpg as dpg

"""
Funzioni ancora da portare
FUNCTION GetExportPath 
PRO Split3DProd 
FUNCTION Check3DProd 
PRO AddFormat 
FUNCTION GetAllFormatsAndUsers 
PRO DPB_EXPORT 
"""


def getExportName(prod_path, format, date, time):
    """
    Constructs an export file name based on product path, format, date, and time.

    This function generates a file name for exporting data. It takes the base name
    of the product path and appends a formatted date, time, and a file extension
    based on the specified format. The function supports various formats, including
    HDF, TIFF, SHP, BUFR, JPEG, and others. The time string is processed to ensure
    it has the correct format.

    Args:
        prod_path (str): The path of the product file.
        format (str): The format of the export file. Supported formats include 'HDF',
                      'TIFF', 'SHP', 'BUFR', 'JPEG', and others.
        date (str): The date string to be included in the export file name.
        time (str): The time string to be included in the export file name. This
                    string is processed to fit a specific format.

    Returns:
        str: The constructed export file name, including the base product name, date,
             formatted time, and appropriate file extension.

    Note:
        The function uses the `os.path.basename` method to extract the base name of
        the product file. The time string is processed using `dpg.times.checkTime`
        to ensure it follows a specific format. If the format is 'PAR' or 'SITES',
        the product name is set to 'PAR' or 'SITES', respectively, regardless of the
        input product path.
    """

    prod_name = os.path.basename(prod_path)
    suffix = ".txt"

    format = format.upper()

    if format in ["HDF", "HDF5", "ODIM"]:
        suffix = ".hdf"
    elif format in ["TIF", "TIFF"]:
        suffix = ".tif"
    elif format == "SHP":
        suffix = ".shp"
    elif format in ["BUFR", "BFR"]:
        suffix = ".bfr"
    elif format in ["JPEG", "JPG"]:
        suffix = ".jpg"
    elif format == "PAR":
        prod_name = "PAR"
    elif format == "SITES":
        prod_name = "SITES"
    else:
        suffix = "." + format.lower()

    ttt, _, _ = dpg.times.checkTime(time, sep="-")
    name = f"{prod_name}_{date}_{ttt}{suffix}"

    return name
