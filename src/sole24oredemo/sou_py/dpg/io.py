from pathlib import Path

import geopandas
import pandas as pd
import fiona
from shapely import Polygon
from shapely.geometry import shape

import sou_py.dpg as dpg
from math import prod
import numpy as np
import os
import struct
from io import StringIO
import geopandas as gpd

from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION get_xml_values 
FUNCTION GetHmlHead 
FUNCTION GetHmlTail 
FUNCTION is_graphic 
FUNCTION read_graphic 
FUNCTION read_lut 
FUNCTION read_polyline 
FUNCTION read_shape 
FUNCTION read_xml_values 
FUNCTION save_hml_coords 
PRO executeCommand 
FUNCTION check_polyline_name    // UNUSED
FUNCTION is_valid_format        // UNUSED
"""


def read_values(path: str, name: str, pathname: str = "", mode_256: bool = False):
    """
    Reads values from a file specified by the given path and name, and returns them as a NumPy array.

    This function attempts to read a file containing floating-point numbers, one per line.
    If 'pathname' is not provided or empty, it constructs the full path using 'dpg.path.getFullPathName'.
    If the file does not exist, it returns None and an error code.
    If 'mode_256' is True and the number of values in the file is 255 or less, the function pads the array with NaN
    values to make its length 256.

    Args:
        path (str): The directory path where the file is located.
        name (str): The name of the file to be read.
        pathname (str, optional): The full path to the file. If empty, it will be constructed using 'path' and
        'name'. Defaults to ''.
        mode_256 (bool, optional): If True, pads the array with NaN values up to a length of 256 if necessary.
        Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - np.ndarray: A NumPy array of the values read from the file, or None if an error occurred.
            - int: An error code, where 0 indicates no error and 1 indicates an error occurred.

    Raises:
        IOError: If there is an error opening the file.
        ValueError: If there is an error parsing a value in the file.
    """
    err = 0
    if (pathname == "") or pathname is None:
        pathname = dpg.path.getFullPathName(path, name)
    if not os.path.isfile(pathname):
        print("Error: File", pathname, "does not exists")
        err = 1
        return None, err
    # endif

    try:
        with open(pathname, "r") as file:
            values = [float(line.strip()) for line in file if line.strip()]

            if len(values) <= 255 and mode_256:
                values.extend([float("nan")] * (256 - len(values)))

            arr = np.array(values)
    except IOError:
        return None, err
    except ValueError as e:
        # Handle the '-Inf' case if necessary
        print(f"Error parsing a value: {e}")
        return None, err

    return arr, err


def read_strings(path: str, name: str, pathname: str = "", record: int = None):
    """
    Read strings from a file and return them as a list or a single string if a record number is provided.

    Args:
        path (str): The directory path where the file is located.
        name (str): The file name to read strings from.
        pathname (str, optional): The full path to the file. If not provided, it's constructed from `path` and `name`.
        record (int, optional): The record number to read from the file. If provided, reads a specific line.

    Returns:
        list/str: A list of strings read from the file, or a single string if a record number is provided.
    """
    err = 0
    strings = []
    if (pathname == "") or pathname is None:
        pathname = dpg.path.getFullPathName(path, name)
    if not os.path.isfile(pathname):
        # in fase di sviluppo va bene perchè mancano i dati effettivamente
        # TODO: è stato rimosso il warning su dati mancanti nei test di high_res ma questa cosa va sistemata in futuro
        path_ = Path(pathname)
        file_name = path_.name
        if not file_name in ["generic.txt", "sites.txt", "definition.txt"]:
            log_message(f"File {pathname} does not exists", level="WARNING")
        err = 1
        return strings, err
    # endif

    # strings e' la lista di righe
    with open(pathname, "r") as file:
        if record is not None:
            # Read a specific line based on the record number
            for i, line in enumerate(file):
                if i == record:
                    return line.strip()
            # If the record number is larger than the number of lines, return an empty string
            return ""
        else:
            # Read all lines into a list
            strings = [line.replace("\n", "") for line in file]

    return strings, err


def type_idl2py(dtype: int) -> np.dtype:
    """
    Converts an IDL data type to its corresponding NumPy data type.

    This function maps data type codes from IDL (Interactive Data Language) to equivalent data types in NumPy. The
    supported IDL types are byte, short, integer, float, double, unsigned short, and unsigned long. If an unsupported
    data type is passed, the function prints a warning message and returns 0.

    Args:
        dtype (int): The IDL data type code to be converted. The supported codes are:
            - 1: byte
            - 2: short
            - 3: integer
            - 4: float
            - 5: double
            - 12: unsigned short
            - 13: unsigned long

    Returns:
        dtype: The corresponding NumPy data type, or 0 if the IDL data type is not supported.

    Raises:
        None: This function does not raise any exceptions but prints a warning for unsupported data types.
    """
    if dtype == 1:  # byte
        return np.uint8
    elif dtype == 2:  # short
        return np.int16
    elif dtype == 3:  # integer
        return np.int32
    elif dtype == 4:  # float
        return np.float32
    elif dtype == 5:  # double
        return np.float64
    elif dtype == 12:  # unsigned short
        return np.uint16
    elif dtype == 13:  # unsigned long
        return np.uint32
    else:
        log_message(f"Data type IDL {dtype} not implemented!", level='WARNING+', all_logs=True)
    return 0


def type_py2idl(dtype: np.dtype) -> int:
    """
    Converts a NumPy data type to its corresponding IDL (Interactive Data Language) data type code.

    This function maps data types from NumPy to their equivalent data type codes in IDL. The supported NumPy types
    are uint8, int16, int32, float32, float64, uint16, and uint32. If a NumPy data type that is not supported is
    provided, the function prints a warning message and returns 0.

    Args:
        dtype (dtype): The NumPy data type to be converted. Supported NumPy data types are:
            - np.uint8: byte
            - np.int16: short
            - np.int32: integer
            - np.float32: float
            - np.float64: double
            - np.uint16: unsigned short
            - np.uint32: unsigned long

    Returns:
        int: The corresponding IDL data type code, or 0 if the NumPy data type is not supported.

    Raises:
        None: This function does not raise any exceptions but prints a warning for unsupported data types.
    """
    if dtype == np.uint8:  # byte
        return 1
    elif dtype == np.int16:  # short
        return 2
    elif dtype == np.int32:  # integer
        return 3
    elif dtype == np.float32:  # float
        return 4
    elif dtype == np.float64:  # double
        return 5
    elif dtype == str:
        return 7
    elif dtype == pd.DataFrame or dtype == geopandas.geodataframe.GeoDataFrame:
        return 8
    elif dtype == list:
        return 11
    elif dtype == np.uint16:  # unsigned short
        return 12
    elif dtype == np.uint32:  # unsigned long
        return 13
    else:
        log_message(f"Data type python {dtype} not implemented!", level='WARNING+', all_logs=True)
    return 0


def read_binary(pathname, dtype, dim):
    dtype_map = {
        1: (np.uint8, 1),    # byte
        2: (np.int16, 2),    # short
        3: (np.int32, 4),    # integer
        4: (np.float32, 4),  # float
        5: (np.float64, 8),  # double
        12: (np.uint16, 2),  # unsigned short
        13: (np.uint32, 4),  # unsigned long
    }

    if dtype not in dtype_map:
        print("Warning: unknown dtype", dtype)
        return None

    np_dtype, byte_size = dtype_map[dtype]

    total_elements = prod(dim)
    total_bytes = total_elements * byte_size

    with open(pathname, "rb") as f:
        file_content = f.read(total_bytes)

    array = np.frombuffer(file_content, dtype=np_dtype)

    if len(dim) > 1 or dim[0] != total_elements:
        array = array.reshape(dim)

    return array


def read_array(
        path: str,
        name,
        dtype: int,
        dim: tuple,
        format: str = "",
        endian: str = None,
        silent: bool = False,
        no_palette: bool = True,
        get_file_date: bool = False,
        names: list = None,
):
    """
    Read data from a file into an array, with support for various data formats and options.

    This function supports reading data in ASCII, binary, and other specified formats. If a file
    is in binary format, it reads the file according to the specified data type and dimensions.
    For ASCII data, the contents are read into a NumPy array. The function also manages color
    palettes and transparency if provided.

    Args:
        path (str): The directory path where the file is located.
        name (str or list): The name of the file to read.
        dtype (int): Data type specifier for reading binary data.
        dim (tuple): Dimensions of the data to read.
        format (str, optional): Format of the file. If empty, the format is inferred from the file extension.
        endian (str, optional): Endian-ness of the binary data. Use 'little' or 1 for little-endian.
        silent (bool, optional): If True, suppresses print statements for errors.
        no_palette (bool, optional): If True, does not apply color palettes to the data.
        get_file_date (bool, optional): If True, fetches the file modification date.
        names (list, optional): Names of data fields, if applicable.

    Returns:
        tuple: A tuple containing:
            - data (numpy.ndarray): The read data.
            - palette (numpy.ndarray): The color palette, if applicable.
            - coords (any): The coordinates, if provided.
            - file_date (datetime, optional): The file modification date, if `get_file_date` is True.

    If an error occurs or the data cannot be read, the function will return None for the data and palette,
    along with the original coords and file_date values. If `get_file_date` is True, the modification
    date of the file is also returned.

    Raises:
        Exception: If the file cannot be opened or an error occurs during reading.
    """

    other_fromat = 1
    coords = None
    data = None
    palette = None
    file_date = None

    if not isinstance(name, str) or name == "":
        print("Error: name not valid")
        if get_file_date:
            return data, palette, coords, file_date
        else:
            return data, palette, coords
    if isinstance(name, list) and isinstance(name[0], str) and name[0] != "":
        pathname = dpg.path.getFullPathName(path, name[0])
    else:
        pathname = dpg.path.getFullPathName(path, name)
    if not os.path.isfile(pathname):
        if get_file_date:
            return data, palette, coords, file_date
        else:
            return data, palette, coords

    if format == "":
        if pathname.upper().find(".TXT") > 0:
            format = "txt"
        if pathname.upper().find(".SHP") > 0:
            format = "shp"
    # endif

    if format.upper() == "ASCII":
        array = read_ascii(pathname)
    elif format.upper() == "COORDS":
        array = read_coords(path, name, dim)
    elif format.upper() == "POLYLINE":
        pass  # TODO
    elif format.upper() == "WIND":
        array = read_wind(path, name)
    elif format.upper() == "SHP":
        coords_poly, array, names = read_shape(path, name)
        coords = dpg.attr__define.Attr(name='coords_attr', owner='None', pointer=coords_poly)

    elif format.upper() == "DBF":
        tmp = name
        position = tmp.find(".dbf")
        if position > 0:
            tmp = tmp[:position] + ".shp"
            coords, array, names = dpg.io.read_shape(path, tmp)

    elif format.upper() == "TXT":
        array = read_strings(path, name)
    else:
        other_fromat = 0

    if (
            format == ""
            or format.upper() == "DAT"
            or format.upper() == "AUX"
            or format.upper() == "RAW"
    ):
        triedBinary = 1
        array = read_binary(pathname, dtype=dtype, dim=dim)
        if endian is not None:
            if endian == 1 or endian == "little":
                array.byteswap(True)  # from little endian to big endian
            else:
                array.byteswap(False)  # from big endian to little endian

        format = "dat"

    dim = array.shape
    if all(d == 0 for d in dim) and other_fromat == 0:
        # TODO array = read_graphic(pathname, r, g, b, ORDER=order, FORMAT=format, TRANSPARENT=transparent)
        dim = array.shape
    # endif
    if all(d == 0 for d in dim) and coords is None:
        if (
                not silent and names is None
        ):  # TODO controllare che names non assuma altri valori non validi
            print(f"read_array: Cannot read datafile {pathname}")
        format = ""
        if get_file_date:
            return data, palette, coords, file_date
        else:
            return data, palette, coords

    try:
        nColors = len(r)
    except:
        nColors = 0
    if nColors > 0 and len(dim) == 2 and no_palette:
        if len(transparent) == nColors:
            tmp = np.zeros((4,) + dim, dtype=np.uint8)
        else:
            tmp = np.zeros((3,) + dim, dtype=np.uint8)
        tmp[0,] = r[array]
        tmp[1,] = g[array]
        tmp[2,] = b[array]
        if len(transparent) == nColors:
            tmp[3] = transparent[array]
        array = tmp.copy()
        nColors = 0
    # endif
    if nColors > 0:
        if nColors < 256:
            spare = np.arange(256 - nColors, dtype=np.uint8)
            r = np.concatenate((r, spare))
            g = np.concatenate((g, spare))
            b = np.concatenate((b, spare))
            if len(transparent) > 1:
                transparent = np.concatenate((transparent, spare))
        # endif
        if len(transparent) > 1:
            palette = np.vstack(
                (r, g, b, transparent)
            ).T  # dovrebbe essere corretta questa, da controllare
            # palette = np.array([[r], [g], [b], [transparent]]).T
        else:
            palette = np.vstack((r, g, b)).T
            # palette = np.array([[r], [g], [b]]).T
    # endif

    if get_file_date:
        file_date = dpg.attr.compareDate(
            0, path=path, name=name, pathname=pathname
        )  # TODO gestire name che può essere una lista
    # endif

    if not isinstance(dim, list):
        dim = [dim]
    if all(d == 0 for d in dim) and coords is not None:
        if get_file_date:
            return data, palette, coords, file_date, names
        else:
            return data, palette, coords, names

    data = array.copy()
    if get_file_date:
        return data, palette, coords, file_date, names
    else:
        return data, palette, coords, names


def loadAsciiData(path: str, name: str):
    """
    Loads ASCII data from a file located at the specified path and file name.

    This function constructs the full path to the file using the provided path and file name. It then checks if the
    file exists. If the file exists, it reads the file and returns its contents as a list of lines. If the file does
    not exist, it returns None.

    Args:
        path (str): The directory path where the file is located.
        name (str): The name of the file to be read.

    Returns:
        list[str] or None: A list of strings, each representing a line in the file, if the file exists. Returns None
        if the file does not exist.

    Raises:
        None: This function does not explicitly raise any exceptions but may raise exceptions related to file
        handling, such as IOError if an issue occurs while opening or reading the file.
    """

    # pathname = os.path.join(path, name)
    pathname = dpg.path.getFullPathName(path, name)

    if not os.path.isfile(pathname):
        return None

    with open(pathname, "r") as f:
        data = f.readlines()

    return data


def load_ascii(path: str, name: str, pathname: str = None):
    """
    Loads ASCII data from a file, given a directory path and file name, or a full pathname.

    This function reads the contents of a specified file and returns the data as a list of lines. If the 'pathname'
    argument is None or not provided, the full path is constructed using 'dpg.path.getFullPathName' and the given
    'path' and 'name'. The function checks if the file exists at the specified location; if it does not,
    None is returned. If the file exists, it reads and returns the file's content.

    Args:
        path (str): The directory path where the file is located. Used if 'pathname' is None.
        name (str): The name of the file to be read. Used if 'pathname' is None.
        pathname (str, optional): The full path to the file. If provided, 'path' and 'name' are not used.

    Returns:
        list[str] or None: A list of strings, each representing a line in the file, if the file exists. Returns None
        if the file does not exist.

    Raises:
        IOError: If an error occurs during file opening or reading.
    """

    if pathname is None:
        # pathname = os.path.join(path, name)
        pathname = dpg.path.getFullPathName(path, name)

    if not os.path.isfile(pathname):
        return None

    with open(pathname, "r") as f:
        data = f.readlines()

    return data


def load_rgba(pathname: str):
    """
    Loads RGBA data from a specified file and returns the number of channels and the RGBA data.

    This function reads a file specified by 'pathname' to load RGBA (Red, Green, Blue, Alpha) data.
    The file is expected to have whitespace-separated integer values representing channel data on each line.
    The function first checks if the file exists; if it does not, it returns (0, 0). If the file exists,
    it reads the data, interprets it as RGBA values, and returns the number of channels and the RGBA data
    in a transposed format.

    Args:
        pathname (str): The full path to the file containing RGBA data.

    Returns:
        tuple: A tuple containing:
            - int: The number of channels in the RGBA data (0 if the file does not exist or an error occurs).
            - list[list[int]]: A list of lists, each inner list containing integers representing one channel of RGBA
            data. Empty if the file does not exist or an error occurs.

    Raises:
        None: While the function handles exceptions internally, it does not raise any. Errors during file reading
        result in a return value of (0, 0).
    """

    if not os.path.isfile(pathname):
        return 0, 0

    try:
        with open(pathname, "r") as file:
            datachannels = file.readline().strip()
            channels = datachannels.split()

            n_channels = len(channels)
            rgba = [[int(channel) for channel in channels]]

            for line in file:
                datachannels = line.strip().split()
                rgba_channel = [int(channel) for channel in datachannels]
                rgba.append(rgba_channel)

            # Transpose the 2D list
            rgba = list(map(list, zip(*rgba)))
            return n_channels, rgba

    except Exception as e:
        print(f"Error reading the file: {e}")
        return 0, 0


def save_values(
        path: str,
        name: str,
        values: str,
        append: bool = False,
        pathname: str = None,
        str_format: str = None,
        filepointer=None,
) -> int:
    """Save a list of values to a file.

    If `pathname` is not provided, it creates a full path using `path` and `name`.
    If the directory does not exist, it is created. The function can append to an existing file
    if `append` is set to True. If `str_format` is provided, it formats each value according to
    this format before writing to the file. If a `filepointer` is not provided, the file is opened
    in the function and closed before the function returns.

    Args:
        path (str): Directory path where the file is to be saved.
        name (str): The name of the file to save the values in.
        values (list): A list or list of lists containing values to be written to the file.
        append (bool): If True, the values will be appended to the file if it exists.
                       Defaults to False.
        pathname (str): Full file path. If None, it's constructed from `path` and `name`.
                        Defaults to None.
        str_format (str): A format string to format the values. Defaults to None.
        filepointer (file object): A file object to write the values to. If None, a file object
                                   will be created within the function. Defaults to None.

    Returns:
        int: 1 if the operation was successful, 0 otherwise.
    """
    # result = save_values('/path/to/dir', 'file.txt', [1, 2, 3], append=True)

    if len(values) == 0:
        return False

    if not pathname:
        pathname = dpg.path.getFullPathName(path, name)

    # Ensure the directory exists, create it if not
    os.makedirs(os.path.dirname(pathname), exist_ok=True)

    # Open the file if filepointer is not provided
    if filepointer is None:
        filepointer = dpg.path.openWriteFile(pathname, append=append)

    if filepointer is None:
        return False

    try:
        # If values is a 1D list or array
        if isinstance(values, list) or isinstance(values, np.ndarray):
            for value in values:
                formatted_value = (
                    format(value, str_format) if str_format else str(value)
                )
                filepointer.write(formatted_value + "\n")
        # If values is a 2D list or array
        else:
            for row in values:
                formatted_row = [
                    format(value, str_format) if str_format else str(value)
                    for value in row
                ]
                filepointer.write(" ".join(formatted_row) + "\n")

        # Only close the file if it was opened in this function
        if "filepointer" in locals():
            filepointer.close()

        return True
    except Exception as e:
        print(f"Error writing to file {pathname}: {e}")
        return False


def SaveRGBA(pathname: str, rgba) -> bool:
    """Save RGBA data to a file.

    Each row of the RGBA data is written to the file in a space-separated string format.
    The file will be overwritten if it exists, or created if it does not.

    Args:
        pathname (str): The full path of the file where the RGBA data will be saved.
        rgba (list of list of int): The RGBA data to be saved. Expected to be a 2D list where each sublist represents
        a channel.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    # Sample usage TODO ok?
    # rgba_data = [
    #     [255, 0, 0, 255], # Red channel
    #     [0, 255, 0, 255], # Green channel
    #     [0, 0, 255, 255], # Blue channel
    #     [255, 255, 255, 0] # Alpha channel
    # ]
    # result = SaveRGBA('/path/to/file.txt', rgba_data)
    try:
        with open(pathname, "w") as filepointer:
            for row in zip(*rgba):  # Transpose the 2D list to iterate by rows
                # Convert all numbers to strings, remove all spaces (strcompress equivalent)
                # and join them with four spaces in between
                values = "    ".join(str(int(val)).replace(" ", "") for val in row)
                filepointer.write(values + "\n")  # Add a newline at the end of each row
        return True
    except IOError as e:
        print(f"Error writing to file {pathname}: {e}")
        return False


def save_lut(path: str, name: str, rgba: str, pathname: str = None) -> bool:
    """Save an RGBA lookup table to a file.

    This function constructs a full pathname from the given path and name, if not provided,
    and then calls `save_rgba` to write the RGBA data to a file.

    Args:
        path (str): The directory path where the file is to be saved.
        name (str): The name of the file to save the RGBA data in.
        rgba (list of list of int): The RGBA data to be saved.
        pathname (str, optional): The full path to the file. If not provided, it's constructed from `path` and `name`.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    # Sample usage TODO ok?
    # rgba_data = [
    #     [255, 0, 0, 255], # Red channel
    #     [0, 255, 0, 255], # Green channel
    #     [0, 0, 255, 255], # Blue channel
    #     [255, 255, 255, 0] # Alpha channel
    # ]
    # result = save_lut('/path/to/dir', 'lut.txt', rgba_data)
    if pathname is None:
        pathname = dpg.path.getFullPathName(path, name)

    return SaveRGBA(pathname, rgba)


def read_ascii(pathname: str) -> list:
    """
    Read the contents of an ASCII file into a list.

    Args:
        pathname (str): The path to the ASCII file.

    Returns:
        list: A list where each element is a line from the file.
    """
    try:
        with open(pathname, "r") as file:
            array = [line.rstrip() for line in file]
        return array
    except IOError as e:
        print(f"Error reading file {pathname}: {e}")
        return []


def read_ulongs(path: str, name: str, pathname: str = None) -> list:
    """
    Read unsigned long integers from a binary file.

    Args:
        path (str): The directory path where the file is located.
        name (str): The file name to read values from.
        pathname (str, optional): The full path to the file. If not provided, it's constructed from `path` and `name`.

    Returns:
        list: A list of unsigned long integers read from the file.
    """
    if pathname is None:
        pathname = dpg.path.getFullPathName(path, name)

    if not os.path.isfile(pathname):
        return []

    values = []
    try:
        with open(pathname, "rb") as filepointer:
            while True:
                # Read 4 bytes (unsigned long) from the file
                data = filepointer.read(4)
                if not data:
                    break
                # Unpack the binary data into an unsigned long (32-bit)
                val = struct.unpack("I", data)[0]
                values.append(val)
        return values
    except IOError as e:
        print(f"Error reading file {pathname}: {e}")
        return []
    except struct.error as e:
        print(f"Error unpacking data: {e}")
        return []


def read_wind(path: str, names: list):
    """
    Read wind speed and direction data from binary files.

    Args:
        path (str): The directory path where the files are located.
        names (list): A list containing the names of the direction and speed files.

    Returns:
        list of list: A 2D list of wind speed and direction values, or an empty list if an error occurs.
    """
    # Sample usage
    # wind_data = read_wind('/path/to/dir', ['wind_dir.bin', 'wind_speed.bin'])
    dirname = dpg.path.getFullPathName(path, names[0])
    speedname = dpg.path.getFullPathName(path, names[1])

    if not os.path.isfile(dirname) or not os.path.isfile(speedname):
        return []

    try:
        with open(dirname, "rb") as dirfile, open(speedname, "rb") as speedfile:
            values = []
            while True:
                # Assuming the data is stored in a binary format that can be read directly as floats
                # TODO vero? è in formato binario?
                dir_data = dirfile.read(4)
                speed_data = speedfile.read(4)
                if not dir_data or not speed_data:
                    break
                # Unpack the binary data into floats
                dir_val = struct.unpack("f", dir_data)[0]
                speed_val = struct.unpack("f", speed_data)[0]
                values.append([speed_val, dir_val])
        return values
    except IOError as e:
        print(f"Error reading files: {e}")
        return []
    except struct.error as e:
        print(f"Error unpacking data: {e}")
        return []


def read_coords(path: str, names: list, dim: tuple) -> np.ndarray:
    # TODO da ricontrollare
    """
    Read coordinates from binary files.

    Args:
        path (str): The directory path where the files are located.
        names (list): A list containing the names of the latitude, longitude, and altitude files.
        dim (tuple): The dimensions of the array to read into.

    Returns:
        numpy.ndarray: A 2D array of coordinates.
    """
    # Sample usage
    # coords = read_coords('/path/to/dir', ['lat.dat', 'lon.dat', 'alt.dat'], (100, 100))
    latname = dpg.path.getFullPathName(path, names[0])
    lonname = dpg.path.getFullPathName(path, names[1])
    altname = dpg.path.getFullPathName(path, names[2])

    # Check if files exist
    if not os.path.isfile(latname) or not os.path.isfile(lonname):
        return None

    # Read binary data from files
    # TODO nel caso il file non sia binario, bisogna modificare la lettura
    #     ad esempio np.dtype('f4') credo non vada più bene
    try:
        with open(latname, "rb") as latfile, open(lonname, "rb") as lonfile:
            # Optionally read altitude data if provided
            altfile = (
                open(altname, "rb") if names[2] and os.path.isfile(altname) else None
            )

            # Determine total size to read based on 'dim'
            # TODO verificare che dim sia una tupla, altrimenti credo fallisca
            total_size = np.prod(dim)
            dtype = np.dtype("f4")  # Assuming float32 for binary data
            values = np.empty((3, total_size), dtype=dtype)

            # TODO mi aspetto che, essendo i valori letti in ordine contrario, andranno invertiti [0] e [2]
            # Read latitude and longitude values
            values[0] = np.fromfile(latfile, dtype=dtype, count=total_size)
            values[1] = np.fromfile(lonfile, dtype=dtype, count=total_size)

            # Read altitude values if file is provided
            if altfile:
                values[2] = np.fromfile(altfile, dtype=dtype, count=total_size)
                altfile.close()

            # Reshape the array to the specified dimensions
            values = values.reshape((3,) + dim)
            return values
    except IOError as e:
        print(f"Error reading coordinate files: {e}")
        return np.array([])
    except struct.error as e:
        print(f"Error unpacking data: {e}")
        return np.array([])


proj_dict = {
        "Stereographic": "stere",
        "Orthographic": "ortho",
        "Lambert Conic": "lcc",
        "Lambert Azimuthal": "laea",
        "Gnomonic": "gnom",
        "Azimuthal Equidistant": "aeqd",
        "Satellite": "geos",
        "Cylindrical": "eqc",
        "Mercator": "merc",
        "Mollweide": "moll",
        "Sinusoidal": "sinu",
        "Aitoff": "aitoff",
        "Hammer Aitoff": "hammer",
        "Albers Equal Area Conic": "aea",
        "Transverse Mercator": "tmerc",
        "Miller Cylindrical": "mill",
        "Robinson": "robin",
        "Lambert Ellipsoid Conic": "leac",
        "Goodes Homolosine": "goode",
        "Geographic": "longlat",
        "UTM": "utm",
        "State Plane": "lcc",
        "Polar Stereographic": "stere",
        "Polyconic": "poly",
        "Equidistant Conic A": "eqdc",
        "Equidistant Conic B": "eqdc",
        "Near Side Perspective": "nsper",
        "Sinusoidal": "sinu",
        "Equirectangular": "eqc",
        "Miller Cylindrical": "mill",
        "Van der Grinten": "vandg",
        "Hotine Oblique Mercator A": "omerc",
        "Hotine Oblique Mercator B": "omerc",
        "Robinson": "robin",
        "Space Oblique Mercator A": "somerc",
        "Space Oblique Mercator B": "somerc",
        "Alaska Conformal": "aea",
        "Interrupted Goode": "igh",
        "Mollweide": "moll",
        "Interrupted Mollweide": "imoll",
        "Hammer": "hammer",
        "Wagner IV": "wag4",
        "Wagner VII": "wag7",
        "Oblated Equal Area": "ocea",
        # "latlon": "latlong",
        "PLATCAR": "latlong",
        "latlon": "latlon",
        "GEOS": "geos",
    }


def get_python_proj_from_idl(input_proj: str):
    """
    Maps a projection type from IDL (Interactive Data Language) to its corresponding type in Python.

    This function uses a dictionary to map projection types from IDL to equivalent projection codes used in Python
    libraries (such as Basemap or Cartopy). The function takes an IDL projection type as input and returns the
    corresponding Python projection code. If the input projection type is not in the predefined dictionary,
    a KeyError will be raised.

    Args:
        input_proj (str): The IDL projection type to be converted to its Python equivalent.

    Returns:
        str: The corresponding Python projection code.

    Raises:
        KeyError: If the input projection type is not found in the predefined dictionary of projection mappings.
    """

    if input_proj in proj_dict.keys():
        return proj_dict[input_proj]
    if input_proj.capitalize() in proj_dict.keys():
        return proj_dict[input_proj.capitalize()]
    else:
        log_message("Projection NOT FOUND. Error", level="ERROR", all_logs=True)
        return None

def get_python_proj_from_idl(input_proj: str):
    """
    Maps a projection type from IDL (Interactive Data Language) to its corresponding type in Python.

    This function uses a dictionary to map projection types from IDL to equivalent projection codes used in Python libraries (such as Basemap or Cartopy). The function takes an IDL projection type as input and returns the corresponding Python projection code. If the input projection type is not in the predefined dictionary, a KeyError will be raised.

    Args:
        input_proj (str): The IDL projection type to be converted to its Python equivalent.

    Returns:
        str: The corresponding Python projection code.

    Raises:
        KeyError: If the input projection type is not found in the predefined dictionary of projection mappings.
    """

    if input_proj in proj_dict.keys():
        return proj_dict[input_proj]
    if input_proj.capitalize() in proj_dict.keys():
        return proj_dict[input_proj.capitalize()]
    else:
        log_message("Projection NOT FOUND. Error", level="ERROR", all_logs=True)
        return None

def get_idl_proj_from_python(input_proj):
    """
    Finds the key in the projection dictionary that corresponds to a given value.

    Args:
        input_proj: The value in the dictionary to search for.

    Returns:
        key or None: the key associated with the given value, or None if not found.
    """
    # questo input_proj corrisponde ad uno degli elementi del dizionario e non delle chiavi
    for key, input in proj_dict.items():
        if input == input_proj:
            return key
    return None


def read_shape(path, name):
    """
    Reads a shapefile, fixes any unclosed polygon geometries, and returns its geometry and attributes.

    Args:
        path (str): The directory path of the shapefile.
        name (str): The name of the shapefile.

    Returns:
        geometry (GeoSeries): The fixed geometries of the shapefile.
        gdf (GeoDataFrame): A GeoDataFrame containing the shapefile's attributes and geometries.
        orig_columns (list): A list of the original attribute column names in the shapefile.
    """
    # coords = 0
    pathname = dpg.path.getFullPathName(path, name)

    gdf = gpd.read_file(pathname, ignore_geometry=True)
    orig_columns = [elem for elem in gdf.columns]

    # shape = fiona.open(pathname)
    fixed_geometries = []
    with fiona.open(pathname) as source:
        for feature in source:
            geom = shape(feature["geometry"])
            if isinstance(geom, Polygon) and not geom.exterior.is_closed:
                closed_coords = list(geom.exterior.coords)
                closed_coords.append(closed_coords[0])
                geom = Polygon(closed_coords)
            fixed_geometries.append(geom)

    gdf["geometry"] = fixed_geometries

    gdf = gpd.GeoDataFrame(gdf, geometry=gdf['geometry'])

    return gdf["geometry"], gdf, orig_columns
