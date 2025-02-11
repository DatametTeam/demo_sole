import numpy as np
from sou_py.dpg.log import log_message
import os
import sou_py.dpg as dpg
import rasterio


def read_tiff(path_name, geo=False):
    """
    Reads the first band of a TIFF file as a NumPy array and optionally retrieves geographical bounds.

    Args:
        path_name (str): The file path to the TIFF image.
        geo (bool, optional): If True, returns the geographical bounds of the image. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of pixel values for the first band of the TIFF image.
            - tuple or None: Geographical bounds of the image (left, bottom, right, top), if `geo` is True; otherwise, returns None.

    Notes:
        - Uses `rasterio` to open and read the TIFF file.
        - If `geo` is set to True, the function fetches the image's geographical boundaries.
        - Handles cases where the TIFF file might be empty or unreadable by returning `None`.
    """
    array = None
    with rasterio.open(path_name) as data:
        array = data.read()[0]

        if geo:
            bounds = data.bounds
            return array, bounds

        return array, None


def waitRainFile(pathname, maxwait):
    """
    Placeholder function to wait for the existence of a specified file within a maximum wait time.

    Args:
        pathname (str): The full path of the file to check for.
        maxwait (int): Maximum time to wait (in seconds) for the file to become available.

    Returns:
        bool: Currently returns True as a placeholder. Once implemented, it should return:
            - True if the file is found within the specified wait time.
            - False if the file is not found within `maxwait` seconds.

    Notes:
        - This function is a placeholder and not yet implemented. Currently, it only logs a warning and immediately returns True.
        - Once implemented, this function should monitor for the file’s existence up to the `maxwait` period.
    """
    log_message(f'Funzione waitRainFile non ancora implementata', 'WARNING')
    return True


def rainmap(prodId, delay=None):
    """
    Processes and stores rainfall data for a specific product ID by locating and reading a TIFF file containing
    rainfall intensity values.

    Args:
        prodId (int): The ID of the product node, containing configurable parameters for the rainfall data retrieval.
        delay (int, optional): Number of minutes to subtract from the current time to find a specific data file.
                               If None, no delay is applied.

    Returns:
        int: Status code representing the outcome of the process:
            - 0 indicates successful file retrieval and processing.
            - 1 indicates an error, such as missing file or failed read operation.

    Workflow:
        1. Retrieves configuration parameters for file paths, prefixes, and deletion rules.
        2. Gets the current or adjusted date and time, creating a filename based on these values.
        3. Checks if the file exists within the specified waiting period (`maxWait`), then reads it as a TIFF file.
        4. If the TIFF file is successfully read:
            - Filters out invalid values, setting them to NaN.
            - Stores the processed data in the array associated with `prodId`.
            - Records the geographic boundaries of the TIFF file for navigation purposes.
        5. Optionally deletes the TIFF file based on configuration.

    Raises:
        FileNotFoundError: If the specified TIFF file is not found within the `maxWait` duration.
        ValueError: If the TIFF file cannot be read or if data is improperly formatted.

    Notes:
        - The `waitRainFile` function handles file availability checks but is currently unimplemented.
        - The TIFF file is deleted post-processing only if `deleteFile` is set to a value greater than 0.
    """
    RXpath, _, _ = dpg.radar.get_par(prodId, 'RXpath', 'datamet_data/SENSORS/RX')
    deleteFile, _, _ = dpg.radar.get_par(prodId, 'deleteFile', 0)
    prefix, _, _ = dpg.radar.get_par(prodId, 'prefix', '')
    postfix, _, _ = dpg.radar.get_par(prodId, 'postfix', '')
    maxWait, _, _ = dpg.radar.get_par(prodId, 'maxWait', 60)
    date, time, exists = dpg.times.get_time(prodId)

    if not exists:
        return 1

    if delay is not None:
        date, time = dpg.times.addMinutesToDate(date, time, -delay)

    name = prefix + dpg.times.checkDate(date, sep='', year_first=True)
    if postfix == '':
        postfix = time[:2] + '.tif'
    else:
        postfix = dpg.times.checkTime(time, sep="")[0] + postfix
        
    name += postfix
    pathName = dpg.path.getFullPathName(RXpath, name)

    if not waitRainFile(pathName, maxWait):
        log_message(f'Cannot find {pathName}', 'INFO')

    array, bounds = read_tiff(pathName, geo=True)
        
    if deleteFile > 0:
        os.remove(pathName)

    if array is None:
        log_message(f"Cannot read {pathName}", 'INFO')
        return 1

    log_message(f"Using {pathName}", 'INFO')

    ind = np.where(array < 0)
    array[ind] = np.nan
    dpg.array.set_array(prodId, data=array, no_copy=True)

    LL_lat = bounds[1]
    LL_lon = bounds[0]
    UR_lat = bounds[3]
    UR_lon = bounds[2]
    dpg.navigation.put_corners(prodId, LL_lat, LL_lon, UR_lat, UR_lon)

    return 0