import numpy as np


def dynamicWhere(dataX: np.ndarray, dataY: np.ndarray, x0: float, x1: float, y0: float, y1: float, count: int) \
        -> np.ndarray:
    """
    Filters indices of data points (dataX, dataY) based on dynamic thresholding.

    This function calculates a dynamic threshold line based on the input coordinates
    (x0, y0) and (x1, y1). It returns the indices of points (dataX, dataY) that lie
    below this threshold line.

    Args:
        dataX (np.ndarray): An array of x-coordinates.
        dataY (np.ndarray): An array of y-coordinates. Must be the same size as dataX.
        x0 (float): The starting x-coordinate for the threshold line.
        x1 (float): The ending x-coordinate for the threshold line.
        y0 (float): The starting y-coordinate for the threshold line.
        y1 (float): The ending y-coordinate for the threshold line.
        count (int): A counter that is set to 0 if the sizes of dataX and dataY do not match. N.B. this parameter is not
                    used.

    Returns:
        np.ndarray: Indices of data points that lie below the threshold line, or -1 if the sizes of dataX and dataY do
        not match.
    """

    if np.size(dataX) != np.size(dataY):
        count = 0
        return -1

    if x1 <= x0:
        ind = np.where((dataX < x0) & (dataY < y0))
        return ind

    coeff = float(y1 - y0) / float(x1 - x0)
    thresh = coeff * (dataX - x0) + y0

    if y1 < y0:
        ind = np.where(thresh < y1)
        thresh[ind] = y1
        ind = np.where(thresh > y0)
        thresh[ind] = y0
    else:
        ind = np.where(thresh > y1)
        thresh[ind] = y1
        ind = np.where(thresh < y0)
        thresh[ind] = y0

    return np.where(dataY < thresh)


def removeUnsignificantZeros(string: str | list[str], all_zeros: bool = False) -> str | list[str]:
    """
    Removes trailing insignificant zeros from decimal strings, optionally keeping zeros after the decimal point.

    This function processes a string or list of strings representing decimal numbers and removes any trailing
    zeros that are not significant. It also removes the decimal point if no decimal digits remain. Optionally,
    it can keep a single zero after the decimal point if 'all_zeros' is set to True.

    Args:
        strings (str or list[str]): A string or list of strings representing decimal numbers.
        all_zeros (bool, optional): If True, retains a single zero after the decimal point even if it's insignificant. Defaults to False.

    Returns:
        str or list[str]: The processed string or list of strings with insignificant zeros removed.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    if isinstance(strings, str):
        strings = [strings]
    processed_strings = []

    for s in strings:
        point_pos = s.find(".")

        if not all_zeros:
            point_pos += 1

        # Check if there is no exponent part
        if point_pos > 0 and "e" not in s:
            while s.endswith("0") and len(s) - 1 > point_pos:
                s = s[:-1]
            # Remove trailing dot if no decimal values left
            if s.endswith("."):
                s = s[:-1]

        processed_strings.append(s)

    # Return single string or list of strings based on input
    return processed_strings if len(processed_strings) > 1 else processed_strings[0]


def getComplement(dims: int, array: np.ndarray) -> np.ndarray:
    """
    Calculates the complement of an array within a given dimension.

    This function finds the complement of a specified array within a range defined by 'dims'. The complement
    consists of all indices within 'dims' that are not present in 'array'.

    Args:
        dims (int): The size of the dimension in which to find the complement.
        array (np.ndarray): An array of indices within the given dimension.

    Returns:
        np.ndarray: An array containing the indices that are not in 'array' but within the range specified by 'dims'.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    """
    # Sample usage TODO da controllare
    dims = 10
    array = np.array([2, 4, 6])
    result = getComplement(dims, array)
    print(result)
    """
    pos = np.full(dims, -1, dtype=int)
    pos[array] = 0
    complement = np.where(pos == -1)[0]
    return complement
