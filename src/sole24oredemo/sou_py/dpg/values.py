import numpy as np

import sou_py.dpg as dpg

"""
Funzioni ancora da portare
FUNCTION IDL_rv_get_parname 
FUNCTION GetBitplanes                   // UNUSED
FUNCTION IDL_rv_are_equivalent_par      // UNUSED
FUNCTION IDL_rv_check_array_values      // UNUSED
FUNCTION IDL_rv_check_for_reload        // UNUSED
FUNCTION IDL_rv_get_scaling_options     // UNUSED
FUNCTION IDL_rv_get_valids              // UNUSED
"""


def minmax(array, exclude_invalid=False):
    """
    Computes the minimum and maximum values of an array along with their corresponding indices.
    Optionally excludes invalid values (e.g., NaN or infinite) from the computation.

    Parameters:
        array (numpy.ndarray): The input array to evaluate.
        exclude_invalid (bool, optional): If True, excludes NaN and infinite values from the computation (default is False).

    Output:
        min_value (float): The minimum value in the array.
        min_index (int): The index of the minimum value in the array.
        max_value (float): The maximum value in the array.
        max_index (int): The index of the maximum value in the array.
    """
    if exclude_invalid:
        # Exclude NaN values from the array
        valid_indices = ~np.isnan(array) & ~np.isinf(array)
        valid_array = array[valid_indices]
        original_indices = np.where(valid_indices)[0]  # Get the indices of non-NaN values
    else:
        valid_array = array
        original_indices = np.arange(len(array))

    min_value = np.min(valid_array)
    min_index = original_indices[np.argmin(valid_array)]

    max_value = np.max(valid_array)
    max_index = original_indices[np.argmax(valid_array)]

    return min_value, min_index, max_value, max_index


# TODO: to be removed in future
def minmax_old(values: np.ndarray):
    """
    Calculates the minimum and maximum values in an array, along with their indices.

    This function computes the minimum and maximum values in the given array, excluding infinite values.
    It also determines the indices of these minimum and maximum values. The function flattens the array
    and applies a mask to exclude infinite values before performing the calculations.

    Args:
        values (np.ndarray): An array from which the minimum and maximum values and their indices are to be found.

    Returns:
        tuple: A tuple containing four elements:
            - minVal (float): The minimum value in the array.
            - minInd (int): The index of the minimum value.
            - maxVal (float): The maximum value in the array.
            - maxInd (int): The index of the maximum value.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    array = values.flatten(order="F")
    mask = np.multiply(array != np.inf, array != -1 * np.inf)
    minVal = np.nanmin(array[mask])
    maxVal = np.nanmax(array[mask])
    (minInd,) = np.where(array == minVal)
    (maxInd,) = np.where(array == maxVal)
    if minInd:
        minInd = minInd[0]
    if maxInd:
        maxInd = maxInd[0]
    return minVal, minInd, maxVal, maxInd


def count_invalid_values(array: np.ndarray, f_null: float = None, sign: int = None):
    """
    Counts and identifies the indices of null (NaN) and infinite values in an array.

    This function analyzes an array to count and locate null (NaN) and infinite values. It can additionally
    identify values that are above or below a specified threshold ('f_null') and treat them as null.
    The 'sign' parameter determines whether to look for positive or negative infinite values.

    Args:
        array (np.ndarray): The array to be analyzed.
        f_null (float, optional): A threshold value to consider as null. Defaults to None.
        sign (int, optional): If -1, counts negative infinite values; if positive, counts positive infinite values.
        Defaults to -1.

    Returns:
        tuple: A tuple containing four elements:
            - ind_null (np.ndarray): Indices of null values (including those beyond 'f_null', if specified).
            - count_null (int): The number of null values.
            - ind_void (np.ndarray): Indices of infinite values.
            - count_void (int): The number of infinite values.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    count_null = 0
    count_void = 0
    ind_null = []
    ind_void = []

    if len(array) == 0:
        return ind_null, count_null, ind_void, count_void

    ind_null = np.where(np.isnan(array))
    count_null = len(ind_null[0])

    if sign is None:
        ind_void = np.where(np.isneginf(array) | np.isposinf(array))

    elif sign == -1 or sign == "neg":
        ind_void = np.where(np.isneginf(array))
    else:
        ind_void = np.where(np.isposinf(array))
    count_void = len(ind_void[0])

    if f_null is None:
        return ind_null, count_null, ind_void, count_void

    if f_null > 0.0:
        ind1 = np.where(array >= f_null)
    else:
        ind1 = np.where(array <= f_null)

    if count_null > 0 and len(ind1[0]) > 0:
        idx = 0
        ind_null_for = ind_null
        ind_null = ()
        for elm in ind_null_for:
            elm = np.concatenate([ind1[idx], elm])
            ind_null = ind_null + (elm,)
            idx += 1
        count_null += len(ind1[0])
    elif len(ind1) > 0:
        ind_null = ind1
        count_null = len(ind1[0])

    return ind_null, count_null, ind_void, count_void


def get_valued_array(
    array: np.ndarray,
    values: np.ndarray = None,
    preserve_void: bool = False,
    simmetric: bool = False,
    set_void_null: bool = False,
):
    """
    Processes an array by applying a set of values or converting it to float, and handles null and infinite values.

    This function takes an array and optionally applies a mapping from an array of 'values'. It can convert
    the array to a float type and handle null and infinite values in specific ways, based on the given parameters.
    The function can also preserve infinite values or replace them with a minimum value or NaN.

    Args:
        array (np.ndarray): The input array to be processed.
        values (np.ndarray, optional): An array of values to map onto 'array'. Defaults to None.
        preserve_void (bool, optional): If True, preserves infinite values in the array. Defaults to False.
        simmetric (bool, optional): If True, uses symmetric handling for infinite values. Defaults to False.
        set_void_null (bool, optional): If True, replaces infinite values with NaN. Defaults to False.

    Returns:
        tuple: A tuple containing five elements:
            - valued_array (np.ndarray): The processed array.
            - ind_null (np.ndarray): Indices of null values in the array.
            - count_null (int): The number of null values.
            - ind_void (np.ndarray): Indices of infinite values.
            - count_void (int): The number of infinite values.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if values is not None:
        dtype = array.dtype
        if dtype in [np.int8, np.int16, np.uint16]:
            valued_array = np.array([values[int(x)] for x in array])
        else:
            valued_array = array.astype(float)
    else:
        valued_array = array.astype(float)

    ind_null, count_null, ind_void, count_void = count_invalid_values(valued_array)

    if preserve_void:
        return valued_array

    if count_void > 0:
        min_val = 0.0
        if not simmetric and values is not None:
            min_val = np.nanmin(values)
        if set_void_null:
            min_val = np.nan
        valued_array[ind_void] = min_val

    return valued_array, ind_null, count_null, ind_void, count_void


def checkBitplanes(p_array: np.ndarray, bitplanes: list = None):
    """
    Determines the bitplane information for an array based on its data type and specified bitplanes.

    This function checks the data type of the provided array and calculates the corresponding number of
    bitplanes, minimum and maximum indices for the bitplanes. If the 'bitplanes' argument is provided,
    the function uses it to adjust the calculations; otherwise, it defaults based on the array's data type.

    Args:
        p_array (np.ndarray): The array for which bitplane information is needed.
        bitplanes (list[int], optional): A list of bitplanes to consider. Defaults to None.

    Returns:
        tuple: A tuple containing three elements:
            - The number of bitplanes (int).
            - The minimum index for the bitplanes (int).
            - The maximum index for the bitplanes (int).

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if p_array is None:
        return 0

    data_type = p_array.dtype

    # Default values
    if bitplanes is None:
        if data_type == np.int8 or data_type == np.uint8:
            bitplanes = 8
            min_ind = 0
            max_ind = 2**bitplanes - 1
            return bitplanes, min_ind, max_ind
        else:
            bitplanes = 16
            min_ind = 0
            max_ind = 2**bitplanes - 1
            return bitplanes, min_ind, max_ind

    tot = 0
    max_b = max(bitplanes)

    # Adjust for int8 type
    if data_type == np.int8 or data_type == np.uint8 and max_b > 8:
        max_b = 8

    ind = bitplanes.index(max_b)
    if ind > 0:
        tot = sum(bitplanes[:ind])

    min_ind = 2**tot - 1
    tot += max_b
    max_ind = 2**tot - 1

    if data_type == np.int8 or data_type == np.uint8:
        return 8, min_ind, max_ind
    else:
        return 10, min_ind, max_ind


def threshold_index(
    threshold: float, values: list, thresh_type: int, offset: int, as_is: bool = False
) -> int:
    """
    Determines the index in a list of values where the threshold is first crossed.

    This function finds the index in 'values' where the specified 'threshold' is first crossed, based on the
    'thresh_type'. The search starts from an 'offset' index. If 'as_is' is True, the function returns the
    index where the threshold is exactly met, if such an index exists.

    Args:
        threshold (float): The threshold value to be crossed.
        values (list[float]): A list of numeric values to search.
        thresh_type (int): The type of thresholding to perform. If 1, returns the threshold as an integer.
        offset (int): The index in 'values' from which to start searching.
        as_is (bool, optional): If True, returns the index where the threshold is exactly met. Defaults to False.

    Returns:
        int: The index in 'values' where the threshold is first crossed or met, or the last index if the end of
        'values' is reached first.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    thresh = float(threshold)

    # If threshType is 1, just return the threshold casted to an integer
    if thresh_type == 1:
        return int(thresh)

    n_val = len(values) - 1
    if n_val < 0:
        return int(thresh)

    if n_val == 0:
        return 0

    i = offset
    if i >= n_val:
        return n_val

    if i < 0:
        i = 0

    while not np.isfinite(values[i]):
        i += 1
        if i == n_val:
            return i

    while values[i] <= thresh:
        i += 1
        if i == n_val:
            if as_is:
                if thresh > values[i - 1]:
                    return i
                return i - 1
            if values[i] <= thresh:
                return i
            return i - 1

    if i == 0:
        return 0

    if as_is:
        if thresh > values[i - 1]:
            return i
        return i - 1

    if np.isfinite(values[i - 1]):
        return i - 1
    return i


def get_equivalent_par(par: str) -> str:
    """
    Maps certain parameter identifiers to their equivalent forms.

    This function takes a parameter identifier (e.g., 'CZ', 'UZ') and returns its equivalent form, based on
    predefined mappings. For example, 'CZ' and 'UZ' are both mapped to 'Z'.

    Args:
        par (str): The parameter identifier to be mapped.

    Returns:
        str: The equivalent parameter identifier, if a mapping exists; otherwise, returns the original identifier.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if par == "CZ" or par == "UZ":
        return "Z"
    else:
        return par


def set_parname(node, parname: str, unit: str = None):
    """
    Sets the parameter name and optionally its unit in the calibration attribute of a node.

    This function updates the parameter name and, if provided, the unit in the calibration attribute of the
    specified node. It retrieves the calibration attribute for the node and uses it to replace the existing
    parameter name and unit tags with the new values.

    Args:
        node (Node object): The node whose calibration attribute is to be updated.
        parname (str): The new parameter name to set.
        unit (str, optional): The new unit for the parameter. Defaults to None.

    Returns:
        None: The function does not return anything, but it updates the node's calibration attribute.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """

    idcalibration = dpg.calibration.get_idcalibration(node)
    tag = "parname"
    val = parname
    if unit is not None:
        tag = [tag, "unit"]
        val = [val, unit]
    ret = dpg.attr.replaceTags(idcalibration, tag, val)


def save_array_values(node, str_format: str = None) -> bool:
    """
    Saves the values of an attribute of a specified node.

    This function takes a node and saves the values of a specific attribute. It checks if the provided node
    is of the correct type and then retrieves the attribute based on a predefined configuration. If the
    attribute is valid, its values are saved.

    Args:
        node (Node object): The node whose attribute values are to be saved.
        str_format (str, optional): The format to use when saving the values. Defaults to None.

    Returns:
        bool: True if the attribute values are successfully saved, False otherwise.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if not isinstance(node, dpg.node__define.Node):
        return False
    attr_name = dpg.cfg.getValueDescName()
    o_attr = node.getSingleAttr(attr_name)
    if isinstance(o_attr, dpg.attr__define.Attr):
        o_attr.save()
    return True


def copy_values_info(
    fromNode,
    toNode,
    only_if_not_exists: bool = False,
    to_save: bool = False,
    to_not_create: bool = False,
    no_values: bool = False,
):
    """
    Copies values information from one node to another.

    This function copies the values information from 'fromNode' to 'toNode'. It supports conditional copying
    (only if the target node doesn't have the values information), saving the copied information, and the option
    to not create new attributes. It can also copy without the actual values.

    Args:
        fromNode (Node object): The source node from which values information is copied.
        toNode (Node object): The target node to which values information is copied.
        only_if_not_exists (bool, optional): If True, copies information only if it doesn't exist in 'toNode'.
        Defaults to False.
        to_save (bool, optional): If True, saves the copied information in 'toNode'. Defaults to False.
        to_not_create (bool, optional): If True, does not create new attributes in 'toNode' during copying. Defaults
        to False.
        no_values (bool, optional): If True, copies information without the actual values. Defaults to False.

    Returns:
        None: The function does not return anything but performs the copying operation.

    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    if only_if_not_exists:
        idcalibration = dpg.calibration.get_idcalibration(toNode, only_current=True)
        if idcalibration is not None:
            return
    _, _ = dpg.tree.copyAttr(fromNode, toNode, dpg.cfg.getValueDescName())
    if not no_values:
        calib_dict = dpg.calibration.get_array_values(
            fromNode, to_not_create=to_not_create
        )
        _, _ = dpg.tree.copyAttr(fromNode, toNode, name=None)
    if to_save:
        _ = save_array_values(toNode)
