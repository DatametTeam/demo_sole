import numpy as np
import sou_py.dpg as dpg

"""
Funzioni ancora da portare
PRO COMPUTE_ORIZ_BEAMS 
PRO UPDATE_AVERAGE_MAP      // UNUSED
PRO UPDATE_VALID_COUNTER    // UNUSED
"""


def getCheckNullSum(add1: np.ndarray, add2: np.ndarray) -> np.ndarray:
    """
    Computes the summation of two arrays, handling non-finite and NaN values.

    This function sums two arrays and handles cases where the summation results in
    non-finite (e.g., inf, -inf) or NaN values. In such cases, it replaces the
    non-finite or NaN values in the summation with corresponding values from one of
    the input arrays.

    Args:
        add1 (np.ndarray): The first array to be summed.
        add2 (np.ndarray): The second array to be summed.

    Returns:
        np.ndarray: The resultant array after summation, with non-finite and NaN
                    values replaced by corresponding values from `add1` or `add2`.

    Note:
        The function first calculates the summation of `add1` and `add2`. It then
        checks for non-finite values in the summation, replacing them with values
        from `add2` and then `add1`. After that, it checks for NaN values and
        replaces them with values from `add2`. This approach ensures that the
        resultant array avoids non-finite and NaN values, favoring valid numeric
        data from either of the input arrays.
    """
    summation = add1 + add2

    # Check for non-finite values (infinity or NaN)
    non_finite_indices = np.where(~np.isfinite(summation))[0]

    if len(non_finite_indices) > 0:
        summation[non_finite_indices] = add2[non_finite_indices]

    non_finite_indices = np.where(~np.isfinite(summation))[0]

    if len(non_finite_indices) > 0:
        summation[non_finite_indices] = add1[non_finite_indices]

    # Check specifically for NaN
    nan_indices = np.where(np.isnan(summation))[0]

    if len(nan_indices) > 0:
        summation[nan_indices] = add2[nan_indices]

    return summation


def checkColIndex(
        col: np.ndarray, lin: int, p_array: np.ndarray, sampling: float = None
):
    """
    Adjusts column indices in an array based on a specified sampling rate.

    This function modifies the column indices in the 'col' array according to a
    specified or calculated sampling rate. The function iteratively adjusts the
    indices and updates the corresponding values in the 'col' array based on
    maximum values from the 'p_array'.

    Args:
        col (np.ndarray): An array of column indices to be adjusted.
        lin (int): A specific line index in 'p_array' used for calculations.
        p_array (np.ndarray): The primary array from which values are extracted
                              for adjusting 'col'.
        sampling (float, optional): The sampling rate for index adjustment. If
                                    None, it is calculated based on 'col'. Defaults
                                    to None.

    Note:
        The function first calculates the sampling rate if not provided. It then
        iterates over each index in 'col', adjusting it within a range determined
        by the sampling rate. For each index, a range of indices ('ro_ind') is
        created and used to extract maximum values from 'p_array'. These maximum
        values are then used to update the corresponding indices in 'col'. The
        function is useful for adjusting indices in an array based on sampling
        criteria, particularly in data analysis and processing tasks.
    """
    dim = len(col)

    if sampling is None:
        sampling = np.max(col) / (dim - 1)

    if sampling <= 1:
        return col

    p_ind = sampling / 2
    length = 0

    for xxx in range(dim):
        _from = col[xxx] - p_ind
        if _from >= 0:
            if xxx < dim - 1:
                length = col[xxx + 1] - col[xxx]
            if length > 0:
                ro_ind = _from + np.arange(length)
                indices = np.where(ro_ind >= dim)
                if len(indices[0]) > 0:
                    ro_ind[indices] = dim - 1
                mmm = np.max(p_array[ro_ind.astype(int), lin], axis=0)
                col[xxx] = _from + indices[0]

    return col


def update_max_map(
        in_pointer: np.ndarray,
        out_pointer: np.ndarray,
        col_ind: np.ndarray,
        lin_ind: np.ndarray,
        first: int = None,
        last: int = None,
        valids: np.ndarray = None,
        sampling: float = None,
        values: np.ndarray = None,
        absolute: bool = False,
        min_val: bool = False,
) -> np.ndarray:
    """
    Updates an output array with maximum or minimum values from an input array based on column and line indices.

    This function processes an input array and updates an output array with either
    maximum or minimum values. It uses specified column and line indices to determine
    where to take values from the input array and applies constraints such as valid
    indices, sampling rate, and range limitations.

    Args:
        in_pointer (np.ndarray): The input array from which values are extracted.
        out_pointer (np.ndarray): The output array to be updated with max/min values.
        col_ind (np.ndarray): Array of column indices for value extraction.
        lin_ind (np.ndarray): Array of line indices corresponding to each column index.
        first (int, optional): The first index to consider in the update. Defaults to None.
        last (int, optional): The last index to consider in the update. Defaults to None.
        valids (np.ndarray, optional): Array of valid indices. Defaults to None.
        sampling (float, optional): The sampling rate for index checking. Defaults to None.
        values (np.ndarray, optional): Array of values for value replacement. Defaults to None.
        absolute (bool, optional): Flag to use absolute values. Defaults to False.
        min_val (bool, optional): If True, updates with minimum values instead of maximum.
                                  Defaults to False.

    Returns:
        np.ndarray: The updated output array with maximum or minimum values.

    Note:
        The function first verifies dimensions and indices. It then iterates through
        each line index, adjusting column indices using 'checkColIndex' and extracting
        values from the input array. These values are used to update the output array
        with either the maximum or minimum value, based on the 'min_val' flag. The
        function is versatile in handling various constraints and can be used in
        scenarios like image processing or data analysis where max/min value extraction
        is required.
    """

    if in_pointer is None:
        in_size = [0]
    else:
        in_size = in_pointer.shape
    if out_pointer is None:
        out_size = [0]
    else:
        out_size = out_pointer.shape

    if len(in_size) != 2:
        return

    if len(col_ind) != out_size[1]:
        col_ind = np.arange(out_size[0])

    n_lines = out_size[0]
    if len(lin_ind) != n_lines:
        lin_ind = np.arange(n_lines)

    if valids is not None and len(valids) == out_size[1]:
        weight_beam = valids
        indices = np.where(col_ind < 0)
        weight_beam[indices] = 0
        indices = np.where(weight_beam > 0)
        if len(indices[0]) <= 0:
            return
    else:
        app_ind = col_ind.copy()
        if first is not None:
            app_ind[:first] = -1
        if last is not None:
            app_ind[last + 1:] = -1
        indices = np.where(app_ind >= 0)
        if len(indices[0]) <= 0:
            return

    p1, p2 = indices[0][0], indices[0][-1] + 1
    out_size = (2, *out_size)
    type_ = dpg.io.type_py2idl(out_pointer.dtype)
    out_arr = np.zeros(out_size, dtype=dpg.io.type_idl2py(type_))
    ind = col_ind[indices]
    inType = dpg.io.type_py2idl(in_pointer.dtype)
    if inType == 4:
        type_ = 0

    out_arr[0, :, :] = out_pointer
    for lll in range(n_lines):
        if lin_ind[lll] >= 0:
            col = col_ind[indices].copy()
            col = checkColIndex(col, lin_ind[lll], in_pointer, sampling=sampling)
            if type_ == 4 and values is not None:
                out_arr[0, lll, p1:p2] = values[in_pointer[lin_ind[lll], col]]
            else:
                out_arr[0, lll, p1:p2] = in_pointer[lin_ind[lll], col]

    out_arr[1, :, :] = out_pointer

    out_abs = np.abs(out_arr)
    valid_mask = np.isfinite(out_abs)
    out_abs[~valid_mask] = np.nan
    if not min_val:
        out_pointer[:, :] = np.nanmax(out_abs, axis=0)
    else:
        out_pointer[:, :] = np.nanmin(out_abs, axis=0)

    return out_arr[0, :, :]


def update_weighted_map(
        in_pointer: np.ndarray,
        out_pointer: np.ndarray,
        w_pointer: np.ndarray,
        weights: np.ndarray,
        col_ind: np.ndarray,
        lin_ind: np.ndarray,
        values: np.ndarray = None,
):
    """
    Updates an output map with weighted values from an input array.

    This function processes an input array and updates an output array with weighted values,
    using specified column and line indices. It applies weights to the input array values
    before adding them to the output array. The function optionally updates a weight array
    to track the total weights applied at each point in the output array.

    Args:
        in_pointer (np.ndarray): The input array from which values are extracted.
        out_pointer (np.ndarray): The output array to be updated with weighted values.
        w_pointer (np.ndarray): The weight array to be updated with the sum of weights applied.
        weights (np.ndarray): An array of weights to be applied to input values.
        col_ind (np.ndarray): Array of column indices for value extraction.
        lin_ind (np.ndarray): Array of line indices corresponding to each column index.
        values (np.ndarray, optional): Array of values for value replacement. Defaults to None.

    Returns:
        None: The function does not return any value.

    Note:
        The function first checks the dimensions of the input and output arrays. It then
        iterates through each line index, applying weights to the corresponding values in
        the input array and updating the output array. If a weights pointer ('w_pointer')
        is provided, the function also updates this array with the total weights applied.
        This function is useful for combining weighted data from different sources or
        for weighted averaging.
    """

    in_size = in_pointer.shape
    out_size = out_pointer.shape

    if len(in_size) != 2 or len(out_size) < 2:
        return

    n_lines = out_size[1]

    if len(lin_ind) != n_lines:
        lin_ind = np.arange(n_lines)
    if len(col_ind) != out_size[0]:
        col_ind = np.arange(out_size[0])

    weight_beam = weights.copy()

    if len(col_ind) == len(weight_beam):
        indices = np.where(col_ind < 0)
        weight_beam[indices] = 0
        indices = np.where(weight_beam > 0)
        if len(indices[0]) <= 0:
            return
        first, last = indices[0][0], indices[0][-1]
        weight_beam = weight_beam[indices]
        col_ind = col_ind[indices]
    else:
        first, last = 0, len(col_ind) - 1
        weight_beam = weight_beam[0]

    for lll in range(n_lines):
        if 0 <= lin_ind[lll] < in_size[1]:
            in_beam = in_pointer[col_ind, lin_ind[lll]]
            if in_pointer.dtype != np.float32 and values is not None:
                in_beam = values[in_beam.astype(int)]
            curr_beam = out_pointer[first: last + 1, lll]
            out_pointer[first: last + 1, lll] = getCheckNullSum(
                curr_beam, in_beam * weight_beam
            )
            if w_pointer is not None:
                curr_w = w_pointer[first: last + 1, lll]
                w_pointer[first: last + 1, lll] = curr_w + weight_beam
