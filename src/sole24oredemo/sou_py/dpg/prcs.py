import numpy as np
from numba import njit, prange
from scipy import ndimage
from scipy.ndimage import uniform_filter

import sou_py.dpg as dpg
from numbers import Number
from deprecated import deprecated

"""
Funzioni ancora da portare
FUNCTION GetFlex 
FUNCTION GetFreqDistr 
FUNCTION GetMask 
PRO FIND_FLEX 
PRO GET_NUMERIC_ARRAY 
PRO MASK_CENTER 
PRO MEDIAN_INDEX 
PRO RotateArray 
PRO SET_FREQ_HIST 
PRO SmoothArray 
FUNCTION CheckNewDimForResampling   // UNUSED
FUNCTION SAMPGRID                   // UNUSED
PRO COMPUTE_HIST                    // UNUSED
PRO FILTER_ARRAY                    // UNUSED
PRO FIND_CENTER                     // UNUSED
PRO FIND_PROFILE                    // UNUSED
PRO GET_REG_PLOT                    // UNUSED
PRO PROFILE_INTERP                  // UNUSED
PRO RETTA_REG                       // UNUSED
PRO RotateData                      // UNUSED
PRO SET_FREQ_PLOT                   // UNUSED
PRO UnsetVoid                       // UNUSED
PRO VOLUME_PROFILE                  // UNUSED
"""


def linearizeValues(values: np.ndarray, scale, set_void: bool = False) -> np.ndarray:
    """
    Linearizes the given values by a specified scale and handles void or null values.

    This function linearizes an array of values ('values') based on a given scale ('scale'). It optionally
    sets void values in the array based on the 'set_void' flag. The function scales the values, converts them
    to a linear scale, and then handles null and void values by setting them to NaN or a specified void value.

    Args:
        values (np.ndarray): An array of values to be linearized.
        scale (int or float): The scale factor for linearization.
        set_void (bool, optional): If True, void values are set to -infinity or a specified void value. Defaults to
        False.

    Returns:
        np.ndarray: The linearized array of values, with null and void values handled.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'count_invalid_values' from
        'dpg.values'.
    """
    if values is None or scale is None:
        return
    if values.size == 0 or scale < 1:
        return

    ind_null, count_null, ind_void, count_void = dpg.values.count_invalid_values(values)

    if scale == 1 or scale == 2:
        values = values.astype(np.float32)
        values /= 10
    values = 10.0 ** values
    if count_null > 0:
        values[ind_null] = np.nan
    if set_void:
        void = -np.inf
    else:
        void = 0.0

    if count_void > 0:
        values[ind_void] = void

    return values


def unlinearizeValues(values: np.ndarray, scale) -> np.ndarray:
    """
    Converts linearized values back to their original scale and handles invalid or void values.

    This function unlinearizes an array of values ('values') using a given scale ('scale'). It handles
    null, void, and extremely small (negligible) values by setting them appropriately. The function applies
    a logarithmic transformation to bring the values back to the original scale, considering the specified scale factor.

    Args:
        values (np.ndarray): An array of linearized values to be unlinearized.
        scale (int or float): The scale factor used during the initial linearization.

    Returns:
        np.ndarray: The unlinearized array of values, with null, void, and negligible values handled.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'count_invalid_values' from
        'dpg.values'.
    """
    if values is None or scale is None:
        return
    if values.size == 0 or scale < 1:
        return

    ind_null, count_null, ind_void, count_void = dpg.values.count_invalid_values(values)

    ind_neg = np.where(values < 0.001)
    count_neg = ind_neg[0].size
    if count_neg > 0:
        values[ind_neg] = 0.001

    values = np.log10(values)

    if scale == 1 or scale == 2:
        values *= 10.0

    if count_null > 0:
        values[ind_null] = np.nan

    if count_void > 0:
        values[ind_void] = -np.inf

    if count_neg > 0:
        values[ind_neg] = -np.inf

    return values


@njit(parallel=True, cache=True, fastmath=True)
def internal_texture_numba(Weight=None, App=None, Text=None, Sum_weight=None):
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i * i + j * j > 0.0:
                weight_first_roll = np.roll(Weight, j)
                weight_second_roll = np.roll(weight_first_roll.T, i).T
                sh = np.multiply(Weight, weight_second_roll)

                app_first_roll = np.roll(App, j)
                app_second_roll = np.roll(app_first_roll.T, i).T
                appSh = np.multiply(sh, np.power(app_second_roll - App, 2))

                Text += appSh
                Sum_weight += sh

    return appSh, Text, Sum_weight


def internal_texture(Weight, App, Text, Sum_weight):
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i * i + j * j > 0.0:
                sh = np.multiply(Weight, np.roll(Weight, (j, i), axis=(0, 1)))
                appSh = np.multiply(
                    sh, np.power(np.roll(App, (j, i), axis=(0, 1)) - App, 2)
                )
                Text += appSh
                Sum_weight += sh

    return appSh, Text, Sum_weight


def texture(array: np.ndarray, minVal: float = 0.0) -> np.ndarray:
    """
    Computes the texture of a given array by considering the variance around each element.

    This function calculates the texture of an array ('array') by examining the variance of values in the
    neighborhood of each element. The texture is defined as the square root of the average squared difference
    between each element and its neighbors. The function handles invalid values by setting them to specified
    minimum values or zero.

    Args:
        array (np.ndarray): The input array for which the texture is to be computed.
        minVal (float, optional): The minimum value to assign to void elements in 'array'. Defaults to 0.0.

    Returns:
        np.ndarray: An array representing the texture of the input array.

    Raises:
        None: This function does not explicitly raise any exceptions but relies on 'count_invalid_values' from
        'dpg.values'.
    """

    App = array.copy()
    Text = np.zeros_like(array)
    Sum_weight = np.zeros_like(array)
    Weight = np.full_like(array, 1)

    indNull, _, indVoid, _ = dpg.values.count_invalid_values(array)

    App[indNull] = 0
    Weight[indNull] = 0
    App[indVoid] = minVal

    # Versione non ottimizzata con numba del ciclo interno della texture
    # appSh, Text, Sum_weight = internal_texture(Weight, App, Text, Sum_weight)
    #
    appSh, Text, Sum_weight = internal_texture_numba(Weight, App, Text, Sum_weight)

    # np.divide produce risultato double anche quando i dati sono interi
    Text = np.sqrt(np.divide(Text, Sum_weight))

    index = np.where(Sum_weight < 3.0)
    Text[index] = 0.0

    Text[indNull] = 0
    Text[indVoid] = 0

    indNull, _, indVoid, _ = dpg.values.count_invalid_values(Text, sign=0)
    Text[indNull] = 0.0
    Text[indVoid] = 0.0

    return Text


def trapez(
        x: np.ndarray,
        a: float,
        b: float,
        s: float,
        t: float,
        inverse: bool = False,
        maxVal: float = None,
        setVoid: bool = False,
        setNull: bool = False,
) -> np.ndarray:
    """
    Applies a trapezoidal transformation to an array and optionally sets void or null values.

    This function transforms the input array 'x' using a trapezoidal function defined by parameters 'a', 'b', 's',
    and 't'.
    It also provides options to invert the transformation, to scale the output by 'maxVal', and to set specific
    values for
    void or null elements in 'x'.

    Args:
        x (np.ndarray): The input array to be transformed.
        a (float): The starting point of the increasing line of the trapezoid.
        b (float): The starting point of the decreasing line of the trapezoid.
        s (float): The slope of the increasing line.
        t (float): The slope of the decreasing line.
        inverse (bool, optional): If True, inverts the trapezoidal transformation. Defaults to False.
        maxVal (float, optional): A value to scale the transformed array. Defaults to None.
        setVoid (bool, optional): If True, sets a specific value for void elements in the array. Defaults to False.
        setNull (bool, optional): If True, sets a specific value for null elements in the array. Defaults to False.

    Returns:
        np.ndarray: The transformed array after applying the trapezoidal function and handling void or null values.

    Raises:
        ZeroDivisionError: Handles division by zero errors in the transformation.
        NameError: Handles exceptions if certain variables are not defined.
    """
    """
    a,b,s,t sono scalari 
    x matrice 
    """
    # il caso in cui s=0 o t=0 va gestito

    if np.isfinite(a):
        out = x.copy()
        try:
            out = np.divide((x - a + s), s)
        except ZeroDivisionError:
            out = np.full(x.shape, np.inf)
    # endif

    if np.isfinite(b):
        ind = np.where(x > b)
        try:
            out[ind] = np.divide(b + t - x[ind], t)
        except NameError:
            out = (b + t - x) / t
        except ZeroDivisionError:
            out = np.full(x.shape, np.inf)
        # endif
    # endif

    ind = np.where(out < 0)
    out[ind] = 0.0
    ind = np.where(out > 1.0)
    out[ind] = 1.0

    if inverse:
        out = 1.0 - out

    if (setVoid) or (setNull):
        indNull, countNull, indVoid, countVoid = dpg.values.count_invalid_values(x)
        if setVoid:
            out[indVoid] = setVoid
        if setNull:
            out[indNull] = setNull
    # endif

    if isinstance(maxVal, Number):
        out *= maxVal

    return out


def maximize_data(data: np.ndarray, box: int) -> np.ndarray:
    """
    Applies a maximum filter to a 2D array over a specified box size.

    This function processes a 2D array ('data') by replacing each element with the maximum value in its
    neighborhood, defined by a square of side length 'box'. The function operates on the central part of the
    array, avoiding the edges by a margin equal to half the box size. The edges of the array are left unchanged.

    Args:
        data (np.ndarray): The input 2D array on which the maximum filter is to be applied.
        box (int): The side length of the square used for calculating the maximum filter.

    Returns:
        np.ndarray: The 2D array after applying the maximum filter.

    Raises:
        None: This function does not explicitly raise any exceptions but requires that 'data' is a 2D array.
    """
    dim = data.shape
    if len(dim) != 2:
        return

    tmp = data.copy()
    bbb = box // 2
    ind = np.arange(box)
    yInit = np.ones(box, dtype=int)  # ind
    xInd = ind  # np.ones(box, dtype=int)

    for xxx in range(bbb, dim[0] - bbb):
        yInd = yInit.copy()
        for yyy in range(bbb, dim[1] - bbb):
            tmp[xxx, yyy] = np.nanmax(data[xInd, yInd])
            yInd += 1
        xInd += 1

    data = (
        tmp.copy()
    )  # TODO valutare se ha senso tenere la copia o restituire direttamente il tmp

    return data


@njit(cache=True, parallel=True, fastmath=True)
def nanmedian_filter_explicit(image, size):
    """
    Apply a NaN-aware median filter to an image, skipping explicit padding.

    Parameters:
        image (ndarray): Input 2D array with NaN values.
        size (int): The size of the filter window (must be odd).

    Returns:
        ndarray: Filtered array with internal regions processed.
    """
    if size % 2 == 0:
        raise ValueError("Filter size must be odd.")

    rows, cols = image.shape
    half_size = size // 2

    # Create an output array
    output = np.zeros(image.shape)

    # Iterate over internal region only
    for i in prange(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            # Extract the neighborhood
            window = image[i - half_size:i + half_size, j - half_size:j + half_size]
            # Compute the median excluding NaNs
            output[i, j] = np.nanmedian(window)

    return output


@deprecated(reason="Not working as intended.")
@njit(cache=False, parallel=False, fastmath=True)
def smooth_data_opt1_numba(data, d3, kdim):
    for ddd in prange(d3):
        filtered_matrix = nanmedian_filter_explicit(data[ddd, :, :], kdim)
        data[ddd, :, :] = filtered_matrix
    return data


@njit(parallel=True, fastmath=True, cache=True)
def smooth_data_opt1_numba_v2(data, d3, kdim):
    identical_rows = kdim // 2  # Precompute the number of rows/columns to copy
    for ddd in prange(d3):  # Use prange for parallel loops
        tmp = data[ddd, :, :]

        # Get dimensions
        rows, cols = tmp.shape

        # Create the filtered matrix
        filtered_matrix = np.empty_like(tmp)

        # Apply a median filter manually
        for i in range(rows):
            for j in range(cols):
                # Define the kernel bounds
                row_start = max(0, i - identical_rows)
                row_end = min(rows, i + identical_rows + 1)
                col_start = max(0, j - identical_rows)
                col_end = min(cols, j + identical_rows + 1)

                # Compute the median within the kernel
                filtered_matrix[i, j] = np.median(tmp[row_start:row_end, col_start:col_end])

        # Copy edges from the original matrix
        filtered_matrix[:identical_rows, :] = tmp[:identical_rows, :]
        filtered_matrix[-identical_rows:, :] = tmp[-identical_rows:, :]
        filtered_matrix[:, :identical_rows] = tmp[:, :identical_rows]
        filtered_matrix[:, -identical_rows:] = tmp[:, -identical_rows:]

        # Update the original data
        data[ddd, :, :] = filtered_matrix

    return data


def smooth_data_opt1(data, d3, kdim):
    for ddd in range(d3):
        tmp = data[ddd, :, :]
        filtered_matrix = ndimage.median_filter(tmp, kdim)
        identical_rows = int(np.floor(kdim / 2))
        filtered_matrix[0:identical_rows, :] = tmp[0:identical_rows, :]  # Copy the first row
        filtered_matrix[-identical_rows:, :] = tmp[-identical_rows:, :]  # Copy the last row
        filtered_matrix[:, 0:identical_rows] = tmp[:, 0:identical_rows]  # Copy the first column
        filtered_matrix[:, -identical_rows:] = tmp[:, -identical_rows:]
        data[ddd, :, :] = filtered_matrix
    return data


def smooth_opt0(data, kdim):
    nan_mask = ~np.isfinite(data)
    data_filled = np.where(nan_mask, 0, data)  # Replace NaNs with 0 for smoothing
    smoothed = uniform_filter(data_filled, size=kdim)  # Apply uniform filter
    # Normalize to account for NaNs (avoiding their influence on averages)
    normalization = uniform_filter((~nan_mask).astype(float), size=kdim)
    with np.errstate(invalid='ignore'):  # Ignore division warnings due to NaN presence
        smoothed /= normalization
    edge = kdim // 2
    if data.ndim == 1:
        smoothed[:edge] = data[:edge]  # Set start boundary to NaN
        smoothed[-edge:] = data[-edge:]  # Set end boundary to NaN
    elif data.ndim == 2:
        smoothed[:edge, :] = data[:edge, :]  # Set top boundary to NaN
        smoothed[-edge:, :] = data[-edge:, :]  # Set bottom boundary to NaN
        smoothed[:, :edge] = data[:, :edge]  # Set left boundary to NaN
        smoothed[:, -edge:] = data[:, -edge:]  # Set right boundary to NaN


    return smoothed


def smooth_data(
        data: np.ndarray,
        box: int,
        opt: int = None,
        no_null=None,
        filter=None,
        kdim=None,
        min_val=None,
):
    '''
    Applies a smoothing filter to the input data.

    Args:
        data (np.ndarray): Array of data to be smoothed.
        box (int): Size of the box for the filter (if used).
        opt (int, optional): Option for the type of smoothing.
            - 0: Mean
            - 1: Median
            - 2: Sobel (not implemented)
            - 3: Maximization
            - others: Convolution
        no_null (None or bool, optional): If True, ignore null values in calculations.
        filter (np.ndarray or None, optional): Filter to be applied to the data.
        kdim (int or None, optional): Dimension of the kernel for the filter.
        min_val (float or None, optional): Minimum value to use for null data.

    Returns:
        np.ndarray: Smoothed data.
    '''

    if data is None:
        return

    if box is not None:
        box = int(box)

    dim = data.shape
    if filter is None:
        if kdim == 1:
            filter = np.ones((kdim, kdim))
        if box is not None:
            if box <= 0:
                return
            filter = np.ones((int(box), int(box)))

    kernel = filter
    boxDim = kernel.shape
    d3 = 1

    if len(dim) == 3:
        d3 = dim[0]

        if kdim is not None:
            if kdim >= 3:
                kdim = d3 - 1
    elif len(dim) == 1:
        kernel = kernel[:, 0]

    sc = np.sum(kernel)
    if sc == 0:
        sc = 1

    dtype_attr = getattr(data, 'dtype', None)
    dtype = dpg.io.type_py2idl(dtype_attr if dtype_attr is not None else type(data))

    if opt is None:
        opt = -1

    countNull = 0
    countVoid = 0
    count9 = 0

    if no_null is None and dtype == 4:
        indNull, countNull, indVoid, countVoid = dpg.values.count_invalid_values(data)

    if opt == 0:
        if kdim is None or len(kdim) == 0:
            kdim = box
        if countVoid > 0:
            data[indVoid] = np.nan

        data = smooth_opt0(data, kdim)

    elif opt == 1:
        if kdim is None or len(kdim) == 0:
            kdim = box
        mDim = min(dim)
        kdim = min(kdim, mDim)
        if countNull > 0:
            data[indNull] = np.inf

        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)

        identical_rows = kdim // 2

        # versione vecchia non ottimizzata
        # data = smooth_data_opt1(data, d3, kdim)

        data = smooth_data_opt1_numba_v2(data, d3, kdim)

        data = np.squeeze(data)

        if dpg.io.type_py2idl(data.dtype) == 4:
            ind9 = np.where(data == np.inf)
            data[ind9] = np.nan

    elif opt == 2:
        for ddd in range(d3):
            tmp = data[:, :, ddd]
            # Sobel filter not directly available in NumPy, using a placeholder
            # You may replace this with an appropriate Sobel filter implementation
            tmp = tmp  # Placeholder for Sobel filter  # TODO: da fare sobel filter
            data[:, :, ddd] = tmp
    elif opt == 3:
        if len(box) != 1 and kdim is not None and len(kdim) == 1:
            data = dpg.prcs.maximize_data(data, kdim)
            return
        data = dpg.prcs.maximize_data(data, box)
    else:
        if min_val is not None and countVoid > 0:
            data[indVoid] == min_val
        for ddd in range(d3):
            tmp = data[:, :, ddd]
            if float(np.version.version) > 6.1:
                tmp = np.convolve(
                    tmp, kernel, mode="same", boundary="wrap"
                )  # CONVOL function in IDL
            else:
                tmp = np.convolve(
                    tmp, kernel, mode="same", boundary="wrap", normalize_kernel=True
                )
            data[:, :, ddd] = tmp

    if countNull > 0:
        data[indNull] = np.nan
    if countVoid > 0:
        data[indVoid] = -np.inf

    return data


def thresh_array(array, min=None, max=None, checkNull=None, checkVoid=None):
    if np.size(array) <= 1:
        return array

    if min is not None:
        ind = np.where(array < min)
        array[ind] = min

    if max is not None:
        ind = np.where(array > max)
        array[ind] = max

    if checkNull is not None:
        ind = np.where(np.isnan(array))
        array[ind] = checkNull

    if checkVoid is not None:
        indVoid = np.where(np.isneginf(array))
        array[indVoid] = checkVoid

    return array


def get_numeric_array(node, name=None, linear=None, values=None, scale=None, reload=None, unit=None, parname=None,
                      aux=None, dim=None):
    main = node
    if name is not None:
        son = dpg.tree.findAllDescendant(node, name)
        if son is None:
            return None
        if isinstance(son, list):
            main = son[0]

    var = dpg.array.get_numeric_array(main, linear=linear, values=values, scale=scale, reload=reload, unit=unit,
                                      parname=parname, aux=aux, dim=dim)

    return var
