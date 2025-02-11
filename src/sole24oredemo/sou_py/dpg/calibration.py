import copy
import os
import sys

import numpy as np
from numbers import Number
import sou_py.dpg as dpg
from sou_py.dpg.attr__define import Attr
from sou_py.dpg.log import log_message

"""
Funzioni ancora da portare
FUNCTION GET_CORRECTED_SOLAR_CONST 
FUNCTION GET_CORRECTION_FACTOR 
FUNCTION GET_COS_SAT_ANGLE 
FUNCTION GET_COS_SOLAR_ANGLE 
FUNCTION GET_LATLON_ARRAY 
FUNCTION GET_RADIANCE 
FUNCTION GET_SOLAR_ANGLE 
FUNCTION GET_SOLAR_CONST 
FUNCTION GET_TEMP_ARRAY 
FUNCTION GET_TEMPERATURE 
FUNCTION MARSHALL_PALMER_f 
FUNCTION SCALE_PERCENT_ARRAY 
FUNCTION TEMP_TO_RAD 
FUNCTION CHECK_CALIB_MODE           // UNUSED
PRO CALIBRATE_VISIBLE               // UNUSED
PRO SUN_CORRECTION                  // UNUSED
PRO Z_TO_R                          // UNUSED
"""


def convertData(
        data: np.ndarray, values: np.ndarray, linear: bool = False, scale: int = None
) -> np.ndarray:
    """
    Convert data array by optional linearization and mapping through a values array.

    Args:
        data (numpy.ndarray): The array of data to be converted.
        values (numpy.ndarray): The array of values used for conversion mapping.
        linear (bool, optional): Indicates if linearization should be applied. Defaults to False.
        scale (int, optional): Scale factor used in linearization. Defaults to 2.

    Returns:
        numpy.ndarray: The converted data array.
    """
    # TODO da ricontrollare gestione dei tipi
    # Sembra corretto perché gestisce adeguatamente i casi integer e float
    # mentre restituisce i dati non convertiti se il tipo non ricade in nessuno
    # di questi due casi
    if np.size(data) <= 1:
        return None

    data_type = data.dtype.type

    if np.issubdtype(data_type, np.floating):
        # if (data_type == np.float32) or (data_type == np.float64):
        if linear:
            if not isinstance(scale, Number):
                scale = 2
            data = dpg.prcs.linearizeValues(data, scale=scale)
        return data

    if np.size(values) <= 1:
        return data

    if np.issubdtype(data_type, np.integer):
        if linear:
            if not isinstance(scale, Number):
                scale = 2
            l_values = dpg.prcs.linearizeValues(values, scale=scale)
            data = l_values[data]
        else:
            data = values[data]

    return data


def power_func(
        Z: np.ndarray,
        a: Number = None,
        b: Number = None,
        thresh: Number = None,
        linear: bool = False,
        inverse: bool = False,
        max_val=None,
) -> np.ndarray:
    """
    Applies a power-law transformation to an input array, with options for linear, inverse, and threshold-based
    transformations.

    This function performs a power-law transformation on the input array 'Z'. It allows for
    linear scaling, inverse transformation, and applies a threshold to determine which
    values to transform. The function can also cap the transformed values to a maximum value.

    Args:
        Z (np.ndarray): The input array to be transformed.
        a (Number, optional): The 'a' parameter in the power-law relationship. Defaults to 200.0.
        b (Number, optional): The 'b' parameter in the power-law relationship. Defaults to 1.6.
        thresh (Number, optional): A threshold value to determine which array values to transform.
                                   Defaults to None.
        linear (bool, optional): If True, applies linear scaling to the data before transformation.
                                 Defaults to False.
        inverse (bool, optional): If True, applies the inverse of the power-law transformation.
                                  Defaults to False.
        max_val (Number, optional): A maximum cap for the transformed values. Defaults to None.

    Returns:
        np.ndarray: The transformed array with the power-law (and optional linear/inverse)
                    transformation applied.

    Note:
        The function first identifies NaN values in 'Z' to preserve them. It then applies
        the power-law transformation to values above the specified threshold. The function
        handles linear scaling by converting to/from dB, and inverse transformation by
        applying the inverse power-law. If 'max_val' is provided, values exceeding this
        maximum are capped. NaN values in the original array are maintained in the output.
    """
    if np.size(Z) <= 0:
        return None
    indNull = np.where(np.isnan(Z))

    if not isinstance(a, Number):
        a = 200.0
        b = 1.6

    nT = 0 if thresh is None else 1

    if nT <= 1:
        if nT == 1:
            th = float(thresh)
        else:
            if linear:
                th = 10.0
            else:
                th = 0.01

        ind = np.where(Z >= th)
        if len(ind[0]) > 0:
            tmp = Z[ind].copy()
            if linear:
                tmp = np.power(10.0, tmp / 10.0)
            if inverse:
                tmp = np.power(tmp / a, 1.0 / b)
            else:
                tmp = a * np.power(tmp, b)

            R = np.zeros_like(Z)
            R[ind] = tmp.copy()
        else:
            R = np.zeros_like(Z)
    else:
        # TODO case three
        # thresh è un numero o può essere anche un array?
        # Da IDL pare possa essere anche un array
        # Attenzione: la funzione è ricorsiva
        print("TODO: to be implemented case threes is passed")

    if isinstance(max_val, Number):
        ix, iy, iz = np.where(R > max_val)
        R[ix, iy, iz] = max_val

    if len(indNull[0]) > 0:
        R[indNull] = np.nan

    return R


def marshall_palmer(
        Z: np.ndarray, a: Number = None, b: Number = None, thresh: Number = None
) -> np.ndarray:
    """
    Applies the Marshall-Palmer relation to convert radar reflectivity to rain rate.

    This function uses the Marshall-Palmer relation, a power-law transformation, to convert
    radar reflectivity (Z) to rain rate (R). It internally calls 'power_func' with linear
    scaling and inverse transformation options enabled.

    Args:
        Z (np.ndarray): The radar reflectivity array.
        a (Number, optional): The 'a' parameter in the Marshall-Palmer relation. Defaults to None.
        b (Number, optional): The 'b' parameter in the Marshall-Palmer relation. Defaults to None.
        thresh (Number, optional): A threshold value below which the reflectivity is considered
                                   too low to calculate the rain rate. Defaults to None.

    Returns:
        np.ndarray: The calculated rain rate array based on the input reflectivity.

    Note:
        The Marshall-Palmer relation is widely used in meteorology to estimate rain rate
        from radar reflectivity. The function can be customized with different 'a' and 'b'
        parameters for different radar types or meteorological conditions. The 'thresh'
        parameter allows ignoring reflectivity values below a certain threshold, which might
        correspond to no or very light precipitation.
    """
    # TODO ma che senso ha?
    R = power_func(Z, a=a, b=b, thresh=thresh, linear=True, inverse=True)
    return R


def createValuesTable(
        offset: float,
        slope: float,
        bitplanes: int,
        nullInd: int = -1,
        voidInd: int = -1,
        bottom: int = 0,
        top: int = 0,
        log_scale: bool = False,
        maxVal: float = None,
):
    """
    Creates a table of real values from digital number (DN) values based on specified conversion parameters.

    This function generates a table mapping digital number (DN) values to real values
    based on provided offset, slope, and bitplane information. It supports linear and
    logarithmic scaling and handles special cases for null and void values, as well as
    custom top and bottom limits.

    Args:
        offset (float): The offset to be applied in the value conversion.
        slope (float): The slope used in linear scaling of DN values.
        bitplanes (int): The number of bitplanes in the DN values.
        nullInd (int, optional): The index to be assigned NaN. Defaults to -1 (no NaN assignment).
        voidInd (int, optional): The index to be assigned negative infinity. Defaults to -1 (no assignment).
        bottom (int, optional): The starting index for real value assignment. Defaults to 0.
        top (int, optional): The ending index for real value assignment. Defaults to 0 (max value of bitplanes - 1).
        log_scale (bool, optional): If True, applies logarithmic scaling. Defaults to False.
        maxVal (float, optional): The maximum real value to be assigned. Defaults to None.

    Returns:
        np.ndarray: An array of real values corresponding to each DN value.

    Note:
        The function first checks and adjusts the 'bottom' and 'top' values to ensure they
        are within the valid range. It then calculates real values for each DN value based
        on the specified offset and slope. If 'log_scale' is True, logarithmic scaling is
        applied instead of linear. Special handling is included for 'nullInd' and 'voidInd'
        indices, as well as for setting values outside the 'bottom' to 'top' range to NaN.
        The function is useful for converting raw sensor data to meaningful real-world values.
    """

    maxpix = 2 ** bitplanes
    if bottom < 0 or bottom > maxpix:
        bottom = 0
    if top < 1 or top >= maxpix:
        top = maxpix - 1

    minV = offset
    slp = slope
    if log_scale:
        slp = 0.0
    if slp <= 0.0:
        maxV = 1.0 * (maxpix - 1)
        if maxVal is not None:
            if maxVal > minV:
                maxV = maxVal
        # endif
        if log_scale:
            maxV = np.log10(maxV)
            minV = 0.0
            slope = 0.0
            slp = (maxV - minV) / float(top - bottom)
        else:
            slope = (maxV - minV) / float(top - bottom)
            slp = slope
        # endif
    # endif

    realValues = minV + slp * np.arange(maxpix)
    realValues = realValues.astype(np.float32)
    if log_scale:
        realValues = np.power(10, realValues) - 1.0

    if bottom > 0:
        realValues = np.roll(realValues, bottom)
        realValues[0: bottom - 1] = np.nan
    # endif

    if nullInd >= 0 and nullInd < maxpix:
        realValues[nullInd] = np.nan
    # endif

    if voidInd >= 0 and voidInd < maxpix:
        realValues[voidInd] = -np.inf
    # endif

    if top + 1 < maxpix - 1:
        realValues[top + 1:] = np.nan
    # endif

    if maxVal is not None:
        if maxVal > realValues[top - 1]:
            realValues[top] = maxVal
    # endif

    return realValues, slope


def createArrayValues(
        calib, data: np.ndarray = None, bitplanes: int = None, log_scale: bool = False
):
    """
    Generates a mapping of raw data values to calibrated real values based on calibration attributes.

    This function creates a table mapping digital number (DN) values to calibrated real values,
    considering various calibration parameters like scaling, offset, slope, and bitplanes.
    It supports both linear and logarithmic scaling and incorporates special handling for
    null and void indices.

    Args:
        calib: The calibration attribute object containing calibration parameters.
        data (np.ndarray, optional): The data array to be used for calibration. Defaults to None.
        bitplanes (int, optional): The number of bitplanes to be considered in calibration.
                                   Defaults to 8.
        log_scale (bool, optional): If True, applies logarithmic scaling. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - bitplanes, log_scale, offset, slope, maxVal, nullInd, voidInd, bottom, and top:
                The calibration parameters used.
            - values: the calibrated value table.

    Note:
        The function first retrieves calibration parameters like offset, slope, and bitplanes
        from the 'calib' object. It then calculates the calibrated values using these parameters,
        handling different scaling scenarios and special index cases. The output dictionary
        includes the values table and all the relevant calibration parameters, facilitating the
        conversion of raw sensor data into meaningful calibrated values.
    """
    scaling, _, _ = dpg.attr.getAttrValue(calib, "scaling", 1)
    if scaling <= 0:
        return None, None, None, None, None, None, None, None, None, None

    if bitplanes is None:
        bitplanes = 8
    bitplanes, _, _ = dpg.attr.getAttrValue(calib, "bitplanes", bitplanes)
    if bitplanes <= 0 or bitplanes > 16:
        return None, None, None, None, None, None, None, None, None, None

    offset, exists, _ = dpg.attr.getAttrValue(calib, "offset", 0.0)
    if not exists:
        offset, exists, _ = dpg.attr.getAttrValue(calib, "minVal", 0.0)
    maxVal, _, _ = dpg.attr.getAttrValue(calib, "maxVal", offset)
    slope, _, _ = dpg.attr.getAttrValue(calib, "slope", 0.0)
    minSlope = slope
    if data is not None:
        if scaling == 2 or scaling == 3:
            slope = 0.0
            maxVal = offset

    if scaling > 4:
        log_scale = True
    else:
        if slope <= 0.0:
            if maxVal <= offset and (data is not None):
                minVal, _, maxVal, _ = dpg.values.minmax(data, exclude_invalid=True)
                if (not exists) and np.isfinite(minVal):
                    offset = minVal
                if scaling == 3 and bitplanes > 8 and minSlope > 0.0:
                    if (maxVal - offset) / 255.0 < minSlope:
                        slope = minSlope
                        bitplanes = 8

    nullInd, _, _ = dpg.attr.getAttrValue(calib, "nullInd", np.int64(-1))
    voidInd, _, _ = dpg.attr.getAttrValue(calib, "voidInd", np.int64(-1))
    bottom, _, _ = dpg.attr.getAttrValue(calib, "bottom", 0)
    top = np.int64(2) ** bitplanes - 1
    top, _, _ = dpg.attr.getAttrValue(calib, "top", top)

    if (bottom > 0) and (nullInd != 0) and (voidInd < 0):
        voidInd = 0
    if offset == 0.0 and bottom == 0:
        if nullInd == 0:
            bottom = 1
        if voidInd >= 0:
            bottom = voidInd + 1

    values, slope = createValuesTable(
        offset,
        slope,
        bitplanes,
        nullInd,
        voidInd,
        bottom,
        top,
        log_scale=log_scale,
        maxVal=maxVal,
    )

    return (
        bitplanes,
        log_scale,
        offset,
        slope,
        maxVal,
        nullInd,
        voidInd,
        bottom,
        top,
        values,
    )


def quantizeArray(fArray: np.ndarray, values: np.ndarray, log_scale: bool = False):
    """
    Quantizes a floating-point array into discrete levels based on provided calibration values and parameters.

    This function converts a floating-point array into a quantized array with discrete
    levels. It uses a set of calibration values and supports both linear and logarithmic
    scaling. Special handling is included for void and null indices, and the function
    adjusts for offsets and range limits.

    Args:
        fArray (np.ndarray): The floating-point array to be quantized.
        values (np.ndarray): An array of calibration values used for quantization.
        log_scale (bool, optional): If True, applies logarithmic scaling. Defaults to False.
        voidInd (int, optional): Index to be used for void values. Defaults to 0.
        nullInd (int, optional): Index to be used for null (NaN) values. Defaults to 0.
        offset (float, optional): The offset value for quantization. Defaults to 0.
        maxVal (float, optional): The maximum value for quantization. Defaults to 0.
        bottom (int, optional): The lower bound index for quantized values. Defaults to 0.
        top (int, optional): The upper bound index for quantized values. Defaults to 0.

    Returns:
        tuple: A tuple containing:
               - np.ndarray: The quantized array.
               - float: The offset used for quantization.

    Note:
        The function first calculates the range of values to be used for quantization.
        It then applies either logarithmic or linear scaling to transform the floating-point
        values to the corresponding quantized levels. Special cases for void and null values
        are handled, and the values are scaled and offset to fit within the specified bottom
        and top range limits. The function is useful in processing and visualization of
        sensor data, where raw measurements need to be converted into discrete levels for analysis.
    """

    if not isinstance(values, np.ndarray):
        return fArray, None, None, None, None, None, None

    indNull, countNull, indVoid, countVoid = dpg.values.count_invalid_values(fArray)
    nullInd, countNullV, voidInd, countVoidV = dpg.values.count_invalid_values(values)
    # minVal, bottom, maxVal, top = dpg.values.minmax(values)
    minVal, bottom, maxVal, top = dpg.values.minmax(values, exclude_invalid=True)

    indMin = np.where(fArray <= minVal)
    indMax = np.where(fArray >= maxVal)

    if log_scale:
        minV = 0.0
        maxV = np.log10(maxVal)
        qArray = np.log10(fArray + 1.0)
        ind = np.where(qArray < 0.0)
        qArray[ind] = 0.0
        offset = 0.0
    else:
        minV = minVal
        maxV = maxVal
        qArray = fArray - minV
        offset = minVal
    # endelse

    fact = (top - bottom) / (maxV - minV)
    btm = bottom + 0.9999

    if top < 256:
        # Require numpy == 1.22.3 to avoid warnings
        qArray = np.uint8(np.nan_to_num(qArray * fact + btm, neginf=0))
    else:
        # Warning dovuto a valori che superavano i limiti di uint16
        cast_input = np.nan_to_num(qArray * fact + btm, neginf=0)
        cast_input = np.clip(cast_input, 0, 65535)  # Valori nel range di uint16
        qArray = np.uint16(cast_input)
    # endelse

    qArray[indMin] = bottom
    qArray[indMax] = top

    if np.size(voidInd) == 0:
        voidInd = -1
    else:
        voidInd = voidInd[0][0]

    if np.size(nullInd) == 0:
        nullInd = -1
    else:
        nullInd = nullInd[0][0]

    if countVoid > 0 and voidInd >= 0:
        qArray[indVoid] = voidInd
    if countNull > 0 and nullInd >= 0:
        qArray[indNull] = nullInd

    return qArray, offset, maxVal, nullInd, voidInd, bottom, top


def quantizeData(
        data: np.ndarray, calib, log_scale: bool = False, out_calib: Attr = None
):
    """
    Quantizes input data based on calibration parameters and optionally updates an output calibration attribute.

    This function applies quantization to input data using a set of calibration parameters
    and generates a quantized data array. It supports optional logarithmic scaling and can
    update an output calibration attribute with the quantization parameters.

    Args:
        data (np.ndarray): The input data array to be quantized.
        calib: The calibration attribute object containing calibration parameters.
        log_scale (bool, optional): If True, applies logarithmic scaling. Defaults to False.
        out_calib (optional): The output calibration attribute object to be updated with
                              quantization parameters. Defaults to None.

    Returns:
        dict: A dictionary containing the quantized data array ('quant'), values table ('values'),
              and a flag indicating whether logarithmic scaling was applied ('log_scale').

    Note:
        The function first checks the validity of the input data and calibration attribute.
        It then retrieves and applies calibration parameters such as offset, slope, and bitplanes
        to quantize the data. The quantized data and corresponding values table are stored in the
        output dictionary. If an output calibration attribute ('out_calib') is provided, the
        function updates it with the quantization parameters. This function is essential for
        processing raw sensor data and converting it into quantized form suitable for further analysis.
    """
    out = {}

    if not isinstance(data, np.ndarray):
        return out, out_calib

    parname, _, _ = dpg.attr.getAttrValue(calib, "parname", "")
    if parname == "":
        return out, out_calib

    (
        bitplanes,
        log_scale,
        offset,
        slope,
        maxVal,
        nullInd,
        voidInd,
        bottom,
        top,
        values,
    ) = createArrayValues(calib, data=data)
    if values is None:
        return out, out_calib

    quant, offset, maxVal, nullInd, voidInd, bottom, top = quantizeArray(
        data, values, log_scale=log_scale
    )

    out["quant"] = quant
    out["values"] = values
    out["log_scale"] = log_scale
    if out_calib is None:
        return out, out_calib

    bitplanes = str(bitplanes).strip()
    offset = str(offset).strip()
    slope = str(slope).strip()
    maxVal = str(round(maxVal, 6)).strip()
    nullInd = str(nullInd).strip()
    voidInd = str(voidInd).strip()
    bottom = str(bottom).strip()
    top = str(top).strip()

    unit, _, _ = dpg.attr.getAttrValue(calib, "unit", "")
    parname, _, _ = dpg.attr.getAttrValue(out_calib, "parname", parname)
    unit, _, _ = dpg.attr.getAttrValue(out_calib, "unit", unit)

    tags = [
        "parname",
        "unit",
        "bitplanes",
        "offset",
        "slope",
        "maxVal",
        "nullInd",
        "voidInd",
        "bottom",
        "top",
    ]
    vals = [
        parname,
        unit,
        bitplanes,
        offset,
        slope,
        maxVal,
        nullInd,
        voidInd,
        bottom,
        top,
    ]

    if isinstance(out_calib, list):
        out_calib = out_calib[0]

    dpg.attr.replaceTags(out_calib, tags, vals)
    dpg.attr.removeTags(out_calib, ["minVal", "inherits"])

    return out, out_calib


def get_array_values(
        node,
        calib=None,
        to_not_create: bool = False,
        getMinMaxVal: bool = False,
        reload: bool = False,
        recompute: bool = False,
):
    """
    Retrieves array values and calibration attributes for a given node.

    This function obtains the array values and various calibration attributes associated
    with a node. It supports options such as recomputing values, reloading, and controlling
    whether to create calibration attributes if they do not exist.

    Args:
        node: The node from which array values and calibration attributes are to be retrieved.
        calib (optional): A specific calibration attribute to be used. Defaults to None.
        to_not_create (bool, optional): If True, does not create calibration attributes if they do not exist.
                                        Defaults to False.
        getMinMaxVal (bool, optional): If True, retrieves the minimum and maximum values. Defaults to False.
        reload (bool, optional): If True, reloads the values. Defaults to False.
        recompute (bool, optional): If True, recomputes the values. Defaults to False.

    Returns:
        tuple: A tuple containing:
               - The calibration attribute object.
               - A dictionary with calibration attributes such as 'slope', 'parname', 'offset',
                 'bitplanes', 'bottom', 'nullInd', 'voidInd', 'top', 'simmetric', 'scaling',
                 'channel', 'fNull', 'str_format', 'pointer', 'scale', 'maxVal', 'minVal', and 'unit'.

    Note:
        The function first retrieves the calibration object and then extracts various calibration
        parameters like slope, offset, bitplanes, etc., from this object. It can optionally compute
        and include the minimum and maximum values. This function is essential for obtaining
        detailed calibration information and array values from a node, particularly in applications
        involving data processing and analysis based on calibrated sensor data.
    """
    if to_not_create:
        to_create = 0
    else:
        to_create = 1

    values, calib, out_dict = get_values(
        node, calib=calib, recompute=recompute, reload=reload, to_create=to_create
    )

    parname = out_dict["parname"]
    if parname == '' or parname is None:
        return values, calib, out_dict

    out_dict["slope"], _, _ = dpg.attr.getAttrValue(calib, "slope", 0.0)
    offset, exists, _ = dpg.attr.getAttrValue(calib, "offset", 0.0)
    if not exists:
        offset, _, _ = dpg.attr.getAttrValue(calib, "minVal", 0.0)
    out_dict["offset"] = offset

    out_dict["bitplanes"], _, _ = dpg.attr.getAttrValue(calib, "bitplanes", 8)
    out_dict["bottom"], _, _ = dpg.attr.getAttrValue(calib, "bottom", 0)
    out_dict["nullInd"], _, _ = dpg.attr.getAttrValue(calib, "nullInd", -1.0)
    out_dict["voidInd"], _, _ = dpg.attr.getAttrValue(calib, "voidInd", -1.0)

    tmp, exists, _ = dpg.attr.getAttrValue(calib, "top", 0.0)
    if exists:
        out_dict["top"] = tmp

    out_dict["simmetric"], _, _ = dpg.attr.getAttrValue(calib, "simmetric", 0)
    out_dict["scaling"], _, _ = dpg.attr.getAttrValue(calib, "scaling", 1)
    out_dict["channel"], _, _ = dpg.attr.getAttrValue(calib, "channel", "")

    tmp, exists, _ = dpg.attr.getAttrValue(calib, "fNull", 0.0)
    if exists:
        out_dict["fNull"] = tmp

    tmp, exists, _ = dpg.attr.getAttrValue(calib, "str_format", 0.0)
    if exists and tmp != "":
        out_dict["str_format"] = tmp
    else:
        out_dict["str_format"] = None

    if getMinMaxVal:
        nValues = np.size(values)
        out_dict["minVal"], _, _ = dpg.attr.getAttrValue(calib, "minVal", offset)
        out_dict["maxVal"], exists, _ = dpg.attr.getAttrValue(
            calib, "maxVal", 255.0
        )
        if not exists and nValues > 1:
            out_dict["maxVal"] = np.nanmax(values)
            out_dict["minVal"] = np.nanmin(values)

    return values, calib, out_dict


def get_values(
        node,
        calib=None,
        get_slope: bool = False,
        to_create: bool = False,
        reload: bool = False,
        recompute: bool = False,
):
    """
    Retrieves values and calibration information associated with a given node.

    This function fetches various values and calibration attributes for a specified node,
    including the values array, parameter name, unit, scale, and optionally, the slope.
    It supports options to create, reload, or recompute the values.

    Args:
        node (dpg.node__define.Node): The node from which values and calibration information
                                      are to be retrieved.
        calib (optional): A specific calibration attribute to be used. Defaults to None.
        get_slope (bool, optional): If True, retrieves the slope value from calibration.
                                    Defaults to False.
        to_create (bool, optional): If True, creates calibration attributes if they do not exist.
                                    Defaults to False.
        reload (bool, optional): If True, reloads the values. Defaults to False.
        recompute (bool, optional): If True, recomputes the values. Defaults to False.

    Returns:
        tuple: A tuple containing:
               - calib: The calibration attribute object.
               - dict: A dictionary with keys 'values', 'parname', 'unit', 'scale', and optionally 'slope'.

    Note:
        The function first checks if the node is a valid 'dpg.node__define.Node' and retrieves the
        calibration dictionary. It then extracts and populates the output dictionary with the required
        information, including the values array, parameter name, unit, and scale. If 'get_slope' is
        True, it also fetches the slope from calibration. This function is essential for accessing
        calibrated values and related information from a node, which is particularly useful in data
        processing and analysis tasks.
    """
    out_dict = {}
    # out_dict["values"] = None
    out_dict["parname"] = ''
    out_dict["unit"] = ''
    out_dict["scale"] = 0
    out_dict["slope"] = 0.
    out_dict['offset'] = 0.
    out_dict["bitplanes"] = 8
    out_dict["bottom"] = 0
    out_dict["nullInd"] = -1
    out_dict["voidInd"] = -1
    out_dict["simmetric"] = None
    out_dict["scaling"] = 1
    out_dict["channel"] = None
    out_dict["top"] = None
    out_dict["fNull"] = None
    out_dict["str_format"] = ''
    out_dict["maxVal"] = None

    values = None

    if isinstance(node, dpg.node__define.Node):
        calib, values = node.getValues(
            to_create=to_create, recompute=recompute, reload=reload
        )
        # calib_pointer = calib.pointer
        # if calib_pointer is None:
        #     return values, calib, out_dict

        if get_slope:
            tmp, exists, _ = dpg.attr.getAttrValue(calib, "slope", 0.0)
            if exists:
                out_dict["slope"] = tmp
    tmp, exists, _ = dpg.attr.getAttrValue(calib, "parname", "")
    if not exists:
        return values, calib, out_dict
    out_dict["parname"] = tmp

    if values is None:
        _, _, _, _, _, _, _, _, _, values = createArrayValues(calib)

    tmp, exists, _ = dpg.attr.getAttrValue(calib, "unit", "")
    if exists:
        out_dict["unit"] = tmp
    out_dict["scale"], _, _ = dpg.attr.getAttrValue(calib, "scale", 0)

    return values, calib, out_dict


def get_idcalibration(
        node, owner=None, name=None, only_current: bool = False, reload: bool = False
) -> Attr:
    """
    Retrieves the calibration attribute for a given node.

    This function fetches the calibration attribute ('idcalibration') associated with a specified node.
    It allows for specification of the attribute owner, name, and supports options to retrieve only
    the current attribute or to reload the attribute.

    Args:
        node: The node for which the calibration attribute is to be retrieved.
        owner (optional): The owner of the calibration attribute. Defaults to None.
        name (optional): The name of the calibration attribute. Defaults to the value
                         descriptor name from the configuration.
        only_current (bool, optional): If True, retrieves only the current calibration
                                       attribute. Defaults to False.
        reload (bool, optional): If True, reloads the calibration attribute. Defaults to False.

    Returns:
        The calibration attribute object associated with the specified node.

    Note:
        The function first determines the appropriate name for the calibration attribute
        (typically a value descriptor name from the configuration). It then uses this
        information to retrieve the calibration attribute from the node, applying any
        specified constraints such as owner, only_current, and reload options. This
        function is crucial for accessing calibration information for a node, which is
        integral in many data processing and analysis operations.
    """
    name = dpg.cfg.getValueDescName()
    idcalibration = dpg.tree.getAttr(
        node,
        name,
        owner=owner,
        only_current=only_current,
        stop_at_first=True,
        reload=reload,
    )
    return idcalibration


def getValuesFile(node, alt_calId=None):
    """
    Retrieves the file name containing value descriptions associated with a node's calibration.

    This function fetches the name of a file (typically containing value descriptions) that is
    associated with the calibration of a given node. It allows for the use of an alternative
    calibration ID and optionally retrieves the attribute index.

    Args:
        node: The node for which the values file name is to be retrieved.
        alt_calId (optional): An alternative calibration ID to be used if the node's calibration
                              is not found. Defaults to None.
        get_attr_ind (bool, optional): If True, also retrieves the attribute index. Defaults to False.

    Returns:
        str or tuple: The name of the values file associated with the node's calibration.
                      If 'get_attr_ind' is True, returns a tuple containing the file name
                      and the attribute index.

    Note:
        The function first attempts to retrieve the calibration ID for the node using
        'get_idcalibration'. If the calibration ID is not found and an alternative calibration
        ID is provided, it uses the alternative ID. It then fetches the 'valuesfile' attribute
        value from the calibration attribute. This function is useful in scenarios where
        calibration values are stored externally and need to be referenced or accessed.
    """
    valuesfile = "values.txt"
    calId = get_idcalibration(node)
    if calId is None:
        if alt_calId is not None:
            calId = alt_calId
    valuesfile, exists, attr_ind = dpg.attr.getAttrValue(calId, "valuesfile", default=valuesfile)
    return valuesfile, calId, attr_ind


def set_array_values(
        node,
        in_dict: dict,
        file: str = None,
        str_format: str = "",
        rem_inherits: bool = False,
        attr=None,
        others=None,
) -> bool:
    """
    Sets array values and attributes for a given node based on the input dictionary.

    This function updates a node with specified array values and attributes, including calibration
    parameters and other related information. It can also handle the creation of a new attribute file
    and optionally remove inherited attributes.

    Args:
        node: The node to be updated with array values and attributes.
        in_dict (dict): A dictionary containing the values and attributes to be set for the node.
        file (str, optional): The name of the file to store values if required. Defaults to None.
        str_format (str, optional): The string format for the data. Defaults to ''.
        rem_inherits (bool, optional): If True, removes inherited attributes. Defaults to False.
        attr (optional): A specific attribute to be updated. Defaults to None.

    Returns:
        bool: True if the operation was successful, False otherwise.

    Note:
        The function iterates through the input dictionary, applying each key-value pair as an
        attribute to the node. It handles special cases such as 'values', 'unit', 'parname',
        'offset', 'slope', 'minVal', 'maxVal', 'bottom', 'top', 'nullInd', 'voidInd', 'scaling',
        'bitplanes', 'scale', 'simmetric', 'channel', and 'name'. This function is integral in
        scenarios where node attributes need to be dynamically set or updated based on varying data
        or calibration requirements.
    """

    values = in_dict["values"] if "values" in in_dict.keys() else None

    if not isinstance(in_dict, dict):
        return False

    varnames = []
    vars = []
    if "scaling" in in_dict and in_dict["scaling"] is not None:
        sc = in_dict["scaling"]
    else:
        sc = -1

    if (isinstance(values, list) or isinstance(values, np.ndarray)) and sc != 0:
        pData = copy.deepcopy(values)  # pData = ptr_new(values)
        if file is None:
            file, _, _ = getValuesFile(node)
        oAttr = dpg.tree.addAttr(
            node, file, pData, format="VAL", str_format=str_format
        )  # /ONLY_CURRENT #TODO
        if sc != 6 and isinstance(oAttr, dpg.attr__define.Attr):
            oAttr.setProperty(to_not_save=True)
        varnames = varnames + ["valuesfile"]
        vars = file
    else:
        if sc == 0:
            file, _, _ = getValuesFile(node)
            _ = dpg.tree.removeAttr(node, name=file)

    if "unit" in in_dict and in_dict["unit"] is not None:
        varnames = varnames + ["unit"]
        vars = vars + [in_dict["unit"]]

    if "parname" in in_dict and in_dict["parname"] is not None:
        varnames = varnames + ["parname"]
        vars = vars + [in_dict["parname"]]

    if "offset" in in_dict and in_dict["offset"] is not None:
        varnames = varnames + ["offset"]
        vars = vars + [in_dict["offset"]]

    if "slope" in in_dict and in_dict["slope"] is not None:
        varnames = varnames + ["slope"]
        vars = vars + [in_dict["slope"]]

    if "minVal" in in_dict and in_dict["minVal"] is not None:
        if isinstance(in_dict["minVal"], int):
            vvv = int(in_dict["minVal"])
        else:
            vvv = in_dict["minVal"]

        varnames = varnames + ["minVal"]
        vars = vars + [str(vvv)]

    if "maxVal" in in_dict and in_dict["maxVal"] is not None:
        if isinstance(in_dict["maxVal"], int):
            vvv = int(in_dict["maxVal"])
        else:
            vvv = in_dict["maxVal"]

        varnames = varnames + ["maxVal"]
        vars = vars + [str(vvv)]

    if "bottom" in in_dict and in_dict["bottom"] is not None:
        varnames = varnames + ["bottom"]
        vars = vars + [in_dict["bottom"]]

    if "top" in in_dict and in_dict["top"] is not None:
        varnames = varnames + ["top"]
        vars = vars + [in_dict["top"]]

    if (
            "nullInd" in in_dict and np.size(in_dict["nullInd"]) >= 1
    ):  # TODO giusto considerarlo numpy array?
        if in_dict["nullInd"] >= 0:
            ccc = int(in_dict["nullInd"])
        else:
            ccc = -1

        null_ind_labels = ["nullInd"] * np.size(in_dict["nullInd"])

        varnames = varnames + null_ind_labels
        vars = vars + [str(ccc)]

    if (
            "voidInd" in in_dict and np.size(in_dict["voidInd"]) >= 1
    ):  # TODO giusto considerarlo numpy array?
        if in_dict["voidInd"] >= 0:
            ccc = int(in_dict["voidInd"])
        else:
            ccc = -1

        null_ind_labels = ["voidInd"] * np.size(in_dict["voidInd"])

        varnames = varnames + null_ind_labels
        vars = vars + [str(ccc)]

    if "scaling" in in_dict and in_dict["scaling"] is not None:
        varnames = varnames + ["scaling"]
        vars = vars + [in_dict["scaling"]]

    if "bitplanes" in in_dict and in_dict["bitplanes"] is not None:
        varnames = varnames + ["bitplanes"]
        vars = vars + [in_dict["bitplanes"]]

    if "scale" in in_dict and in_dict["scale"] is not None:
        varnames = varnames + ["scale"]
        vars = vars + [in_dict["scale"]]

    if "simmetric" in in_dict and in_dict["simmetric"] is not None:
        varnames = varnames + ["simmetric"]
        vars = vars + [in_dict["simmetric"]]

    if "channel" in in_dict and in_dict["channel"] is not None:
        varnames = varnames + ["channel"]
        vars = vars + [in_dict["channel"]]

    if "name" in in_dict and in_dict["name"] is not None:
        varnames = varnames + ["name"]
        vars = vars + [in_dict["name"]]

    # TODO others viene mai usato??
    if others is not None:
        log_message("DA SISTEMARE IL CASO OTHERS", level="ERROR")
        sys.exit()

    calname = dpg.cfg.getValueDescName()

    _ = dpg.tree.replaceAttrValues(
        node=node,
        name=calname,
        varnames=varnames,
        values=vars,
        only_current=True,
        rem_inherits=rem_inherits,
    )
    if isinstance(attr, dpg.attr__define.Attr):  # TODO or dict?
        _ = dpg.attr.replaceTags(attr, varnames, vars, rem_inherits=rem_inherits)

    return True


def set_array_values_OLD_VERSION(
        node,
        in_dict: dict,
        file: str = None,
        str_format: str = "",
        rem_inherits: bool = False,
        attr=None,
) -> bool:
    """
    Old version
    """

    values = in_dict["values"] if "values" in in_dict.keys() else None

    if not isinstance(in_dict, dict):
        return False

    varnames = None
    if "scaling" in in_dict and in_dict["scaling"] is not None:
        sc = in_dict["scaling"]
    else:
        sc = -1

    if (isinstance(values, list) or isinstance(values, np.ndarray)) and sc != 0:
        pData = copy.deepcopy(values)  # pData = ptr_new(values)
        if file is None:
            file, _, _ = getValuesFile(node)
        oAttr = dpg.tree.addAttr(
            node, file, pData, format="VAL", str_format=str_format
        )  # /ONLY_CURRENT #TODO
        if sc != 6 and isinstance(oAttr, dpg.attr__define.Attr):
            oAttr.setProperty(to_not_save=True)
        varnames = "valuesfile"
        vars = file
    else:
        if sc == 0:
            file, _, _ = getValuesFile(node)
            _ = dpg.tree.removeAttr(node, name=file)

    if "unit" in in_dict and in_dict["unit"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "unit"]
            vars = [vars, in_dict["unit"]]
        elif isinstance(varnames, list):
            varnames.extend("unit")
            vars.extend(in_dict["unit"])
        else:
            varnames = "unit"
            vars = in_dict["unit"]

    if "parname" in in_dict and in_dict["parname"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "parname"]
            vars = [vars, in_dict["parname"]]
        elif isinstance(varnames, list):
            varnames.extend("parname")
            vars.extend(in_dict["parname"])
        else:
            varnames = "parname"
            vars = in_dict["parname"]

    if "offset" in in_dict and in_dict["offset"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "offset"]
            vars = [vars, in_dict["offset"]]
        elif isinstance(varnames, list):
            varnames.extend("offset")
            vars.extend(in_dict["offset"])
        else:
            varnames = "offset"
            vars = in_dict["offset"]

    if "slope" in in_dict and in_dict["slope"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "slope"]
            vars = [vars, in_dict["slope"]]
        elif isinstance(varnames, list):
            varnames.extend("slope")
            vars.extend(in_dict["slope"])
        else:
            varnames = "slope"
            vars = in_dict["slope"]

    if "minVal" in in_dict and in_dict["minVal"] is not None:
        if isinstance(in_dict["minVal"], int):
            vvv = int(in_dict["minVal"])
        else:
            vvv = in_dict["minVal"]
        if isinstance(varnames, str):
            varnames = [varnames, "minVal"]
            vars = [vars, str(vvv)]
        elif isinstance(varnames, list):
            varnames.extend("minVal")
            vars.extend(str(vvv))
        else:
            varnames = "minVal"
            vars = str(vvv)

    if "maxVal" in in_dict and in_dict["maxVal"] is not None:
        if isinstance(in_dict["maxVal"], int):
            vvv = int(in_dict["maxVal"])
        else:
            vvv = in_dict["maxVal"]
        if isinstance(varnames, str):
            varnames = [varnames, "maxVal"]
            vars = [vars, str(vvv)]
        elif isinstance(varnames, list):
            varnames.extend("maxVal")
            vars.extend(str(vvv))
        else:
            varnames = "maxVal"
            vars = str(vvv)

    if "bottom" in in_dict and in_dict["bottom"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "bottom"]
            vars = [vars, in_dict["bottom"]]
        elif isinstance(varnames, list):
            varnames.extend("bottom")
            vars.extend(in_dict["bottom"])
        else:
            varnames = "bottom"
            vars = in_dict["bottom"]

    if "top" in in_dict and in_dict["top"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "top"]
            vars = [vars, in_dict["top"]]
        elif isinstance(varnames, list):
            varnames.extend("top")
            vars.extend(in_dict["top"])
        else:
            varnames = "top"
            vars = in_dict["top"]

    if (
            "nullInd" in in_dict and in_dict["nullInd"].size >= 1
    ):  # TODO giusto considerarlo numpy array?
        if in_dict["nullInd"][0] >= 0:
            ccc = in_dict["nullInd"].astype(int)
        else:
            ccc = " "

        null_ind_labels = ["nullInd"] * len(in_dict["nullInd"])

        if isinstance(varnames, str):
            varnames = [varnames, "nullInd"]
            vars = [vars, in_dict["nullInd"]]
        elif isinstance(varnames, list):
            varnames.extend(null_ind_labels)
            vars.extend([str(ccc)] * len(in_dict["nullInd"]))
        else:
            varnames = null_ind_labels
            vars = [str(ccc)] * len(in_dict["nullInd"])

    if (
            "voidInd" in in_dict and in_dict["voidInd"].size >= 1
    ):  # TODO giusto considerarlo numpy array?
        if in_dict["voidInd"][0] >= 0:
            ccc = in_dict["voidInd"].astype(int)
        else:
            ccc = " "

        null_ind_labels = ["voidInd"] * len(in_dict["voidInd"])

        if isinstance(varnames, str):
            varnames = [varnames, "voidInd"]
            vars = [vars, in_dict["voidInd"]]
        elif isinstance(varnames, list):
            varnames.extend(null_ind_labels)
            vars.extend([str(ccc)] * len(in_dict["voidInd"]))
        else:
            varnames = null_ind_labels
            vars = [str(ccc)] * len(in_dict["voidInd"])

    if "scaling" in in_dict and in_dict["scaling"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "scaling"]
            vars = [vars, in_dict["scaling"]]
        elif isinstance(varnames, list):
            varnames.extend("scaling")
            vars.extend(in_dict["scaling"])
        else:
            varnames = "scaling"
            vars = in_dict["scaling"]

    if "bitplanes" in in_dict and in_dict["bitplanes"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "bitplanes"]
            vars = [vars, in_dict["bitplanes"]]
        elif isinstance(varnames, list):
            varnames.extend("bitplanes")
            vars.extend(in_dict["bitplanes"])
        else:
            varnames = "bitplanes"
            vars = in_dict["bitplanes"]

    if "scale" in in_dict and in_dict["scale"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "scale"]
            vars = [vars, in_dict["scale"]]
        elif isinstance(varnames, list):
            varnames.extend("scale")
            vars.extend(in_dict["scale"])
        else:
            varnames = "scale"
            vars = in_dict["scale"]

    if "simmetric" in in_dict and in_dict["simmetric"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "simmetric"]
            vars = [vars, in_dict["simmetric"]]
        elif isinstance(varnames, list):
            varnames.extend("simmetric")
            vars.extend(in_dict["simmetric"])
        else:
            varnames = "simmetric"
            vars = in_dict["simmetric"]

    if "channel" in in_dict and in_dict["channel"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "channel"]
            vars = [vars, in_dict["channel"]]
        elif isinstance(varnames, list):
            varnames.extend("channel")
            vars.extend(in_dict["channel"])
        else:
            varnames = "channel"
            vars = in_dict["channel"]

    if "name" in in_dict and in_dict["name"] is not None:
        if isinstance(varnames, str):
            varnames = [varnames, "name"]
            vars = [vars, in_dict["name"]]
        elif isinstance(varnames, list):
            varnames.extend("name")
            vars.extend(in_dict["name"])
        else:
            varnames = "name"
            vars = in_dict["name"]

    # TODO others viene mai usato??
    """
    if n_elements(others) gt 0 then begin
        if n_elements(others) eq n_elements(tagOthers) then begin
            if n_elements(varnames) ne 0 then begin
                ptr = ptr_new(CreateAttrStruct(tagOthers, others))
                nt = RemoveTags(ptr, varnames)
                varnames = [varnames, (*ptr).(0)]
                vars = [vars, (*ptr).(1)]
                ptr_free, ptr
            endif else begin
                varnames = tagOthers
                vars = string(others)
            endelse
        endif
    endif
    """

    calname = dpg.cfg.getValueDescName()

    _ = dpg.tree.replaceAttrValues(
        node, calname, vars, only_current=True, rem_inherits=rem_inherits
    )
    if isinstance(attr, dpg.attr__define.Attr):  # TODO or dict?
        _ = dpg.attr.replaceTags(attr, varnames, vars, rem_inherits=rem_inherits)

    return True


def set_values(
        node,
        calib: Attr = None,
        alt_node=None,
        bottom=None,
        nullInd=None,
        voidInd=None,
        rem_inherits=None,
) -> bool:
    """
    Sets values and attributes for a given node based on calibration information.

    This function updates a node with values and attributes derived from a calibration object.
    It allows for alternative nodes to be specified for additional value derivation and supports
    the specification of bottom, null, and void indices.

    Args:
        node: The node to be updated with values and attributes.
        calib (dpg.attr__define.Attr): The calibration attribute object containing calibration information.
        alt_node (dpg.node__define.Node, optional): An alternative node for deriving additional values.
                                                    Defaults to None.
        bottom (optional): The bottom index for value assignment. Defaults to None.
        nullInd (optional): The index to be assigned NaN. Defaults to None.
        voidInd (optional): The index to be assigned negative infinity. Defaults to None.
        rem_inherits (bool, optional): If True, removes inherited attributes. Defaults to False.

    Returns:
        bool: True if the operation was successful, False otherwise.

    Note:
        The function first extracts calibration parameters from the 'calib' object, such as the parameter
        name, values, and slope. If alternative nodes are specified, it attempts to derive additional
        values from them. The function then updates the given node with these values and attributes,
        taking into account the specified bottom, null, and void indices. It is integral in scenarios
        where nodes require dynamic updating based on calibration data or when merging information from
        multiple sources.
    """
    # input param: node, calib, alt_node, bottom, bottom, voidInd, rem_inherits
    # output param: bool

    btm = None
    nInd = None
    vInd = None
    offset = None
    slope = None
    unit = None
    parname = None
    maxVal = None
    bitplanes = None
    scale = None
    simmetric = None
    scaling = None
    str_format = None

    if isinstance(calib, dpg.attr__define.Attr) and calib is not None:
        values, calib, out_dict = get_array_values(None, calib, getMinMaxVal=True)
        if out_dict["parname"] is None or out_dict["parname"] == "":
            return None
        unit = out_dict["unit"]
        parname = out_dict["parname"]
        offset = out_dict["offset"]
        slope = out_dict["slope"]
        maxVal = out_dict["maxVal"]
        bitplanes = out_dict["bitplanes"]
        scale = out_dict["scale"]
        simmetric = out_dict["simmetric"]
        btm = out_dict["bottom"]
        nInd = out_dict["nullInd"]
        vInd = out_dict["voidInd"]
        scaling = out_dict["scaling"]
        str_format = out_dict["str_format"]

    if bottom is None and btm is not None:
        bottom = btm
    if nullInd is None and nInd is not None:
        nullInd = nInd
    if voidInd is None and vInd is not None:
        voidInd = vInd

    if isinstance(alt_node, dpg.node__define.Node):
        values, _, out_dict = get_values(alt_node, to_create=True, get_slope=True)
        if values is not None:
            maxVal = np.nanmax(np.ma.masked_invalid(values))
            minVal = np.nanmin(np.ma.masked_invalid(values))
            slp = out_dict["slope"]
            if offset is None:
                offset = minVal
            if slope is None and slp is not None:
                slope = slp

    in_dict = {}
    in_dict["unit"] = unit
    in_dict["parname"] = parname
    in_dict["offset"] = offset
    in_dict["slope"] = slope
    in_dict["maxVal"] = maxVal
    in_dict["bottom"] = bottom
    in_dict["bitplanes"] = bitplanes
    in_dict["scale"] = scale
    in_dict["simmetric"] = simmetric
    in_dict["nullInd"] = nullInd
    in_dict["voidInd"] = voidInd
    in_dict["scaling"] = scaling
    in_dict["str_format"] = str_format
    in_dict["rem_inherits"] = rem_inherits

    result = set_array_values(node, in_dict, rem_inherits=rem_inherits)

    return result


def copy_calibration(fromNode, toNode, to_save: bool = False):
    """
    Copies calibration attributes from one node to another.

    This function is used to replicate the calibration attributes from a source node
    to a destination node. It provides an option to save the changes to the destination node.

    Args:
        fromNode: The source node from which calibration attributes are to be copied.
        toNode: The destination node to which calibration attributes are to be copied.
        to_save (bool, optional): If True, saves the destination node after copying the attributes.
                                  Defaults to False.

    Returns:
        None: This function does not return any value.

    Note:
        The function utilizes 'dpg.tree.copyAttr' to perform the attribute copying, targeting
        the calibration attributes as defined by 'dpg.cfg.getValueDescName'. This functionality
        is particularly useful in scenarios where calibration settings need to be replicated across
        multiple nodes for consistency in data processing or analysis.
    """
    _, _ = dpg.tree.copyAttr(
        fromNode, toNode, dpg.cfg.getValueDescName(), to_save=to_save
    )


def timeToForecast(path, time, maxHours, delta=None, prefix=None):
    """
    Searches for the path of data to be imported, based on the time and maxHours parameter given

    Args:
        path: The path to the directory containing the data.
        time: The time of the data to be imported.
        maxHours: The maximum amount of hours to span if no data is found at current time.
        delta:
        prefix: The prefix of the data directory to be imported.

    Returns:
        sub: The path of the directory containing the data to be imported. If no directory is found,
            it returns an empty string.
    """
    if prefix is None:
        prefix = "FC_"
    time, hh, _ = dpg.times.checkTime(time)
    if delta is not None:
        hh += delta
    if maxHours is None:
        maxHours = 24

    ii = 0
    while hh > 0 and ii < maxHours:
        hhh = str(hh).strip()
        if prefix.startswith("DF"):
            kkk = str(hh + 6).strip()
            strSub = prefix + hhh + "-" + kkk + "hr"
        else:
            strSub = prefix + hhh + "hr"

        sub = os.path.join(path, strSub)
        log_message(f"Trying {sub}", level="INFO")

        if os.path.isdir(sub):
            return sub

        hh -= 1
        ii += 1

    return ""


def getModelPath(
        current,
        searchPath=None,
        date=None,
        time=None,
        parName=None,
        prefix=None,
        maxHours=None,
):
    """
    Searches for the path of data to be imported. Calls the function timeToForecast to find the data nearest
    to a given time. If no data is found for the provided date, it searches for data in the previous date.

    Args:
        current: The current Node.
        searchPath: The path to the parent directory where the data to be imported is located.
        date: The date to which the data refers.
        time: The time to which the data refers.
        parName: A string containing the name of the subdirectory of data to search.
        prefix: The prefix of the data directory to be imported.
        maxHours: The maximum amount of hours to span if no data is found at current time.

    Returns:
        fpath: The path of the directory containing the data to be imported.
    """
    mPath = ""
    if dpg.tree.node_valid(current):
        date, time, _ = dpg.times.get_time(current)
    if searchPath is not None:
        mPath = searchPath
    else:
        mPath, _, _ = dpg.radar.get_par(
            current, "searchPath", "", parFile=dpg.cfg.getScheduleDescName()
        )

    if mPath == "":
        mPath = dpg.path.getDir("MODELS", with_separator=True)

    path = dpg.path.checkPathname(mPath, with_separator=False)

    if date is not None:
        strDate = dpg.times.checkDate(date, sep=os.sep, year_first=True)
        strTime = "0000"
        path = os.path.join(path, strDate, strTime)

    if parName is None:
        return path

    if maxHours is None:
        maxHours, _, _ = dpg.radar.get_par(current, "searcHours", 3)

    path = os.path.join(path, parName)
    fpath = timeToForecast(path, time, maxHours, prefix=prefix)

    if fpath != "":
        log_message(f"...using {fpath}", level="INFO")
        return fpath

    if date is None:
        return path

    # Da controllare questo ramo, perché la funzione getPrevDay va sistemata
    # ritorna la data in un formato sbagliato per l'utilizzo che viene fatto qui
    log_message(f"Cannot find forecast in {path}", level="INFO")
    prev = dpg.times.getPrevDay(date)
    strDate = dpg.times.checkDate(prev, sep=os.sep, year_first=True)
    strTime = "0000"
    path = os.path.join(mPath, strDate, strTime, parName)
    fpath = timeToForecast(path, time, maxHours, delta=24, prefix=prefix)
    if fpath == "":
        return fpath

    log_message(f"...using {fpath}", level="INFO")
    return fpath


def get_temperature(R, vc, A, B, unit=None):
    """
    Converts radiance to temperature using calibration coefficients and optionally adjusts the temperature
    to a specified unit (Celsius or Pixel).

    Args:
        R: The radiance value.
        vc: Calibration coefficient for wavelength (usually in microns).
        A: Calibration coefficient A.
        B: Calibration coefficient B.
        unit: Desired unit for the temperature ('C' for Celsius, 'PIXEL' for pixel-based unit, default is None).

    Returns:
        t: The computed temperature in the specified unit.
        unit: The unit of the temperature.
    """
    # h = 6.626176e-34 plank
    # c = 299792458 speed
    # k = 1.380662e-23 boltzman
    # C1 = 2. * h * (c**2)  1.1910620e-016
    # C2 = h * c / k         0.014387863

    if vc <= 0.0:
        return R

    C1 = 1.1910400e-05
    C2 = 1.4387700
    C1v = C1 * (vc ** 3)
    C2v = C2 * vc

    cv = C1v / R
    den = np.log(cv + 1.0)
    t = C2v / den
    toCent = 0.0

    if unit is not None:
        if unit.upper() == "C":
            toCent = 273.15
        elif unit.upper() == "PIXEL":
            unit = "K"

    if A > 0.0:
        t -= B
        t /= A

    t -= toCent
    t = t.astype(np.float32)
    return t, unit


def get_calib_par(channel, origin=None):
    """
    Fetches the calibration parameters for a specific channel by loading them from a calibration file.

    Parameters:
        - channel: The channel name (e.g., 'channel1').
        - origin: Optional origin path or source to fetch the calibration from (default is None).

    Returns:
        - coeffs: A list of calibration coefficients (vc, a, b, calCoeff, maxSZA).
        - origin: The origin path used for calibration (if provided).
        - name: The name of the calibration.
        - par: The complete parameters of the calibration.
        - tagPar: Additional tags for the calibration.
        - defParName: Default parameter name.
        - defParUnit: Default unit of the parameter.
        - slopeBits: The number of bits for slope information.
        - auxCalChan: Auxiliary calibration channel if available.
    """
    path = dpg.path.getDir("CALIBRATIONS", sep=os.sep, with_separator=True)
    if origin is not None:
        path = os.path.join(path, origin)
    attr = dpg.attr.loadAttr(path, channel + ".txt")
    slopeBits, _, _ = dpg.attr.getAttrValue(attr, "slopeBits", 10.0)
    if attr is None:
        return 0
    tagPar, par, _ = dpg.attr.getAllTags(attr)
    name, _, _ = dpg.attr.getAttrValue(attr, "name", "")
    defParName, _, _ = dpg.attr.getAttrValue(attr, "parname", "")
    defParUnit, _, _ = dpg.attr.getAttrValue(attr, "unit", "")
    calCoeff, _, _ = dpg.attr.getAttrValue(attr, "calCoeff", 0.0)
    maxSZA, _, _ = dpg.attr.getAttrValue(attr, "maxSZA", 80.0)
    vc, _, _ = dpg.attr.getAttrValue(attr, "vc", 0.0)
    a, _, _ = dpg.attr.getAttrValue(attr, "a", 0.0)
    b, _, _ = dpg.attr.getAttrValue(attr, "b", 0.0)
    coeffs = [vc, a, b, calCoeff, maxSZA]
    aux, exists, _ = dpg.attr.getAttrValue(attr, "auxCalChan", "")
    auxCalChan = aux if exists else None

    return (
        coeffs,
        origin,
        name,
        par,
        tagPar,
        defParName,
        defParUnit,
        slopeBits,
        auxCalChan,
    )


def rad_to_temp(rad, channel, unit=None, origin=None):
    """
    Converts radiance values to temperature based on the calibration parameters of the given channel.

    Args:
        - rad: The radiance values (can be a numpy array).
        - channel: The calibration channel name.
        - unit: The unit for the temperature ('C' for Celsius, 'PIXEL' for pixel-based unit, default is None).
        - origin: The origin source of calibration data (default is None).

    Returns:
        - temp: The computed temperature corresponding to the radiance.
        - unit: The unit for the temperature ('C' or 'PIXEL').
        - origin: The source of the calibration.
    """

    if np.size(rad) < 1 or np.size(channel) != 1:
        return
    if origin is None:
        origin = "MSG3"
    if unit is None:
        unit = "C"

    # Da mappe di radianza a temperatura
    calpar, origin, _, _, _, _, _, _, _ = get_calib_par(channel, origin=origin)
    if np.size(calpar) < 3:
        log_message(f"Cannot find calibration coeffs for {channel}", "WARNING")
        return
    temp, unit = get_temperature(rad, calpar[0], calpar[1], calpar[2], unit=unit)

    return temp, unit, origin


def remove_array_values(node, clean=False):
    # TODO: Aggiungere head comment
    valuesfile, calId, attr_ind = getValuesFile(node)
    dpg.tree.removeAttr(node, name=valuesfile, delete_file=True)
    if clean:
        dpg.tree.removeAttr(node, name=dpg.cfg.getValueDescName(), delete_file=True)
        return
    if calId is not None:
        dpg.attr.removeTags(calId[attr_ind], 'valuesfile')

    return
