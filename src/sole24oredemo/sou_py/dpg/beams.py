import math
from cmath import sin
import sou_py.dpg as dpg
import numpy as np

"""
Funzioni ancora da portare
FUNCTION GetBeamIndexWithinHeight 
FUNCTION GetProjectedBeamIndex 
PRO ComputeBeamIndexAndWeights 
PRO ComputeWeights 
FUNCTION GetWeightBeam              // UNUSED
FUNCTION IDL_rv_get_ppi_heights     // UNUSED
FUNCTION IDL_rv_get_surf_array      // UNUSED 
PRO PROJECT_BEAM                    // UNUSED
"""


def getHeightBeam(
    rangeRes: float,
    nBins: int,
    elevation: float,
    ECC: float = 1,
    site_height: float = None,
    projected: bool = False,
    range_off: float = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Calculates the height of a radar beam based on various radar parameters.

    This function computes the height of each bin in a radar beam given the range
    resolution, number of bins, and elevation angle. It supports adjustments for
    Earth curvature correction (ECC), site height, and range offset. Optionally,
    heights can be calculated in projected mode, considering the beam's elevation
    angle.

    Args:
        rangeRes (float): The range resolution of the radar in meters.
        nBins (int): The number of bins along the radar beam.
        elevation (float): The elevation angle of the radar beam in degrees.
        ECC (float, optional): Earth curvature correction factor. Defaults to 1.
        site_height (float, optional): The height of the radar site above mean sea
                                       level in meters. Defaults to None.
        projected (bool, optional): If True, adjusts the slant range for elevation
                                    angle to project onto a flat surface. Defaults
                                    to False.
        range_off (float, optional): The range offset to be applied. Defaults to None.
        dtype (data-type, optional): The desired data-type for the calculations
                                     (e.g., np.float32). Defaults to np.float32.

    Returns:
        np.ndarray: An array of heights for each bin in the radar beam, calculated
                    based on the provided parameters.

    Note:
        The function first calculates the slant range for each bin using 'dpg.map.getPolarRange'.
        In 'projected' mode, it adjusts the slant range based on the elevation angle.
        The final height of each bin is then computed using 'dpg.map.slantRangeToHeight',
        taking into account Earth curvature, site height, and additional radar parameters.
        This function is essential in radar data processing to determine the geometric
        height of radar targets.
    """
    slant_range = dpg.map.getPolarRange(
        rangeRes, nBins, range_off=range_off, dtype=dtype
    )
    if projected:
        cosEl = np.cos(elevation * np.pi / 180, dtype=dtype)  # !DTOR*elevation
        if cosEl < 0.1:
            cosEl = 0.1
        slant_range /= cosEl
    # endif
    h = dpg.map.slantRangeToHeight(
        slant_range, elevation, ECC=ECC, site_height=site_height
    )
    return h, slant_range


def getRangeBeamIndex(
    inDim: int,
    outDim: int,
    inRes: float,
    outRes: float,
    in_el: float = None,
    out_el: float = None,
    range_off: float = None,
) -> np.ndarray:
    """
    Calculates the index mapping between two sets of radar ranges with different resolutions and elevations.

    This function computes an array of indices that maps radar data from one set of
    dimensions and resolution to another. It accounts for changes in resolution and
    adjustments due to radar beam elevation angles. The function is useful for
    re-gridding or interpolating radar data from one resolution or geometry to another.

    Args:
        inDim (int): The number of bins in the input radar data.
        outDim (int): The desired number of bins in the output radar data.
        inRes (float): The range resolution of the input radar data in meters.
        outRes (float): The desired range resolution for the output radar data in meters.
        in_el (float, optional): The elevation angle of the input radar beam in degrees. Defaults to None.
        out_el (float, optional): The elevation angle of the output radar beam in degrees. Defaults to None.
        range_off (float, optional): The range offset to be applied. Defaults to None.

    Returns:
        np.ndarray: An array of indices that map from the input range to the output range.

    Note:
        The function first adjusts the output range resolution based on the output elevation
        angle. It then calculates the polar range for the output dimensions using 'dpg.map.getPolarRange'.
        Next, it adjusts the input range resolution based on the input elevation angle and
        scales the index array accordingly. The function handles cases where the index
        exceeds the input dimensions, setting such indices to -1 to indicate invalid mapping.
    """
    fattRes = outRes
    if np.size(out_el) == 1 and out_el is not None:
        fattRes *= np.cos(np.deg2rad(out_el))

    index = dpg.map.getPolarRange(fattRes, outDim, range_off=range_off)

    fattRes = inRes
    if np.size(in_el) == 1 and in_el is not None:
        fattRes *= np.cos(np.deg2rad(in_el))

    if fattRes is not None and fattRes > 0:
        index /= fattRes
        index = index.astype(int)
        ind = np.where(index >= inDim)
        if np.size(ind) > 0:
            index[ind] = -1
    else:
        index[:] = -1

    return index


def getBeamIndex(range, rangeRes: float, range_off: float = None):
    """
    Calculates the beam index for a given radar range and resolution.

    This function computes the index of a radar beam based on the radar range and range
    resolution. It optionally adjusts for a range offset before calculating the index.
    This function is typically used to map a physical range to its corresponding index
    in a radar data array.

    Args:
        range (np.ndarray or float): The physical range(s) for which the beam index is to be calculated.
        rangeRes (float): The range resolution of the radar system in meters.
        range_off (float, optional): An optional range offset to be subtracted from the
                                     input range before index calculation. Defaults to None.

    Returns:
        np.ndarray or int: The calculated beam index (or indices) corresponding to the
                           input range(s). The data type is int.

    Note:
        The function first adjusts the input range by subtracting the range offset if it
        is provided. It then calculates the beam index by dividing the adjusted range
        by the range resolution and taking the floor of the result. This calculation
        reflects the position of the range within the radar system's discretized range bins.
    """
    index = range
    if range_off is not None:
        index -= range_off
    return np.floor(index / rangeRes).astype(int)


def slantRangeToOrizRange(slantRange, elevationDeg: float, ecc: float = None):
    """
    Converts slant range to horizontal (or ground) range for radar data.

    This function calculates the horizontal range from a given slant range and elevation
    angle, considering the Earth's curvature. It optionally applies an Earth curvature
    correction (ECC) factor.

    Args:
        slantRange (np.ndarray or float): The slant range distance(s) in meters.
        elevationDeg (float): The elevation angle in degrees.
        ecc (float, optional): Earth curvature correction factor. If set to 0, curvature
                               is ignored. If None, it defaults to 1. Defaults to None.

    Returns:
        np.ndarray or float: The horizontal (or ground) range corresponding to the
                             given slant range and elevation angle.

    Note:
        The function first calculates the sine and cosine of the elevation angle. It then
        computes the horizontal distance by multiplying the slant range with the cosine
        of the elevation. If the ECC is not 0, the function additionally calculates the
        height component of the slant range and adjusts the horizontal range based on the
        Earth's radius. This calculation is important in radar systems for converting
        slant range measurements into actual ground distances.
    """
    """
    ; Punto X di intersezione tra la retta passante per i punti
    ; P1 = [sr*cos(el) , sr*sin(el)]
    ; P0 = [   0       ,    -er    ]
    ; con la retta Y = 0.
    ; Equazione della retta Y = mX + q
    ; m = (er + sr*sin(el)) / er*cos(el)
    ; q = -er
    ; da cui
    ; X = -q/m
    """
    sinE1 = np.sin(np.radians(elevationDeg))
    cosE1 = np.cos(np.radians(elevationDeg))

    if ecc is None:
        ecc = 1
    dist = slantRange * cosE1
    if ecc == 0:
        return dist

    height = slantRange * sinE1
    er = dpg.map.getEarthRadius()

    num = dist * er
    den = height * er
    orizRange = num / den

    return orizRange


def heightToSlantRange(
    height: float, elevationDeg: float, ecc: float = None, site_height: float = None
) -> float:
    """
    Converts a height above ground to slant range in a radar system.

    This function calculates the slant range corresponding to a given height above the
    ground level, considering the elevation angle of the radar beam. It optionally
    applies an Earth curvature correction (ECC) factor and takes into account the radar
    site's height above mean sea level.

    Args:
        height (float): The height above ground level in meters.
        elevationDeg (float): The elevation angle of the radar beam in degrees.
        ecc (float, optional): Earth curvature correction factor. If set to 0, curvature
                               is ignored. If None, it defaults to 1. Defaults to None.
        site_height (float, optional): The height of the radar site above mean sea level
                                       in meters. Defaults to 0.0.

    Returns:
        float: The calculated slant range in meters.

    Note:
        The function calculates the sine of the elevation angle and adjusts the input
        height by subtracting the radar site's height. If ECC is set to 0, a simplified
        calculation is used, ignoring Earth's curvature. Otherwise, the Earth's radius
        is factored in to compute the slant range accurately. This calculation is crucial
        for radar systems in determining the distance to an object located at a certain
        height above the ground.
    """
    sinE1 = np.sin(np.radians(elevationDeg))
    if ecc is None:
        ecc = 1
    if site_height is None:
        site_height = 0.0

    hhh = height - site_height

    if ecc == 0:
        slantRange = hhh / sinE1
        if slantRange < 0:
            return 0.0
        return slantRange

    er = dpg.map.getEarthRadius()
    bbb = sinE1 * 2.0 * er
    ccc = hhh * hhh + hhh * 2.0 * er
    delta = bbb * bbb + 4.0 * ccc
    slantRange = (np.sqrt(delta) - bbb) / 2

    if slantRange < 0:
        return 0.0

    return slantRange


def orizRangeToSlantRange(
    orizRange: float, elevationDeg: float, ecc: float = None
) -> float:
    """
    Converts horizontal (ground) range to slant range in a radar system.

    This function calculates the slant range from a given horizontal range and elevation
    angle, considering the Earth's curvature. It optionally applies an Earth curvature
    correction (ECC) factor.

    Args:
        orizRange (float): The horizontal range distance in meters.
        elevationDeg (float): The elevation angle of the radar beam in degrees.
        ecc (float, optional): Earth curvature correction factor. If set to 0, curvature
                               is ignored. If None, it defaults to 1. Defaults to None.

    Returns:
        float: The calculated slant range corresponding to the given horizontal range
               and elevation angle.

    Note:
        The function first calculates the sine and cosine of the elevation angle. If the
        ECC is set to 0, a simplified calculation is performed that directly converts the
        horizontal range to slant range using trigonometric relationships. Otherwise, the
        Earth's radius is factored in to adjust for the Earth's curvature effect. This
        calculation is essential in radar systems to determine the actual slant distance
        to a target based on its observed horizontal distance.
    """
    sinE1 = np.sin(np.radians(elevationDeg))
    cosE1 = np.cos(np.radians(elevationDeg))

    if ecc is None:
        ecc = 1

    if ecc == 0:
        slantRange = orizRange / cosE1
        return slantRange

    er = dpg.map.getEarthRadius()
    num = orizRange * er
    height = orizRange * sinE1
    den = er * cosE1 + height
    slantRange = num / den

    return slantRange


def getAzimutBeamIndex(
    nAzIn,
    nAzOut,
    azOffIn,
    azOffOut,
    azResIn,
    azResOut,
    az_coords_in=None,
    tolerance=None,
):
    """
    Parameters:
        nAzIn, nAzOut: Number of azimuth points in input and output.
        azOffIn, azOffOut: Azimuth offset for input and output.
        azResIn, azResOut: Azimuth resolution for input and output.
        az_coords_in: Input azimuth coordinates (optional).
        tolerance: Tolerance for index matching (optional).

    Returns:
        Output from GetAzIndex function.
    """
    # Create an array of azimuth values for output
    azimut = np.arange(nAzOut) * azResOut + (azOffOut + azResOut / 2.0)

    return dpg.map.get_az_index(
        azimut, azResIn, azOffIn, az_coords_in, max_index=nAzIn, tolerance=tolerance
    )


def getWidthBeam(
    slantRange: float,
    width: float,
    elevation: float,
    site_height: float,
    half: bool = False,
):
    """
    This method calculates two values, up and down, which represent distances or angles related to a slant range (slantRange) and a certain elevation.

    Args:
        slantRange (float): The calculated slant range corresponding to the given horizontal range and elevation angle.
        width (float): Angle used for the calculation of the "angle" variable.
        elevation (float): The elevation angle of the radar beam in degrees.
        site_height (float): The height of the radar site above mean sea level in meters.

    :keywords:
        - half (boolean): Flag to approve or not the calculation for the "angle" variable with half width.

    :return:
        - up (float): Distance calculated on the beam direction.
        - down (float): Distance calculated on the beam direction.
    """

    down = None
    if half is not None:
        angle = (width / 2) * (np.pi / 180)
    else:
        angle = width * (np.pi / 180)

    num = slantRange * np.sin(angle)

    R = dpg.map.getEarthRadius(real=True)
    sinteta = np.sin(elevation * (np.pi / 180))
    if sinteta < 0:
        sinteta = 0
    costeta = np.cos(elevation * (np.pi / 180))
    if np.size(site_height) == 1:
        R += site_height
    gamma = np.arctan((R * costeta) / (slantRange + (R * sinteta)))

    up = gamma + angle
    up = num / np.sin(up)

    if half is not None:
        down = gamma - angle
        down = num / np.sin(down)

    return up, down
