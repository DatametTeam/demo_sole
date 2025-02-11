import os

import numpy as np

import sou_py.dpb as dpb
import sou_py.dpg as dpg
import sou_py.preprocessing as preprocessing
from sou_py.dpg.log import log_message


def get_blockage(dem, altitude, elevation, range_res, beam_width):
    '''
    This function calculates the radar beam blockage based on a digital elevation model (DEM),
    radar altitude, elevation angle, range resolution, and beam width. It determines how much
    of the radar signal is blocked by terrain at different ranges and angles.

    Args:
        dem:           2D array of terrain heights (Digital Elevation Model) in meters.
        altitude:      Radar site altitude (height of the radar above sea level) in meters.
        elevation:     Radar elevation angle in degrees.
        range_res:     Range resolution of the radar in meters.
        beam_width:    Beam width of the radar in degrees.

    Returns:
        blockage (np.ndarray): 2D array of blockage values, representing the fraction
        of the radar beam blocked by the terrain. Values range
        between 0 (no blockage) and 1 (completely blocked).
    '''

    dim = dem.shape
    blockage = np.zeros(dim, dtype=np.float32)
    hhh = np.zeros(dim, dtype=np.float32)
    range_arr = np.zeros(dim, dtype=np.float32)

    heightBeam, slant_range = dpg.beams.getHeightBeam(
        range_res, dim[-1], elevation=elevation, site_height=altitude, ECC=True
    )

    for aaa in range(dim[0]):
        hhh[aaa, :] = heightBeam
        range_arr[aaa, :] = slant_range

    radius = range_arr * (beam_width * np.pi / 180.0) / 2.0
    yyy = dem - hhh
    index = np.where(yyy > -radius)
    if len(index[0]) <= 0:
        return blockage

    # Per rimuovere il warning si verifica che sqrt non sia applicato su valori negativi
    sqrt_input = radius[index] ** 2 - yyy[index] ** 2
    if np.any(sqrt_input < 0):
        sqrt_input[sqrt_input < 0] = 0

    # Per rimuovere il warning si obbliga ad avere l'input di arcsin compreso tra -1 e 1
    arcsin_input = np.clip(yyy[index] / radius[index], -1.0, 1.0)

    blockage[index] = (
                              yyy[index] * np.sqrt(sqrt_input)
                              + radius[index] ** 2.0 * np.arcsin(arcsin_input)
                              + 0.5 * np.pi * radius[index] ** 2
                      ) / (np.pi * radius[index] ** 2)

    index = np.where(blockage > 1)
    blockage[index] = 1
    index = np.where(blockage < 0)
    blockage[index] = 0
    index = np.where(~np.isfinite(blockage))
    blockage[index] = 0

    for rrr in range(dim[-1]):
        index = np.where(blockage[:, rrr] < blockage[:, rrr - 1])
        blockage[index, rrr] = blockage[index, rrr - 1]

    return blockage


def pbb_corr(
        scanId, threshold, blockage=np.array(()), dem_path=None, dim=None, tolerance=None
):
    '''
    Apply Partial Beam Blockage (PBB) correction to radar scan data based on terrain blockage.

    Args:
        scanId (object):       Identifier for the radar scan to be corrected.
        threshold (float):     Threshold value for the blockage correction. Cells with blockage above this value
                               will be corrected up to the threshold.
        blockage (np.ndarray): (Optional) Precomputed blockage array. If not provided, it will be calculated.
        dem_path (str):        (Optional) Path to the digital elevation model (DEM) used to compute blockage.
        dim (tuple):           (Optional) Dimensions of the radar scan. If not provided, it will be inferred.
        tolerance (float):     (Optional) Tolerance to adjust the DEM values (e.g., subtracting tolerance from heights).

    Returns:
        blockage (np.ndarray): The updated blockage array after correction.
        no_corr (int):         Indicator whether a correction was applied (0 if corrected, 1 if not).
        dim (tuple):           Dimensions of the radar scan data.
    '''

    no_corr = 1
    if np.size(blockage) > 0:
        tmp = blockage.copy()

    radar_dict = dpg.navigation.get_radar_par(scanId, get_az_coords_flag=True)
    site_coords = radar_dict["site_coords"]
    range_res = radar_dict["range_res"]
    az_coords = radar_dict["az_coords"]
    elevation = radar_dict["elevation_off"]
    beam_width = radar_dict["beam_width"]
    map = radar_dict["map"]
    par = radar_dict["par"]

    if threshold > 0:
        _, _, dim, _, _, _, _, _ = scanId.getArrayInfo()
        if dim is None:
            return
    else:
        if len(dim.shape) < 2:
            return
        if len(az_coords.shape) > 0:
            tmp = az_coords.copy()
        par[8] = 0

    dem = dpg.warp.get_dem(
        destMap=map,
        destPar=par,
        destDim=dim,
        path=dem_path,
        numeric=True,
        hr=True,
        az_coords=az_coords,
    )

    if np.size(dem) < 1:
        return

    if tolerance:
        dem -= tolerance

    blockage = get_blockage(dem, site_coords[2], elevation, range_res, beam_width)

    if threshold <= 0:
        return
    index = np.where(blockage > 0)
    if len(index[0]) <= 0:
        return blockage, no_corr, dim
    data = dpb.dpb.get_data(scanId, linear=True)

    if np.size(data) != np.size(blockage):
        return blockage, no_corr, dim

    index = np.where(blockage > threshold)
    blockage[index] = threshold

    data /= 1 - blockage

    dpb.dpb.put_data(scanId, data, linear=True)
    no_corr = 0

    log_message(f"PBB: corrected {len(index[0])} cells")

    return blockage, no_corr, dim


def pbb(prodId, lbm=None, volume=np.array(()), moment=None, main=None):
    '''
    Perform Partial Beam Blockage (PBB) correction or calculation on radar scan data.

    Args:
        prodId (object):      Identifier for the radar product to be processed.
        lbm (np.ndarray):     (Optional) Blockage map used for correction. If not provided, it will be computed.
        volume (np.ndarray):  (Optional) Radar data volume. If not provided, it will be retrieved automatically.
        moment (str):         (Optional) Radar moment (e.g., reflectivity). Used when retrieving volume data.
        main (object):        (Optional) Reference to the main radar scan node.

    Returns:
        None
    '''

    dem_path = None
    dim = None

    if not isinstance(prodId, dpg.node__define.Node):
        return

    max_el, _ = dpb.dpb.get_par(prodId, "max_el", default=10.0)

    if max_el < 0:
        log_message("max_el < 0 ... PBB ignored.", level="WARNING")
        return

    threshold, site = dpb.dpb.get_par(prodId, "threshold", default=0.0)
    tolerance, _ = dpb.dpb.get_par(prodId, "tolerance", default=0.0)
    create_dem, _ = dpb.dpb.get_par(prodId, "create_dem", default=0)
    maxVal, _ = dpb.dpb.get_par(prodId, "maxVal", default=100.0)

    home = os.path.join(dpg.cfg.getClutterHome(), site, "DEM")
    if os.path.isdir(home):
        dem_path = home

    if lbm is not None:
        if np.size(lbm) > 0:
            tmp = lbm.copy()

        set_null, _ = dpb.dpb.get_par(prodId, "set_null", default=0)
        if np.size(volume) <= 1:
            dpb.dpb.get_last_volume(
                prodId, moment, volume, main=main, projected=True
            )  # TODO: da implementare
        if np.size(volume) <= 1:
            volume, _, node = preprocessing.sampling.sampling(
                prodId=prodId, projected=True, moment=moment, here=True, get_volume=True
            )
            if np.size(volume) <= 1:
                return

        radar_dict = dpg.navigation.get_radar_par(
            main, get_el_coords_flag=True
        )  # TODO: da controllare perchè forse non torna map
        map = radar_dict["map"]
        par = radar_dict["par"]
        el_coords = radar_dict["el_coords"]
        site_coords = radar_dict["site_coords"]
        range_res = radar_dict["range_res"]
        beam_width = radar_dict["beam_width"]

        block, height = pbb_lbm(
            volume,
            map,
            par,
            el_coords,
            site_coords[2],
            threshold,
            beam_width,
            maxVal,
            dem_path,
            set_null,
            lbm,
        )  # TODO: da implementare

        return

    any_par = threshold <= 0
    volId = dpb.dpb.get_volumes(prodId=prodId, moment=moment, any=any_par)
    scan_dict = dpb.dpb.get_scans(volId=volId, max_el=max_el)
    scans = scan_dict["scans"]
    coord_set = scan_dict["coord_set"]
    scan_dim = scan_dict["scan_dim"]
    count = len(scans)
    if count <= 0:
        return

    if threshold <= 0 or create_dem > 0:
        nBins, _ = dpb.dpb.get_par(prodId, "nBins", default=1000)
        nAz, _ = dpb.dpb.get_par(prodId, "nAz", default=360)
        if nBins > 0:
            if create_dem > 0:
                count = 1
                dim = [nAz, nBins]  # TODO: da controllare
            else:
                dim = [count, nAz, nBins]  # TODO: da controllare
        else:
            dim = scan_dim
            count = 1
            coord_set = coord_set[0]

        outPointer, par, _ = dpg.radar.check_in(
            node_in=scans[0], node_out=prodId, dim=dim, type=4
        )
        if outPointer is None:
            return
        if create_dem > 0:
            radar_dict = dpg.navigation.get_radar_par(scans[0])
            par = radar_dict["par"]  # TODO: DA CONTROLLARE!
            map = radar_dict["map"]
            dem = dpg.warp.get_dem(
                destMap=map, destPar=par, destDim=dim, numeric=True, hr=True
            )

            if np.size(dem) > 1:
                outPointer = dem

            dpg.radar.check_out(outId=prodId, pointer=outPointer, par=par)
            in_dict = {}
            in_dict["unit"] = "m"
            in_dict["parname"] = "Quota"
            in_dict["offset"] = 0.0
            in_dict["slope"] = 0
            in_dict["maxVal"] = 0
            in_dict["scaling"] = 0
            dpg.calibration.set_array_values(prodId, in_dict=in_dict)
            return

        outPointer[:] = np.nan

        if len(dim.shape) == 2:  # TODO: da controllare
            nEl = dim[0] * dim[1]
        elif len(dim.shape) == 3:
            nEl = dim[-1] * dim[-2]

    threshold /= 100

    blockage = np.array(())
    for sss in range(count):
        blockage, no_corr, dim = pbb_corr(
            scans[sss],
            threshold,
            blockage=blockage,
            dem_path=dem_path,
            dim=dim,
            tolerance=tolerance,
        )
        if threshold <= 0:
            if np.size(blockage) == nEl:
                outPointer[sss, :, :] = (1 - blockage) * maxVal
                tmp = blockage.copy()

        else:
            if no_corr > 0:
                return

    if threshold > 0:
        return

    dpg.radar.check_out(outId=prodId, pointer=outPointer, par=par, el_coords=coord_set)

    return
