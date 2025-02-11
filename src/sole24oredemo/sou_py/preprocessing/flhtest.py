import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb
import sou_py.preprocessing as pre
from sou_py.dpg.log import log_message


def getFLHTest(
        range_res: float,
        nBins: int,
        nAz: int,
        FLH: np.ndarray,
        elevation: float,
        beam_width: float,
        FLthickDown: float,
        FLthickUp: float,
        site_height: float,
):
    """
    Calculates the FLH (Freezing Level Height) test values.

    This function computes the upper and lower boundaries for the freezing level height. It returns a matrix of test
    values indicating the presence of the freezing level.

    Args:
        range_res (float): The radar's range resolutions.
        nBins (int): The number of range bins.
        nAz (int): The number of azimuth angles.
        FLH (np.ndarray): An array representing the freezing level height values for each azimuth and bin.
        elevation (float): The elevation angle of the radar beam (in degrees).
        beam_width (float): The width of the radar beam (in degrees).
        FLthickDown (float): The thickness of the freezing level region below the FLH value.
        FLthickUp (float): The thickness of the freezing level region above the FLH value.
        site_height (float): The height of the radar site.

    Returns:
        tuple:
            - flTest (np.ndarray): A matrix of test values (shape: nAz x nBins).
            - values_height (np.ndarray): A matrix (shape: nAz x nBins) of radar beam heights corresponding to
              each range bin and azimuth angle.
            - Returns (None, None) if the size of FLH is less than or equal to 1.
    """
    if np.size(FLH) <= 1:
        return None, None

    h, _ = dpg.beams.getHeightBeam(
        rangeRes=range_res, nBins=nBins, elevation=elevation, site_height=site_height
    )

    slantRange = dpg.map.getPolarRange(rangeRes=range_res, nBins=nBins)
    wup, wdown = dpg.beams.getWidthBeam(
        slantRange, beam_width, elevation, half=True, site_height=site_height
    )
    hUp = np.zeros((nAz, nBins), dtype=np.float32)
    hDown = np.zeros((nAz, nBins), dtype=np.float32)
    values_height = np.zeros((nAz, nBins), dtype=np.float32)

    for aaa in range(nAz):
        values_height[aaa, :] = h
        hUp[aaa, :] = h + wup
        hDown[aaa, :] = h - wdown

    den = hUp - hDown
    flUp = FLH + FLthickUp
    flDown = FLH - FLthickDown
    qUp = hUp - flUp
    qDown = flDown - hDown
    qUp /= den
    qDown /= den
    qUp /= 2.0

    ind = np.where(qUp > 0.5)
    qUp[ind] = 0.5
    ind = np.where(qUp < 0)
    qUp[ind] = 0

    ind = np.where(qDown > 1)
    qDown[ind] = 1
    ind = np.where(qDown < 0)
    qDown[ind] = 0

    flTest = np.maximum(qUp, qDown)
    return flTest, values_height


def flhtest(prodId, attr):
    """
    Executes the Freezing Level Height (FLH) quality control test for a given product ID. The test processes radar
    volume data based on configured thickness and altitude parameters, assessing the quality of measurements and
    adjusting them against known FLH values and geographic data.

    Args:
        prodId (int): Product ID containing radar and site-specific parameters.
        attr (str): Attribute parameter, used for custom adjustments to the process.

    Returns:
        None

    Raises:
        KeyError: If required parameters are missing from the configuration.
        ValueError: If volume data or coordinates are improperly formatted.

    Notes:
        - Unused portions of the volume data can be cleared based on the `toRemove` flag.
        - Several helper functions are placeholders (`getFLHTest`, `correctFLHQuality`) and require implementations
          to support specific validation and quality checks.
    """
    schedule, site_name = dpb.dpb.get_par(prodId, "schedule", "")
    FLthickDown, _ = dpb.dpb.get_par(prodId, "FLthickDown", 750.0, prefix=site_name)
    FLthickUp, _ = dpb.dpb.get_par(prodId, "FLthickUp", 250.0, prefix=site_name)
    FLH, _ = dpb.dpb.get_par(prodId, "FLH", 3000.0, prefix=site_name)
    hMax, _ = dpb.dpb.get_par(prodId, "hMax", 0.0, prefix=site_name)
    hMin, _ = dpb.dpb.get_par(prodId, "hMin", 0.0, prefix=site_name)
    sampled, _ = dpb.dpb.get_par(prodId, "sampled", 1, prefix=site_name)
    max_el, _ = dpb.dpb.get_par(prodId, "max_el", -1.0, prefix=site_name)
    toRemove, _ = dpb.dpb.get_par(prodId, "toRemove", 0, attr=attr)
    quality_name, _ = dpb.dpb.get_par(prodId, "quality_name", "Quality")

    if FLthickDown <= 0:
        log_message("FLHTEST Ignored", level="Warning")
        return

    if sampled > 0:
        volume, _, qVolId = pre.sampling.sampling(prodId, measure=quality_name)
        out_dict = dpg.navigation.get_radar_par(qVolId, get_el_coords_flag=True)
        site_coords = out_dict["site_coords"]
        par = out_dict["par"]
        coord_set = out_dict["el_coords"]
        range_res = out_dict["range_res"]
        beam_width = out_dict["beam_width"]

        count = len(coord_set)
        if count <= 0:
            return
        dim = np.size(volume)
        volId = qVolId

    else:
        if max_el <= 0:
            tmp = max_el
            max_el = None

        volId = dpb.dpb.get_volumes(prodId=prodId, any=True)
        scans_dict = dpb.dpb.get_scans(volId=volId)
        range_res = scans_dict["range_res"]
        beam_width = scans_dict["beam_width"]
        coord_set = scans_dict["coord_set"]
        site_coords = scans_dict["site_coords"]
        scan_dim = scans_dict["scan_dim"]
        best_scan_ind = scans_dict["best_scan_ind"]
        scans = scans_dict["scans"]

        count = len(scans)
        if count <= 0:
            return

        volId = scans[best_scan_ind]
        dim = [count] + scan_dim

    date, time, _ = dpg.times.get_time(prodId)
    outPointer, par, _ = dpg.radar.check_in(
        node_in=volId,
        node_out=prodId,
        dim=dim,
        type=4,
        filename="volume.dat",
        no_values=True,
    )
    if outPointer is None:
        return

    nEl = dim[-1] * dim[-2]
    maxVal = 100

    if schedule != "":
        n_hours, site_name = dpb.dpb.get_par(prodId, "n_hours", 12.0)
        min_step, _ = dpb.dpb.get_par(prodId, "min_step", 60.0, prefix=site_name)
        site, _ = dpb.dpb.get_par(prodId, "site", "ITALIA", prefix=site_name)
        time = time.split(":")[0] + ":00"
        nprods, prod_path = dpg.access.get_aux_products(
            prodId=prodId,
            schedule=schedule,
            site_name=site,
            last_date=date,
            last_time=time,
            n_hours=n_hours,
            min_step=min_step,
        )
        if nprods <= 0:
            log_message("Cannot find FLH Data", level="Warning")
            if toRemove == 1:
                del outPointer
            return
        tree = dpg.tree.createTree(prod_path[0])
        flhNode = dpg.radar.find_site_node(tree=tree, name=site)
        flhMap = dpg.warp.warp_map(source=flhNode, destNode=prodId, numeric=True)
        if len(flhMap.shape) > 1:
            FLH = flhMap.copy()
            log_message(f"Using {prod_path[0]}")
    if np.size(FLH) == 1:
        if FLH <= FLthickDown:
            log_message("FLH too low!", level="WARNING")
    if hMax > hMin:
        outMap = dpg.map.check_map(prodId)  # TODO: da implementare
        dem = dpg.warp.get_dem(outMap, par, dim, numeric=True)

    site_height = site_coords[2]
    # coord_set = str(coord_set)

    for sss in range(count):
        quality, values_height = getFLHTest(
            range_res,
            dim[-1],
            dim[-2],
            FLH,
            coord_set[sss],
            beam_width,
            FLthickDown,
            FLthickUp,
            site_height=site_height,
        )
        quality *= maxVal
        if hMax > hMin:
            correctFLHQuality(
                quality, values_height, dem, hMax, hMin
            )  # TODO: da fare, non sembra entrarci mai per ora
        outPointer[sss, :, :] = quality

    if sampled > 0:
        done = pre.quality.quality(
            prodId=prodId,
            testValues=outPointer,
            update=True,
            test_name="FLHTest",
            coordSet=coord_set,
            maxVal=maxVal,
            qVolId=qVolId,
            qScanId=qVolId,
        )  # TODO: da controllare
    else:
        done = pre.quality.quality(
            prodId=prodId,
            update=True,
            test_name="FLHTest",
            coordSet=coord_set,
            maxVal=maxVal,
        )

    if toRemove == 1:
        del outPointer
        return

    dpg.radar.check_out(
        outId=prodId, pointer=outPointer, par=par, el_coords=coord_set, attr=attr
    )
    return
