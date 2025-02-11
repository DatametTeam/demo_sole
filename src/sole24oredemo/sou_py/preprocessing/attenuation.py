"""
Gestisce l’attenuazione del segnale radar durante eventi precipitativi in base al tipo di scansione.
Nel caso di singola polarizzazione, l'attenuazione non viene corretta, ma i dati affetti subiscono una riduzione
della qualità.
Con la doppia polarizzazione, l'attenuazione specifica (Ah) è calcolata mediante il metodo ZPHI, con parametri
configurabili,
ed è applicata solo sotto il Freezing Level (se disponibile) e con qualità sopra una soglia definita (threshQ).
In entrambi i casi, la qualità del volume è ridotta proporzionalmente all'attenuazione rispetto a soglie configurabili.
"""
import os
import pickle

import numpy as np
import sou_py.dpg as dpg
import sou_py.products as prd
import sou_py.preprocessing as preprocessing
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message
import sou_py.dpv as dpv
import warnings

data_to_plot = {}


def check_flh(flh, zh, flh_tolerance, range_res, elevation, site_height):
    """
    Modifies the input array `zh` by setting elements to zero where the
    corresponding elements in `flh` are less than or equal to the `height`
    plus a `flh_tolerance`.

    Args:
        flh (np.ndarray): 2D array of freeboard level height.
        zh (np.ndarray): 2D array that will be modified.
        flh_tolerance (float): Tolerance value to be added to height beams.
        range_res (float): Range resolution.
        elevation (float): Elevation angle of radar.
        site_height (float): Height of radar site.
    Returns:
        zh (np.ndarray): input array modified.
    """
    if np.size(zh) <= 0 or flh.shape != zh.shape:
        return

    dim = zh.shape
    n_az = dim[-2]
    height_beams = dpg.access.get_height_beams(
        elevation, dim[-1], range_res, site_height=site_height
    )
    height = zh.copy()
    height_beams += flh_tolerance

    for bbb in range(n_az):
        height[bbb, :] = height_beams

    ind = np.where(height >= flh)
    zh[ind] = 0

    return zh


def get_ah(z, k, res, bb, gamma, mask=None):
    """
            Calculate the AH array based on input parameters

    Args:
            z (np.ndarray):              Input data array
            k (np.ndarray):              An array of coefficients
            res (float):                 Resolution or step size for calculations
            bb (float):                  A constant used in the power function
            gamma (float):               Another constant used in the calculations
            mask (np.ndarray, optional): An array to filter the input data. Defaults to None

    Returns:
            np.ndarray:                  The calculated AH array.

    Note:
            The function calculates the specific attenuation along a radar path. Uses radar reflectivity and
            the attenuation coefficients to determine the specific attenuation. If a mask array is provided,
            the calculations are limited to the elements indicated by the mask. A power function is used for
            convert the reflectivity, and the results are integrated along the way to obtain the specific attenuation
    """
    n_z = np.size(z)
    ah = np.zeros(n_z, dtype=np.float32)

    if mask is not None and np.size(mask) == n_z:
        index = np.where(mask > 0)
        if len(index[0]) < 2:
            return ah
        r1 = index[0][0]
        rn = index[0][-1]
    else:
        r1 = 0
        rn = n_z - 1

    tmp_k = np.nan_to_num(k, neginf=0)
    pdp = 2 * res * np.nancumsum(tmp_k)
    dpdp = pdp[rn] - pdp[r1]

    if dpdp > 0:
        zp = dpg.calibration.power_func(z[r1: rn + 1], 1.0, bb, linear=True)
        tmp = zp * res
        ir1rn = 0.46 * bb * np.nansum(tmp)
        factor = 10 ** (0.1 * bb * gamma * dpdp) - 1
        irrn = ir1rn - 0.46 * bb * np.nancumsum(tmp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # RuntimeWarning: divide by zero encountered in divide
            ah[r1: rn + 1] = factor * zp / (ir1rn + factor * irrn)

    return ah


def get_pia(
        z,
        k,
        mask,
        bb,
        gamma,
        range_res,
        flh=None,
        elevation=None,
        site_height=None,
        delta_h=None,
):
    """
    Calculate Path Integrated Attenuation (PIA) for radar data.

    Args:
            z (np.ndarray):                2D array of radar reflectivity data.
            k (np.ndarray):                2D array of coefficients.
            mask (np.ndarray):             2D mask array.
            bb (float):                    Constant used in calculations.
            gamma (float):                 Another constant used in calculations.
            range_res (float):             Range resolution in meters.
            flh (np.ndarray, optional):    2D array of freeboard level height.
            elevation (float, optional):   Elevation angle of the radar beam.
            site_height (float, optional): Height of the radar site.
            delta_h (float, optional):     Delta height for calculations.

    Returns:
            np.ndarray:                    2D array of Path Integrated Attenuation (PIA).

    Note:
            The function calculates integrated attenuation along the path for 2D radar data using reflectivity and i
            specific attenuation coefficients. If provided, the flh array is used to determine a height limit
            above which data is ignored. The specific attenuation is calculated and integrates these values along the
            path to obtain the PIA.
            The results are useful for correcting attenuation in radar data and improving precipitation estimation.
    """
    h_beam = None

    res_km = range_res / 1000.0
    dim = z.shape
    n_bins = dim[1]
    n_az = dim[0]

    if flh is not None and flh.shape == z.shape:
        h_beam = dpg.access.get_height_beams(
            elevation, n_bins, range_res, site_height=site_height, projected=True
        )

    ah = np.zeros(dim, dtype=np.float32)
    pia = np.zeros(dim, dtype=np.float32)

    for aaa in range(n_az):
        tmp_mask = mask[aaa, :].copy()
        if h_beam is not None and np.size(h_beam) == n_bins:
            ind = np.where(h_beam > flh[:, aaa] - delta_h)
            if len(ind[0]) > 0:
                tmp_mask[ind] = 0
        ah[aaa, :] = get_ah(z[aaa, :], k[aaa, :], res_km, bb, gamma, tmp_mask)
        pia[aaa, :] = 2 * res_km * np.nancumsum(ah[aaa, :])

    return pia


def deattenuate(
        scanId,
        qScanId,
        kScanId,
        pScanId,
        B,
        gamma,
        smoothBox,
        thresh_q,
        thresh_k,
        Amin,
        Amax,
        par_node,
        flh_map,
        flh_tolerance,
        to_save,
        overwrite,
        elev,
        save_pkl=0
):
    out_map = dpb.dpb.get_data(scanId, numeric=True)
    if np.size(out_map) <= 1:
        log_message(f"Invalid Node! @ {dpg.tree.getNodePath(scanId)}")
        return None

    kdp = dpb.dpb.get_data(kScanId, numeric=True)
    phidp = dpb.dpb.get_data(pScanId, numeric=True)

    if np.size(kdp) != np.size(out_map):
        log_message(f"Invalid Node! @ {dpg.tree.getNodePath(kScanId)}")
        return None

    indK = np.where(kdp > thresh_k)
    if len(indK[0]) <= 1:
        log_message(
            f"No attenuation @ scan {dpg.tree.getNodePath(scanId)}", level="WARNING"
        )
        return None

    nav_dict = dpg.navigation.get_radar_par(scanId)
    range_res = nav_dict["range_res"]
    elevation = nav_dict["elevation_off"]
    site_coords = nav_dict["site_coords"]
    par = nav_dict["par"]

    currData = out_map.copy()
    if smoothBox > 0:
        currData = dpg.prcs.smooth_data(
            currData, box=smoothBox, opt=1
        )  # TODO: forse qua succede qualcosa di strano, mimmo usa solo la median....

    currData = check_flh(
        flh_map, currData, flh_tolerance, range_res, elevation, site_coords[2]
    )

    if thresh_q > 0:
        q = dpb.dpb.get_data(qScanId, numeric=True)
        if np.size(q) == np.size(currData):
            currData = np.where(q < thresh_q, 0, currData)

    dim = kdp.shape
    mask = np.zeros(dim)
    mask[indK] = 1
    ind = np.where((currData <= 0) | (~np.isfinite(currData)))
    mask[ind] = 0

    att = get_pia(currData, kdp, mask, B, gamma, range_res)

    maxAtt = np.nanmax(att)
    if maxAtt <= 0.1:
        log_message(
            f"No attenuation @ scan {dpg.tree.getNodePath(scanId)}", level="WARNING"
        )
        return

    log_message(f"Max attenuation = {maxAtt}")
    log_message(f"Attenuation Corrected using K @ {dpg.tree.getNodePath(scanId)}")

    if Amax > Amin:
        q_map = (Amax - att) / (Amax - Amin)
        ind = np.where(q_map >= 1)
        count = len(ind[0])
        if count < np.size(q_map):
            if count > 0:
                q_map[ind] = 1.0
            ind = np.where(q_map < 0)
            q_map[ind] = 0.0
            att[ind] = Amax

            maxVal = 100.0
            q_map *= maxVal
            weight, _, _ = dpg.radar.get_par(par_node, "q_weight", 0.5)
            _ = preprocessing.quality.quality(
                par_node,
                testValues=q_map,
                update=True,
                test_name="AttenuationTest",
                maxVal=maxVal,
                elevation=elevation,
                weight=weight,
            )

        else:
            log_message("Quality not updated")

    out_map += att

    if overwrite:
        _ = dpg.radar.set_par(
            scanId, "ATTENUATION", 1, only_current=True, to_save=to_save
        )
        dpg.radar.set_corrected(scanId)
        dpb.dpb.put_data(scanId, out_map, to_save=to_save)

        if save_pkl:
            data_to_plot[f"elev_{elev}"]["uz_corr"] = out_map.copy()
            data_to_plot[f"elev_{elev}"]["att"] = att.copy()
            data_to_plot[f"elev_{elev}"]["kdp"] = kdp.copy()
            data_to_plot[f"elev_{elev}"]["phidp"] = phidp.copy()

    return out_map


def attenuation(prodId, moment, zphi=None, nameK=None, attr=None):
    """
    Procedure to correct cells affected by signal attenuation.
    Only high-resolution volumes are used.

    Args:
        prodId: Node where the processing is performed.
        moment: Quantity to be corrected.
        zphi: If set on polarimetric volumes, the ZPHI method is used.
              For non-polarimetric volumes, the quality is reduced.
        nameK: Optional name of the KDP volume to be used.
        attr (optional):  A specific attribute to use. Defaults to None.

    Returns:
        None
    """

    flhId = None

    volId = dpb.dpb.get_volumes(prodId=prodId, moment=moment)
    tmp, site_name, exists = dpg.radar.get_par(
        prodId, "max_el", 0
    )
    if exists:
        max_el = tmp

    scans_dict = dpb.dpb.get_scans(volId=volId)
    scans = scans_dict["scans"]
    # max_el = scans_dict['max_el']
    scan_dim = scans_dict["scan_dim"]
    best_scan_ind = scans_dict["best_scan_ind"]
    coord_set = scans_dict["coord_set"]

    nScans = len(scans)
    if nScans <= 0:
        return

    Amin, _, _ = dpg.radar.get_par(prodId, "Amin", 1.0, prefix=site_name)
    Amax, _, _ = dpg.radar.get_par(prodId, "Amax", 1.0, prefix=site_name)

    if zphi is not None:
        if nameK is not None:
            volIdK = dpb.dpb.get_volumes(prodId, moment=nameK)
        if volIdK is None:
            volIdK = dpb.dpb.get_volumes(prodId, moment="KDP")
        if isinstance(volIdK, dpg.node__define.Node):
            _, _, kScans, scans, coord_set = dpg.access.check_coord_set(volId, volIdK)
        if np.size(kScans) != nScans:
            zphi = 0

    pVolId = dpb.dpb.get_volumes(prodId, moment="PHIDP")
    _, _, pScans, _, _ = dpg.access.check_coord_set(volId, pVolId)

    qualId = dpg.access.get_quality_volume(
        prodId
    )  # to_create=True TODO: non sembra venga usato nella funzione?
    _, _, qScans, scans, coord_set = dpg.access.check_coord_set(volId, qualId)
    if np.size(scans) != nScans:
        qScans = np.zeros(nScans)

    overwrite, _, _ = dpg.radar.get_par(prodId, "overwrite", 1, prefix=site_name)
    to_save, _, _ = dpg.radar.get_par(prodId, "to_save", 0, prefix=site_name)
    Zmin, _, _ = dpg.radar.get_par(prodId, "Zmin", 20.0, prefix=site_name)
    Zmax, _, _ = dpg.radar.get_par(prodId, "Zmax", 60.0, prefix=site_name)
    threshH, _, _ = dpg.radar.get_par(prodId, "threshH", 5000.0, prefix=site_name)
    A, _, _ = dpg.radar.get_par(prodId, "A", 1.08e-6, prefix=site_name)
    B, _, _ = dpg.radar.get_par(prodId, "B", 0.798, prefix=site_name)
    alpha, _, _ = dpg.radar.get_par(prodId, "alpha", 0.08, prefix=site_name)
    gamma, _, _ = dpg.radar.get_par(prodId, "gamma", alpha, prefix=site_name)
    threshQuality, _, _ = dpg.radar.get_par(prodId, "threshQuality", 0.0, prefix=site_name)
    threshK, _, _ = dpg.radar.get_par(prodId, "threshK", 0.1, prefix=site_name)
    flh_prod_name, _, _ = dpg.radar.get_par(prodId, "flh_prod_name", "", prefix=site_name)
    flh_site_name, _, _ = dpg.radar.get_par(
        prodId, "flh_site_name", "ITALIA", prefix=site_name
    )
    flh_tolerance, _, _ = dpg.radar.get_par(prodId, "flh_tolerance", 0.0, prefix=site_name)
    smoothBox, _, _ = dpg.radar.get_par(prodId, "smooth", 5, prefix=site_name, attr=attr)
    toRemove, _, _ = dpg.radar.get_par(prodId, "toRemove", 1, prefix=site_name, attr=attr)
    storeIndex, _, _ = dpg.radar.get_par(prodId, "index", -1, prefix=site_name)
    save_pkl, _, _ = dpg.radar.get_par(prodId, "save_pkl", 0)

    if storeIndex < 0:
        elevation, _, _ = dpg.radar.get_par(prodId, "elevation", -1, prefix=site_name)
        if elevation >= 0:
            storeNode = dpg.radar.selectScan(scans, elevation)
            storeIndex = np.where(storeNode == scans)  # TODO: da controllare
            if len(storeIndex) == 1:
                storeIndex = storeIndex[0]
            else:
                storeIndex = -1

    if storeIndex >= nScans:
        storeIndex = nScans - 1

    if toRemove == 0:
        if storeIndex >= 0:
            dim = scan_dim
        else:
            dim = [nScans] + nScans  # TODO: da controllare

        inp = scans[best_scan_ind]
        if Amax > Amin and isinstance(qScans[best_scan_ind], dpg.node__define.Node):
            inp = qScans[best_scan_ind]
        outPointer, par, _ = dpg.radar.check_in(
            node_in=inp, node_out=prodId, dim=dim, type=4, filename="volume.dat"
        )
        nEl = dim[-1] * dim[-2]

    if flh_prod_name != "":
        date, time, _ = dpg.times.get_time(prodId)
        n_hours = 12.0
        min_step = 60
        time = time[:3] + "00"
        ret, prod_path = dpg.access.get_aux_products(
            prodId=prodId,
            schedule=flh_prod_name,
            site_name=flh_site_name,
            last_date=date,
            last_time=time,
            n_hours=n_hours,
            min_step=min_step,
        )

        if ret > 0:
            flhTree = dpg.tree.createTree(prod_path[0])
            flhId = dpg.radar.find_site_node(flhTree, flh_site_name)
            if flhId is not None and isinstance(flhId, dpg.node__define.Node):
                log_message(f"Using {prod_path[0]}", level="INFO")

    for sss in range(nScans):

        navigation_dict = dpg.navigation.get_radar_par(
            scans[sss], get_az_coords_flag=True
        )
        if save_pkl:
            data_to_plot[f"elev_{sss}"] = {}
            data_to_plot[f"elev_{sss}"]["navigation_dict_uz"] = navigation_dict
            data_to_plot[f"elev_{sss}"]["node_path"] = dpg.tree.getNodePath(scans[sss])

        if flh_prod_name != "":
            flh_map = dpg.warp.warp_map(flhId, scans[sss], numeric=True)
            if np.size(flh_map) <= 1:
                log_message("Invalid FLH!", level="WARNING")
                return
        if zphi is None:
            attenuate_quality()  # TODO. da fare
        else:
            out_map = deattenuate(
                scans[sss],
                qScans[sss],
                kScans[sss],
                pScans[sss],
                B,
                gamma,
                smoothBox,
                threshQuality,
                threshK,
                Amin,
                Amax,
                par_node=prodId,
                flh_map=flh_map,
                flh_tolerance=flh_tolerance,
                to_save=to_save,
                overwrite=overwrite,
                elev=sss,
                save_pkl=save_pkl
            )

        if not toRemove:
            if np.size(out_map) > 0:
                if sss == storeIndex or sss == 0 or len(dim) == 3:
                    out = dpg.warp.warp_map(
                        scans[sss],
                        prodId,
                        source_data=out_map,
                        regular=True,
                        numeric=True,
                    )
                tmp = out_map.copy()
            if np.size(out) == nEl:
                if np.size(dim) == 3:
                    outPointer[sss, :, :] = out
                else:
                    outPointer[:, :] = out
        else:
            if np.size(out_map) > 0:
                if out_map is not None:
                    tmp = out_map.copy()

    dpg.tree.removeTree(flhTree)

    if save_pkl:
        with open(os.path.join(volId.parent.path, "kdp_att_uz_data.pkl"), "wb") as handle:
            pickle.dump(data_to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if toRemove > 0:
        return

    if storeIndex >= 0:
        coord_set = coord_set[storeIndex]

    dpg.radar.check_out(
        outId=prodId, pointer=outPointer, par=par, el_coords=coord_set, attr=attr
    )

    return
