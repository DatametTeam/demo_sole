"""
Il modulo KDP ricostruisce il volume di Specific Differential Phase (KDP) a partire da PHIDP, filtrata
con un filtro mediano 3x3 ad alta risoluzione. L'algoritmo utilizza una finestra mobile lungo il fascio radar
per calcolare KDP come derivata di PHIDP, assumendo precipitazione omogenea all'interno della finestra.
La procedura iterativa, attualmente eseguita 3 volte con una finestra di 2.5 km,
esclude dati con qualità inferiore al 50% per PHIDP e valori di KDP fuori dal range -1 a 30 deg/km.
Per raggi con fase massima superiore a 50 gradi, la finestra è ridotta del 50%.
"""
import os
import pickle
import numpy as np
from numba import njit, prange

import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message
import warnings

data_to_plot = {}
use_numba = False


def get_phi_from_k(kdp, dr):
    """
    Computes the differential phase (`phi`) from specific differential phase (`kdp`) data.

    Args:
        - kdp: 2D array of specific differential phase values.
        - dr: Range resolution (distance increment between points).

    Returns:
        - phi: 2D array of cumulative phase values.
    """
    kkk = kdp.copy()
    ind = np.where(~np.isfinite(kkk))
    kkk[ind] = 0

    phi = np.cumsum(2.0 * kkk * dr, axis=1)

    # phi = np.where(phi >= 360, phi - 360, phi)

    return phi


@njit(parallel=False, fastmath=False, cache=False)
def get_phi_from_k_numba(kdp, dr):
    """
    Computes the differential phase (`phi`) from specific differential phase (`kdp`) data using Numba for optimization.

    Args:
        - kdp: 2D array of specific differential phase values.
        - dr: Range resolution (distance increment between points).

    Returns:
        - phi: 2D array of cumulative phase values.
    """
    # Create a copy of the input array
    kkk = np.copy(kdp)
    rows, cols = kkk.shape

    # Replace non-finite values with 0
    for i in range(rows):
        for j in range(cols):
            if not np.isfinite(kkk[i, j]):
                kkk[i, j] = 0

    # Initialize phi
    phi = np.zeros_like(kkk)

    # Compute cumulative sum along the second axis (axis=1)
    for i in range(rows):
        for j in range(1, cols):
            phi[i, j] = phi[i, j - 1] + 2.0 * kkk[i, j] * dr

    return phi


@njit(cache=False, parallel=True, fastmath=True)
def get_k_from_phi_numba(pdpf, dr, upThresh, downThresh, init_reset, win_range=0):
    """
    Derives specific differential phase (`kdp`) from cumulative phase (`phi`) data using Numba.

    Parameters:
        - pdpf: 2D array of cumulative phase values.
        - dr: Range resolution (distance increment between points).
        - upThresh: Upper threshold for `kdp`.
        - downThresh: Lower threshold for `kdp`.
        - init_reset: Number of initial bins to reset to zero.
        - win_range: Window size for differential calculation (default is 0).

    Returns:
        - kdp: 2D array of specific differential phase values.
    """
    dim = pdpf.shape
    if len(dim) != 2:
        return np.array([[0], [0]], dtype=np.float32)

    if win_range < 1:
        win_range = 1

    # Transpose to handle roll without axis
    pdpf_t = pdpf.T
    kdp_t = np.roll(pdpf_t, -win_range) - np.roll(pdpf_t, win_range)
    kdp = kdp_t.T

    const = 0.5 / (2 * dr * win_range)
    kdp *= const

    # Reset initial and end regions to 0
    kdp[:, :init_reset * win_range + 1] = 0
    kdp[:, dim[1] - 2 * win_range:] = 0

    # Flatten, apply non-finite checks, and reshape back
    kdp_flat = kdp.ravel()
    pdpf_flat = pdpf.ravel()

    non_finite_indices = np.where(~np.isfinite(kdp_flat) | ~np.isfinite(pdpf_flat))
    kdp_flat[non_finite_indices] = 0

    # Handle negative wrapping
    neg_indices = np.where(kdp_flat < -180)
    kdp[neg_indices] += 360

    # Apply thresholds
    if downThresh < 0:
        below_thresh_indices = np.where(kdp_flat < downThresh)
        kdp_flat[below_thresh_indices] = 0.0

    if upThresh > 0:
        above_thresh_indices = np.where(kdp_flat > upThresh)
        kdp_flat[above_thresh_indices] = 0

    kdp = kdp_flat.reshape(dim)

    return kdp


def get_k_from_phi(pdpf, dr, upThresh, downThresh, init_reset, win_range=0):
    """
    Derives specific differential phase (`kdp`) from cumulative phase (`phi`) data.

    Parameters:
        - pdpf: 2D array of cumulative phase values.
        - dr: Range resolution (distance increment between points).
        - upThresh: Upper threshold for `kdp`.
        - downThresh: Lower threshold for `kdp`.
        - init_reset: Number of initial bins to reset to zero.
        - win_range: Window size for differential calculation (default is 0).

    Returns:
        - kdp: 2D array of specific differential phase values.
    """
    dim = pdpf.shape
    if len(dim) != 2:
        return -1

    if win_range < 1:
        win_range = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # RuntimeWarning: roll porta a valori inf o -inf
        kdp = np.roll(pdpf, -win_range, axis=1) - np.roll(pdpf, win_range, axis=1)
    const = np.float32(0.5 / (2 * dr * win_range))

    kdp *= const

    kdp[:, 0: init_reset * win_range + 1] = 0
    kdp[:, dim[1] - 2 * win_range: dim[1]] = 0

    ind = np.where(~np.isfinite(kdp))
    kdp[ind] = 0

    ind = np.where(~np.isfinite(pdpf))
    kdp[ind] = 0

    indNeg = np.where(kdp < -180)
    kdp[indNeg] += 360

    if downThresh < 0:
        ind = np.where(kdp < downThresh)
        kdp[ind] = 0.0

    if upThresh > 0:
        ind = np.where(kdp > upThresh)
        kdp[ind] = 0

    return kdp


@njit(cache=False)
def recompute_phi_numba(
        array,
        niter,
        k_win,
        k_outwin,
        range_res,
        upThresh,
        downThresh,
        phiThresh,
        winReduction,
        initReset,
        elev,
        save_pkl=0
):
    """
    Iteratively refines `kdp` and `phi` values over multiple iterations.


    Args:
        - array: Initial `phi` array to refine.
        - niter: Number of iterations to perform.
        - k_win: Initial window size for `kdp` computation.
        - k_outwin: Optional larger window size for outer iterations.
        - range_res: Range resolution in meters.
        - upThresh, downThresh: Thresholds for filtering `kdp` values.
        - phiThresh: Phase threshold for selective recomputation.
        - winReduction: Reduction factor for the window size.
        - initReset: Number of initial bins to reset to zero.
        - elev: Elevation angle for metadata storage.

    Returns:
        - kdp: Refined 2D array of specific differential phase values.
        - array: Refined 2D array of cumulative phase values.
    """

    if k_win <= 0:
        return

    fbox = int(np.floor(k_win * 1000.0 / range_res))
    dr = range_res / 1000

    kdp = get_k_from_phi_numba(
        array,
        dr,
        win_range=fbox,
        upThresh=upThresh,
        downThresh=downThresh,
        init_reset=initReset,
    )

    if k_outwin > 0:
        fbox = int(np.floor((k_outwin * 1000.0 / range_res)))

    for iii in prange(niter):
        array = get_phi_from_k_numba(kdp, dr)

        if phiThresh > 0 and winReduction > 0:
            maxP = np.array([np.nanmax(row) for row in array])
            ind = np.where(maxP > phiThresh)[0]  # Flatten the index

            if len(ind) > 0:
                kdpr = get_k_from_phi_numba(
                    array,
                    dr,
                    win_range=int(np.floor(fbox * winReduction)),
                    upThresh=upThresh,
                    downThresh=downThresh,
                    init_reset=initReset,
                )
                for ccc in range(len(ind)):
                    kdp[ind[ccc], :] = kdpr[ind[ccc], :]
                array = get_phi_from_k_numba(kdp, dr)

        kdp = get_k_from_phi_numba(
            array,
            dr,
            win_range=fbox,
            upThresh=upThresh,
            downThresh=downThresh,
            init_reset=initReset,
        )

    return kdp, array


def recompute_phi(
        array,
        niter,
        k_win,
        k_outwin,
        range_res,
        upThresh,
        downThresh,
        phiThresh,
        winReduction,
        initReset,
        elev,
        save_pkl=0
):
    """
    Iteratively refines `kdp` and `phi` values over multiple iterations.


    Args:
        - array: Initial `phi` array to refine.
        - niter: Number of iterations to perform.
        - k_win: Initial window size for `kdp` computation.
        - k_outwin: Optional larger window size for outer iterations.
        - range_res: Range resolution in meters.
        - upThresh, downThresh: Thresholds for filtering `kdp` values.
        - phiThresh: Phase threshold for selective recomputation.
        - winReduction: Reduction factor for the window size.
        - initReset: Number of initial bins to reset to zero.
        - elev: Elevation angle for metadata storage.

    Returns:
        - kdp: Refined 2D array of specific differential phase values.
        - array: Refined 2D array of cumulative phase values.
    """

    if k_win <= 0:
        return

    fbox = int(np.floor(k_win * 1000.0 / range_res))
    dr = range_res / 1000

    kdp = get_k_from_phi(
        array,
        dr,
        win_range=fbox,
        upThresh=upThresh,
        downThresh=downThresh,
        init_reset=initReset,
    )

    if save_pkl:
        data_to_plot[f"elev_{elev}"]["kdp_corr"] = kdp.copy()
        data_to_plot[f"elev_{elev}"]["phi_corr"] = array.copy()

    if k_outwin > 0:
        fbox = int(np.floor((k_outwin * 1000.0 / range_res)))

    for iii in range(niter):
        array = get_phi_from_k_numba(kdp, dr)

        if save_pkl:
            data_to_plot[f"elev_{elev}"][f"phi_{iii}"] = array.copy()

        if phiThresh > 0 and winReduction > 0:
            maxP = np.nanmax(array, axis=1)
            ind = np.where(maxP > phiThresh)

            if len(ind[0]):
                kdpr = get_k_from_phi(
                    array,
                    dr,
                    win_range=int(np.floor(fbox * winReduction)),
                    upThresh=upThresh,
                    downThresh=downThresh,
                    init_reset=initReset,
                )
                # for ccc in range(len(ind[0])):
                #     kdp[ind[0][ccc], :] = kdpr[ind[0][ccc], :]
                kdp[ind[0]] = kdpr[ind[0]]
                array = get_phi_from_k(kdp, dr)

        kdp = get_k_from_phi(
            array,
            dr,
            win_range=fbox,
            upThresh=upThresh,
            downThresh=downThresh,
            init_reset=initReset,
        )

        if save_pkl:
            data_to_plot[f"elev_{elev}"][f"kdp_{iii}"] = kdp.copy()

    return kdp, array


def compute_kdp(
        pScanId,
        zScanId,
        niter,
        k_win,
        k_outwin,
        PHImedianwidth,
        medianwidth,
        qScanId,
        qThresh,
        zThresh,
        upThresh,
        downThresh,
        phiThresh,
        winReduction,
        initReset,
        fill,
        elev,
        save_pkl=0
):
    """
    Calculates and refines `kdp` and `phi` values for radar data.

    Parameters:
        - pScanId: Product scan identifier for phase data.
        - zScanId: Product scan identifier for reflectivity data.
        - niter: Number of refinement iterations.
        - k_win, k_outwin: Window sizes for `kdp` computation.
        - PHImedianwidth, medianwidth: Smoothing window sizes for `phi` and `kdp`.
        - qScanId: Identifier for quality control data.
        - qThresh: Threshold for quality control.
        - zThresh: Threshold for reflectivity filtering.
        - upThresh, downThresh: Thresholds for `kdp` filtering.
        - phiThresh: Phase threshold for selective recomputation.
        - winReduction: Reduction factor for window size.
        - initReset: Number of initial bins to reset to zero.
        - fill: Flag to determine if invalid values should be filled.
        - elev: Elevation angle for metadata storage.

    Returns:
        - kdp: Final 2D array of specific differential phase values.
        - phi: Final 2D array of cumulative phase values.
    """
    currData = dpb.dpb.get_data(pScanId, numeric=True)
    if np.size(currData) <= 1:
        return

    if save_pkl:
        data_to_plot[f"elev_{elev}"]["phi_raw"] = currData.copy()

    par = dpg.navigation.get_radar_par(pScanId)
    range_res = par["range_res"]
    az_off = par["azimut_off"]
    zscan_dict = dpg.navigation.get_radar_par(zScanId)
    zpar = zscan_dict["par"]

    if np.size(zpar) > 6 and zpar is not None:
        if az_off != zscan_dict["azimut_off"]:
            dpg.warp.warp_map(
                pScanId, zScanId, currData, source_data=currData, regular=True
            )

    if isinstance(qScanId, dpg.node__define.Node) and qScanId is not None:
        qData = dpb.dpb.get_data(qScanId, numeric=True)
        if np.size(qData) == np.size(currData):
            indNullq = np.where(qData < qThresh)
            currData[indNullq] = np.nan

    indNull, countNull, indVoid, countVoid = dpg.values.count_invalid_values(currData)
    zData = dpb.dpb.get_data(zScanId, numeric=True)

    countZ = 0
    countVoidZ = 0
    countNullZ = 0

    if np.size(zData) == np.size(currData):
        indNullZ, countNullZ, indVoidZ, countVoidZ = dpg.values.count_invalid_values(
            zData
        )
        if zThresh > -20:
            indZ = np.where(zData < zThresh)
            countZ = len(indZ[0])
            currData[indZ] = -np.inf

    # indNeg = np.where(currData > 180)
    # currData[indNeg] -= 360
    # indNeg = np.where(currData < -180)
    # currData[indNeg] += 360

    if PHImedianwidth > 0:
        currData = dpg.prcs.smooth_data(currData, PHImedianwidth, opt=1)

    if not use_numba:
        kdp, phi = recompute_phi(
            currData,
            niter,
            k_win,
            k_outwin,
            range_res,
            upThresh,
            downThresh,
            phiThresh,
            winReduction,
            initReset,
            elev=elev,
            save_pkl=save_pkl
        )
    else:
        kdp, phi = recompute_phi_numba(
            currData,
            niter,
            k_win,
            k_outwin,
            range_res,
            upThresh,
            downThresh,
            phiThresh,
            winReduction,
            initReset,
            elev=elev,
            save_pkl=save_pkl
        )

    if np.size(kdp) <= 1 or fill:
        return

    if countNullZ > 0:
        kdp[indNullZ] = np.nan
        phi[indNullZ] = np.nan
    if countNull > 0:
        kdp[indNull] = np.nan
        phi[indNull] = np.nan
    if countVoid > 0:
        kdp[indVoid] = -np.inf
        phi[indVoid] = -np.inf
    if countZ > 0:
        kdp[indZ] = -np.inf
        phi[indZ] = -np.inf
    if countVoidZ > 0:
        kdp[indVoidZ] = -np.inf
        phi[indVoidZ] = -np.inf

    if medianwidth > 0:
        kdp = dpg.prcs.smooth_data(kdp, medianwidth, opt=1)

    phi = np.where(phi >= 360, phi - 360, phi)

    return kdp, phi


def kdp(prodId, attr=None, aux=None):
    """
    Compute the specific differential phase (KDP) and optionally the differential phase (PHIDP) for a given product ID.

    Args:
        prodId (str): Product ID for which the KDP computation is to be performed.
        attr (dpg.attr__define.Attr, optional): Calibration attributes for the KDP data.
        aux (str, optional): Auxiliary moment (e.g., reflectivity) to use in KDP computation.

    Returns:
        None
    """

    phiVol = None
    niter, _ = dpb.dpb.get_par(prodId, "niter", 3)
    if niter < 1:
        log_message("No iterations ... KDP ignored!", level="Warning")
        return

    Pvol = dpb.dpb.get_volumes(prodId=prodId, moment="PHIDP")

    if aux is not None:
        Zvol = dpb.dpb.get_volumes(prodId=prodId, moment=aux)
    else:
        Zvol = dpb.dpb.get_volumes(prodId=prodId, moment="CZ")

    _, _, zScans, pScans, coordSet = dpg.access.check_coord_set(Pvol, Zvol)

    nScans = len(pScans)
    if nScans < 1:
        return

    to_save, _ = dpb.dpb.get_par(prodId, "to_save", 1)
    qThresh, _ = dpb.dpb.get_par(prodId, "threshQuality", 90.0)
    medianwidth, _ = dpb.dpb.get_par(prodId, "medianwidth", 3)
    PHImedianwidth, _ = dpb.dpb.get_par(prodId, "PHImedianwidth", 3)
    k_win, _ = dpb.dpb.get_par(prodId, "k_win", 2.5)
    k_outwin, _ = dpb.dpb.get_par(prodId, "k_outwin", 2.5)
    phiThresh, _ = dpb.dpb.get_par(prodId, "phiThresh", 50.0)
    winReduction, _ = dpb.dpb.get_par(prodId, "winReduction", 0.5)
    initReset, _ = dpb.dpb.get_par(prodId, "initReset", 2)
    downThresh, _ = dpb.dpb.get_par(prodId, "downThresh", -1.0)
    upThresh, _ = dpb.dpb.get_par(prodId, "upThresh", 30.0)
    zThresh, _ = dpb.dpb.get_par(prodId, "zThresh", 5.0)
    fill, _ = dpb.dpb.get_par(prodId, "fill", 0)
    createK, _ = dpb.dpb.get_par(prodId, "createK", "KDP")
    createPHI, _ = dpb.dpb.get_par(prodId, "createPHI", "")
    quality_name, _ = dpb.dpb.get_par(prodId, "quality_name", "Quality")
    save_pkl, _ = dpb.dpb.get_par(prodId, "save_pkl", 0)

    if qThresh > 0:
        Qvol = dpb.dpb.get_volumes(prodId, moment=quality_name)
        _, _, qScans, _, _ = dpg.access.check_coord_set(Pvol, Qvol)

    kVol, _ = dpg.tree.addNode(Pvol.parent, createK)
    if createPHI != "" and createPHI is not None:
        phiVol, _ = dpg.tree.addNode(Pvol.parent, createPHI)

    _, calib, calib_dict = dpg.calibration.get_values(prodId)
    if calib is None and isinstance(attr, dpg.attr__define.Attr):
        calib = attr

    filename = "SCAN.dat"
    bitplanes, _, _ = dpg.attr.getAttrValue(calib, "bitplanes", 8)

    for sss in range(nScans):
        if len(qScans) > sss:
            qScanId = qScans[sss]
        else:
            qScanId = None

        if save_pkl:
            navigation_dict = dpg.navigation.get_radar_par(
                pScans[sss], get_az_coords_flag=True
            )

            data_to_plot[f"elev_{sss}"] = {}
            data_to_plot[f"elev_{sss}"]["navigation_dict_phi"] = navigation_dict
            data_to_plot[f"elev_{sss}"]["node_path"] = dpg.tree.getNodePath(pScans[sss])

        kdp, phi = compute_kdp(
            pScans[sss],
            zScans[sss],
            niter,
            k_win,
            k_outwin,
            PHImedianwidth,
            medianwidth,
            qScanId=qScanId,
            qThresh=qThresh,
            zThresh=zThresh,
            upThresh=upThresh,
            downThresh=downThresh,
            phiThresh=phiThresh,
            winReduction=winReduction,
            initReset=initReset,
            fill=fill,
            elev=sss,
            save_pkl=save_pkl
        )

        kScan, _ = dpg.tree.addNode(kVol, dpg.tree.getNodeName(pScans[sss]))

        if np.size(kdp) > 1:
            dpb.dpb.put_data(
                kScan,
                kdp,
                attr=calib,
                main=zScans[sss],
                no_copy=True,
                filename=filename,
            )
            dpg.array.set_array_info(kScan, bitplanes=bitplanes)

            if save_pkl:
                kdp_navigation_dict = dpg.navigation.get_radar_par(
                    kScan, get_az_coords_flag=True
                )
                data_to_plot[f"elev_{sss}"]["navigation_dict_kdp"] = kdp_navigation_dict
        else:
            log_message(f"Unreliable KDP...removed! @ path: {kScan.path}")
            dpg.tree.removeNode(kScan, directory=to_save, shared=True)

        if phiVol is not None:
            _, phi_calib, phi_calib_dict = dpg.calibration.get_values(pScans[sss])
            phiScan, _ = dpg.tree.addNode(phiVol, dpg.tree.getNodeName(pScans[sss]))

            dpb.dpb.put_data(
                phiScan,
                phi,
                attr=phi_calib,
                main=pScans[sss],
                no_copy=True,
                filename=filename,
            )

    if to_save:
        dpg.tree.saveNode(kVol)

    if phiVol is not None:
        dpg.tree.saveNode(phiVol)

        if save_pkl:
            # Save the cicle data as pkl TODO: da sistemare nel caso in MimmoStyle
            with open(os.path.join(phiVol.parent.path, "kdp_phi_data.pkl"), "wb") as handle:
                pickle.dump(data_to_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dpg.calibration.copy_calibration(kScan, kVol, to_save=to_save)
