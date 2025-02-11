"""
Introduce il concetto di qualità del dato radar, espresso in percentuale (0-100) e codificato su 8 bit.
La qualità viene stimata attraverso una serie di test indipendenti, ciascuno associato a un peso specifico
per valutare la probabilità che una cella radar non sia affetta da errore. Il valore di qualità è
aggiornato iterativamente in base ai risultati dei test e ai relativi pesi. Le celle filtrate nella fase di
declutter partono con qualità pari a 0.
"""
from numbers import Number

import numpy as np
import os
import sou_py.dpg as dpg
import sou_py.dpb as dpb
import sou_py.products as prd
import sou_py.preprocessing as preproc
from sou_py.dpg.log import log_message


def create_quality(tree, name, reset=False):
    """
    Creates a new quality volume within a given tree structure, optionally resetting it if it already exists.

    Args:
        tree: The data tree to search within. Can be a 'dpg.node__define.Node' or other
              types that are then converted into a node, where the quality volume is to be created.
        name (str): The name of the new node representing the quality volume to be added.
        reset (bool, optional): If True, resets the volume if it already exists by clearing its contents. Defaults to
        False.

    Returns:
        newVol: The created or updated quality volume node.
    """
    calPath = dpg.path.getDir("CALIBRATIONS", with_separator=True)
    volId = dpg.access.find_volume(tree, "UZ")
    parent = volId.parent
    newVol, exists = dpg.tree.addNode(parent, name)
    if exists:
        if not reset and newVol.getSons() is not None:
            return
        log_message(
            f"Resetting Volume {name} at node {newVol.path}",
            all_logs=False,
            level="INFO",
        )

    _, err = dpg.tree.copySonsToNode(volId, newVol, and_files=True, overwrite=reset)
    if err != 0:
        return

    vName = dpg.cfg.getValueDescName()
    volPath = dpg.tree.getNodePath(newVol)
    dpg.utility.copyFile(
        os.path.join(calPath, "quality.txt"), os.path.join(volPath, vName)
    )
    file, _, _ = dpg.calibration.getValuesFile(newVol)
    dpg.tree.removeAttr(newVol, name=file, delete_file=True)
    scans = dpg.tree.getSons(newVol)
    if len(scans) == 0:
        scans = [newVol]

    parFile = dpg.cfg.getParDescName()
    for scan in scans:
        destPath = dpg.tree.getNodePath(scan)
        dpg.utility.copyFile(
            os.path.join(calPath, "quality.txt"), os.path.join(destPath, vName)
        )
        file, _, _ = dpg.calibration.getValuesFile(scan)
        dpg.tree.removeAttr(scan, name=file, delete_file=True)
        dpg.tree.removeAttr(scan, name=parFile, delete_file=True)
        pointer, array_dict = dpg.array.get_array(scan)
        dtype = array_dict["type"]
        if pointer is not None:
            # Assuming 'pointer' is a reference to a NumPy array
            if dtype != 1:
                pointer = pointer.astype(
                    np.uint8
                )  # Convert the array to byte type (unsigned 8-bit integer)
            pointer[:] = 255
            dpg.array.set_array(scan, pointer)
        dpg.array.set_array_info(scan, name=name, dtype=1, bitplanes=8)

    dpg.tree.saveNode(newVol)
    log_message(f"Created Volume {name} @ {newVol.path}")
    return newVol


def update_q_scan(
        prodId,
        qualityScanId,
        weight,
        level=None,
        maxval=None,
        subtract=None,
        test_name=None,
        to_save=None,
        done=None,
        updateValues=np.array(()),
):
    """
    Updates the quality scan node based on specified parameters, adjusting quality values according to given weights
    and criteria.

    Args:
        prodId (node): The node associated with the quality scan.
        qualityScanId: The identifier of the quality scan node to be updated.
        weight (float): The weight factor used in the quality update process.
        level (int, optional): The level parameter for mapping quality values. Defaults to None.
        maxval (float, optional): The maximum allowable value for quality coefficients. Defaults to 100.
        subtract (bool, optional): If True, subtracts the computed coefficient from the quality values. Defaults to
        None.
        test_name (str, optional): The name of the test to check if it has already been performed. Defaults to None.
        to_save (bool, optional): If True, saves the updated quality scan node. Defaults to None.
        done (int, optional): Indicates whether the update has already been performed (0 for no, 1 for yes). Defaults
        to None.
        updateValues (np.ndarray, optional): An array of values used to update the quality scan. Defaults to an empty
        array.

    Returns:
        int: The status of the update process (1 if done, 0 if not).
    """
    if done != 0 and done != 1:
        done = 0

    if not isinstance(weight, Number):
        weight, _ = dpb.dpb.get_par(prodId, "weight", 1.0)

    if weight <= 0:
        return done

    if not isinstance(to_save, Number):
        to_save, _ = dpb.dpb.get_par(prodId, "save_quality", 1)

    if test_name is not None:
        test, _ = dpb.dpb.get_par(qualityScanId, test_name, 0)
        if test > 0:
            log_message(
                f"{test_name} already performed at {qualityScanId.path}",
                level="WARNING",
            )
            return done
        dpg.radar.set_par(qualityScanId, test_name, 1, only_current=True)

    pQuality, _, _ = dpb.dpb.get_pointer(qualityScanId)
    if pQuality is None or not isinstance(pQuality, np.ndarray):
        log_message(f"Invalid not at {qualityScanId.path}", level="WARNING")
        return done

    nEL = np.size(pQuality)
    if nEL <= 1:
        return done

    if np.size(updateValues) != nEL:
        updateValues = dpg.warp.warp_map(
            prodId, qualityScanId, level=level, regular=True, numeric=True
        )
        if np.size(updateValues) != nEL:
            return done

    coeff = updateValues

    if maxval is None:
        maxval = 100
    ind = ~np.isfinite(coeff)
    count = np.sum(ind)
    if count > 0:
        coeff[ind] = maxval

    if subtract:
        if maxval != 100:
            coeff *= 100 / maxval
        coeff = weight * (100 - coeff)
        ind = np.where(coeff < 0)
        if len(ind[0]) > 0:
            coeff[ind] = 0
        ind = np.where(coeff > 100)
        if len(ind[0]) > 0:
            coeff[ind] = 100

        pQuality -= coeff
        ind = np.where(pQuality < 0)
        if len(ind[0]) > 0:
            pQuality[ind] = 0

    else:
        coeff *= weight / maxval
        if weight < 1:
            coeff += 1 - weight
            coeff = np.where(coeff < 0, 0, coeff)
            coeff = np.where(coeff > 1, 1, coeff)
            pQuality *= coeff

    if to_save:
        dpg.tree.saveNode(qualityScanId, only_current=True)

    if test_name is None:
        log_message(
            f"{test_name}: quality updated. Node: {qualityScanId.path}", level="INFO"
        )

    done = 1
    return done


def quality_check(
        scanId,
        qScan,
        threshQuality,
        medianBox,
        altScanId,
        threshVoid=None,
        threshQVoid0=None,
        threshVoid0=None,
        threshQVoid1=None,
        threshVoid1=None,
        voidQuality=None,
        meanThresh=None,
        init_reset=None,
        setNull=False,
):
    """
    Performs a quality check on a radar scan, modifying the scan data based on various thresholds and quality measures.

    Args:
        scanId (node): The identifier for the scan to be checked.
        qScan: The identifier for the quality scan associated with the data.
        threshQuality (float): Threshold value below which data quality is considered poor.
        medianBox (int): Size of the median filter box for smoothing data.
        altScanId: The identifier for an alternative scan to adjust based on quality.
        threshVoid (float, optional): Threshold below which data is considered void. Defaults to None.
        threshQVoid0 (float, optional): Initial quality threshold for void detection. Defaults to None.
        threshVoid0 (float, optional): Initial data threshold for void detection. Defaults to None.
        threshQVoid1 (float, optional): Secondary quality threshold for void detection. Defaults to None.
        threshVoid1 (float, optional): Secondary data threshold for void detection. Defaults to None.
        voidQuality (float, optional): Value to assign to void data in the alternative scan. Defaults to None.
        meanThresh (float, optional): Mean threshold for detecting anomalies in the data. Defaults to None.
        init_reset (int, optional): Index or indices for initializing reset operations. Defaults to None.
        setNull (bool, optional): If True, sets certain data points to null based on quality checks. Defaults to False.

    Returns:
        None
    """
    currPointer, _, _ = dpb.dpb.get_pointer(scanId)
    if currPointer is None:
        return

    qPointer, _, _ = dpb.dpb.get_pointer(qScan)
    if qPointer is None:
        currPointer[:] = np.nan
        log_message("No quality... Resetting!", level="WARNING", all_logs=True)
        return

    if np.size(currPointer) != np.size(qPointer):
        currPointer[:] = np.nan
        log_message("Anomalous quality... Resetting!", level="WARNING", all_logs=True)
        return

    if medianBox > 1:
        currPointer = dpg.prcs.smooth_data(currPointer, medianBox, opt=1, no_null=True)
        qPointer = dpg.prcs.smooth_data(qPointer, medianBox, opt=1, no_null=True)

    if threshVoid > 0:
        ind = np.where(currPointer < threshVoid)
        currPointer[ind] = -np.inf
        qPointer[ind] = 100

    count1 = 0
    count2 = 0

    if threshQVoid0 > 0:
        ind1 = dpg.utilityArray.dynamicWhere(
            currPointer,
            qPointer,
            threshVoid0,
            threshVoid1,
            threshQVoid0,
            threshQVoid1,
            count=count1,
        )
        count1 = len(ind1[0])

    if threshQuality > 0:
        if threshVoid1 > 0:
            ind2 = np.where((qPointer < threshQuality) & (currPointer > threshVoid1))
        else:
            ind2 = np.where(qPointer < threshQuality)
        count2 = len(ind2[0])

    currPointer[ind1] = -np.inf

    val = -np.inf
    if setNull:
        if count2 > 0:
            qPointer[ind2] = 0
        indNull = np.where(np.isnan(currPointer))
        qPointer[indNull] = 0
        if altScanId != qScan:
            altQuality, _, _ = dpb.dpb.get_pointer(altScanId)
            if altQuality is not None:
                altQuality[ind1] = voidQuality
                altQuality[ind2] = 0
                altQuality[indNull] = 0

        val = np.nan

    currPointer[ind2] = val

    log_message(f"{scanId.path} filtered", level="INFO", general_log=True)

    if init_reset is not None and init_reset:
        ind = np.where(currPointer[:, init_reset] > 10)
        countInit = len(ind[0])
        countInit *= 3

        for rrr in range(len(init_reset)):
            ind = np.where(currPointer[:, rrr] > 10)
            count = len(ind[0])
            if count > 200 and count > countInit:
                qPointer[ind, rrr] = 0
                currPointer[ind, rrr] = np.nan

    if meanThresh is not None and meanThresh:
        mmm = np.nanmean(np.isfinite(currPointer))
        if mmm > meanThresh:
            ind = np.where(currPointer >= mmm)
            count = len(ind[0])
            if count / float(np.size(currPointer)) > 0.5:
                currPointer[:] = np.nan
                log_message(
                    f"Anomalous PPI: mean value = {mmm} ... Resetting!",
                    level="ERROR",
                    all_logs=True,
                )

    # dpg.array.set_array(scanId, currPointer)
    # dpg.array.set_array(qScan, qPointer)
    # dpg.array.set_array(altScanId, altQuality)

    return


def check_quality(prodId, volId, qualId, altId, to_save=None, threshQuality=None):
    _, _, qScans, scans, coordSet = dpg.access.check_coord_set(
        volId, qualId, coordSet=None, reverse_order=True
    )
    """
    Performs a quality check on radar volume data by applying various quality thresholds and parameters.

    Args:
        prodId (node): The product identifier associated with the volume data.
        volId: The volume identifier for the radar data to be checked.
        qualId: The identifier for the quality node associated with the radar data.
        altId: The identifier for an alternative node that may be used during the quality check.
        to_save (int, optional): Determines whether to save the updated quality node. Defaults to None.
        threshQuality (float, optional): The threshold value below which data quality is considered poor. Defaults to 
        None.

    Returns:
        float: The threshold value for quality used in the checks.
    """
    if len(scans) < 1:
        return

    if isinstance(altId, dpg.node__define.Node):
        _, _, altScans, scans, coordSet = dpg.access.check_coord_set(
            volId, altId, coordSet=None, reverse_order=True
        )

    if len(altScans) != len(scans):
        altScans = qScans

    if threshQuality is None:
        threshQuality, site_name = dpb.dpb.get_par(prodId, "threshQuality", 0.0)

    threshVoid, site_name = dpb.dpb.get_par(prodId, "threshVoid", 0.0)
    threshQVoid0, _ = dpb.dpb.get_par(prodId, "threshQVoid0", 0.0, prefix=site_name)
    threshVoid0, _ = dpb.dpb.get_par(prodId, "threshVoid0", 0.0, prefix=site_name)
    threshQVoid1, _ = dpb.dpb.get_par(prodId, "threshQVoid1", threshQVoid0, prefix=site_name)
    threshVoid1, _ = dpb.dpb.get_par(prodId, "threshVoid1", threshVoid0, prefix=site_name)
    voidQuality, _ = dpb.dpb.get_par(prodId, "voidQuality", 80.0, prefix=site_name)
    medianBox, _ = dpb.dpb.get_par(prodId, "medianBox", 0, prefix=site_name)
    meanThresh, _ = dpb.dpb.get_par(prodId, "meanThresh", 0.0, prefix=site_name)
    init_reset, _ = dpb.dpb.get_par(prodId, "init_reset", 0, prefix=site_name)
    setNull, _ = dpb.dpb.get_par(prodId, "setNull", 1, prefix=site_name)

    log_message(f"Quality check using: {qualId.path}", level="INFO", all_logs=True)

    for sss in range(len(scans)):
        quality_check(
            scans[sss],
            qScans[sss],
            threshQuality,
            medianBox,
            altScans[sss],
            threshVoid,
            threshQVoid0,
            threshVoid0,
            threshQVoid1,
            threshVoid1,
            voidQuality,
            meanThresh=meanThresh,
            init_reset=init_reset,
            setNull=setNull,
        )

    if to_save is None:
        to_save, _ = dpb.dpb.get_par(prodId, "save_quality", 0)
    if to_save > 0:
        dpg.tree.saveNode(qualId)

    return threshQuality


def quality(
        prodId,
        update=False,
        name=None,
        test_name=None,
        subtract=False,
        maxVal=None,
        weight=None,
        to_save=False,
        qVolId=None,
        check=None,
        moment=None,
        measure=None,
        testValues=np.array(()),
        elevation=np.array(()),
        coordSet=np.array(()),
        done=0,
):
    """
    Performs quality control on radar volume data, either by creating a new quality volume or updating an existing one.

    Args:
        prodId: The product identifier associated with the volume data.
        update (bool, optional): If True, updates the quality volume with new data. Defaults to False.
        name (str, optional): The name of the quality volume. Defaults to None.
        test_name (str, optional): The name of the test to check if it has already been performed. Defaults to None.
        subtract (bool, optional): If True, subtracts the computed coefficient from the quality values. Defaults to
        False.
        maxVal (float, optional): The maximum allowable value for quality coefficients. Defaults to None.
        weight (float, optional): The weight factor used in the quality update process. Defaults to None.
        to_save (bool, optional): If True, saves the updated quality volume. Defaults to False.
        qVolId: The identifier for the quality volume node. Defaults to None.
        check (str, optional): The name of an alternative scan to be used during quality check. Defaults to None.
        moment (str, optional): The name of the moment to be used. Defaults to None.
        measure (str, optional): The measure name if the moment is not provided. Defaults to None.
        testValues (np.ndarray, optional): An array of values used to update the quality scan. Defaults to an empty
        array.
        elevation (np.ndarray, optional): The elevation angle for the scan. Defaults to an empty array.
        coordSet (np.ndarray, optional): The coordinate set for the scan. Defaults to an empty array.
        done (int, optional): Indicates whether the quality update has already been performed (0 for no, 1 for yes).
        Defaults to 0.

    Returns:
        int: The status of the quality update process (1 if done, 0 if not).
    """
    tree = None
    volId = None

    if name is None:
        name, _ = dpb.dpb.get_par(prodId, "quality_name", "Quality")

    if qVolId is None:
        qVolId, tree = dpg.access.find_volume(
            measure=name, prod_id=prodId, get_tree=True
        )

    if check is not None:
        if dpg.io.type_py2idl(type(check)) == 7:
            altId, tree = dpg.access.find_volume(
                tree, measure=check, prod_id=prodId, get_tree=True
            )
            altId = create_quality(tree, check, reset=True)
        if moment is not None:
            measure = moment
        if measure is None:
            measure, _ = dpb.dpb.get_par(prodId, "measure", "CZ")
        volId, tree = dpg.access.find_volume(tree, measure=measure, get_tree=True)
        if not isinstance(volId, dpg.node__define.Node):
            return
        threshquality = check_quality(prodId, volId, qVolId, altId, to_save=to_save)
        dpg.radar.set_corrected(volId, prodId=prodId)
        done = 1
        return done

    reset, _ = dpb.dpb.get_par(prodId, "reset_quality", 0)
    if not isinstance(qVolId, dpg.node__define.Node) or reset:
        qVolId = create_quality(tree=tree, name=name, reset=reset)

    if not update:
        if to_save:
            dpg.tree.saveNode(qVolId)
        return done

    if len(testValues) > 1:
        if np.size(elevation) == 1 and elevation is not None:
            qScanId = dpg.access.get_single_scan(qVolId, elevation)
        done = update_q_scan(
            prodId=prodId,
            qualityScanId=qScanId,
            updateValues=testValues,
            weight=weight,
            maxval=maxVal,
            subtract=subtract,
            test_name=test_name,
            to_save=to_save,
        )
        return done

    if len(coordSet) <= 0:
        out = dpg.navigation.get_radar_par(prodId, get_el_coords_flag=True)
        coordSet = out["el_coords"]

    _, values, qScans, scans, coordSet = dpg.access.check_coord_set(
        volId, qVolId, coordSet=coordSet
    )
    nScans = len(qScans)

    for sss in range(nScans):
        done = update_q_scan(
            prodId,
            qScans[sss],
            weight=weight,
            level=sss,
            maxval=maxVal,
            subtract=subtract,
            test_name=test_name,
            to_save=to_save,
            done=done,
        )

        # if done == 0:
        #     log_message("Done == 0. Qualcosa non va", level='ERROR', all_logs=True)

    return done
