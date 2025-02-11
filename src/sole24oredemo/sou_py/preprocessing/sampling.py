"""
Il campionamento è una procedura fondamentale che trasforma i dati radar grezzi, memorizzati in matrici 2D,
in un unico volume tridimensionale. Questo processo permette l'applicazione di algoritmi sui volumi corretti e regolari.
Il campionamento include parametri configurabili come risoluzione in azimuth, in range e portata.
I volumi risultanti mantengono un passo azimutale regolare e vengono proiettati al suolo per garantire coerenza lungo
la verticale.
"""
# pylint: disable= locally-disabled, line-too-long, trailing-whitespace
# l'utente dovrebbe accedere solo a parameters, e teoricamente sugli altri file non ha visibilità
# non dovrebbe quindi sapere i nomi specifici dei parametri contenuti negli altri file
# ma solo avere un metodo per accederci
import numpy as np
import os

from numba import njit, prange
import numba as nb

from sou_py import dpg
from sou_py import dpb
import sou_py.products as prd
import sou_py.preprocessing as preproc
from sou_py.dpg.log import log_message


def sampling(
        prodId,
        moment=None,
        sampled=None,
        outName=None,
        attr=None,
        here=False,
        remove_on_error=False,
        check_null=None,
        no_save=False,
        reload=False,
        linear=False,
        projected=False,
        copy=False,
        force=False,
        add=False,
        check_type=None,
        out_null=None,
        get_volume=False,
        measure=None,
):
    """
    Handles the access and processing of a 3D volume of type [elevation, azimuth, range], checking for the existence
    of a low-resolution volume based on the current product's date and time. If the low-resolution volume does not
    exist,
    it is created with a resolution defined by range_res and az_res.

    Args:
        prodId: Node of the product, used to access various optional parameters contained in parameters.txt
        moment (str, optional): Name of the required quantity. Defaults to None
        sampled (int, optional): If None, any existing sampled volume is ignored, and a new sampling is performed.
        Defaults to None
        outName (str, optional): Optional name of the output node. Defaults to None
        attr (optional): Attributes for volume encoding (process.txt). Defaults to None
        here (bool, optional): If True, re-samples a volume with different resolution from the standard. Defaults to
        False
        remove_on_error (bool, optional): If True, removes the node on error. Defaults to False
        check_null (optional): Check for null values. Defaults to None
        no_save (bool, optional): If True, disables saving of the sampled node. Defaults to False
        reload (bool, optional): If True, data is reloaded from the file. Defaults to False
        linear (bool, optional): If True, enables conversion to linear scale for logarithmic quantities. Defaults to
        False
        projected (bool, optional): If True, enables ground projection of beams, equivalent to an additional range
        sampling factor of cos(elev). Defaults to False
        copy (bool, optional): If True, copies the node. Defaults to False
        force (bool, optional): If True, forces the operation even if conditions are not met. Defaults to False
        add (bool, optional): If True, adds the node to prodId. Defaults to False
        check_type (optional): Type checking parameter. Defaults to None
        out_null (optional): Output for null values. Defaults to None
        get_volume (bool, optional): If True, returns the volume. Defaults to False
        measure (str, optional): Name of the quantity if the 'moment' keyword is not defined. Defaults to None

    Returns:
        tuple: A tuple containing:
               - volume: The copied or sampled volume (if applicable).
               - pointer: Pointer to the original array containing the data (byte or int type).
               - node: Node of the sampled volume (useful for accessing the volume's properties).

    Example:
        To change from a range resolution of 100 meters to a resolution of 1000 meters, the 10 values are sampled using
        the value corresponding to the indicated decile; to change from an azimuth resolution of 0.9 degrees to a
        step of 1 degree, the nearest azimuth is considered.
    """
    # Verificare input e output

    # Unused -> numeric
    # Output -> node, pointer, volume

    # measure = None
    if moment is not None and moment != "":
        measure = moment
    if measure is None:
        measure, _, _ = dpg.radar.get_par(prodId, "measure", "")
    if measure == "":
        measure, _, _ = dpg.radar.get_par(prodId, "measure", "CZ")
    if sampled is None:
        sampled, _, _ = dpg.radar.get_par(prodId, "sampled", 1)
    if sampled > 0:
        rangeRes_default = 1000.0
    else:
        rangeRes_default = 0
    rangeRes, _, _ = dpg.radar.get_par(prodId, "rangeRes", rangeRes_default)
    if rangeRes == 1000.0:
        if not no_save:
            no_save = False
        if not here:
            here = False
    else:
        if not no_save:
            no_save = True
        if not here:
            here = True

    if not here:
        raw_tree = dpg.access.getRawTree(prod_id=prodId, sampled=True)
        if outName is None:
            outName = measure
        node, exists = dpg.tree.addNode(
            raw_tree, outName, to_not_save=no_save
        )
    else:
        if outName is not None:
            node, exists = dpg.tree.addNode(
                prodId, outName, to_not_save=no_save
            )
        else:
            node = prodId
            exists = False

    if not exists:
        max_el, _, _ = dpg.radar.get_par(prodId, "max_el", -1.0)
        min_el, _, _ = dpg.radar.get_par(prodId, "min_el", -1.0)
        min_scans, _, _ = dpg.radar.get_par(prodId, "min_scans", 0)
        decile, _, _ = dpg.radar.get_par(prodId, "decile", 7)
        tmp, _, _ = dpg.radar.get_par(prodId, "offset", 0.0)
        if tmp != 0.0:
            offset = tmp
        else:
            offset = None
        medianBox, _, _ = dpg.radar.get_par(prodId, "medianBox", 0.0)
        volId = dpg.access.find_volume(None, measure, prod_id=prodId, force=force)
        if dpg.tree.node_valid(volId):
            tmp_dict = {
                "min_el": min_el,
                "max_el": max_el,
                "decile": decile,
                "medianBox": medianBox,
                "min_scans": min_scans,
                "offset": offset,
            }
            # TODO capire outputs e valutare di risistemare come la funzione viene chiamata
            scans = sample_scans(
                volId,
                node,
                tmp_dict,
                no_save=no_save,
                reload=reload,
                check_type=check_type,
                attr=attr,
                check_null=check_null,
                out_null=out_null,
            )
        else:
            node = None

    pointer, _, _ = dpb.dpb.get_pointer(node=node)

    if pointer is None:
        if dpg.tree.node_valid(node):
            log_message("Error: Invalid pointer", level="WARNING")
        if remove_on_error:
            dpg.tree.removeNode(prodId, directory=True)
        log_message(f"Cannot find {measure} at path {prodId.path}", level='WARNING+', all_logs=True)
        return None, pointer, node

    if not here and copy:
        dpg.tree.copyNode(node, prodId, add=add)

    if not get_volume:
        return None, pointer, node

    volume = pointer

    if projected:
        dpb.dpb.project_volume(node, volume)

    if linear:
        _ = dpg.prcs.linearizeValues(volume, scale=2)
    return volume, pointer, node


@njit(parallel=False, cache=True, fastmath=True)  # Disable parallel processing for exact comparison
def checkSamplingIndex(ro, alpha, p_array, sampling, decile, offset):
    """
    Adjusts the sampling indices for a given array based on the specified sampling rate, decile, and offset.

    Args:
        ro (np.ndarray): Array representing range indices, modified in place based on the sampling.
        alpha (np.ndarray): Array representing azimuth indices.
        p_array (np.ndarray): The array from which the data is being sampled.
        sampling (float): The sampling rate for adjusting the indices.
        decile (int): The decile value used to select the appropriate index.
        offset (float): The initial offset for the range indices, modified in place.

    Returns:
        float: The updated offset after processing all indices. If the inputs are invalid, returns None.
    """
    # Input validation
    if ro is None or p_array is None or alpha is None:
        return None

    dim = ro.shape
    p_dim = p_array.shape

    # Determine index dimensions based on input shape
    if len(dim) == 3:
        nlines_ind = 1
        ncols_ind = 2
    else:
        nlines_ind = 0
        ncols_ind = 1

    # Exactly match the p_ind calculation from the original function
    p_ind = round(sampling * decile / 10)
    p_ind = max(1, p_ind)
    p_ind = min(int(sampling) - 1, p_ind)

    # Create range array
    rrr = np.arange(int(sampling))

    # Local copy of offset to modify
    local_offset = offset

    # Iterate through columns and lines
    for xxx in range(dim[ncols_ind]):
        # Calculate row indices with clipping
        ro_ind = rrr + int(local_offset)
        ro_ind = np.clip(ro_ind, 0, p_dim[ncols_ind] - 1)

        for yyy in prange(dim[nlines_ind]):
            a_ind = int(alpha[yyy, xxx])

            # Only process valid azimuth indices
            if a_ind >= 0:
                # Ensure sorting uses the exact same method
                ind = np.argsort(p_array[a_ind, ro_ind])
                ro[yyy, xxx] = ro_ind[ind[p_ind]]

        # Update offset
        local_offset += sampling

    return local_offset


def sample_volume(
        scans,
        outPointer,
        out_res,
        azimut_res,
        decile,
        medianBox,
        range_off,
        azimut_off,
        check_type=None,
        reload=False,
):
    """
Samples a set of radar scans and stores the processed data in the provided output array.

    Args:
        scans (list): A list of scan identifiers from which data will be sampled.
        outPointer (np.ndarray): The output array where the sampled data will be stored.
        out_res (float): The desired output resolution for the range dimension.
        azimut_res (float): The desired azimuth resolution.
        decile (int): The decile value used for sampling.
        medianBox (int): The size of the median filter box.
        range_off (float): The offset to be applied in the range dimension.
        azimut_off (float): The offset to be applied in the azimuth dimension.
        check_type (optional): Parameter to check the type of data (used for error handling). Defaults to None.
        reload (bool, optional): If True, forces reloading of data. Defaults to False.

    Returns:
        np.ndarray: The output array `outPointer` filled with the sampled data.
    """
    nScans = len(scans)
    if azimut_off is None:
        azimut_off = 0.0
    dim = outPointer.shape

    if len(dim) == 3:
        dim_nlines_ind = 1
        dim_ncols_ind = 2
    else:
        dim_nlines_ind = 0
        dim_ncols_ind = 1

    for idx, scan in enumerate(scans):
        inPointer, _, file_type = dpb.dpb.get_pointer(scan, reload=reload)
        if check_type and file_type is not None:
            log_message(f"Unreliable file {scan}", level="WARNING")
            inPointer = None
        if inPointer is not None:
            pDim = inPointer.shape
            if len(pDim) == 3:
                pDim_nlines_ind = 1
                pDim_ncols_ind = 2
            else:
                pDim_nlines_ind = 0
                pDim_ncols_ind = 1
            par_dict = dpg.navigation.get_radar_par(scan, get_az_coords_flag=True)
            rngResIn = par_dict["range_res"]
            azOffIn = par_dict["azimut_off"]
            azResIn = par_dict["azimut_res"]
            az_coords = par_dict["az_coords"]
            # non usato rngOffIn = par_dict["range_off"]
            # non usato elevation_off = par_dict["elevation_off"]

            azInd = dpg.beams.getAzimutBeamIndex(
                pDim[pDim_nlines_ind],
                dim[dim_nlines_ind],
                azOffIn,
                azimut_off,
                azResIn,
                azimut_res,
                az_coords_in=az_coords,
            )
            rngInd = dpg.beams.getRangeBeamIndex(
                pDim[pDim_ncols_ind],
                dim[dim_ncols_ind],
                rngResIn,
                out_res,
                range_off=range_off,
            )
            y = np.outer(azInd, np.ones(dim[dim_ncols_ind], dtype=int))
            # Perform matrix multiplication
            # y = (np.ones(dim[1])) @ azInd # Not working this way
            x = np.outer(np.ones(dim[dim_nlines_ind], dtype=int), rngInd)
            sampling = np.float32(out_res / rngResIn)
            if sampling > 1.0:
                offset = np.float32(range_off / rngResIn)
                offset = checkSamplingIndex(x, y, inPointer, sampling, decile, offset)

            # Assuming 'inPointer' is a NumPy array
            appo = inPointer[y.astype("int32"), x.astype("int32")]

            # Find indices where 'azInd' is less than 0, and set corresponding elements in 'appo' to NaN
            ind = np.where(azInd < 0)
            if ind[0].size > 0:
                appo[:, ind] = np.nan

            # Find indices where 'rngInd' is less than 0, and set corresponding elements in 'appo' to NaN
            ind = np.where(rngInd < 0)
            if ind[0].size > 0:
                appo[:, ind] = np.nan

            if medianBox > 0:
                appo = dpg.prcs.smooth_data(appo, medianBox, opt=1)

            outPointer[idx, :, :] = appo
        else:
            outPointer[idx, :, :] = np.nan

    return outPointer


# dpg.array.set_array(node, pointer=pointer) per settare il valore dentro il nodo


def sample_scans(
        volId,
        outId,
        in_dict,
        no_save=False,
        check_type=None,
        reload=False,
        check_null=None,
        attr=None,
        out_null=None,
):
    """
    Samples a set of scans from a 3D radar volume based on specified parameters and stores the processed data in an
    output volume.

    Args:
        volId: The identifier for the input volume from which scans are sampled.
        outId: The identifier for the output volume where sampled data will be stored.
        in_dict (dict): A dictionary containing parameters for sampling:
                        - min_el: Minimum elevation angle.
                        - max_el: Maximum elevation angle.
                        - decile: Decile value to use for sampling.
                        - medianBox: Size of the median filter box.
                        - min_scans: Minimum number of scans required.
                        - offset: Offset value to add to the sampled volume (optional).
        no_save (bool, optional): If True, prevents saving the output node after processing. Defaults to False.
        check_type (optional): Parameter for checking the type of data (purpose unclear). Defaults to None.
        reload (bool, optional): If True, forces reloading of data. Defaults to False.
        check_null (np.ndarray, optional): An array used to check for null values in the output. Defaults to None.
        attr (optional): Attributes for the output volume. Defaults to None.
        out_null (optional): Value to use for null entries in the output array. Defaults to None.

    Returns:
        list: A list of the sampled scans, or None if the sampling fails.
    """

    min_el = in_dict["min_el"]
    max_el = in_dict["max_el"]
    decile = in_dict["decile"]
    medianBox = in_dict["medianBox"]
    min_scans = in_dict["min_scans"]
    offset = in_dict["offset"]

    out_dict = dpg.access.get_scans(volId=volId, min_el=min_el, max_el=max_el)
    scans = out_dict["scans"]
    coord_set = out_dict["coord_set"]
    site_coords = out_dict["site_coords"]
    best_scan_ind = out_dict["best_scan_ind"]
    scan_dim = out_dict["scan_dim"]
    mode = out_dict["mode"]

    if mode != 1:
        log_message(f"Cannot Sample Volume {volId.path}", level="WARNING")
        return None
    nScans = len(scans)
    if nScans <= 0:
        log_message(f"Cannot Sample Volume {volId.path}", level="WARNING")
        return None
    if nScans < min_scans:
        log_message(f"Insufficient Scans {volId.path}", level="WARNING")
        return None

    if isinstance(scan_dim, list):
        dim = [nScans] + scan_dim
        # dim = scan_dim
        # dim.append(nScans)
    else:
        dim = [scan_dim, nScans]
    in_value = scans[best_scan_ind]
    type_value = 4

    id_calib = dpg.calibration.get_idcalibration(outId, only_current=True)
    if id_calib is not None:
        no_values = True
    else:
        no_values = False

    outPointer, par, _ = dpg.radar.check_in(
        in_value, outId, dim=dim, type=type_value, no_values=no_values
    )
    if outPointer is None:
        log_message(f"Cannot Create Volume {outId}", level="WARNING")
        return None

    par_dict = dpg.navigation.get_radar_par(None, par=par)
    range_res = par_dict["range_res"]
    azimut_res = par_dict["azimut_res"]
    range_off = par_dict["range_off"]
    azimut_off = par_dict["azimut_off"]

    outPointer = sample_volume(
        scans,
        outPointer,
        range_res,
        azimut_res,
        decile,
        medianBox,
        range_off,
        azimut_off,
        check_type=check_type,
        reload=reload,
    )

    dpb.dpb.put_radar_par(
        outId,
        par=par,
        site_coords=site_coords,
        el_coords=coord_set,
        remove_coords=(nScans == 1),
    )

    if offset:
        # TODO ramo da controllare
        outPointer += offset  # ?????

    if not no_values:
        nullInd = 0
        bottom = 1
        voidInd = 0
        _, _, _, countVoid = dpg.values.count_invalid_values(outPointer)
        if countVoid > 0:
            bottom = 2
            voidInd = 1
        dpb.dpb.put_values(
            outId, attr, alt_node=volId, bottom=bottom, nullInd=nullInd, voidInd=voidInd
        )

    if check_null is not None and outPointer is not None:
        if check_null.size == outPointer.size:
            # Find indices where check_null has nan value
            ind_null = np.where(np.isnan(check_null))
            if len(ind_null[0]) > 0:
                if out_null is None:
                    out_null = np.nan
                # Update the specified indices in out_pointer with out_null
                outPointer[ind_null] = out_null
                # dpg.array.set_array(outId, pointer=outPointer) non serve

    if not no_save:
        dpg.tree.saveNode(outId)

    log_message(f"Sampled {volId.path} in {outId.path}", level="WARNING")

    return scans
