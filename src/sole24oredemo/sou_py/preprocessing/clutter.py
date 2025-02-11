import os.path

import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb
import sou_py.products as products
from sou_py.dpg.log import log_message


def clutter_getmap(prodId, moment):
    """
    Retrieves and processes a clutter map for the specified radar moment, associating it with the product ID.

    Args:
        prodId: The product identifier associated with the radar data.
        moment (str): The radar moment (e.g., 'CZ', 'UZ') for which the clutter map is being retrieved.

    Returns:
        None: The function performs processing on the clutter map and updates the product but does not return a value.
    """
    volId, nVol = dpg.access.get_needed_volumes(
        prodId, measure=moment, remove_if_not_exists=True
    )

    if nVol <= 0:
        return

    max_el, site_name, _ = dpg.radar.get_par(prodId, "max_el", 10.0)

    scans_dict = dpb.dpb.get_scans(volId=volId, max_el=max_el)
    scans = scans_dict["scans"]
    coord_set = scans_dict["coord_set"]

    if len(scans) <= 0:
        return

    dim = scans_dict["scan_dim"]
    dim = [len(scans)] + dim

    outPointer, par, _ = dpg.radar.check_in(
        node_in=scans[scans_dict["best_scan_ind"]],
        node_out=prodId,
        dim=dim,
        type=4,
        filename="volume.dat",
    )

    if outPointer is None:
        return

    outPointer.fill(np.nan)
    nEl = dim[1] * dim[2]

    clutterRoot = dpg.access.getClutterMapRoot(site_name=site_name, class_name="map")

    for i, sss in enumerate(scans):
        clutterNode = dpg.access.getClutterMapNode(sss, clutterRoot)
        cl_map = dpg.warp.warp_map(clutterNode, prodId, regular=True, numeric=True)
        if np.size(cl_map) == nEl:
            outPointer[i, :, :] = cl_map
        else:
            if np.size(cl_map) > 0:
                tmp = cl_map
        dpg.tree.removeNode(clutterNode)

    dpg.tree.removeTree(clutterRoot)

    dpg.radar.check_out(outId=prodId, pointer=outPointer, par=par, el_coords=coord_set)
    return


def copy_ppi_to_clutter(scan, root, elevation):
    err = 0
    path = dpg.tree.getNodePath(root)
    date, time, _ = dpg.times.get_time(scan)
    if date is None:
        err = 1
        log_message(f"Cannot find date @ node {scan.path}", level="WARNING")
        return err

    next = date + '.' + time
    next = "".join(next.split(":"))
    if os.path.isdir(os.path.join(path, next)):
        err = 1
        return err

    out, _ = dpg.tree.addNode(root, next, remove_if_exists=True)
    pointer, _, _ = dpg.radar.check_in(node_in=scan, node_out=out, filename='ppi.dat')
    if pointer is None:
        err = 1
        log_message(f"Not valid PPI @ {scan.path}", level='WARNING')
        return err

    dpg.radar.set_par(out, 'elevation', elevation, only_current=True, format='(F4.1)')
    dpg.radar.check_out(outId=out)
    dpg.tree.saveNode(out)
    return err


def add_volume_to_clutter(samplesRoot, volId, max_el, reset, max_samples, delay):
    scans_dict = dpb.dpb.get_scans(volId=volId, max_el=max_el)
    if scans_dict is not None:
        scans = scans_dict['scans']
        coord_set = scans_dict['coord_set']
    else:
        log_message(f"Can't find scans @ {volId.path}", level='ERROR')
        return

    added = 0
    nH = 1
    nScans = len(scans)
    date, time, _ = dpg.times.get_time(volId)

    for sss in range(nScans):
        samplesNode = dpg.access.getClutterMapNode(scans[sss], samplesRoot)
        if samplesNode is not None and isinstance(samplesNode, dpg.node__define.Node):
            if delay > 0:
                last_date, last_time, _ = dpg.times.get_time(samplesNode)
                nH, _ = dpg.times.getNHoursBetweenDates(date, time, last_date, last_time, double=True)
            if nH > delay:
                err = copy_ppi_to_clutter(scans[sss], samplesNode, coord_set[sss])
                if err == 0:
                    dpg.access.remove_old_samples(samplesNode, max_samples)
                    added += 1
                    dpg.times.set_time(samplesNode, date=date, time=time, to_save=True)

    return added


def compute_clutter_map(destNode, samplesNode, outRes=None, threshold=None, perc_thresh=0):
    valids = 0
    scans_dict = dpb.dpb.get_scans(samplesNode)
    if scans_dict is not None:
        scans = scans_dict['scans']
        scan_dim = scans_dict['scan_dim']
        scale = scans_dict['scale']
        nScans = len(scans)
    else:
        log_message(f"Cannot find scans @ {samplesNode.path}", level='ERROR')
        return 0

    type_ = 4
    outPointer, par, values = dpg.radar.check_in(node_in=scans[-1], node_out=destNode, dim=scan_dim, type=type_,
                                                 no_par=True, values=True, filename='clutter.dat')

    if outPointer is None or np.size(outPointer) <= 1:
        return 0

    par_dict = dpg.navigation.get_radar_par(par=par)
    if par_dict is not None:
        par = par_dict['par']
        site_coords = par_dict['site_coords']
        range_off = par_dict['range_off']
        range_res = par_dict['range_res']
        azimut_off = par_dict['azimut_off']
        azimut_res = par_dict['azimut_res']
    else:
        log_message("Cannot create par_dict. ERROR", level='ERROR')
        return 0

    if outRes is not None and outRes != 0:
        range_res = outRes

    azOffIn = np.float32(0)
    azResIn = np.float32(1)
    azimut_off = np.float32(0)
    parFile = dpg.cfg.getItemDescName()
    elevation = None

    if np.size(azimut_res) == 0 or azimut_res is None:
        azimut_res = 1
    azimut_res = np.abs(azimut_res)
    if np.size(threshold) == 1 and threshold is not None:
        validCounter = np.zeros(scan_dim)

    for sss in range(nScans):
        hide, _, ok = dpg.radar.get_par(scans[sss], 'hide', 0, parFile=parFile, prefix=dpg.tree.getNodeName(scans[sss]))
        if hide == 0:
            inPointer, arr_dict = dpg.array.get_array(scans[sss])
            if inPointer is not None:
                pDim = arr_dict['dim']
                ok = 1
            else:
                ok = 0
        else:
            ok = 0
        if ok:
            in_par_dict = dpg.navigation.get_radar_par(scans[sss], get_az_coords_flag=True)
            if in_par_dict is not None:
                rngOffIn = in_par_dict['range_off']
                rngResIn = in_par_dict['range_res']
                azOffIn = in_par_dict['azimut_off']
                azResIn = in_par_dict['azimut_res']
                az_coords = in_par_dict['az_coords']
                elevation = in_par_dict['elevation_off']
            else:
                log_message(f"Cannot red parameters.txt from {scans[sss].path}. ERROR", 'ERROR')
                return
            if azimut_off is None:
                azimut_off = azOffIn
            if azimut_res is None:
                azimut_res = azResIn
            azInd = dpg.beams.getAzimutBeamIndex(pDim[0], scan_dim[0], azOffIn=azOffIn, azOffOut=azimut_off,
                                                 azResIn=azResIn, azResOut=azimut_res, az_coords_in=az_coords)
            rngInd = dpg.beams.getRangeBeamIndex(pDim[1], scan_dim[1], rngResIn, range_res, range_off=range_off)
            sampling = int(range_res / rngResIn)
            out = dpg.base.update_max_map(in_pointer=inPointer, out_pointer=outPointer, col_ind=rngInd, lin_ind=azInd,
                                          sampling=sampling, values=values)

            if np.size(out) > 0 and out is not None:
                if np.size(out) == np.size(validCounter):
                    ind = np.where(out >= threshold)
                    validCounter[ind] += 1
            valids += 1

    _, count_null, _, _ = dpg.values.count_invalid_values(outPointer)
    if np.size(validCounter) > 0:
        ind = np.where(validCounter < perc_thresh * valids / 100)
        outPointer[ind] = -np.inf

    elevation_res = 0

    dpg.navigation.set_radar_par(scan_dim[1], site_coords=site_coords, range_off=range_off, range_res=range_res,
                                 azimut_off=azimut_off, azimut_res=azimut_res, elevation_off=elevation,
                                 elevation_res=elevation_res, par=par)

    dpg.radar.check_out(outId=destNode, pointer=outPointer, par=par)

    if count_null > 0:
        nullInd = 0
        voidInd = 1
    else:
        nullInd = -1
        voidInd = 0

    dpg.calibration.set_values(destNode, nullInd=nullInd, voidInd=voidInd)

    log_message(f"{valids} samples in {dpg.tree.getNodePath(samplesNode)}", level='INFO')
    dpg.tree.saveNode(destNode, only_current=True)
    return valids


def update_clutter_map(samplesRoot, sitename, outRes, threshold=None, perc_thresh=None, paths=[]):
    sons = dpg.tree.getSons(samplesRoot)
    if len(sons) <= 0:
        elev = 90.
        return paths, elev

    elev = []

    mapRoot = dpg.access.getClutterMapRoot(sitename, 'map')
    for sss in range(len(sons)):
        name = dpg.tree.getNodeName(sons[sss])
        clutter, _ = dpg.tree.addNode(mapRoot, name)
        elNodes = dpg.tree.getSons(sons[sss])
        for eee in range(len(elNodes)):
            el = dpg.tree.getNodeName(elNodes[eee])
            outNode, _ = dpg.tree.addNode(clutter, el)
            valids = compute_clutter_map(outNode, elNodes[eee], outRes, threshold=threshold, perc_thresh=perc_thresh)
            if valids > 0:
                if np.size(paths) == 0:
                    paths = [dpg.tree.getNodePath(outNode)]
                    elev = [float(el)]
                else:
                    paths = paths + [dpg.tree.getNodePath(outNode)]
                    elev = elev + [float(el)]
            dpg.tree.removeNode(elNodes[eee])
            dpg.tree.removeNode(outNode)

    dpg.tree.removeTree(mapRoot)
    if np.size(paths) > 0:
        sorted_pairs = sorted(zip(elev, paths))
        elev, paths = zip(*sorted_pairs)
        elev = list(elev)
        paths = list(paths)
    else:
        elev = 90

    return paths, elev


def clutter_update(prodId, moment, attr):
    nSamples, site_name, _ = dpg.radar.get_par(prodId, 'samples_to_add', 0)
    max_samples, _, _ = dpg.radar.get_par(prodId, 'max_samples', 20, prefix=site_name)
    max_el, _, _ = dpg.radar.get_par(prodId, 'max_el', 3., prefix=site_name)
    max_el_to_show, _, _ = dpg.radar.get_par(prodId, 'max_el_to_show', 2., prefix=site_name)
    reset, _, _ = dpg.radar.get_par(prodId, 'reset', 0, prefix=site_name)
    if nSamples <= 0:
        return

    outRes, _, _ = dpg.radar.get_par(prodId, 'outRes', 0., prefix=site_name)
    threshold, _, _ = dpg.radar.get_par(prodId, 'threshold', 0., prefix=site_name)
    perc_thresh, _, _ = dpg.radar.get_par(prodId, 'perc_thresh', 50., prefix=site_name)
    delay, _, _ = dpg.radar.get_par(prodId, 'delay', 0., prefix=site_name)
    date, time, _ = dpg.times.get_time(prodId)

    samplesRoot = dpg.access.getClutterMapRoot(site_name, 'samples')
    nSamplesToAdd, _, _ = dpg.radar.get_par(samplesRoot, 'samples_to_add', nSamples)
    nSamplesAdded, _, _ = dpg.radar.get_par(samplesRoot, 'samples_added', 0)

    status = 0
    if nSamples == nSamplesToAdd:
        if nSamplesAdded >= nSamplesToAdd:
            status = 1
            if delay > 0:
                status = products.weatherstatus.weatherStatus(site_name)
            nSamplesAdded = nSamplesToAdd
        else:
            delay = 0
        reset = 0
    else:
        nSamplesAdded = 0

    if status != 0:
        log_message("ClutterMap not updated!", level='WARNING')
        dpg.tree.removeTree(samplesRoot)
        dpg.tree.removeNode(prodId, directory=True)
        return

    volId, nVol = dpg.access.get_needed_volumes(prod_id=prodId, measure=moment, remove_if_not_exists=True)
    if nVol <= 0:
        return

    added = add_volume_to_clutter(samplesRoot, volId, max_el=max_el, reset=reset, max_samples=max_samples, delay=delay)

    toRemove, _, _ = dpg.radar.get_par(prodId, 'toRemove', 1, attr=attr)

    if added > 0:
        paths, elev = update_clutter_map(samplesRoot, site_name, outRes, threshold=threshold, perc_thresh=perc_thresh)
        log_message("ClutterMap Updated", level='INFO')
        if toRemove == 0:
            if len(paths) > 0 and elev[0] <= max_el_to_show:
                scan = dpg.tree.createTree(paths[0])
                dpg.radar.check_in(node_in=scan, node_out=prodId)
                dpg.times.set_time(prodId, date=date, time=time)
                dpg.calibration.remove_array_values(prodId)
                dpg.tree.removeTree(scan)
            else:
                toRemove = 1
        nSamplesAdded += 1
    else:
        log_message("ClutterMap is already Updated!", level='INFO')
        toRemove = 1

    if toRemove > 0:
        dpg.tree.removeNode(prodId, directory=True)

    tags = ['samples_to_add', 'samples_added']
    values = [nSamples, nSamplesAdded]

    dpg.radar.set_par(samplesRoot, tags, values, only_current=True, to_save=True)
    dpg.tree.removeTree(samplesRoot)

    return


def clutter(prodId, moment=None, getMap=False, update=False, attr=None, remove=False):
    """
     Handles clutter map operations for radar data, including retrieving, updating, and removing clutter maps.

    Args:
        prodId: The product identifier associated with the radar data.
        moment (str, optional): The radar moment, if not provided, it is retrieved from the product configuration.
        getMap (bool, optional): If True, retrieves and processes the clutter map. Defaults to False.
        update (bool, optional): If True, updates the clutter map with new data. Defaults to False.
        attr (Attr, optional): Attribute data to be used during clutter map updates or removals. Defaults to None.
        remove (bool, optional): If True, removes the clutter map associated with the specified moment. Defaults to
        False.

    Returns:
        None: The function performs the specified operation (get, update, remove) but does not return any value.
    """
    if moment is None:
        moment, _, _ = dpg.radar.get_par(prodId, name="measure", default="")
    if moment == "":
        moment, _, _ = dpg.radar.get_par(prodId, "moment", default="")

    if getMap:
        clutter_getmap(prodId, moment=moment)
        return

    if update:
        clutter_update(prodId, moment=moment, attr=attr)
        return

    if remove:
        clutter_remove(prodId, moment=moment, attr=attr)
        return

    return
