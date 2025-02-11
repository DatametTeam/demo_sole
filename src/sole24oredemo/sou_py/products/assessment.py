import os.path

import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message


def get_last_assessment(node, warp=False, linear=False):
    schedule, _, _ = dpg.radar.get_par(node, 'assSchedule', '$SCHEDULES/SENSORS/VALID/Assessment')
    site_name, _, _ = dpg.radar.get_par(node, 'assSite', 'ITALIA')
    date, time, _ = dpg.times.get_time(node)
    var = dpb.dpb.get_data(node, numeric=True, schedule=schedule, site_name=site_name, warp=warp, linear=linear,
                           date=date, time=time, n_hours=2)

    if len(np.shape(var)) <= 1:
        return None

    if linear:
        ind = np.where((~np.isfinite(var)) | (var <= 0))
        var[ind] = 1
        var = 1 / var

    return var


def assessment(prodId, data=None, warp=None, linear=None, ass=None, node_name=None, origin=None, attr=None):
    if data is not None:
        data = get_last_assessment(prodId, warp=warp, linear=linear)
        return data

    if ass is not None:
        ass = None

    delay, site_name, _ = dpg.radar.get_par(prodId, 'delay', 0)
    thresh, _, _ = dpg.radar.get_par(prodId, 'thresh', 1., prefix=site_name)
    mode, _, _ = dpg.radar.get_par(prodId, 'mode', 0, prefix=site_name)
    linear, _, _ = dpg.radar.get_par(prodId, 'linear', 1, prefix=site_name)
    inverse, _, _ = dpg.radar.get_par(prodId, 'inverse', 0, prefix=site_name)
    log, _, _ = dpg.radar.get_par(prodId, 'log', 1, prefix=site_name)
    schedule, _, _ = dpg.radar.get_par(prodId, 'schedule', '', prefix=site_name)
    date, _, _ = dpg.radar.get_par(prodId, 'date', '', prefix=site_name)
    time, _, _ = dpg.radar.get_par(prodId, 'time', '', prefix=site_name)
    min_step, _, _ = dpg.radar.get_par(prodId, 'min_step', 60, prefix=site_name)
    site_name, _, _ = dpg.radar.get_par(prodId, 'site', site_name)

    if node_name is None:
        node_name, _, _ = dpg.radar.get_par(prodId, 'node_name', node_name)

    if date == '':
        date, time, _ = dpg.times.get_time(prodId)

    if delay != 0:
        date, time = dpg.times.addMinutesToDate(date=date, time=time, minutes=-60 * delay)
        dpg.times.set_time(prodId, date=date, time=time)

    nProds, prod_path = dpg.access.get_aux_products(prodId, schedule=schedule, site_name=site_name, last_date=date,
                                                    last_time=time, min_step=min_step, origin=origin)
    if nProds <= 0:
        log_message(f"Nothing to do for {site_name} @ {date} {time}", level='WARNING')
        return None

    if isinstance(prod_path, list):
        prod_path = prod_path[0]

    if not os.path.isdir(prod_path):
        log_message(f"Cannot find {prod_path}", level='WARNING+')
        return None

    log_message(f"{date}_{time} : using {prod_path}")

    tree = dpg.tree.createTree(prod_path)
    node = dpg.radar.find_site_node(tree, node_name, origin=origin)
    if not isinstance(node, dpg.node__define.Node):
        log_message(f"Cannot find {node_name}", level='WARNING+')
        return None

    radar = dpg.prcs.get_numeric_array(node)
    if np.shape(radar) <= 1:
        return None

    no_values = mode == 0

    pointer, _, _ = dpg.radar.check_in(node_in=node, node_out=prodId, type=4, filename='assesment.dat',
                                       no_values=no_values)

    outMap, _, outDim, outPar, _, _, _, _, _ = dpg.navigation.check_map(prodId, destMap=True)

    dpg.tree.removeTree(tree=tree)

    if pointer is None:
        return

    gaugeSchedule, _, _ = dpg.radar.get_par(prodId, 'gaugeSchedule', '')
    gaugeSite, _, _ = dpg.radar.get_par(prodId, 'gaugeSite', 'ITALIA')

    if gaugeSchedule != '':
        gauge = dpb.dpb.get_data(prodId, numeric=True, warp=True, schedule=gaugeSchedule, site_name=gaugeSite,
                                 date=date, time=time)

        if gauge is None or np.shape(gauge) != np.shape(radar):
            log_message(f"Cannot get data for {gaugeSchedule}", level='WARNING+')
            return None

    if mode != 4 and np.shape(gauge) != np.shape(radar):
        nHours, _, exists = dpg.radar.get_par(node, 'nhours', 1)
        if exists <= 0:
            n_hours, _, _ = dpg.radar.get_par(node, 'n_hours', 1)
        count, values = get_all_srt()  # TODO: ??? dove è sta funzione in IDL?
        if count >= 3:
            ind = np.where(values > 0)
        if count < 3:
            log_message(f"Insufficient number of samples for {site_name}", level='WARNING')
            summary_rain_path, site_name, _ = dpg.radar.get_par(prodId, 'summary_rain_path',
                                                                '/data1/SENSORS/SUMMARY/raingauges')
            count, values = GET_SRT_FROM_SUMMARY(summary_rain_path, date, time, CODES=codes, NHOURS=nHours)
            if count <= 0:
                log_message("Cannot read summary", level='WARNING')
                return None

        ret = GET_STATION_LOCATIONS(regions, paths, lats, lons, CODES=codes)
        CREATE_SCATTER_IMAGE(values, lats, lons, outDim, outMap, outPar, gauge, LINEAR=linear, check_values=True)
        if np.shape(gauge) != np.shape(radar):
            return None
        ind = np.where(~np.isfinite(gauge))
        gauge[ind] = 0.

        dem = dpg.warp.get_dem(outMap, outPar, outDim, numeric=True, hr=True)
        if np.shape(dem) == np.shape(gauge):
            ind = np.where(dem <= 0)
            gauge[ind] = -np.inf

    if mode == 1:
        if isinstance(attr, dpg.attr.Attr):
            calib = attr
            dpg.attr.replaceTags(calib, ['parname', 'scaling'], ['Error', '4'])
        if inverse:
            ass = radar - gauge

    elif mode == 2:
        ass = gauge

    elif mode == 3:
        ass = radar

    elif mode == 4:
        up_thresh, _, _ = dpg.radar.get_par(prodId, 'up_thresh', 4.)
        down_thresh, _, _ = dpg.radar.get_par(prodId, 'down_thresh', 1.)
        warp, _, _ = dpg.radar.get_par(prodId, 'warp', 0)
        ass = get_last_assessment(prodId, down_thresh=down_thresh, up_thresh=up_thresh, warp=warp,
                                  linear=True)  # Ignorare, non viene mai chiamata così
        if np.shape(ass) == np.shape(radar):
            ass *= radar
        else:
            ass = radar

    else:
        if isinstance(attr, dpg.attr.Attr):
            calib = attr
        if inverse:
            ass = gauge / radar
        else:
            ass = radar / gauge

        ind = np.where(gauge == radar)
        ass[ind] = 1
        if log:
            ass = 10 * np.log10(ass)
        ind = np.where(gauge <= thresh)
        ass[ind] = -np.inf
        ind = np.where(gauge > thresh and radar <= 0)
        ass[ind] = np.nan

    pointer = ass
    dpg.radar.check_out(outId=prodId, par=outPar, reset_values=True, attr=calib)

    return ass


def thresh_assessment(ass, aux, x0, x1, y0, y1, down_thresh=None, up_thresh=None):
    if ass is None or np.size(ass) <= 1:
        return None

    if np.size(aux) == np.size(ass):
        coeff = (y1 - y0) / (x1 - x0)
        thresh = aux - x0
        thresh *= coeff
        thresh += y0

        if y1 < y0:
            ind = np.where(thresh < y1)
            thresh[ind] = y1
            ind = np.where(thresh > y0)
            thresh[ind] = y0
        else:
            ind = np.where(thresh > y1)
            thresh[ind] = y1
            ind = np.where(thresh < y0)
            thresh[ind] = y0

        ass_tmp = ass.copy()
        ass = np.minimum(ass, thresh)
        ind = np.where(np.isnan(thresh))
        ass[ind] = ass_tmp[ind]

    ind = np.where((~np.isfinite(ass)) | (ass <= 0))
    ass[ind] = 1.

    if down_thresh is not None:
        ind = np.where(ass < down_thresh)
        ass[ind] = down_thresh

    if up_thresh is not None:
        ind = np.where(ass > up_thresh)
        ass[ind] = up_thresh

    return ass
