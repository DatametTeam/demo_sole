import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message


def LoadCurrentStorms(node, nHours, delta_t, nZones, refDate, refTime):
    nCols = np.fix(nHours * 60. / delta_t)  # TODO: da controllare

    pointer, out_dict = dpg.array.get_array(node)
    dtype = out_dict['type'] if 'type' in out_dict else None
    dim = out_dict['dim'] if 'dim' in out_dict else None

    if pointer is not None:
        if np.shape(dim) != 2:
            pointer = None
        elif dim[-1] != nCols or dim[-2] != nZones:  # TODO: da controllare
            pointer = None

    if pointer is None:
        filename = 'SRT.dat'
        dim = [nCols, nZones]
        dpg.array.create_array(node, pointer, dtype=4, dim=dim, filename=filename, format='ascii')

    array = np.float(pointer)
    dates, times = dpg.times.get_times(node=node)
    if dates is None or times is None:
        array[:] = np.nan
        return array, nCols

    nH = dpg.times.getNHoursBetweenDates(dates[1], times[1], refDate, refTime) # TODO: controllare
    if nH == 0:
        return array, nCols
    if nH > 0:
        return -1, nCols
    if nH <= -nHours:
        array[:] = np.nan
        return array, nC

    nH *= 60
    if nH > -delta_t:
        nH = -delta_t
    nC = np.fix(nH/delta_t)

    if nC != 0:
        array = np.roll(array, nC)
        if nC < 0:
            array[nCols+nC:, :] = np.nan
        if nC > 0:
            array[0:nC, :] = np.nan

    return array, nCols


def storms(path, data, date, time, delta_t, coeff=None):
    if data is None:
        log_message("Cannot update Storms!", level='WARNING')
        return None

    tree = dpg.tree.createTree(path)
    maskNode = dpg.tree.getSon(tree, 'mask')
    pointer, _ = dpg.array.get_array(maskNode)

    if pointer is None or np.size(pointer) <= 1:
        log_message("Cannot load Mask!", level='WARNING')
        dpg.tree.removeTree(tree)
        return None

    if np.shape(pointer) != np.shape(data):
        log_message('Mask does not match Data!', level='WARNING')
        dpg.tree.removeTree(tree)
        return None

    nHours, prefix, _ = dpg.radar.get_par(tree, 'nHours', 12)
    par_ind, _, _ = dpg.radar.get_par(tree, 'par_ind', 3, prefix=prefix)
    percentile, _, _ = dpg.radar.get_par(tree, 'percentile', 90., prefix=prefix)
    thresh_filter, _, _ = dpg.radar.get_par(tree, 'thresh_filter', 10., prefix=prefix)
    count_thresh, _, _ = dpg.radar.get_par(tree, 'count_thresh', 5, prefix=prefix)
    valid_thresh, _, _ = dpg.radar.get_par(tree, 'valid_thresh', 1., prefix=prefix)

    if coeff is None:
        coeff = delta_t / 60.

    nZones = max(pointer)
    newArray, nCols = LoadCurrentStorms(tree, nHours, delta_t, nZones, date, time)

    if np.size(newArray) <= 1:
        dpg.tree.removeTree(tree)
        return None

    # TODO: da finire di fare il porting
