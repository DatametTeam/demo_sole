import numpy as np

import sou_py.dpg as dpg
import sou_py.dpg as dpb
from sou_py.dpg.log import log_message


def update_weather_status(siteName, test_name, test_val, thresh, log, status):
    status = 1
    curr_val = 0
    status_file = dpg.cfg.getDefaultStatusFile(siteName)
    attr = dpg.attr.loadAttr(pathname=status_file)
    curr_stat, _, _ = dpg.attr.getAttrValue(attr, 'status', 0)
    new_line = 1

    if test_val == 0:
        curr_val, _, _ = dpg.attr.getAttrValue(attr, test_name, 0)
        curr_val += 1
        if np.size(thresh) == 1:
            tags, values, nTags = dpg.attr.getAllTags(attr)
            if nTags > 0:
                ind = np.where(np.float32(values) > thresh)
                if len(ind[0]) >= nTags - 1:
                    status = 0
        else:
            status = curr_stat

    attr = dpg.attr.replaceTags(attr, [test_name, 'status'], [str(curr_val), str(status)], to_create=True)
    dpg.attr.writeAttr(attr, pathname=status_file, format='txt')

    if np.size(log) == 1:
        log_message(f"{test_name}: {log}", level='INFO')

    if test_val == 0:
        log_message(f"{test_name} is good for {curr_val} times")

    if status != curr_stat:
        msg = siteName + ' ... status changed: '
    else:
        msg = siteName + ' ... status not changed: '
    if status <= 0:
        msg += 'GOOD WEATHER!'
    else:
        msg += 'BAD WEATHER!'

    log_message(msg, level='INFO')

    return


def weatherStatus(siteName, data=None, test_name='', thresh_val=None, thresh_count=None, n_times=None, update=False):
    status = -1
    if np.size(siteName) != 1:
        return
    if siteName == '':
        return

    if update:
        if np.size(data) > 1:
            if thresh_count <= 0:
                return
            ind = np.where(data > thresh_val)
            count = len(ind[0])
            test_val = count > thresh_count
            log = f"{str(count).strip()} pixel greater than {str(thresh_val).strip()}"
            update_weather_status(siteName, test_name, test_val, n_times, log=log, status=status)
            return

        log_message('Cannot evaluate weather status!', level='WARNING')

    file = dpg.cfg.getDefaultStatusFile(siteName)
    attr = dpg.attr.loadAttr(pathname=file)
    status, _, _ = dpg.attr.getAttrValue(attr, 'status', -1)

    return status
