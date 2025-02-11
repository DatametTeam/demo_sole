import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb


def Persistence(prodId):
    """
    Procedure for updating the product containing the persistence time of precipitation.

    Args:
        prodId: Node of the current product, from which you access the optional parameters contained in parameters.txt.

    Returns:
        - **data**: Data and metadata related to node passed as argument.
    """

    thresh, _, _ = dpg.radar.get_par(prodId, "thresh", 10.0)
    minStep, _, _ = dpg.radar.get_par(prodId, "minStep", -10)
    schedule, _, _ = dpg.radar.get_par(prodId, "schedule", "")
    site, _, _ = dpg.radar.get_par(prodId, "site", "MOSAIC")
    date, time, _ = dpg.times.get_time(prodId, nominal=True)

    var = dpb.dpb.get_data(
        prodId, schedule=schedule, date=date, time=time, site_name=site, numeric=True
    )

    if len(var) <= 1:
        return

    null_x, null_y = np.where(np.isnan(var))
    countNull = len(null_x)

    ind_x, ind_y = np.where(var <= thresh)
    count = len(ind_x)

    complement_x, complement_y = np.where(~(var <= thresh))
    nComp = len(complement_x)

    data = dpb.dpb.get_prev_data(prodId, minStep, numeric=False)

    if len(data) != len(var):
        dim = var.shape
        data = np.zeros(dim, dtype=np.uint8)
        if nComp > 0:
            data[complement_x, complement_y] = 2
    elif nComp > 0:
        data[complement_x, complement_y] += 1

    if count > 0:
        data[ind_x, ind_y] = 1

    if countNull > 0:
        data[null_x, null_y] = 0

    return data
