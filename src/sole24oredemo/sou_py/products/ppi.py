import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb
import sou_py.preprocessing as pre
from sou_py.dpg.log import log_message


def PPI(
        prodId,
        ppi=None,
        moment=None,
        sampled=None,
        put=False,
        index=None,
        remove_on_error=False,
        node=None,
        no_save=False,
        elevation=None,
        reload=None,
):
    # TODO: verificare la correttezza del commento
    """
    Procedure for extracting one or more PPIs from a volume


    Args:
        prodId: Node of the current product, from which you access the optional parameters contained in parameters.txt.
        elevation: Elevation angle.
        index: If greater than or equal to 0 elevation is ignored.
        n_elev: Number of elevations to use (if greater than 1 a 3D array is generated).


    :keywords:
        - MOMENT: Size to be used.
        - SAMPLED: If set the sampled volume is used.
        - PUT: If set the matrix is stored on the current node.
        - ELEVATION: If set 'elevation' is ignored.
        - INDEX: If set 'index' is ignored.


    :return:
        - **PPI**: Count the output matrix
    """

    # log_message("DA SISTEMARE PPI", level="ERROR")
    # return
    if ppi is not None:
        tmp = ppi

    if elevation is None:
        elevation, site_name = dpb.dpb.get_par(prodId, "elevation", 0)
    if index is None:
        index, site_name = dpb.dpb.get_par(prodId, "index", -1)
    if sampled is None:
        sampled, site_name = dpb.dpb.get_par(prodId, "sampled", 1)

    here = 1 - sampled
    volume, _, node = pre.sampling.sampling(
        prodId,
        sampled=sampled,
        moment=moment,
        here=here,
        remove_on_error=remove_on_error,
        reload=reload,
        no_save=no_save,
        get_volume=True
    )
    if volume is None:
        scan_dim = [0]
    else:
        scan_dim = volume.shape
    if len(scan_dim) < 2:
        return

    out_dict = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)
    el_coords = out_dict["el_coords"]
    par = out_dict["par"]

    if len(el_coords) <= 0:
        return

    n_elev, site_name = dpb.dpb.get_par(prodId, "n_elev", 1)

    if n_elev > 1:
        if n_elev > scan_dim[0]:
            n_elev = scan_dim[0]
        ppi = volume[0: n_elev - 1, :, :]
        elevation = el_coords[0: n_elev - 1]
        for eee in range(n_elev):
            elevation[eee] = dpg.access.roundClutterElev(elevation[eee])

    else:
        if index < 0:
            diff = np.abs(el_coords - elevation)
            delta = min(diff, index)
        ppi = volume[index, :, :]
        elevation = el_coords[index]

    if put is None:
        return

    dpb.dpb.put_data(prodId, ppi, main=node)
    dpb.dpb.put_radar_par(prodId, par=par, el_coords=elevation)
    dpb.dpb.put_par(prodId, "elevation", elevation)

    _, _, _, calib, _, _ = dpb.dpb.get_values(node=node)
    dpb.dpb.put_values(prodId, calib=calib)

    return
