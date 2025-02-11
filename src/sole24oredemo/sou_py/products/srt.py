"""
Stima la precipitazione cumulata su un periodo di N ore, utilizzando mappe SRI generate con alta frequenza per maggiore precisione.
Integra i dati di precipitazione istantanea e supporta il calcolo di cumulate successive su intervalli orari.
"""
import numbers
import numpy as np
import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message


def sr_total(
    current,
    max_minutes,
    step,
    weight,
    filterThresh,
    warp=None,
    schedule=None,
    name=None,
    sourceNode=None,
    filterSchedule=None,
):
    """
    Function to calculate the total surface precipitation in a given time interval.
    This function accumulates instantaneous precipitation maps at defined time intervals and applies
    any filters and weights specified.

    Args:
        current (Node):               Current node to get intial date and time.
        max_minutes (int):            Max time (in minute) for accumulation of precipitation
        step (int):                   Time interval (in minute) between precipitation maps to accumulate.
        weight (float):               Weight applied to precipitation maps.
        filterThresh (float):         Threshold to filter precipitation values.
        warp:                         If set, the current node is set as a warp node.
        schedule (str or None):       Schedule for retrieving precipitation maps.
        name (str or None):           Name of site or precipitation product to be used.
        sourceNode (Node):            Node containing the SRT to be integrated.
        filterSchedule:               Schedule for applying specific filters (optional).

    Returns:
        srt (np.ndarray):            Array containing the total accumulated surface precipitation.
    """
    if warp is not None:
        warpNode = current
    destNode = current

    date, time, _ = dpg.times.get_time(current)
    minutes = 0
    min_step = int(np.abs(step))
    if min_step <= 10:
        date, time = dpg.times.addMinutesToDate(date, time, -min_step)

    srt, valids = [], []
    while minutes < max_minutes:
        var = dpb.dpb.search_data(
            schedule,
            name,
            destNode=destNode,
            date=date,
            time=time,
            sourceNode=sourceNode,
        )
        if isinstance(var, numbers.Number):
            var = np.array([var])

        if len(var) > 1:
            ix, iy = np.where(np.isfinite(var))
            var = np.where(np.isnan(var), 0.0, var)
            if len(ix) > 1:
                if filterSchedule is not None:
                    print("TODO: implement filter srt")
                else:
                    if filterThresh > 0:
                        var = np.where(var < filterThresh, 0)
                if weight < 1.0:
                    var *= weight
                if len(srt) == 0:
                    srt = var.astype(float)
                    valids = np.zeros(srt.shape, dtype=np.uint8)
                else:
                    if var.shape == srt.shape:
                        srt += var

                valids[ix, iy] = 1

        minutes += min_step
        date, time = dpg.times.addMinutesToDate(date, time, -min_step)

    if len(srt) <= 0:
        return np.array([0], dtype=np.int32)

    if len(valids) <= 0:
        return np.array([0], dtype=np.int32)

    srt = np.where(valids == 0, np.nan, srt)
    return srt


def SRT(prodId, srtNode=None, prog=None, name=None, origin=None, filterSchedule=None):
    """
    Algorithm for calculating the SRT (Surface Rainfall Total) product, which provides an estimate of cumulative
    precipitation in a given time interval (N hours). The algorithm takes into account all precipitation maps
    snapshot produced by the indicated schedule and the frequency of the schedule itself.

    Args:
        prodId (Node):         Identifier for the radar product.
        srtNode (Node):        None containing the SRT to be integrated (only used in prog mode).
        prog:                  E.G. To calculate a 6 hour SRT you use current SRT_3 + 3 hour ago SRT_3. And so on.
        name:                  Name of site or precipitation product to be used.
        origin:                If set, the origin value is extracted from prodId.
        filterSchedule:        Schedule for applying specific filters (optional)

    Returns:
        srt (np.ndarray):     Array containing the total accumulated surface precipitation.
        minutes (int):        Total time in minutes for which precipitation was calculated.
    """
    n_hours, _, _ = dpg.radar.get_par(prodId, "n_hours", 1)
    step, _, _ = dpg.radar.get_par(prodId, "step", 60)
    max_weight, _, _ = dpg.radar.get_par(prodId, "max_weight", 0.0)
    min_weight, _, _ = dpg.radar.get_par(prodId, "min_weight", 0.0)
    www = step / 60.0

    if max_weight > 0 and www > max_weight:
        www = max_weight
    if min_weight > 0 and www < min_weight:
        www = max_weight

    max_minutes = n_hours * 60
    if prog is not None:
        print("TODO: to be implemented prog not None")

    schedule, _, _ = dpg.radar.get_par(prodId, "schedule", "")
    filterThresh, _, _ = dpg.radar.get_par(prodId, "filterThresh", -1.0)

    if name is None:
        name, _, _ = dpg.radar.get_par(prodId, "site", "MOSAIC")

    if origin is None:
        origin, _, _ = dpg.radar.get_par(prodId, "origin", "")

    if filterSchedule is not None:
        filterSchedule, _, _ = dpg.radar.get_par(prodId, "filterName", "ITALIA")
        filterWarp, _, _ = dpg.radar.get_par(prodId, "filterWarp", 0)

    srt = sr_total(
        prodId,
        max_minutes,
        step,
        www,
        filterThresh,
        schedule=schedule,
        name=name,
        sourceNode=srtNode,
    )
    minutes = max_minutes

    # dpg.radar.check_out(prodId, srt) check_out sostituita con put_data
    dpb.dpb.put_data(prodId, srt)
    return srt, minutes


def SRT_recompute(prodId, srtNode, delays):
    """
    Recomputes the SRT (Signal Recovery Time) for a given product and its associated nodes.

    This function iterates through the provided list of delays, computes the SRT for each corresponding product node,
    and saves the updated result. For each delay, the previous node is retrieved, the SRT computation is performed,
    and the node is saved. Finally, the node is removed from the tree after processing.

    Args:
        prodId (Node): The product ID or node that serves as the starting point for the recomputation.
        srtNode (Node): The node where the SRT computation will be applied.
        delays (list): A list of delay values to process.

    Returns:
        None: The function performs actions on the product node and updates the node tree, but does not return any value.

    """

    for ddd in delays:
        prev = dpb.dpb.get_prev_node(prodId, ddd)
        _, _ = SRT(prev, srtNode)
        dpg.tree.saveNode(prev)
        if isinstance(prev, dpg.node__define.Node):
            path = prev.path
        log_message(f"Recomputed... {path}")
        dpg.tree.removeTree(prev)
