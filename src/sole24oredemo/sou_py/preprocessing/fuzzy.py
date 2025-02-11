"""
Implementa l'algoritmo Fuzzy per classificare i dati radar in base a funzioni di appartenenza (MBF).
Le MBF definiscono il grado di appartenenza di una grandezza a diverse classi, come clutter e non-clutter.
Ogni grandezza (inclusi valori derivati come la tessitura) viene valutata per ogni cella radar.
Le MBF standard sono trapezoidali e possono essere invertite, con parametri come soglie e spread
configurabili per adattarsi ai dati specifici.
"""
import numpy as np

import sou_py.dpb as dpb
import sou_py.dpg as dpg
import sou_py.preprocessing as pre
from sou_py.dpg.log import log_message


def fuzzy(compId, fuzzyId=None, name=None, to_remove=False):
    '''
    This function processes a given component (compId) with fuzzy logic, applying thresholds, smoothing, and quality
    checks.
    It also supports removing the node if certain conditions are met, and can handle linking to a second fuzzy node (
    fuzzyId).

    Args:
        compId:       ID of the component to process.
        fuzzyId:      Optional. ID of a second component to link and apply quality checks.
        name:         Optional. Name for the quality check.
        to_remove:    Optional. If True, removes the node if certain conditions are met.

    Returns:
        None
    '''

    weight, _ = dpb.dpb.get_par(compId, "weight", default=1.0)
    el_index, _ = dpb.dpb.get_par(compId, "el_index", default=-1)

    if weight <= 0 and el_index < 0:
        if to_remove:
            dpg.tree.removeNode(compId, directory=True)

    pointer, dim, _ = dpb.dpb.get_pointer(compId)
    if pointer is None:
        if to_remove:
            dpg.tree.removeNode(compId, directory=True)

    up_tresh, _ = dpb.dpb.get_par(compId, "up_thresh", default=np.nan)
    up_spread, _ = dpb.dpb.get_par(compId, "up_spread", default=0.0)
    down_thresh, _ = dpb.dpb.get_par(compId, "down_thresh", default=np.nan)
    down_spread, _ = dpb.dpb.get_par(compId, "down_spread", default=0.0)
    inverse, _ = dpb.dpb.get_par(compId, "inverse", default=0)
    setVoid, _ = dpb.dpb.get_par(compId, "voidVal", default=1.0)
    setNull, _ = dpb.dpb.get_par(compId, "nullVal", default=1.0)
    maxVal, _ = dpb.dpb.get_par(compId, "maxVal", default=100.0)
    quality_name, _ = dpb.dpb.get_par(compId, "quality_name", default="")
    save_quality, _ = dpb.dpb.get_par(compId, "save_quality", default=0)
    subtract, _ = dpb.dpb.get_par(compId, "subtract", default=1)
    medianBox, _ = dpb.dpb.get_par(compId, "medianBox", default=0)

    if setVoid < 0:
        tmp = setVoid.copy()
    if setNull < 0:
        tmp = setNull.copy()

    pointer = dpg.prcs.trapez(
        pointer,
        down_thresh,
        up_tresh,
        down_spread,
        up_spread,
        maxVal=maxVal,
        inverse=inverse,
        setVoid=setVoid,
        setNull=setNull,
    )

    dpg.array.set_array(compId, pointer=pointer)

    if medianBox > 1:
        pointer = dpg.prcs.smooth_data(pointer, medianBox, opt=1)

    if quality_name != "" and weight > 0:
        if name is None:
            name = dpg.tree.getNodeName(compId)

        pre.quality.quality(
            compId,
            update=True,
            name=quality_name,
            test_name=name,
            subtract=subtract,
            maxVal=maxVal,
            weight=weight,
            to_save=save_quality,
        )
        if to_remove:
            dpg.tree.removeNode(compId, directory=True)

    if len(dim) == 3 and el_index >= 0:
        if el_index >= dim[0]:
            el_index = dim[0]

        pointer = pointer[el_index, :, :]
        dim = dim[-2:]
        dpg.array.set_array_info(compId, dim=dim)
        dpg.array.set_array(compId, pointer=pointer)

    if not isinstance(fuzzyId, dpg.node__define.Node):
        return

    destPointer, _, _ = dpb.dpb.get_pointer(fuzzyId)
    if destPointer is None:
        destPointer, par, _ = dpg.radar.check_in(
            node_in=compId, node_out=fuzzyId, type=4, pointer=destPointer, dim=dim
        )
        if destPointer is None:
            return
        destPointer = maxVal
    else:
        par = dpg.navigation.get_radar_par(compId)

    if pointer.size != destPointer.size:
        return

    pre.quality.quality(
        compId,
        pointer,
        update=True,
        subtract=subtract,
        maxVal=maxVal,
        weight=weight,
        qscanid=fuzzyId,
        qvolid=fuzzyId,
        to_save=False,
    )

    dpg.radar.check_out(out=compId, pointer=pointer, par=par)
    dpg.radar.check_out(out=fuzzyId, pointer=destPointer, par=par)
