import numpy as np

import sou_py.dpg as dpg
from sou_py.dpg.log import log_message


def commanager(point, options=None, all_ent=None, prov=None, area=None, name=None):
    """
    Procedure dedicated to the management of municipal data (Comuni).
    To optimize the search for the municipalities affected by an event, a special pre-indexing is used:
    each pixel of the grid (1200x1400) belongs to exactly one municipality.

    Args:
        point (int): Position of the pixel (row*1200+column) in the grid.
        options (optional): Options for initial setup (default is None).
        all_ent (optional): A collection of all entities (default is None).
        prov (optional): The province associated with the municipality (default is None).
        area (optional): The area of the municipality (default is None).
        name (optional): The name of the municipality (default is None).

    Returns:
        name (str): The name of the municipality.
        area (float): The area of the municipality.
        prov (str): The province associated with the municipality.
        code: The code associated with the province.
    """

    init = dpg.dpg.check_RPG_options(options)
    if init is not None:
        log_message(
            "COMMANAGER INIT SET: TO BE IMPLEMENTED", level="ERROR"
        )  # TODO: da fare

    path = dpg.path.getDir("SHAPES", with_separator=True) + "Comuni"
    tree = dpg.tree.createTree(path, shared=True)

    attr, n_ent, _ = dpg.coords.get_shape_info(tree)

    if n_ent is None or n_ent <= 0:
        log_message(f"Cannot Init Commons @ {tree.path}", level="WARNING+")
        return

    if all_ent is not None:
        all_ent = dpg.map.check_map(tree)

    node = dpg.tree.getSon(tree, "index")
    p_index = dpg.array.get_array(node)
    if p_index is None:
        log_message(f"Cannot fine index @ {node.path}", level="WARNING+")
        return

    if point is not None:
        code = p_index[0][point]
        if code == -1:
            ind = np.where(code == -1)
            if len(ind[0]) > 0:
                name = "Mare"
                if prov > 0:
                    prov = ""
        else:
            if prov is not None:
                prov = attr.loc[code, "PV"]
            if area is not None:
                area = attr.loc[code, "Area"]
            if name is not None:
                name = attr.loc[code, "Name"]
                ind = np.where(code == 0)
                if len(ind[0]) > 0:
                    name = "Stato estero"
                    if len(prov) >= 0:
                        prov = ""
        if np.size(code) > 1:
            ind = np.unique(code)
            if np.size(name) > 1:
                name = name[ind]
            if np.size(area) > 1:
                area = area[ind]
            if np.size(prov) > 1:
                prov = prov[ind]
            code = code[ind]

        return name, code, prov, area
