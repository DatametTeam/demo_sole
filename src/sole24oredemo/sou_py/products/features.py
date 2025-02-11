import os.path

import numpy as np

import sou_py.dpg as dpg
import sou_py.dpb as dpb


def features(prodId):
    """
    Extracts and organizes radar feature data for a given product.

    This function retrieves radar feature data based on the product ID. It processes various parameters such as
    schedule, site, nodeName, and auxiliary information. It also handles time adjustments and organizes the data
    into a structured format for further analysis.

    Args:
        prodId (Node): Node from which radar feature data is extracted.

    Returns:
        tuple: A tuple containing:
            - features_data (numpy.ndarray): A 3D array of radar feature data
            - names (list): A list of names for the features.

        If no valid data is found, the function returns (None, None).
    """

    features_data = None

    date, time = dpg.times.get_times(prodId)
    names, _, _ = dpg.radar.get_par(prodId, "product", "")
    nFeat = len(names)

    _, owner, system = dpg.schedule__define.get_schedule_path(prodId)

    replace = 0
    if system != "RADAR" and system != "EXTRA":
        path = dpg.tree.getNodePath(owner)
        path = os.path.join(os.path.basename(path), "BASIC")
        replace = os.path.isdir(path)

    for ppp in range(nFeat):
        schedule, _, _ = dpg.radar.get_par(prodId, "schedule", "", prefix=names[ppp])
        if schedule != "":
            if replace > 0:
                pos = schedule.find("RADAR")
                if pos >= 0:
                    schedule = schedule[0:pos] + system + schedule[pos + 5 :]

            site, _, _ = dpg.radar.get_par(prodId, "site", "MOSAIC", prefix=names[ppp])
            nodeName, _, _ = dpg.radar.get_par(prodId, "nodeName", "", prefix=names[ppp])
            aux, _, _ = dpg.radar.get_par(prodId, "aux", 0, prefix=names[ppp])
            delta, _, _ = dpg.radar.get_par(prodId, "delta", 0, prefix=names[ppp])
            maximize, _, _ = dpg.radar.get_par(prodId, "maximize", 0, prefix=names[ppp])

            ddd = date
            ttt = time

            if delta > 0:
                ddd, ttt = dpg.times.addMinutesToDate(ddd, ttt, -delta)

            data = dpb.dpb.get_data(
                prodId,
                numeric=True,
                warp=True,
                schedule=schedule,
                name=nodeName,
                site_name=site,
                aux=aux,
                date=ddd,
                time=ttt,
                maximize=maximize,
            )

            if np.size(data) > 1:
                if features_data is None:
                    expected = np.size(data)
                    dim = list(data.shape)
                    dim = [nFeat] + dim
                    features_data = np.zeros(dim, dtype=np.float32)
                    features_data[:] = np.nan
                if np.size(data) == expected:
                    features_data[ppp, :, :] = data

    if features_data is not None:
        return features_data, names

    _, _, dim, _, _, _, _, _ = dpg.navigation.get_geo_info(prodId)
    if len(dim) != 2:
        return None, None

    dim = [nFeat] + list(dim)
    features_data = np.zeros(dim, dtype=np.float32)
    features_data[:] = np.nan

    return features_data, names
