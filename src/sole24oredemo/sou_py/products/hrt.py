import numpy as np
from shapely import Polygon
from skimage.measure import find_contours

from sou_py.dpg.log import log_message
import sou_py.dpb as dpb
import sou_py.dpg as dpg


def computeHRT(
    hrmNode, eee, destMap, par, dim, minutes, vel, dir, cor, index, scalingCorr
):
    """
    This function performs high-resolution tracking (HRT) of dynamic objects in a given map domain.

    Args:
        - hrmNode: The node or element in the hierarchical map for computation.
        - eee: The index or identifier for a specific subset of the map.
        - destMap: The destination map object where the tracking occurs.
        - par: Map projection parameters or transformation configuration.
        - dim: Dimensional parameters for spatial computation.
        - minutes: Time duration (in minutes) used in motion calculations.
        - vel: Velocity vector used in track prediction.
        - dir: Direction vector for object movement.
        - cor: Correction factor to adjust scaling in the track evolution.
        - index: Identifier used to mark areas in the output track.
        - scalingCorr: Scaling factor applied to adjust object evolution.

    Returns:
        - A tuple containing:
            - len(max_contour): Length of the largest contour detected.
            - path_ll: Pathway of the detected object in latitude and longitude coordinates.
            - area: Area of the detected region above the threshold.
            - track: The updated track matrix with indexed areas.
    """

    ret, vertex = dpg.coords.get_points(hrmNode, destMap=destMap, optim=True, index=eee)
    if ret <= 0:
        return

    pVertex = vertex
    evolution = np.ones(6)
    scaling = scalingCorr * (1.0 - cor)
    evolution += scaling
    levels = 1.0

    track = dpg.coords.computeTrack(
        pVertex, vel, dir, minutes, levels, evolution, dim=dim, par=par, get_track=True
    )

    contours = find_contours(
        track, level=levels, fully_connected="high", positive_orientation="low"
    )

    y, x = dpg.map.lincol_2_yx(
        contours[0][:, 0], contours[0][:, 1], par, set_center=True
    )

    max_contour = max(contours, key=len)
    if len(max_contour) < 3:
        log_message("Contour has found irregular objects.", level="WARNING+")
        return 0, None, None, None

    ind = np.where(track >= levels)
    area = len(ind[0])
    track[ind] = index
    lat, lon = dpg.map.yx_2_latlon(y, x, map=destMap)
    path_ll = np.column_stack((lon, lat))

    return len(max_contour), path_ll, area, track


def hrt(prodId, hrm, indexName=None, track=None, recompute=None):
    """
    This function computes and updates high-resolution tracks (HRT) for dynamic objects, integrating storm tracking data
    into a geospatial framework.

    Args:
        - prodId: Product identifier node for which HRT is to be computed.
        - hrm: Hierarchical map node for spatial and temporal data processing.
        - indexName: Optional, the index name for storm tracking (default: "SSI").
        - track: Optional, an existing track array to be updated.
        - recompute: Optional, flag to trigger reprocessing (not implemented).

    Returns:
        - Updated track matrix with storm indices.
    """

    if not isinstance(prodId, dpg.node__define.Node):
        return

    if indexName is None:
        indexName = "SSI"
    if track is not None:
        track = None

    if not isinstance(hrm, dpg.node__define.Node):
        date, time, _ = dpg.times.get_time(prodId)
        schedule, _, _ = dpg.radar.get_par(prodId, "schedule", "")
        origin, _, _ = dpg.radar.get_par(prodId, "origin", "")
        _, owner, system = dpg.schedule__define.get_schedule_path(prodId=prodId)
        if system != "RADAR" and system != "EXTRA":
            log_message("SYSTEM != RADAR or EXTRA: da fare", level="ERROR")
            path = dpg.tree.getNodePath(owner)

        hrm = dpb.dpb.get_last_node(schedule, origin, date=date, time=time)

    storms, nFeat, parnames = dpg.coords.get_shape_info(hrm, names=True)
    if nFeat <= 0:
        return

    shpfile, _, _ = dpg.radar.get_par(prodId, "shpfile", "hrt.shp")
    outPath = dpg.tree.getNodePath(prodId)
    outFile = dpg.path.getFullPathName(outPath, shpfile)

    valids = 0
    if len(storms) > 0:
        map, _, dim, par, _, _, _, _, _ = dpg.navigation.check_map(
            prodId, destMap=True, mosaic=True, dim=True, par=True
        )
        threshold, _, _ = dpg.radar.get_par(prodId, "threshold", 4.0)
        threshIndex, _, _ = dpg.radar.get_par(prodId, "threshIndex", threshold)
        threshArea, _, _ = dpg.radar.get_par(prodId, "threshArea", 10.0)
        scalingCorr, _, _ = dpg.radar.get_par(prodId, "scalingCorr", 0.5)
        SSImin, _, _ = dpg.radar.get_par(prodId, "minutes", 30, prefix=indexName)
        HRImin, _, _ = dpg.radar.get_par(prodId, "minutes", 30, prefix="HRI")
        vel = storms["Vel"]
        dir = storms["Dir"]
        cor = storms["Cor"]
        if "HRI" in storms.columns:
            index = np.maximum(storms["HRI"], storms["SSI"])
        else:
            index = storms["SSI"]

        storms_to_remove = []
        for sss in range(len(storms)):
            # sss = 5
            ret = 0
            if index[sss] > threshIndex and storms.loc[sss, "Area"] >= threshArea:
                minutes = SSImin
                if "HRI" in storms.columns:
                    if storms.loc[sss, "SSI"] < storms.loc[sss, "HRI"]:
                        minutes = HRImin
                ret, path_ll, area, currTrack = computeHRT(
                    hrm,
                    sss,
                    map,
                    par,
                    dim,
                    minutes,
                    vel[sss],
                    dir[sss],
                    cor[sss],
                    index[sss],
                    scalingCorr,
                )

            if ret <= 0:
                log_message(
                    f"Removing index {sss} from storms. Area {storms.loc[sss, 'Area']} under threshold",
                    level="WARNING",
                )
                storms_to_remove.append(sss)
            else:
                new_geometry = Polygon(path_ll)
                storms.loc[sss, "Area"] = area
                storms.loc[sss, "geometry"] = new_geometry
                storms.loc[sss, "Time"] = storms.loc[sss, "Time"] + f"+{minutes}"

                if track is None:
                    track = currTrack
                else:
                    track = np.maximum(track, currTrack)

        storms = storms.drop(storms_to_remove)
        storms = storms.reset_index(drop=True)
        valids = len(storms)

        if recompute is not None:
            log_message("DA FARE PORTING DI QUESTA PARTE", level="ERROR")
            print(error)

        storms = storms.map(
            lambda x: (
                int(x * 1000) / 1000 if isinstance(x, float) and not np.isnan(x) else x
            )
        )
        # storms = storms.set_geometry('Geometry')
        storms.to_file(outFile)

        dpg.tree.replaceAttrValues(
            prodId,
            dpg.cfg.getGeoDescName(),
            ["coordfile", "format"],
            [shpfile, "shp"],
            only_current=True,
            to_save=True,
        )
        shpfile = shpfile[:-3] + "dbf"
        dpg.tree.replaceAttrValues(
            prodId,
            dpg.cfg.getArrayDescName(),
            ["datafile", "format"],
            [shpfile, "dbf"],
            only_current=True,
            to_save=True,
        )

        return track
