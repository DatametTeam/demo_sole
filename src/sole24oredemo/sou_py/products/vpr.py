from datetime import datetime

import numpy as np
import sou_py.dpg as dpg
import sou_py.dpb as dpb
from sou_py.dpg.log import log_message


def VPR_Get(prodId, step=100.0):
    """
            Questa funzione recupera il VPR per un prodotto radar specificato, utilizzando i parametri configurati
            nel prodotto.
    Args:
            prodId (int): ID del prodotto radar.
            step (float): Risoluzione in altezza per il profilo VPR.

    Returns:
            np.ndarray:   Il profilo verticale di riflettività (VPR) recuperato.

    Note:
            TODO: Commento incompleto perchè in attesa di terminare lo sviluppo di getNHoursBetweenDates
    """
    # PRO VPR_Get, prodId, VPR=vpr, STEP=step
    vpr = None

    max_vpr_hours, prefix, _ = dpg.radar.get_par(prodId, "max_vpr_hours", 3)
    class_, _, _ = dpg.radar.get_par(prodId, "class", "Z", prefix=prefix)

    samplesRoot = dpg.tree.createTree(dpg.cfg.getVPRHome(prefix))
    avgNode, _ = samplesRoot.addNode(class_)
    avgNode, _ = avgNode.addNode("average")

    radar_dict = dpg.navigation.get_radar_par(avgNode)
    step = radar_dict['h_res']

    date, time, exists1 = dpg.times.get_time(avgNode)
    currDate, currTime, exists2 = dpg.times.get_time(prodId)

    if max_vpr_hours <= 0:
        nh = 0
    else:
        nh, _ = dpg.times.getNHoursBetweenDates(currDate, currTime, date, time, double=True)

    if nh <= max_vpr_hours:
        vpr, _ = dpb.dpb.get_var(avgNode, "vpr.txt")

    return vpr, step


def getVPRDelta(profile, threshProfile, minDelta, maxDelta):
    """
    Questa funzione calcola la differenza tra il profilo fornito e il valore iniziale del profilo

    Args:
        profile:        Il profilo da correggere.
        threshProfile:  Soglia al di sotto della quale il profilo è considerato troppo basso per la correzione.
        minDelta:       Correzione minima da applicare.
        maxDelta:       Correzione massima da applicare.

    Returns:
        np.ndarray:     L'array di correzione del profilo VPR.
    """

    if isinstance(profile, type(None)):
        log_message("Old Profile --> No Correction!", 'INFO')
        return np.array([0])

    ind = np.where(np.isfinite(profile))
    if len(ind[0]) <= 4:
        log_message("Few valid values for Profile --> No Correction!", "WARNING")
        return np.array([0])

    maxP = np.nanmax(profile[np.isfinite(profile)])
    if maxP <= threshProfile:
        log_message("Low Profile --> No Correction! " + str(maxP) + " <= " + str(threshProfile), "WARNING")
        return np.array([0])

    profile[: ind[0][0]] = profile[ind[0][0]]
    delta = profile[0] - profile

    ind = np.invert(np.isfinite(delta))
    delta[ind] = 0.0

    ind = np.where(delta < minDelta)
    delta[ind] = minDelta

    log_message("Max VPR correction = " + str(np.nanmax(delta)) + " ==> " + str(maxDelta))

    ind = np.where(delta > maxDelta)
    delta[ind] = maxDelta

    return delta


def VPR_Corr(prodId, volNode, data, heights, vpr, vprStep, maxDelta, prefix=""):
    """
        Questa funzione applica la correzione del profilo VPR ai dati radar utilizzando i parametri configurati nel
        prodotto e le altezze
        dei dati radar. Se i dati o le altezze non sono definiti, genera un errore.

    Args:
        prodId:   ID del prodotto radar
        volNode:  Nodo del volume radar
        data:     Array dei dati radar da correggere
        heights:  Array delle altezze corrispondenti ai dati radar
        vpr:      Profilo verticale di riflettività da applicare
        vprStep:  Passo del profilo VPR
        maxDelta: Correzione massima da applicare
        prefix:   Prefisso per l'accesso ai parametri radar

    Returns:
        None
    """
    # PRO VPR_Corr, prodId, volNode, data, heights, vpr, vprStep, maxDelta

    if data.size <= 1:
        return data

    minDelta, _, _ = dpg.radar.get_par(prodId, "minDelta", -2.0, prefix=prefix)
    threshProfile, _, _ = dpg.radar.get_par(prodId, "threshProfile", 10.0, prefix=prefix)
    threshCorr, _, _ = dpg.radar.get_par(prodId, "threshCorr", 5.0, prefix=prefix)

    VPRDelta = getVPRDelta(vpr, threshProfile, minDelta, maxDelta)
    if VPRDelta.size <= 1:
        return data

    dim = data.shape
    if len(dim) < 3:
        if np.size(heights) != np.size(data) is None:
            log_message("Heights Undefined!", "WARNING")
            return data
        indCorr = np.where(data > threshCorr)
        if len(indCorr[0]) <= 0:
            return data
        ind = heights / vprStep
        ind = ind.astype(np.int64)
        delta = VPRDelta[ind]
        data[indCorr] += delta[indCorr]
        return data
    # endif

    out = dpg.navigation.get_radar_par(volNode, get_el_coords_flag=True)
    site_coords = out["site_coords"]
    range_res = out["range_res"]
    el_coords = out["el_coords"]

    nEl = np.size(el_coords)
    if nEl <= 0:
        log_message("Elevations Undefined!", 'WARNING')
        return data

    if not isinstance(el_coords, list) and not isinstance(el_coords, np.ndarray):
        el_coords = [el_coords]

    delta = np.zeros(dim[1:])
    maxInd = VPRDelta.size
    for eee in range(nEl):
        tmp = data[eee, :, :].copy()
        indCorr = np.where(tmp > threshCorr)
        if len(indCorr[0]) > 0:
            heightBeams = dpg.access.get_height_beams(el_coords[eee], dim[-1], range_res, site_height=site_coords[2],
                                                      projected=True)
            ind = np.int64(heightBeams / vprStep)
            ind[ind >= VPRDelta.size] = maxInd - 1
            deltaBeam = VPRDelta[ind]
            for aaa in range(dim[1]):
                delta[aaa, :] = deltaBeam.copy()
            # endfor
            tmp[indCorr] += delta[indCorr]
            data[eee, :, :] = tmp.copy()
        # endif
    # endfor

    return data


def VPR_CheckSeason(prodId):
    """
    #TODO
    Commento mancante perchè in attesa dello sviluppo della funzione
    """

    auto, _, _ = dpg.radar.get_par(prodId, 'auto', 0)
    if auto <= 0:
        return 1
    date, _, _ = dpg.times.get_time(prodId)
    frmt_date = dpg.times.checkDate(date)
    date_obj = datetime.strptime(frmt_date, '%d-%m-%Y')

    mm = date_obj.month
    if 4 < mm < 11:
        return 0

    return 1


def computeVolumeProfile(values, valuesHeight, scale, thresh, vprStep, vprPlanes, qValues, qThresh, rainyPercThresh,
                         range_off, range_len):
    valids = 0
    profile = np.zeros(vprPlanes, dtype=np.float32)
    profile[:] = np.nan

    if np.size(values) <= 1:
        return profile, valids
    if np.size(values) != np.size(qValues):
        return profile, valids
    if np.size(valuesHeight) <= 1:
        return profile, valids

    dim = np.shape(values)

    ind = np.where((values[0, :, :] < thresh) | (qValues[0, :, :] < qThresh))
    ind_complement = np.where(~((values[0, :, :] < thresh) | (qValues[0, :, :] < qThresh)))
    nComp = len(ind_complement[0])

    if np.float32(nComp / (dim[-1] * dim[-2])) < rainyPercThresh / 100:
        log_message("Few rain values!", level='WARNING')
        return profile, valids

    indH = np.zeros(dim, dtype=int)
    tr = (valuesHeight / vprStep)
    tr = (tr + 0.5).astype(int)

    for aaa in range(dim[1]):
        indH[:, aaa, :] = tr

    nElev = dim[0]
    for eee in range(nElev):
        ppi = indH[eee, :, :]
        ppi[ind] = -1
        if range_off > 0:
            ppi[:, 0:range_off - 1] = -1
        if range_len > 0 and range_len + range_off < dim[-1]:
            ppi[:, range_len + range_off:] = -1
        indH[eee, :, :] = ppi

    for ppp in range(vprPlanes):
        ind = np.where(indH == ppp)
        if len(ind[0]) > 0:
            sel = values[ind]
            sel = dpg.prcs.linearizeValues(sel, scale=scale)
            profile[ppp] = np.nanmean(sel[np.isfinite(sel)])

    profile = dpg.prcs.unlinearizeValues(profile, scale=scale)

    valids = int(np.sum(np.isfinite(profile)))

    return profile, valids


def VPR_Volume(prodId, volume, qVolume, volNode, vprStep, vprPlanes):
    """
    TODO
    Commento mancante perchè in attesa dello sviluppo della funzione
    """

    percThresh, site_name, _ = dpg.radar.get_par(prodId, 'percThresh', 40.)
    rainyPercThresh, _, _ = dpg.radar.get_par(prodId, 'rainyPercThresh', 0.5, prefix=site_name)
    threshVal, _, _ = dpg.radar.get_par(prodId, 'threshVal', 10., prefix=site_name)
    threshQuality, _, _ = dpg.radar.get_par(prodId, 'threshQuality', 0., prefix=site_name)
    range_off, _, _ = dpg.radar.get_par(prodId, 'range_off', 20, prefix=site_name)
    range_len, _, _ = dpg.radar.get_par(prodId, 'range_len', 80, prefix=site_name)
    smoothBox, _, _ = dpg.radar.get_par(prodId, 'smooth', 0, prefix=site_name)
    medianFilter, _, _ = dpg.radar.get_par(prodId, 'medianFilter', 0, prefix=site_name)

    err = 1
    scale = 2

    if volume is None:
        dim = np.array([])
    else:
        dim = volume.shape

    radar_dict = dpg.navigation.get_radar_par(volNode, get_el_coords_flag=True)
    site_coords = radar_dict['site_coords']
    range_res = radar_dict['range_res']
    el_coords = radar_dict['el_coords']

    if len(dim) > 1 and np.size(el_coords) > 1:
        height_beams = dpg.access.get_height_beams(el_coords, dim[-1], range_res, site_height=site_coords[2],
                                                   projected=True)  # TODO: da controllare
        values, calib, out_dict = dpg.calibration.get_array_values(volNode)
        scale = out_dict['scale']
        if range_res != 1000.:
            range_off = np.float32(range_off * 1000. / range_res)
            range_len = np.float32(range_len * 1000. / range_res)

    vpr, valids = computeVolumeProfile(volume, height_beams, scale, threshVal, vprStep, vprPlanes,
                                       qVolume, threshQuality, rainyPercThresh, range_off, range_len)

    if valids > len(vpr) * percThresh / 100:
        err = 0
        if smoothBox > 0:
            vpr = dpg.prcs.smooth_data(vpr, smoothBox, opt=medianFilter)

    dpb.dpb.put_var(prodId, 'vpr.txt', vpr, set_array=True)
    dpg.navigation.set_radar_par(node=prodId, site_coords=site_coords, h_off=0., h_res=vprStep, site_name=site_name)

    return vpr, err


def getHIceMax(prodId):
    err = 0

    schedule, site_name, _ = dpg.radar.get_par(prodId, 'FL_schedule', '')
    hice_max, _, _ = dpg.radar.get_par(prodId, 'hice_max', 3000., prefix=site_name)

    if schedule == '':
        return hice_max

    site_coords, _, _ = dpg.navigation.get_site_coords(prodId)

    date, time, _ = dpg.times.get_time(prodId)
    deltaH1max, _, _ = dpg.radar.get_par(prodId, 'deltaH1max', 1.5, prefix=site_name)
    n_hours, _, _ = dpg.radar.get_par(prodId, 'n_hours', 12., prefix=site_name)
    min_step, _, _ = dpg.radar.get_par(prodId, 'min_step', 60., prefix=site_name)
    site, _, _ = dpg.radar.get_par(prodId, 'site', 'ITALIA', prefix=site_name)

    time = time[:3] + '00' + time[5:]

    nProds, prod_path = dpg.access.get_aux_products(prodId, site_name=site_name, last_date=date, last_time=time,
                                                    n_hours=n_hours, min_step=min_step)

    if nProds <= 0:
        return hice_max, 1

    tree = dpg.tree.createTree(prod_path[0])
    flhNode = dpg.radar.find_site_node(tree, site)

    data = dpb.dpb.get_data(flhNode, numeric=True)

    if data is None or np.size(data) <= 1:
        return hice_max, 1

    sourceDim = np.shape(data)
    _, sourceMap, sourceDim, sourcePar, _, _, _, _, _ = dpg.navigation.check_map(flhNode, sourceMap=True, dim=sourceDim)
    y, x = dpg.map.latlon_2_yx(site_coords[0], site_coords[1], map=sourceMap)
    y, x = dpg.map.yx_2_lincol(y, x, sourcePar, dim=sourceDim)
    if x < 0 or y < 0:
        return hice_max, 1

    FL = data[x, y]
    dpg.tree.removeTree(tree)

    if not np.isfinite(FL):
        return hice_max, 1

    log_message(f"Freezing Level = {FL} m")
    fl_inc = deltaH1max * 1000.
    if FL < site_coords[2] + fl_inc:
        FL = site_coords[2] + fl_inc
        log_message(f"... corrected to {FL} m")

    return FL, err


def getVPRParameters(vp_zhh_avg, hgt_vec_zhh, hice_max, hmin, deltaH1max, delta_zrny_zice, delta_hgtkm_vpr,
                     z_rain_thresh, force_delta, max_hup):
    zrny = np.float32(0)
    zice = np.float32(0)
    hup = np.max(hgt_vec_zhh)
    angcoef = np.float32(0)
    vpr_qual = np.float32(0)

    ind = np.where((hgt_vec_zhh <= hice_max) & (hgt_vec_zhh >= hmin))

    if len(ind[0]) > 0:
        zmax = np.nanmax(vp_zhh_avg[ind][np.isfinite(vp_zhh_avg[ind])])  # Finds the maximum, ignoring NaN
        ind1 = np.nanargmax(vp_zhh_avg[ind])  # Finds the index of the maximum value
        hmax = hgt_vec_zhh[ind[0][ind1]]
    else:
        zmax = np.nanmax(vp_zhh_avg[ind][np.isfinite(vp_zhh_avg[ind])])
        ind = np.nanargmax(vp_zhh_avg)  # Finds the index of the maximum value
        hmax = hgt_vec_zhh[ind]

    hice = hmax + force_delta
    ind = np.where((hgt_vec_zhh <= hice) & (hgt_vec_zhh >= hmax))

    if len(ind[0]) > 0:
        zice = np.nanmin(vp_zhh_avg[ind][np.isfinite(vp_zhh_avg[ind])])

    hrny = hmax - force_delta
    if hrny < 0.5 * hmax:
        hrny = 0.5 * hmax

    ind1 = np.where(hgt_vec_zhh <= hrny)
    if len(ind1[0]) > 0:
        zrny = np.nanmean(vp_zhh_avg[ind1][np.isfinite(vp_zhh_avg[ind1])])

    if not np.isfinite(zrny):
        zrny = zice + delta_zrny_zice
    if zrny < zice:
        zrny = zice
    if zrny > zice + 0.5 * (zmax - zice):
        zrny = zice + 0.5 * (zmax - zice)
    if zrny > zice + delta_zrny_zice:
        zrny = zice + delta_zrny_zice

    if zmax > z_rain_thresh:
        vpr_qual += 0.2
    if zrny > z_rain_thresh:
        vpr_qual += 0.2

    ind1 = np.where(hgt_vec_zhh > hice)

    if len(ind1[0]) > 0:
        nEl = len(vp_zhh_avg)
        std_vpz = np.zeros(nEl, dtype=np.float32)
        ind = np.arange(5) - 2
        for iii in range(2, nEl - 2):
            tmp = vp_zhh_avg[iii + ind]
            fff = np.where(np.isfinite(tmp))
            if len(fff[0]) > 1:
                std_vpz[iii] = np.nanstd(tmp[np.isfinite(tmp)], ddof=1)

        max_std = np.nanmax(std_vpz[ind1])
        ind2 = np.where(std_vpz[ind1] / max_std > 0.5)
        if len(ind2[0]) > 0:
            vpr_qual += 0.2
            hup = hgt_vec_zhh[ind1[0][ind2[0][len(ind2[0]) - 1]]]
            if hup - hice < 2:
                hup = hice + 0.7 * (hgt_vec_zhh[ind1[len(ind1[0]) - 1]] - hice)
        else:
            hup = hice + 0.7 * (hgt_vec_zhh[ind1[len(ind1[0]) - 1]] - hice)

        dif_vpz = np.gradient(vp_zhh_avg) / delta_hgtkm_vpr
        dif_vpz = dpg.prcs.smooth_opt0(dif_vpz, 5)
        if hup > max_hup:
            hup = max_hup

        ind = np.where((hgt_vec_zhh > hice) & (hgt_vec_zhh < hup) & (vp_zhh_avg > -20) & (dif_vpz < 0))
        if len(ind[0]) > 0:
            vpr_qual += 0.2
            XX = hgt_vec_zhh[ind]
            YY = vp_zhh_avg[ind]
            XX -= hice
            YY -= zice
            angcoef = np.polyfit(XX, YY, 1)
            angcoef = np.float32(angcoef[0])

        if angcoef >= 0:
            vpr_qual = 0

        return [zrny, zmax, zice, hrny, hmax, hice, hup, angcoef, np.float32(vpr_qual)]


def getVPRLinearModel(hgt, zmin_ice, zmin_rain, hmin, deltaz, deltaH1, deltaH2, angcoef):
    nEl = np.size(hgt)
    z = np.zeros(nEl, dtype=np.float32)
    z[:] = np.nan

    ind1 = np.where((hgt <= hmin + deltaH1) & (hgt > hmin))  # below the Zhh maximum within BB
    ind2 = np.where((hgt <= hmin + deltaH1 + deltaH2) & (hgt > hmin + deltaH1))  # above the Zhh maximum, within BB
    ind3 = np.where(hgt >= hmin + deltaH1 + deltaH2)  # above the ice parth

    count1 = len(ind1[0])
    count2 = len(ind2[0])
    count3 = len(ind3[0])

    m1 = deltaH1 / deltaz
    m2 = deltaH2 / np.abs(zmin_rain + deltaz - zmin_ice)
    q1 = hmin - m1 * zmin_rain
    q2 = hmin + deltaH1 + deltaH2 + m2 * zmin_ice
    q3 = hmin + deltaH1 + deltaH2 - (1 / angcoef) * zmin_ice

    if count1 > 0:
        z[ind1] = (hgt[ind1] - q1) / m1
        z[0] = zmin_rain
        if ind1[0][0] > 0:
            z[0:ind1[0][0]] = zmin_rain

        if count2 > 0:
            z[ind2] = -(hgt[ind2] - q2) / m2
        if count3 > 0:
            z[ind3] = angcoef * (hgt[ind3] - q3)

    return z


def VPR_Model(prodId, profile, vprStep, hice_max=None):
    err = 1
    vprPlanes = len(profile)
    if vprPlanes <= 1:
        return profile, err, None

    model = np.zeros(vprPlanes, dtype=np.float32) * np.nan

    deltaH1max, site_name, _ = dpg.radar.get_par(prodId, 'deltaH1max', 1.5)
    delta_zrny_zice, _, _ = dpg.radar.get_par(prodId, 'delta_zrny_zice', 3., prefix=site_name)
    min_vpr_quality, _, _ = dpg.radar.get_par(prodId, 'min_vpr_quality', 0.7, prefix=site_name)
    z_rain_thresh, _, _ = dpg.radar.get_par(prodId, 'z_rain_thresh', 10., prefix=site_name)
    z_max_thresh, _, _ = dpg.radar.get_par(prodId, 'z_max_thresh', 15., prefix=site_name)
    h_max_thresh, _, _ = dpg.radar.get_par(prodId, 'h_max_thresh', 0.8, prefix=site_name)
    force_delta, _, _ = dpg.radar.get_par(prodId, 'force_delta', 0.5, prefix=site_name)
    max_hup, _, _ = dpg.radar.get_par(prodId, 'max_hup', 7., prefix=site_name)

    if hice_max is None:
        hice_max, err = getHIceMax(prodId)
        hice_max /= np.float32(1000)

    hmin = np.float32(hice_max - deltaH1max)
    hgt_vec_zhh = np.arange(vprPlanes, dtype=np.float32)
    delta_hgtkm_vpr = np.float32(vprStep / 1000)
    hgt_vec_zhh *= delta_hgtkm_vpr
    hgt_vec_zhh += delta_hgtkm_vpr / 2.

    params = getVPRParameters(profile, hgt_vec_zhh, hice_max, hmin, deltaH1max, delta_zrny_zice, delta_hgtkm_vpr,
                              z_rain_thresh, force_delta, max_hup)

    z_rain = params[0]
    z_max = params[1]
    z_ice = params[2]
    delta_z = z_max - z_rain

    h_rain = params[3]
    h_max = params[4]
    h_ice = params[5]
    deltaH1 = h_max - h_rain
    deltaH2 = h_ice - h_max

    angcoef = params[7]
    qual = params[8]

    if qual < min_vpr_quality:
        return model, err, hice_max

    if z_max <= z_max_thresh:
        return model, err, hice_max

    if h_max <= h_max_thresh:
        return model, err, hice_max

    model = getVPRLinearModel(hgt_vec_zhh, z_ice, z_rain, h_rain, delta_z, deltaH1, deltaH2, angcoef)
    err = 0

    return model, err, hice_max


def computeMeanProfile(root):
    samples = None

    sons = dpg.tree.getSons(root)
    name = 'vpr.txt'

    for sss in sons:
        var, _ = dpb.dpb.get_var(sss, name)
        if np.size(var) > 1:
            tmp = var.T
            if samples is None:
                samples = [tmp]
            else:
                samples = samples + [tmp]

    if samples is None or np.size(samples) == 0:
        log_message("No Valid Samples!", 'WARNING')
        return np.nan

    dim = np.shape(samples)
    n = dim[1]
    m = dim[0]

    if m > 1:
        meanProf = np.zeros(n, dtype=np.float32)
        for hhh in range(n):
            vals = np.array([arr[hhh] for arr in samples])
            meanProf[hhh] = np.nanmean(vals[np.isfinite(vals)])
    else:
        meanProf = np.reshape(samples, -1)

    return meanProf


def VPR_Update(prodId, vprStep, vprPlanes, model, hice_max):
    class_, site_name, _ = dpg.radar.get_par(prodId, 'class', 'Z')
    max_samples, _, _ = dpg.radar.get_par(prodId, 'max_samples', 0, prefix=site_name)
    max_hours, _, _ = dpg.radar.get_par(prodId, 'max_hours', 0, prefix=site_name)

    model = np.repeat(np.nan, vprPlanes)

    currDate, currTime, _ = dpg.times.get_time(prodId)
    samplesRoot = dpg.tree.createTree(dpg.cfg.getVPRHome(site_name))

    outNode, _ = dpg.tree.addNode(samplesRoot, class_)
    avg, _ = dpg.tree.addNode(outNode, 'average')
    samplesNode, _ = dpg.tree.addNode(outNode, 'samples')

    next_ = currDate + '.' + currTime
    next_ = ''.join([part for part in next_.split(':') if part])

    out, _ = dpg.tree.addNode(samplesNode, next_, remove_if_exists=True)
    dpg.tree.copyNode(prodId, out, from_memory=True)

    if max_samples > 0:
        dpg.access.remove_old_samples(samplesNode, max_samples)
    if max_hours > 0:
        dpg.access.remove_old_nodes(samplesNode, currDate, currTime, max_hours)

    meanProf = computeMeanProfile(samplesNode)
    model, _, _ = VPR_Model(prodId, meanProf, vprStep, hice_max=hice_max)
    dpg.tree.copyNode(prodId, avg, from_memory=True)
    dpb.dpb.put_var(avg, 'vpr.txt', model, set_array=True, to_save=True)

    log_message(f"Added VPR sample for class {class_}")
    dpg.tree.removeTree(samplesRoot)

    return


def VPR(
        prodId,
        data,
        qVolume=None,
        node=None,
        update=False,
        corr=False,
        hr=False,
        model=False,
        mean=None,
        heights=None,
):
    """
    NAME:
    VPR

    :Description:
                Questa funzione gestisce il profilo VPR per un prodotto radar, applicando eventuali correzioni basate
                sul profilo e aggiornando il modello

    :Params:
                prodId:         ID del prodotto radar
                data:           Array dei dati radar
                qVolume:        Volume di Qualita' della stessa dimensione di volume
                node:           Nodo del prodotto radar
                update:         Indica se aggiornare il profilo VPR. Default è False
                get_corr:
                hr:
                get_model:
                get_mean:
                heights:
    :Output:
                VPR:            Array contenente il profilo calcolato
                HEIGHTS:        Array contenente la quota delle celle selezionate

    Note:
            TODO: Commento incompleto perchè in attesa di terminare lo sviluppo della funzione
    """
    hice_max = None

    step, site_name, _ = dpg.radar.get_par(prodId, "vprStep", 100.0)
    if corr:
        maxDelta, _, _ = dpg.radar.get_par(prodId, "maxDelta", 10.0, prefix=site_name)
        if maxDelta <= 0.0:
            return data, None, None, None
        if VPR_CheckSeason(prodId) <= 0:
            return data, None, None, None
        mean, step = VPR_Get(prodId, step=step)
        if hr:
            log_message("DA IMPLEMENTARE VPR_CorrHR", level='ERROR')
            print(errore)
            # VPR_CorrHR, prodId, mean, step, maxDelta
            # return
        # endif
        data = VPR_Corr(prodId=prodId, volNode=node, data=data, heights=heights, vpr=mean, vprStep=step,
                        maxDelta=maxDelta)
        return data, heights, model, mean
    # endif

    vprPlanes, _, _ = dpg.radar.get_par(prodId, "vprPlanes", 100, prefix=site_name)
    vpr, err = VPR_Volume(prodId, data, qVolume, node, step, vprPlanes)

    if model:
        if err == 0:
            model, err, hice_max = VPR_Model(prodId, vpr, step)
        if np.size(model) != vprPlanes:
            model = np.zeros(vprPlanes, dtype=np.float32) * np.nan
    if update:
        if err == 0:
            VPR_Update(prodId, step, vprPlanes, model=mean, hice_max=hice_max)
    if mean:
        if np.size(mean) != vprPlanes:
            mean, step = VPR_Get(prodId)
        if np.size(mean) != vprPlanes:
            mean = np.repeat(np.nan, vprPlanes)

    return vpr, heights, model, mean
