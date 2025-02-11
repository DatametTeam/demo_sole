"""
Calcola la mappa della quantità totale di acqua liquida stimata in una colonna verticale sopra ogni pixel.
Combina la media dei valori positivi di riflettività con l'estensione verticale (da ETM) e
converte i dati usando una relazione predefinita per stimare il contenuto di acqua liquida.
"""
import time

import numpy as np
import sou_py.dpg as dpg
from numba import njit

from sou_py.dpg.log import log_message


def compute_vil(volume, heightBeams, zThresh, zMax, A, B, minH, maxH):
    """
                Funzione per il calcolo del Vertical Integrated Liquid (VIL) a partire da un volume radar 3D.
                Ogni pixel della mappa VIL rappresenta il contenuto integrato di acqua liquida stimata lungo la
                verticale del pixel,
                utilizzando i dati di riflettività radar e i valori di altezza dei fasci radar.

    Args:
                volume:      Volume radar 3D da elaborare.
                heightBeams: Array contenente le altezze dei fasci radar per ogni livello verticale.
                zThresh:     Soglia di riflettività. I valori sotto questa soglia vengono ignorati.
                zMax:        Valore massimo di riflettività. I valori superiori a zMax vengono impostati a zMax.
                A:           Costante per la conversione da riflettività a contenuto di acqua liquida.
                B:           Esponente per la conversione da riflettività a contenuto di acqua liquida.
                minH:        Quota minima per considerare un eco significativo.
                maxH:        Quota massima per considerare un eco significativo.

    Returns:
                vil:         Array 2D contenente il contenuto integrato di acqua liquida (VIL) in g/m³.
                etm:         Array 2D contenente l'estensione verticale massima (ETM) in metri.

    Note:
                La funzione converte i valori di riflettività radar in contenuto di acqua liquida utilizzando la
                relazione
                Z = A * Zlin^B, dove Zlin è la riflettività lineare e A e B sono costanti definite dall'utente.
                In particolare di default vengono utilizzati: A = 3.44x10-3 e B = 4/7.
                La riflettività viene media lungo la verticale per ogni pixel, tenendo conto delle altezze dei fasci
                radar.
                I valori di riflettività che superano zMax vengono limitati a zMax, e i valori al di sotto di zThresh
                vengono ignorati.
                La funzione gestisce eventuali valori non finiti (NaN o infiniti) impostandoli a zero nella
                riflettività lineare.
                Per ogni pixel, viene calcolata l'estensione verticale massima (ETM) come l'altezza massima del
                fascio radar che supera la soglia zThresh.
                La funzione include un controllo per impostare i valori VIL non significativi a zero e i valori ETM
                non validi a -inf.
    """

    dim = volume.shape
    nX = dim[2]
    nY = dim[1]
    if len(dim) == 2:
        nZ = 1
    else:
        nZ = dim[0]

    vil = np.zeros((nX, nY), dtype=np.float32)
    etm = np.zeros((nX, nY), dtype=np.float32)
    hhh = np.zeros((nZ, nX), dtype=np.float32)

    Zlin = volume.copy()
    Zlin = np.where(Zlin > zMax, zMax, Zlin)

    Zlin = np.power(10, 0.1 * Zlin)
    Zlin = np.where(~np.isfinite(volume), 0, Zlin)

    hhh[0, :] = heightBeams[:, 0].copy()
    # calculate the average between consecutive layers along the first axis
    Zlin[1:, :, :] = (Zlin[1:, :, :] + Zlin[:-1, :, :]) / 2.0
    hhh[1:, :] = (heightBeams[:, 1:] - heightBeams[:, :-1]).T

    Zlin = A * np.power(Zlin, B)
    hhh = np.where(hhh < 0, 0, hhh)

    etm, vil = support_etm_vil(heightBeams, maxH, volume, zThresh, Zlin, hhh)

    etm = etm.T
    vil = vil.T

    ind = np.where((vil <= 0) | (etm <= minH))
    vil[ind] = 0.0
    etm[ind] = -np.inf

    return vil, etm


# unused right now
def support_etm_vil(heightBeams, maxH, volume, zThresh, Zlin, hhh):
    """
    Description:
        This function computes two matrices, ETM and VIL, using the provided radar data and various thresholds.
        The ETM matrix represents the maximum vertical extent of the radar reflectivity, while the VIL matrix
        indicates the integrated liquid water content.
    Args:
        heightBeams: A 2D array containing the heights of the radar beams for each vertical level.
        maxH:        The maximum height to be considered for calculations.
        volume:      Volume representing the radar volume data.
        zThresh:     Value under this threshold are ignored.
        Zlin:        Volume indicating linear reflectivity.
        hhh:         2D array containing average values calculated between consecutive levels of `heightBeams`.

    Returns:
        etm:         Matrix containing the integrated liquid water (VIL) content in g/m³.
        vil:         Matrix containing the maximum vertical extent (ETM) in meters.
    """
    dim0, dim1, dim2 = volume.shape
    etm = np.zeros((dim2, dim1), dtype=np.float32)
    vil = np.zeros((dim0, dim1, dim2), dtype=np.float32)

    # Number of valid element for each row
    n_of_valid_element = (heightBeams <= maxH).sum(axis=1)

    # Valid element for volume
    depth_range = np.arange(volume.shape[0])[:, None, None]
    idx_mask = depth_range < n_of_valid_element
    valid_element_volume = (volume >= zThresh) & idx_mask

    # Count how many element are valid along the 0 axis
    count_valid = valid_element_volume.sum(axis=0)

    # Find last 'true' element along 0 axis
    reversed_volume = valid_element_volume[::-1]
    idx = np.argmax(reversed_volume, axis=0)
    any_true = np.any(valid_element_volume, axis=0)
    last_indices = np.where(any_true, dim0 - 1 - idx, -1).flatten()

    # Populate etm
    etm_mask = np.where(np.any(valid_element_volume, axis=0))
    etm_res = heightBeams[etm_mask[1], last_indices[last_indices >= 0]]
    etm[etm_mask[1], etm_mask[0]] = etm_res
    idx_mask_zlin = depth_range < count_valid

    # Populate vil
    hhh_3d = np.repeat(hhh[:, None, :], dim1, axis=1)
    vil[idx_mask_zlin] = hhh_3d[idx_mask_zlin] * Zlin[idx_mask_zlin]
    vil = np.sum(vil, axis=0).T

    return etm, vil


def support_etm_vil_backup(
    etm, vil, nX, nY, heightBeams, maxH, Zlin, zThresh, volume, hhh
):
    """
    This function computes enhanced terrain model (ETM) heights and vertically integrated liquid (VIL) values
    from 3D radar volume data.

    Parameters:
        - etm: 2D array to store computed ETM heights for each grid cell.
        - vil: 2D array to store computed VIL values for each grid cell.
        - nX: Number of grid cells in the X-direction.
        - nY: Number of grid cells in the Y-direction.
        - heightBeams: 2D array of beam heights for each X and Z level.
        - maxH: Maximum height threshold for processing beam data.
        - Zlin: 3D array of radar reflectivity values in linear scale.
        - zThresh: Reflectivity threshold to determine significant returns.
        - volume: 3D radar volume data (Z by Y by X).
        - hhh: Array of vertical thicknesses between radar levels.

    Returns:
        - Updated `etm` and `vil` arrays with calculated heights and liquid values.
    """

    for xxx in range(nX):
        hhh1 = np.where(heightBeams[xxx, :] <= maxH)
        last = len(hhh1[0])
        if last > 0:
            for yyy in range(nY):
                ind = np.where(volume[0:last, yyy, xxx] >= zThresh)
                count = len(ind[0])
                if count > 0:
                    etm[xxx, yyy] = heightBeams[xxx, ind[0][-1]]
                    vil[xxx, yyy] = np.sum(
                        np.multiply(Zlin[0:count, yyy, xxx], hhh[0:count, xxx])
                    )
    return etm, vil


def VIL(prodId, volume, main, get_etm=False):
    """
    NAME: VIL

    :Description:
        Algoritmo per il calcolo del contenuto di acqua liquida sulla verticale di ogni pixel


    Args:
        prodId:     Nodo del prodotto, da cui si accede ai seguenti parametri opzionali contenuti in parameters.txt
        zThresh:    Soglia di riflettivita'. Il valore rilevato al Top dovra' essere superiore a tale soglia (default
        = 18 dbZ)
        zMax:       Soglia massima di riflettivita'. I valori superiori a zMax vengono posti a Zmax (default = 56 dbZ)
        minH:       Quota minima. Il Top dovra' essere superiore a tale quota (default = 2000 m)
        maxH:       Quota massima. Il Top dovra' essere inferiore a tale quota (default = 16000 m)
        volume:     Volume da elaborare; generalmente output della procedura GET_CURR_VOLUME con opzione /PROJECTED,
        main:       Nodo del volume campionato (necessario per accedere alle proprieta' del volume)


    Returns:
        VIL:       Array contenente la matrice polare in kg/mq del VIL (float)
        ETM:       Array contenente la matrice polare in metri del Top (float) (puo' essere indifferentemente input o
        output!)
    """

    scan_dim = volume.shape
    if len(scan_dim) < 2:
        log_message("Error: volume has wrong shape", level="ERROR")
        return None, None

    node = main

    zthresh, site_name, _ = dpg.radar.get_par(prodId, "zthresh", 18.0)
    zMax, _, _ = dpg.radar.get_par(prodId, "zMax", 56.0, prefix=site_name)
    A, _, _ = dpg.radar.get_par(prodId, "A", 3.4e-6, prefix=site_name)
    B, _, _ = dpg.radar.get_par(prodId, "B", 0.57, prefix=site_name)
    minH, _, _ = dpg.radar.get_par(prodId, "minH", 2000.0, prefix=site_name)
    maxH, _, _ = dpg.radar.get_par(prodId, "maxH", 16000.0, prefix=site_name)

    out = dpg.navigation.get_radar_par(node, get_el_coords_flag=True)

    site_coords = out["site_coords"]
    range_res = out["range_res"]
    range_off = out["range_off"]
    coord_set = out["el_coords"]

    heightBeams = dpg.access.get_height_beams(
        coord_set,
        scan_dim[2],
        range_res,
        site_height=site_coords[2],
        range_off=range_off,
        projected=True,
    )
    heightBeams = heightBeams.T

    vil, etm = compute_vil(volume, heightBeams, zthresh, zMax, A, B, minH, maxH)

    return vil, etm
