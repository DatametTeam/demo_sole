import numpy as np
import pandas as pd


def CSI(obs, pred, threshold=0.1):
    """
    CSI - critical success index
    details in the paper:
    Woo, W., & Wong, W. (2017).
    Operational Application of Optical Flow Techniques to Radar-Based
    Rainfall Nowcasting.
    Atmosphere, 8(3), 48. https://doi.org/10.3390/atmos8030048
    Args:
        obs (numpy.ndarray): observations
        pred (numpy.ndarray): predictions
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: CSI value
    """

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pred=pred,
                                                           threshold=threshold)

    if (hits + misses + falsealarms) == 0:
        # print("Error: hits + misses + falsealarms == 0. Returning 0")
        return None

    return hits / (hits + misses + falsealarms)


def prep_clf(obs, pred, threshold=0.1):
    obs = np.where(obs >= threshold, 1, 0)
    pred = np.where(pred >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pred == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pred == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pred == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pred == 0))

    return hits, misses, falsealarms, correctnegatives


def compute_CSI(targets, outputs, thresholds=None):
    if thresholds is None:
        thresholds = [1, 5, 10, 20, 50]

    prediction_times = list(targets.keys())
    prediction_offsets = [f'{5 * (i + 1)}_min' for i in range(len(prediction_times))]

    data = pd.DataFrame(index=thresholds, columns=prediction_offsets)

    for idx, pred_time in enumerate(prediction_times):
        for th in thresholds:
            metric_value = CSI(targets[pred_time],
                               outputs[pred_time],
                               threshold=th)

            data.at[th, prediction_offsets[idx]] = metric_value

    data = data.fillna(value=0)
    return data
