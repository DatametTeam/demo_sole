import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from functools import partial

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sole24oredemo.utils import compute_figure


def create_fig_dict_in_parallel(gt_data, pred_data, save_on_disk=False):
    out_dir = Path("/archive/SSD/home/guidim/demo_sole/data/output/ConvLSTM/20250205/20250205/gen_images/gt")

    base_date = datetime.strptime('31-01-2025', "%d-%m-%Y")
    time_step = timedelta(minutes=5)

    max_workers = os.cpu_count()

    progress1_container = st.container()
    with progress1_container:
        gt_progress = st.progress(0)
        gt_status = st.text("Processing GT")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        gt_results = []
        for i, result in enumerate(executor.map(
                partial(save_figure, base_date=base_date, time_step=time_step, out_dir=out_dir,
                        save_on_disk=save_on_disk),
                gt_data[:, 0], range(gt_data.shape[0])
        )):
            gt_results.append(result)
            gt_progress.progress((i + 1) / len(gt_data[:, 0]))
            gt_status.text(f"Processed {i + 1}/{len(gt_data[:, 0])} GT images")
    # st.write(f"Time for GT data: {time.time() - timea} seconds")

    progress_container = st.container()
    with progress_container:
        pred_progress = st.progress(0)
        pred_status = st.text("Processing Predictions")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pred_results = []
        total_sequences = pred_data.shape[0]

        for i, result in enumerate(executor.map(
                partial(save_prediction_sequence, base_date=base_date, time_step=time_step, out_dir=out_dir,
                        save_on_disk=save_on_disk),
                pred_data, range(total_sequences)
        )):
            pred_results.append(result)
            pred_progress.progress((i + 1) / total_sequences)
            pred_status.text(f"Processed {i + 1}/{total_sequences} predictions sequences")

    # Aggregate results into dictionaries
    gt_figures = {key: fig for result in gt_results for key, fig in result.items()}
    pred_figures = {key: value for result in pred_results for key, value in result.items()}

    return gt_figures, pred_figures


def save_figure(data_slice, index, base_date, time_step, out_dir, save_on_disk):
    timestamp = base_date + index * time_step
    fig = compute_figure(data_slice, timestamp.strftime('%d-%m-%Y %H:%M'))

    if save_on_disk:
        file_name = f"{timestamp.strftime('%d%m%Y_%H%M')}.png"
        file_path = os.path.join(out_dir, 'gt', file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return {}
    else:
        plt.close(fig)
        return {timestamp.strftime('%d%m%Y_%H%M'): fig}


def save_prediction_sequence(data_series, element_index, base_date, time_step, out_dir, save_on_disk):
    element_timestamp = base_date + element_index * time_step
    folder_name = element_timestamp.strftime('%d%m%Y_%H%M')
    results = {}

    for i, prediction in enumerate(data_series):
        timestamp_offset = i * 5  # 5 minutes per time step
        if i != 30 or i != 60:
            pass
        fig = compute_figure(prediction, f"+{timestamp_offset}min")

        if save_on_disk:
            folder_path = os.path.join(out_dir, 'pred', folder_name)
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"{folder_name}_+{timestamp_offset}min.png"
            file_path = os.path.join(folder_path, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            results[f"+{timestamp_offset}min"] = fig

    if not save_on_disk:
        return {folder_name: results}
    else:
        return {}
