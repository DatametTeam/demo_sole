import io
from datetime import datetime

from matplotlib import pyplot as plt
from sole24oredemo.metrics import compute_CSI
from sole24oredemo.utils import read_groundtruth_and_target_data


def generate_metrics_plot(selected_date, selected_time, selected_models, config):
    selected_datetime = datetime.combine(selected_date, selected_time)

    thresholds = config.get("csi_thresholds", None)

    csi_df_total = {}
    for model in selected_models:
        _, target_data, pred_dict = read_groundtruth_and_target_data(selected_datetime.strftime("%d%m%Y_%H%M"), model)

        csi_df_model = compute_CSI(target_data, pred_dict, thresholds=thresholds)
        csi_df_total[model] = csi_df_model

    print(csi_df_total.keys())
    plots = []  # List to store plots as BytesIO objects

    for index, row in csi_df_model.iterrows():
        plt.figure(figsize=(10, 6))  # Create a new figure for each row
        x_values = csi_df_model.columns  # The column names for X-axis

        # Plot data for each model
        for model_name, model_df in csi_df_total.items():
            y_values = model_df.loc[index].values  # Get the row of the current model
            plt.plot(x_values, y_values, label=model_name, marker='o', markersize=3)  # Plot the line

        # Customize the plot
        plt.title(f"CSI @ {index} mm/h")
        plt.xlabel("Time Intervals")
        plt.ylabel("Values")
        plt.legend(title="Models")
        plt.grid(True)
        plt.tight_layout()
        plt.ylim([0, 1.1])
        # Save the plot to a BytesIO object
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)  # Reset buffer to the start
        plots.append(buffer)  # Append the buffer to the list
        plt.close()

    return plots
