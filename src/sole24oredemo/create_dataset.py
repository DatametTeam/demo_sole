import os
import h5py
import numpy as np


def read_hdf_files_sequences(root_folder, sequence_length=12):
    """
    Read HDF files and create sequences of images.

    Args:
        root_folder (str): Path to the root folder containing HDF files
        sequence_length (int): Number of consecutive images in each sequence

    Returns:
        numpy.ndarray: 4D array with shape (NUM_SEQUENCES, sequence_length, HEIGHT, WIDTH)
    """
    all_sequences = []
    file_paths = []

    # Walk through all subfolders
    for subdir, _, files in os.walk(root_folder):
        for i, file in enumerate(files):
            if file.endswith(('.h5', '.hdf', '.hdf5')):
                file_paths.append(os.path.join(subdir, file))

    file_paths = sorted(file_paths, key=lambda x: x.split('_')[-1].split('.')[0])

    for i, file_path in enumerate(file_paths[:-12]):
        with h5py.File(file_path, 'r') as hdf_file:
            # Adjust this to match your actual data path

            # Create sequences of images
            file_sequences = []
            for seq in range(12):
                next_elem = file_paths[i + seq]
                with h5py.File(next_elem, 'r') as hdf1_file:
                    data = hdf1_file['dataset1']['data1']['data'][:]
                    file_sequences.append(data)

            all_sequences.append(file_sequences)

    return np.array(all_sequences) if all_sequences else None


# Example usage
root_folder = "/davinci-1/work/protezionecivile/SRI_sole24/SRI_adj_01-25/31"
combined_sequences = read_hdf_files_sequences(root_folder)

if combined_sequences is not None:
    print("Combined sequences shape:", combined_sequences.shape)
    np.save("/davinci-1/home/guidim/demo_sole/data/output/ConvLSTM/20250205/20250205/generations/data.npy",
            combined_sequences)
