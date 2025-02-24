import mne
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from util import window_steps, parse_seizure_file, get_path

from preprocessing import get_bands

from matrix import frobenius_norm, riemannian_metric_spd, ponderated_difference, moving_average

from graph import adjacency_to_incidence, edge_weights, weighted_graph_laplacian

from synchronization import pli, wpli, sliding_window

from plots import plot_seizure



# File name
patient = 4
record = 5
patient_name, record_path = get_path(patient, record)
dir_path = '..\\chb-mit-scalp-eeg-database-1.0.0\\'

# Load the EDF file
raw = mne.io.read_raw_edf(f'{dir_path}{patient_name}\\{record_path}', preload=True, stim_channel='auto', verbose=False)
raw.filter(l_freq=1, h_freq=None, fir_design='firwin', verbose=False)

# Get seizures times
seizures = parse_seizure_file(f'{dir_path}{patient_name}\\{patient_name}-summary.txt')[record_path]

# Get bands
bands_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
bands = [0, 4, 8, 13, 30, 100]
# bands_names = ['all']
# bands = [0, 100]
bands_records = get_bands(raw, bands, dtype='array')

# Parameters
sfreq= raw.info['sfreq']
window_duration = 6
window = int(sfreq * window_duration)
overlap_duration = 1
overlap = int(sfreq * overlap_duration)
step_duration = window_duration - overlap_duration
step = int(sfreq * step_duration)

# Compute PLI and WPLI
wpli_vals = [sliding_window(records, wpli, window_size=window, step_size=step) for records in bands_records]
print("WPLI computed")

# Compute spectral embeddings
all_laplacians = []
for band, wpli_val in zip(bands_names, wpli_vals):
    incidences = [adjacency_to_incidence(mat) for mat in wpli_val]
    weights = [edge_weights(mat) for mat in wpli_val]
    band_laplacians = np.array([weighted_graph_laplacian(incidence, weight) for incidence, weight in zip(incidences, weights)])
    all_laplacians.append(band_laplacians)
    print(f'Laplacians computed for band {band}')

# Compute difference with average
all_avg = [moving_average(band_laplacians, 100) for band_laplacians in all_laplacians]
laplacians_wrt_avg = [np.hstack((np.full(99, np.nan), frobenius_norm(band_laplacians[99:], band_avg))) for band_laplacians, band_avg in zip(all_laplacians, all_avg)]

# Compute dynamic differences
# differences = [ponderated_difference(band_laplacians, frobenius_norm, k=50, alpha=10e-32) for band_laplacians in all_laplacians]

# eig_results = [np.linalg.eig(laplacian) for laplacian in all_laplacians]
# all_eigenvalues = np.array([result[0] for result in eig_results])
# eigenvectors = np.array([result[1]*np.sign(result[1]) for result in eig_results])

# # Sort embeddings
# for band_eigenvalues in all_eigenvalues:
#     sorted_indices = [np.argsort(eigenvalues) for eigenvalues in band_eigenvalues]
#     band_eigenvalues = [eigenvalues[sorted_index] for eigenvalues, sorted_index in zip(band_eigenvalues, sorted_indices)]
# fielder_eigenvalues = [eigenvalue[:, 1] for eigenvalue in all_eigenvalues]
# eigenvectors = np.array([eigenvector[:, sorted_index] for eigenvector, sorted_index in zip(eigenvectors, sorted_indices)])


# Plot k's
N = wpli_vals[0].shape[0]
time_windows = window_steps(0, N, window_duration, step_duration)
_, axes = plt.subplots(max(len(laplacians_wrt_avg), 2), 1, figsize=(10, 10), sharex=True)
axes.flatten()
for i in range(len(laplacians_wrt_avg)):
    plot_seizure(laplacians_wrt_avg[i], time_windows, bands_names[i], seizures=seizures, ax=axes[i])
plt.show()