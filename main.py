import mne
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation


import warnings
warnings.filterwarnings('ignore')

from util import window_steps, parse_seizure_file, get_path

from preprocessing import get_bands

from matrix import frobenius_norm, riemannian_metric_spd, moving_average, spectral_width

from graph import adjacency_to_incidence, edge_weights, weighted_graph_laplacian

from synchronization import wpli, sliding_window

from plots import plot_seizure



# File name
patient = 3
record = 2
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
all_wpli_vals = [
    sliding_window(records, wpli, window_size=window, step_size=step)
    for records in tqdm(bands_records, desc="Computing wPLI")
]

# Compute spectral embeddings
all_laplacians = []
for band, band_wpli_vals in tqdm(list(zip(bands_names, all_wpli_vals)), desc="Computing spectral embeddings"):
    incidences = [adjacency_to_incidence(adjacency) for adjacency in band_wpli_vals]
    weights = [edge_weights(adjacency) for adjacency in band_wpli_vals]
    band_laplacians = np.array([
        weighted_graph_laplacian(incidence, weight, adj_matrix=adjacency)
        for incidence, weight, adjacency in zip(incidences, weights, band_wpli_vals)
    ])
    all_laplacians.append(band_laplacians)

# Compute spectral embeddings
eig_results = [np.linalg.eig(laplacian) for laplacian in all_laplacians]
all_eigenvalues = np.array([result[0] for result in eig_results])
all_eigenvectors = np.array([result[1]*np.sign(result[1]) for result in eig_results])

# Sort embeddings
for band_eigenvalues, band_eigenvectors in zip(all_eigenvalues, all_eigenvectors):
    sorted_indices = [np.argsort(eigenvalues) for eigenvalues in band_eigenvalues]
    band_eigenvalues = [eigenvalues[sorted_index] for eigenvalues, sorted_index in zip(band_eigenvalues, sorted_indices)]
    band_eigenvectors = [eigenvectors[sorted_index] for eigenvectors, sorted_index in zip(band_eigenvectors, sorted_indices)]

# Reduce dimensionality
in_dim = 10
out_dim = 2
if in_dim != out_dim:
    all_reduced_eigenvectors = [
        np.array([
            PCA(n_components=out_dim).fit_transform(band_eigenvector[:, 1:in_dim+1])
            for band_eigenvector in band_eigenvectors
        ])
        for band_eigenvectors in tqdm(all_eigenvectors, desc="Reducing dimensionality")
    ]
else:
    all_reduced_eigenvectors = all_eigenvectors[:, :, 1:]

# Compute clusters
eps = 0.1
min_samples = 3
all_labels = [
    np.array([
        DBSCAN(eps=eps, min_samples=min_samples).fit_predict(band_reduced_eigenvector)
        for band_reduced_eigenvector in band_reduced_eigenvectors
    ])
    for band_reduced_eigenvectors in tqdm(all_reduced_eigenvectors, desc="Computing clusters")
]

# Extract data of interest
delta_eigenvectors = all_reduced_eigenvectors[-1]
data = delta_eigenvectors[:, :, 0:2]


# Plot data
print("Preparing the plot ...")
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(np.min(data[:, :, 0]) - 1, np.max(data[:, :, 0]) + 1)
ax.set_ylim(np.min(data[:, :, 1]) - 1, np.max(data[:, :, 1]) + 1)
scatter = ax.scatter(data[0, :, 0], data[0, :, 1], c='blue', alpha=0.6)
ax.set_title("Spectral Embeddings")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")

def update(frame):
    x = data[frame, :, 0]
    y = data[frame, :, 1]
    scatter.set_offsets(np.column_stack((x, y)))
    ax.set_title(f"Spectral Embeddings — Frame {frame} / {data.shape[0]}")
    return scatter

anim = FuncAnimation(fig, update, frames=data.shape[0], interval=200, blit=False, repeat=True)

anim.save('reduced_normalized_spectral_embeddings.gif', writer='pillow', fps=30)

plt.show()
