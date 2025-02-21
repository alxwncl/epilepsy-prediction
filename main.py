import mne
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from util import window_steps, parse_seizure_file

from matrix import frobenius_norm, airm, kappa, moving_average

from synchronization import pli, wpli, synchronization_matrix, sliding_window

from preprocessing import differential_window

from plots import plot_seizure



# File name
record_path = 'chb-mit-scalp-eeg-database-1.0.0\\chb03\\chb03_02.edf'
# record_path = 'chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01_18.edf'

# Load the EDF file
raw = mne.io.read_raw_edf(record_path, preload=True, stim_channel='auto', verbose=False)
raw.filter(l_freq=1, h_freq=None, fir_design='firwin', verbose=False)
seizures = parse_seizure_file('chb-mit-scalp-eeg-database-1.0.0\\chb03\\chb03-summary.txt')['chb03_02.edf']
# seizures = parse_seizure_file('chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01-summary.txt')['chb01_18.edf']
records, _ = raw[:-1]

# Parameters
sfreq= raw.info['sfreq']
window_duration = 6
window = int(sfreq * window_duration)
overlap_duration = 1
overlap = int(sfreq * overlap_duration)
step_duration = window_duration - overlap_duration
step = int(sfreq * step_duration)

# Compute WPLI
wpli_vals = sliding_window(records, wpli, window_size=window, step_size=step)
pli_vals = sliding_window(records, pli, window_size=window, step_size=step)
mapped_wpli_vals = kappa(wpli_vals)
wpli_mean_vals = wpli_vals.mean(axis=(1, 2))
mapped_wpli_mean_vals = mapped_wpli_vals.mean(axis=(1, 2))
avg = moving_average(wpli_vals, 100)
mapped_avg = moving_average(mapped_wpli_vals, 100)
wpli_wrt_avg = np.hstack((np.full(99, np.nan), frobenius_norm(wpli_vals[99:], avg)))
mapped_wpli_wrt_avg = np.hstack((np.full(99, np.nan), airm(mapped_wpli_vals[99:], mapped_avg)))

# Plot PLI and wPLI
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes = axes.flatten()
N = wpli_vals.shape[0]
time_windows = window_steps(0, N, window_duration, step_duration)
truncation = 500
plot_seizure(wpli_vals[:, 9, -1], time_windows, r'$\text{wPLI}$', seizures, window=truncation, ax=axes[0])
plot_seizure(pli_vals[:, 9, -1], time_windows, r'$\text{PLI}$', seizures, window=truncation, ax=axes[1])
plt.show()

# Plot d(avg, wpli)
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes = axes.flatten()
N = wpli_vals.shape[0]
time_windows = window_steps(0, N, window_duration, step_duration)
truncation = None
plot_seizure(wpli_wrt_avg, time_windows, r'$d(\text{avg}, \text{wpli})$ in Euclidean manifold', seizures, window=truncation, ax=axes[0])
plot_seizure(mapped_wpli_wrt_avg, time_windows, r'$d(\text{avg}, \text{wpli})$ in SPD manifold', seizures, window=truncation, ax=axes[1])
plt.show()