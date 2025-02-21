import mne
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from util import window_steps, parse_seizure_file

from matrix import riemannian_metric_spd

from graph import adjacency_to_incidence, edge_weights, weighted_graph_laplacian

from synchronization import pli, wpli, sliding_window

from plots import plot_seizure



# File name
record_path = 'chb-mit-scalp-eeg-database-1.0.0\\chb03\\chb03_02.edf'

# Load the EDF file
raw = mne.io.read_raw_edf(record_path, preload=True, stim_channel='auto', verbose=False)
raw.filter(l_freq=1, h_freq=None, fir_design='firwin', verbose=False)
seizures = parse_seizure_file('chb-mit-scalp-eeg-database-1.0.0\\chb03\\chb03-summary.txt')['chb03_02.edf']
records, _ = raw[:-1]

# Parameters
sfreq= raw.info['sfreq']
window_duration = 6
window = int(sfreq * window_duration)
overlap_duration = 1
overlap = int(sfreq * overlap_duration)
step_duration = window_duration - overlap_duration
step = int(sfreq * step_duration)

# Compute PLI and WPLI
wpli_vals = sliding_window(records, wpli, window_size=window, step_size=step)
pli_vals = sliding_window(records, pli, window_size=window, step_size=step)