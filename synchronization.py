import mne
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt



def instantaneous_phase(record):
    hilbert_transform = hilbert(record)
    phase = np.angle(hilbert_transform, deg=False)
    return phase


def wpli(ref, comp):
    difference = np.sin(ref - comp)
    return np.abs(np.sum(difference)) / np.sum(np.abs(difference))


def pli(ref, comp):
    difference = np.sign(ref - comp)
    return np.abs(np.sum(difference)) / len(difference)


def synchronization_matrix(data, method):
    n_channels = len(data)

    # Compute instantaneous phase
    phase = np.zeros((n_channels, len(data[0])))
    for i in range(n_channels):
        phase[i] = instantaneous_phase(data[i])

    # Compute WPLI
    matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = method(phase[i], phase[j])
    return matrix


def sliding_window(data, method, window_size, step_size):
    n_channels = len(data)
    n_samples = len(data[0])
    n_windows = int((n_samples - window_size) / step_size) + 1

    # Compute the method for each window
    results = np.zeros((n_windows, n_channels, n_channels))
    for i in range(n_windows):
        window = data[:, i*step_size:i*step_size+window_size]
        results[i] = synchronization_matrix(window, method)
    return results


if __name__ == '__main__':
    # Load the EDF file
    file_path = 'chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01_01.edf'
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Display basic information
    data, times = raw[:]

    # Test functions
    print(instantaneous_phase(data[0][40:60]))