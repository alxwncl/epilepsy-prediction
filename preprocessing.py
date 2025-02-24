import mne
import numpy as np



def get_bands(raw, bands, dtype='array'):
    if dtype not in ['array', 'raw']:
        raise ValueError(f'Invalid dtype. Expected "array" or "raw". Received {dtype}.')
    
    # Get filtered signals
    filtered_signals = []
    for i in range(len(bands) - 1):
        low = bands[i]
        high = bands[i + 1]
        # Create a copy of raw to avoid altering the original data
        filtered_raw = raw.copy().filter(l_freq=low, h_freq=high, fir_design='firwin', verbose=False)
        if dtype == 'array':
            records, _ = filtered_raw[:]
            filtered_signals.append(records)
        elif dtype == 'raw':
            filtered_signals.append(filtered_raw)
    return filtered_signals
        


def differential_window(y, w=100000):
    D = np.gradient(y, axis=1)
    return np.exp(np.abs(D * y) / w)



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from plots import plot_seizure
    from util import parse_seizure_file
    
    # Files names
    record_path = 'chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01_18.edf'

    # Load the EDF file
    raw = mne.io.read_raw_edf(record_path, preload=True, stim_channel='auto', verbose=False)
    seizures = parse_seizure_file('chb-mit-scalp-eeg-database-1.0.0\\chb01\\chb01-summary.txt')['chb01_18.edf']
    raw.filter(l_freq=1, h_freq=None, fir_design='firwin', verbose=False)
    records, times = raw[:-1]

    # Apply differential window
    differentiated_records = differential_window(records, w=1)
    plot_seizure(differentiated_records[0], times, 'Differentiated EEG', seizures)
    plt.figure()
    plot_seizure(records[0], times, 'Raw EEG', seizures)
    plt.show()
    