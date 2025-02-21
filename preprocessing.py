import numpy as np



def differential_window(y, w=100000):
    D = np.gradient(y, axis=1)
    return np.exp(np.abs(D * y) / w)



if __name__ == '__main__':

    import mne
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
    