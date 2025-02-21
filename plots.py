import matplotlib.pyplot as plt


def plot_seizure(record, times, label, seizures=None, window=None, ax=None):
    # Use the provided ax or create a new one if not provided
    if ax is None:
        _, ax = plt.subplots()
    
    # Plot the signal on the ax
    ax.plot(times, record, linestyle='-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(label)
    
    # Window truncation
    extended = False
    if window is not None and seizures is not None:
        if len(seizures) == 1:
            mid = (seizures[0][1] + seizures[0][0]) / 2
            length = seizures[0][1] - seizures[0][0]
        else:
            mid = (seizures[-1][1] + seizures[0][0]) / 2
            length = seizures[-1][1] - seizures[0][0]
        if window < length:
            window = length
            print('Window size is smaller than the seizure length, resulting in window extension.')
            extended = True
        ax.set_xlim([mid - window / 2, mid + window / 2])
    
    # Seizures delineation
    if not extended and seizures is not None:
        for seizure in seizures:
            ax.axvline(x=seizure[0], color='red', linestyle='--', linewidth=1)
            ax.axvline(x=seizure[1], color='red', linestyle='--', linewidth=1)
    
    ax.grid(True)
    return ax