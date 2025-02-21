import os
import re
import mne
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import warnings
import concurrent.futures

warnings.filterwarnings('ignore')

from util import parse_seizure_file
from synchronization import wpli, sliding_window


def true_with_probability(p):
    """
    Returns True with probability p and False with probability 1-p.
    """
    return random.random() < p


def process_file(args):
    """
    Processes a single EDF file.
    
    Args:
        args: A tuple containing:
            file_path (str): Full path to the EDF file.
            seizures (list): List of (start, end) tuples for seizure intervals.
            file_resampling (float): Probability of processing a value.
            window_duration (float): Duration of the sliding window in seconds.
            overlap_duration (float): Duration of the overlap in seconds.
    
    Returns:
        A tuple of three lists: (overall_vals, ictal_vals, interictal_vals)
    """
    file_path, seizures, file_resampling, window_duration, overlap_duration = args
    overall_vals = []
    ictal_vals = []
    interictal_vals = []
    
    try:
        # Load EDF file and apply filtering
        raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto', verbose=False)
        raw.filter(l_freq=1, h_freq=None, fir_design='firwin', verbose=False)
        records, _ = raw[:-1]
        
        # Parameters for sliding window
        sfreq = raw.info['sfreq']
        window = int(sfreq * window_duration)
        step_duration = window_duration - overlap_duration
        step = int(sfreq * step_duration)
        
        # Compute wPLI matrices via sliding window
        wpli_matrices = sliding_window(records, wpli, window_size=window, step_size=step)
        
        # Process each sliding window segment
        for seg_index, wpli_matrix in enumerate(wpli_matrices):
            segment_start = seg_index * step_duration
            segment_midpoint = segment_start + window_duration / 2
            
            # Iterate over the lower triangle of the matrix
            for i in range(1, wpli_matrix.shape[0]):
                for j in range(i):
                    # Only process a fraction of files to avoid memory issues
                    if true_with_probability(file_resampling):
                        val = wpli_matrix[i][j]
                        overall_vals.append(val)
                        if any(seizure_start <= segment_midpoint <= seizure_end for seizure_start, seizure_end in seizures):
                            ictal_vals.append(val)
                        else:
                            interictal_vals.append(val)
                        
        return overall_vals, ictal_vals, interictal_vals
    except Exception as e:
        print("Error processing file:", file_path, e)
        return [], [], []
    

def quantile_resample(values, n_samples):
    """
    Resamples an array to a specified number of samples using quantile-based sampling.
    
    Args:
        values (np.ndarray): The input array to resample.
        n_samples (int): The number of samples to draw.
        
    Returns:
        np.ndarray: The resampled array, preserving the original distribution.
    """
    if len(values) <= n_samples:
        print("Not enough values for resampling")
        return np.array(values)
    
    # Compute quantiles
    quantiles = np.linspace(0, 1, n_samples + 1)
    quantile_values = np.quantile(values[~np.isnan(values)], quantiles)
    
    # Sample from each quantile segment
    samples = []
    for i in range(len(quantile_values) - 1):
        segment = values[(values >= quantile_values[i]) & (values < quantile_values[i+1])]
        if len(segment) > 0:
            sample = np.random.choice(segment, 1, replace=False)
            samples.append(sample[0])
    
    return np.array(samples)


if __name__ == '__main__':

    # Number of threads
    max_workers = 8

    # Resampling
    file_resampling = 0.001   # probability
    box_resampling = 100 # number of samples
    qq_resampling = 100    # number of samples

    # Window and overlap duration in seconds
    window_duration = 6
    overlap_duration = 1

    # Define patterns for directories and files
    dir_pattern = re.compile(r'^chb\d{2}$')
    file_pattern = re.compile(r'^chb\d{2}_\d{2}\.edf$')

    # Directory path
    base_dir = 'chb-mit-scalp-eeg-database-1.0.0-nobug'

    # Lists to store the results
    ictal_values = []
    interictal_values = []
    overall_values = []

    # Gather tasks
    tasks = []
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path) and dir_pattern.match(subdir):
            # Build the summary file pathfor each patient
            summary_file = os.path.join(subdir_path, f"{subdir}-summary.txt")
            all_seizures = parse_seizure_file(summary_file)
            for filename in os.listdir(subdir_path):
                if file_pattern.match(filename):
                    file_path = os.path.join(subdir_path, filename)
                    seizures = all_seizures.get(filename, [])
                    seizures = seizures if seizures is not None else []
                    args = (file_path, seizures, file_resampling, window_duration, overlap_duration)
                    tasks.append(args)
    print(f"Total files to process: {len(tasks)}")

    # Define executor pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks)))
    
    # Get results from all files
    for res in results:
        ov, ict, interict = res
        overall_values.extend(ov)
        ictal_values.extend(ict)
        interictal_values.extend(interict)



    # --- Plot Histograms ---
    bin_size = 0.05
    bin_edges = np.arange(0, 1 + bin_size, bin_size)

    plt.figure(figsize=(12, 10))

    # Overall distribution
    plt.subplot(3, 1, 1)
    ax1 = plt.gca()
    ax1.set_axisbelow(True)
    if overall_values:
        weights = np.ones(len(overall_values)) / len(overall_values) * 100
        plt.hist(overall_values, bins=bin_edges, weights=weights,
                color='blue', alpha=0.7, edgecolor='black')
    plt.title('Overall wPLI Distribution')
    plt.ylabel('Percentage (%)')

    # Ictal distribution
    plt.subplot(3, 1, 2)
    ax2 = plt.gca()
    ax2.set_axisbelow(True)
    if ictal_values:
        weights = np.ones(len(ictal_values)) / len(ictal_values) * 100
        plt.hist(ictal_values, bins=bin_edges, weights=weights,
                color='red', alpha=0.7, edgecolor='black')
    plt.title('Ictal wPLI Distribution')
    plt.ylabel('Percentage (%)')

    # Interictal distribution
    plt.subplot(3, 1, 3)
    ax3 = plt.gca()
    ax3.set_axisbelow(True)
    if interictal_values:
        weights = np.ones(len(interictal_values)) / len(interictal_values) * 100
        plt.hist(interictal_values, bins=bin_edges, weights=weights,
                color='green', alpha=0.7, edgecolor='black')
    plt.title('Interictal wPLI Distribution')
    plt.xlabel('wPLI')
    plt.ylabel('Percentage (%)')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'wpli_distributions_resample_p{file_resampling}.pdf')


    # --- Plot Boxplots for the Three Distributions ---
    plt.figure(figsize=(8, 6))

    # Resample to avoid too many outliers
    box_resampling = 100
    resampled_ictal_values = quantile_resample(np.array(ictal_values), box_resampling)
    resampled_interictal_values = quantile_resample(np.array(interictal_values), box_resampling)
    resampled_overall_values = quantile_resample(np.array(overall_values), box_resampling)
        
    # Prepare data and labels
    data_resampled = [resampled_overall_values, resampled_ictal_values, resampled_interictal_values]
    labels = ['Overall', 'Ictal', 'Interictal']
        
    whis = [5, 95]
    plt.boxplot(data_resampled, labels=labels, whis=whis, showfliers=True)
    plt.ylabel("wPLI")
    ax_box = plt.gca()
    ax_box.set_axisbelow(True)
    plt.savefig(f'wpli_boxplots_resample_p{box_resampling}.pdf')


    # --- Q-Q Plot for Ictal vs. Interictal wPLI Values ---
    plt.figure(figsize=(8, 6))

    # Resample to the same number of quantiles.
    qq_resampling = min(len(ictal_values), len(interictal_values), qq_resampling)
    resampled_ictal_qq = quantile_resample(np.array(ictal_values), qq_resampling)
    resampled_interictal_qq = quantile_resample(np.array(interictal_values), qq_resampling)
    actual_resampling = min(len(resampled_ictal_qq), len(resampled_interictal_qq))
    resampled_ictal_qq = resampled_ictal_qq[:actual_resampling]
    resampled_interictal_qq = resampled_interictal_qq[:actual_resampling]
        
    # Sort the resampled data so that each index corresponds to a quantile
    resampled_ictal_qq = np.sort(resampled_ictal_qq)
    resampled_interictal_qq = np.sort(resampled_interictal_qq)

    # Create the Q-Q plot
    plt.scatter(resampled_interictal_qq, resampled_ictal_qq, facecolors='none', edgecolors='black', alpha=0.7, zorder=2)

    # Plot the reference line
    min_val = min(np.min(resampled_ictal_qq), np.min(resampled_interictal_qq))
    max_val = max(np.max(resampled_ictal_qq), np.max(resampled_interictal_qq))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='dashed', linewidth=1, color='red', zorder=1)

    plt.xlabel('Interictal')
    plt.ylabel('Ictal')
    ax_qq = plt.gca()
    ax_qq.set_axisbelow(True)
    ax_qq.grid(True, which='both', alpha=0.7)
    plt.savefig(f'wpli_qqplot_resample_p{qq_resampling}.pdf')

    plt.show()


