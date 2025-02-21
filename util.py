import numpy as np
import re


def window_steps(start, n_windows, window_size, step_size):
    return np.arange(n_windows) * step_size + (window_size / 2.0) + start


def parse_seizure_file(file_path):
    results = {}
    
    # Regex patterns to match both "Seizure Start Time:" and "Seizure 1 Start Time:" formats.
    seizure_start_pattern = re.compile(r"Seizure(?: \d+)? Start Time:\s*(\d+)")
    seizure_end_pattern   = re.compile(r"Seizure(?: \d+)? End Time:\s*(\d+)")
    
    with open(file_path, 'r') as file:
        # Create an iterator over the lines of the file.
        lines = iter(file)
        for line in lines:
            line = line.strip()
            # Only process lines we care about.
            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
                # Default entry: if there are no seizures, we'll leave it as None.
                results[current_file] = None

            elif line.startswith("Number of Seizures in File:"):
                count_str = line.split(":", 1)[1].strip()
                try:
                    seizure_count = int(count_str)
                except ValueError:
                    seizure_count = 0

                # If there are seizures, read the expected number of seizure time pairs.
                if seizure_count > 0:
                    seizures = []
                    for _ in range(seizure_count):
                        # Read the seizure start time line.
                        start_line = next(lines).strip()
                        match_start = seizure_start_pattern.search(start_line)
                        if match_start:
                            seizure_start = int(match_start.group(1))
                        else:
                            raise ValueError(f"Expected a seizure start time line, got: {start_line}")

                        # Read the seizure end time line.
                        end_line = next(lines).strip()
                        match_end = seizure_end_pattern.search(end_line)
                        if match_end:
                            seizure_end = int(match_end.group(1))
                        else:
                            raise ValueError(f"Expected a seizure end time line, got: {end_line}")
                        
                        seizures.append((seizure_start, seizure_end))
                    results[current_file] = seizures
                    
    return results


if __name__ == "__main__":
    file_path = "chb-mit-scalp-eeg-database-1.0.0\chb03\chb03-summary.txt"
    seizure_info = parse_seizure_file(file_path)
    print(seizure_info)
