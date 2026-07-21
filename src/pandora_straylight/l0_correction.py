import numpy as np
import os

# Load the correction matrix
correction_matrix = np.loadtxt(r"outputs\straylight_matrix_original_method\P63_corr_mat_ib20.csv", delimiter=",")

# Input file path
input_file = r"data\stray_light_afftected_L0\Pandora63s1_GreenbeltMD_20250414_L0.txt"

# Determine the output file name
base, ext = os.path.splitext(input_file)

output_dir = r"data\corrected_L0"
output_file = output_dir + "_corrected" + ext

# Read the file using 'latin-1' encoding to avoid UnicodeDecodeError.
with open(input_file, 'r', encoding='latin-1') as f:
    lines = f.readlines()

# First 44 lines are metadata.
metadata = lines[:44]
data_lines = lines[44:]

# Counters for dark and raw rows (measurement rows that get corrected)
dark_count_rows = 0
raw_count_rows = 0

# Build a dictionary of dark counts from rows where:
# Column 1 == "SQ", Column 9 (filterwheel #1) == "1", and Column 10 (filterwheel #2) == "3".
# key = routine count from Column 3.
dark_dict = {}
for line in data_lines:
    if not line.strip():
        continue
    parts = line.strip().split()
    # if parts[0] == "SQ" and parts[8] == "9" and parts[9] == "9":
    if parts[0] == "SS" and parts[8] < "9" and parts[9] == "3":
        dark_count_rows += 1
        try:
            dark_counts = np.array(parts[24:2072], dtype=float)
        except Exception as e:
            print("Error converting dark counts in line:", line, e)
            continue
        key = parts[2]
        dark_dict[key] = dark_counts

# Process the data lines.
processed_lines = []
for line in data_lines:
    if not line.strip():
        processed_lines.append(line)
        continue

    parts = line.strip().split()
    
    # Process measurement rows: SS with filterwheel #1 = 1 and filterwheel #2 = 5.
    # if parts[0] == "SS" and parts[8] == "2" and parts[9] == "8":
    if parts[0] == "SS" and parts[8] < "9" and parts[9] != "3":
        raw_count_rows += 1
        key = parts[2]
        # Retrieve the dark counts using the routine key.
        # If no matching dark row is found, use an array of zeros.
        dark_counts = dark_dict.get(key, np.zeros(2048))
        
        try:
            raw_counts = np.array(parts[24:2072], dtype=float)
        except Exception as e:
            print("Error converting raw counts in line:", line, e)
            processed_lines.append(line)
            continue
        
        # Subtract dark counts from raw counts, apply the correction matrix,
        # then add the dark counts back, clipping the correction term to avoid negatives.
        corrected_counts = dark_counts + np.clip(np.dot(correction_matrix, (raw_counts - dark_counts)), 0, None)
        
        # Format the corrected values as round numbers (no decimals).
        corrected_counts_str = [f"{val:.0f}" for val in corrected_counts]
        
        # Rebuild the line: keep columns 1-24 and any columns after 2072 unchanged.
        new_parts = parts[:24] + corrected_counts_str + parts[2072:]
        new_line = " ".join(new_parts) + "\n"
        processed_lines.append(new_line)
    else:
        # Leave other rows unchanged.
        processed_lines.append(line)

# Write the updated file.
with open(output_file, 'w', encoding='latin-1') as f:
    f.writelines(metadata)
    f.writelines(processed_lines)

print("File has been processed and saved as:", output_file)
print("Total dark counts rows found:", dark_count_rows)
print("Total bright counts rows found:", raw_count_rows)