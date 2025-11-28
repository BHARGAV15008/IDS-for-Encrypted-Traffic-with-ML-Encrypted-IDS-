import pandas as pd
import os

def count_rows_in_csv(file_path):
    try:
        # Read only the header to get column names, then iterate through chunks
        # This is memory efficient for large files
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            num_lines = sum(1 for line in f) + 1 # +1 for the header
        return num_lines
    except Exception as e:
        return f"Error reading {file_path}: {e}"

base_path_cve = r"E:\IDS for Encrypted Traffic with ML (Encrypted IDS)\01_Data\MachineLearningCVE"
base_path_traffic_labelling = r"E:\IDS for Encrypted Traffic with ML (Encrypted IDS)\01_Data\TrafficLabelling_"

cve_files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]

print("Row counts for MachineLearningCVE files:")
for file_name in cve_files:
    file_path = os.path.join(base_path_cve, file_name)
    rows = count_rows_in_csv(file_path)
    print(f"{file_name}: {rows} rows")

print("\nRow counts for TrafficLabelling_ files:")
for file_name in cve_files: # Assuming same filenames for now
    file_path = os.path.join(base_path_traffic_labelling, file_name)
    rows = count_rows_in_csv(file_path)
    print(f"{file_name}: {rows} rows")
