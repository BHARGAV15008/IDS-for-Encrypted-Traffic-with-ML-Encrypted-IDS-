import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import os

def generate_packets_for_flow(flow_series, flow_id):
    """
    Generates a DataFrame of synthetic packets for a single flow.
    """
    packets = []
    
    # Basic flow features
    total_fwd_packets = int(flow_series.get('Total Fwd Packets', 0))
    total_bwd_packets = int(flow_series.get('Total Backward Packets', 0))
    flow_duration = flow_series.get(' Flow Duration', 1) # Avoid division by zero
    
    # Forward packet characteristics
    fwd_pkt_len_mean = flow_series.get(' Fwd Packet Length Mean', 0)
    fwd_pkt_len_std = flow_series.get(' Fwd Packet Length Std', 0)
    
    # Backward packet characteristics
    bwd_pkt_len_mean = flow_series.get(' Bwd Packet Length Mean', 0)
    bwd_pkt_len_std = flow_series.get(' Bwd Packet Length Std', 0)
    
    # Inter-arrival times
    flow_iat_mean = flow_series.get(' Flow IAT Mean', 0)
    flow_iat_std = flow_series.get(' Flow IAT Std', 0)
    
    # Label
    label = 1 if 'BENIGN' not in str(flow_series.get(' Label', 'BENIGN')).upper() else 0

    # Generate packet timings
    total_packets = total_fwd_packets + total_bwd_packets
    if total_packets == 0:
        return pd.DataFrame()

    # Generate inter-arrival times, ensuring they are non-negative
    iats = np.random.normal(flow_iat_mean, flow_iat_std, total_packets - 1)
    iats[iats < 0] = flow_iat_mean / 10 # Replace negative IATs with a small positive value
    timestamps = np.concatenate(([0], np.cumsum(iats)))

    # Generate packet sizes, ensuring they are non-negative
    fwd_sizes = np.random.normal(fwd_pkt_len_mean, fwd_pkt_len_std, total_fwd_packets)
    fwd_sizes[fwd_sizes < 0] = 0
    bwd_sizes = np.random.normal(bwd_pkt_len_mean, bwd_pkt_len_std, total_bwd_packets)
    bwd_sizes[bwd_sizes < 0] = 0
    
    # Create packet list
    fwd_packets = [{'timestamp': ts, 'size': size, 'direction': 'client_to_server'} for ts, size in zip(timestamps[:total_fwd_packets], fwd_sizes)]
    bwd_packets = [{'timestamp': ts, 'size': size, 'direction': 'server_to_client'} for ts, size in zip(timestamps[total_fwd_packets:], bwd_sizes)]
    
    all_packets = sorted(fwd_packets + bwd_packets, key=lambda x: x['timestamp'])
    
    # Create DataFrame
    df = pd.DataFrame(all_packets)
    df['flow_id'] = flow_id
    df['label'] = label
    
    # Add placeholder columns for TLS features
    df['is_handshake'] = False
    df['tls_version'] = 'unknown'
    df['cipher_suites'] = "[]"
    df['key_exchange'] = ''
    df['certificate_size'] = 0
    df['certificate_issuer'] = ''
    df['certificate_validity_days'] = 0
    df['extensions'] = "[]"
    
    return df

def create_packet_level_dataset(output_path):
    """
    Combines flow-based CSVs and generates a packet-level dataset.
    """
    cve_path = '01_Data/MachineLearningCVE/'
    traffic_path = '01_Data/TrafficLabelling_/'
    
    cve_files = glob.glob(os.path.join(cve_path, '*.csv'))
    traffic_files = glob.glob(os.path.join(traffic_path, '*.csv'))
    all_files = cve_files + traffic_files
    
    if not all_files:
        print("No CSV files found in the specified directories.")
        return

    print(f"Found {len(all_files)} CSV files to process.")
    
    all_packets_dfs = []
    flow_counter = 0
    
    for file in tqdm(all_files, desc="Processing files"):
        try:
            df = pd.read_csv(file, encoding='latin1', low_memory=False)
            df.columns = df.columns.str.strip() # Clean column names
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing flows in {os.path.basename(file)}", leave=False):
                packets_df = generate_packets_for_flow(row, flow_counter)
                if not packets_df.empty:
                    all_packets_dfs.append(packets_df)
                    flow_counter += 1
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if not all_packets_dfs:
        print("No packets were generated. Please check the input data.")
        return

    print("Combining all generated packets...")
    final_df = pd.concat(all_packets_dfs, ignore_index=True)
    
    print(f"Saving packet-level data to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == '__main__':
    output_file = '01_Data/02_Processed/combined_packet_level_data.csv'
    create_packet_level_dataset(output_file)
