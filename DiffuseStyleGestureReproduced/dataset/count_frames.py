import os
import csv
import numpy as np
import argparse

def count_frames(folder):
    # List all npz files
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
    
    # Count frames for each file
    frames_per_file = {}
    for file in all_files:
        with np.load(file, allow_pickle=True) as npz:
            total_frames = npz["bvh_features"].shape[0]
        frames_per_file[file] = total_frames
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(folder), 'frame_counts.csv')
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'frame_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for filename, frame_count in frames_per_file.items():
            writer.writerow({'filename': filename, 'frame_count': frame_count})
    
    print(f"Frame counts saved to {output_path}")
    
    return frames_per_file

if __name__ == "__main__":    
    count_frames('dataset/genea2023_dataset/trn/main-agent/features/')