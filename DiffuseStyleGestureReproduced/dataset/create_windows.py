import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import random

def create_windows_database(input_folder, output_file, 
                          seq_length=150, seed_length=8,
                          num_windows=100000):
    """Pre-extract windows and save them in an optimized database format"""
    
    # Find all NPZ files
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz')]
    print(f"Found {len(files)} NPZ files")
    
    # Process first file to get dimensions
    with np.load(files[0]) as npz:
        gesture_dim = npz["bvh_features"].shape[1]
        audio_dim = npz["audio_features"].shape[1]
        speaker_shape = npz["main_agent_id_one_hot"].shape
    
    # Create storage arrays for extracted windows
    chunk_size = seq_length + seed_length
    
    # Pre-allocate arrays for all windows
    gestures = np.zeros((num_windows, seq_length, gesture_dim), dtype=np.float16)
    seeds = np.zeros((num_windows, seed_length, gesture_dim), dtype=np.float16)
    audio = np.zeros((num_windows, seq_length, audio_dim), dtype=np.float16)
    speakers = np.zeros((num_windows, speaker_shape[0]), dtype=np.float16)
    
    # Track valid files and their frame counts
    valid_files = []
    frames_per_file = {}
    
    for file_path in files:
        with np.load(file_path) as npz:
            frames = len(npz["bvh_features"])
            if frames >= chunk_size:
                valid_files.append(file_path)
                frames_per_file[file_path] = frames
    
    print(f"Found {len(valid_files)} valid files")
    
    # Extract windows
    window_idx = 0
    progress_bar = tqdm(total=num_windows, desc="Extracting windows")
    
    while window_idx < num_windows:
        # Select random file
        file_path = random.choice(valid_files)
        total_frames = frames_per_file[file_path]
        
        # Load the data
        with np.load(file_path) as npz:
            # Extract multiple windows from this file
            windows_to_extract = min(100, num_windows - window_idx)
            
            for i in range(windows_to_extract):
                # Choose random starting frame
                start_frame = random.randint(0, total_frames - chunk_size)
                
                # Extract windows
                gestures[window_idx] = npz["bvh_features"][
                    start_frame + seed_length : start_frame + chunk_size].astype(np.float16)
                seeds[window_idx] = npz["bvh_features"][
                    start_frame : start_frame + seed_length].astype(np.float16)
                audio[window_idx] = npz["audio_features"][
                    start_frame + seed_length : start_frame + chunk_size].astype(np.float16)
                speakers[window_idx] = np.array(npz["main_agent_id_one_hot"], dtype=np.float16)
                
                window_idx += 1
                progress_bar.update(1)
                
                if window_idx >= num_windows:
                    break
    
    progress_bar.close()
    
    # Save arrays to a single file
    print(f"Saving {num_windows} windows to {output_file}")
    np.savez_compressed(
        output_file,
        gestures=gestures,
        seeds=seeds,
        audio=audio,
        speakers=speakers
    )
    
    # Save window database metadata
    meta_file = output_file.replace('.npz', '_meta.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump({
            'num_windows': num_windows,
            'seq_length': seq_length,
            'seed_length': seed_length,
            'gesture_dim': gesture_dim,
            'audio_dim': audio_dim,
            'speaker_shape': speaker_shape,
            'precision': 'float16'
        }, f)
    
    print(f"Window database created successfully!")

if __name__ == "__main__":
    create_windows_database(
        input_folder="dataset/genea2023_dataset/trn/main-agent/features",
        output_file="dataset/genea2023_dataset/trn/main-agent/training_windows_100k.npz",
        seq_length=150,
        seed_length=8,
        num_windows=100000  # Adjust based on your RAM capacity
    )