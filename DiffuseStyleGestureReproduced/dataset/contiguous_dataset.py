import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import random

def create_consolidated_data(input_folder, output_file):
    """Create a single consolidated file with all gesture/audio data"""
    # Find all NPZ files and sort them for consistent order
    files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz')])
    print(f"Found {len(files)} NPZ files to consolidate")
    
    # Process first file to get dimensions
    with np.load(files[0]) as npz:
        gesture_dim = npz["bvh_features"].shape[1]
        audio_dim = npz["audio_features"].shape[1]
        speaker_shape = npz["main_agent_id_one_hot"].shape
    
    # First pass: calculate total frames and collect metadata with exact sizes
    total_frames = 0
    file_segments = []
    speaker_data = []
    file_sizes = {}  # Store exact sizes to prevent mismatches
    
    for file_path in tqdm(files, desc="Analyzing files"):
        try:
            with np.load(file_path) as npz:
                # Get exact frame count from this specific file
                gesture_frames = len(npz["bvh_features"])
                audio_frames = len(npz["audio_features"])
                
                # Verify data consistency
                if gesture_frames != audio_frames:
                    print(f"Warning: {file_path} has mismatched frames - gestures:{gesture_frames}, audio:{audio_frames}")
                    # Use minimum to stay safe
                    num_frames = min(gesture_frames, audio_frames)
                else:
                    num_frames = gesture_frames
                
                # Store the actual size for verification later
                file_sizes[file_path] = num_frames
                
                if num_frames > 0:
                    # Record segment information
                    file_segments.append({
                        'file': os.path.basename(file_path),
                        'full_path': file_path,  # Store full path for matching
                        'start_idx': total_frames,
                        'end_idx': total_frames + num_frames,
                        'frames': num_frames
                    })
                    
                    # Save speaker data
                    speaker_data.append(np.array(npz["main_agent_id_one_hot"], dtype=np.float16))
                    
                    # Update total frame count
                    total_frames += num_frames
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Total frames to consolidate: {total_frames}")
    
    # Create consolidated arrays
    gestures = np.zeros((total_frames, gesture_dim), dtype=np.float16)
    audio = np.zeros((total_frames, audio_dim), dtype=np.float16)
    
    # Second pass: fill the consolidated arrays
    for i, segment in enumerate(tqdm(file_segments, desc="Consolidating data")):
        file_path = segment['full_path']  # Use the stored full path
        start = segment['start_idx']
        end = segment['end_idx']
        expected_frames = segment['frames']
        
        try:
            with np.load(file_path) as npz:
                # Get exact frame count again
                gesture_frames = len(npz["bvh_features"])
                
                # Verify the frame count matches what we recorded
                if gesture_frames != expected_frames:
                    print(f"  WARNING: Frame count mismatch for {file_path}")
                    print(f"  Recorded: {expected_frames} frames, Actual: {gesture_frames} frames")
                    # Use the smaller of the two to avoid errors
                    copy_frames = min(expected_frames, gesture_frames)
                    print(f"  Using {copy_frames} frames to avoid error")
                else:
                    copy_frames = expected_frames
                
                # Explicitly extract the correct number of frames to copy
                gesture_data = npz["bvh_features"][:copy_frames].astype(np.float16)
                audio_data = npz["audio_features"][:copy_frames].astype(np.float16)
                
                # Copy data to consolidated arrays with explicit slicing
                gestures[start:start+copy_frames] = gesture_data
                audio[start:start+copy_frames] = audio_data
                
        except Exception as e:
            print(f"Error copying data from {file_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save consolidated data
    print(f"Saving consolidated data to {output_file}")
    np.savez_compressed(
        output_file,
        gestures=gestures,
        audio=audio,
        speakers=np.array(speaker_data, dtype=np.float16)
    )
    
    # Save metadata separately for quick access
    meta_file = output_file.replace('.npz', '_meta.pkl')
    with open(meta_file, 'wb') as f:
        metadata = {
            'total_frames': total_frames,
            'gesture_dim': gesture_dim,
            'audio_dim': audio_dim,
            'speaker_shape': speaker_shape,
            'file_segments': file_segments
        }
        pickle.dump(metadata, f)
    
    print(f"Consolidated data created successfully!")
    print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

if __name__ == "__main__":
    create_consolidated_data("dataset/genea2023_dataset/trn/main-agent/features", "dataset/genea2023_dataset/trn/main-agent/consolidated.npz")