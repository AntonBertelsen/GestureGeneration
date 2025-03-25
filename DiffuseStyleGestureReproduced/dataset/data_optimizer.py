import os
import numpy as np
import torch
import pickle
from tqdm import tqdm

def convert_to_memmap_fp16(input_folder, output_folder):
    """Convert NPZ files to memory-mappable binary format with float16 precision"""
    os.makedirs(output_folder, exist_ok=True)
    
    # List all NPZ files
    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz')]
    print(f"Found {len(files)} NPZ files to convert to float16")
    
    # Process each file
    for file_path in tqdm(files):
        base_name = os.path.basename(file_path).replace('.npz', '')
        output_path = os.path.join(output_folder, base_name)
        
        # Create directory for this file
        os.makedirs(output_path, exist_ok=True)
        
        # Load NPZ file
        with np.load(file_path) as npz:
            # Save metadata
            metadata = {
                'frames': len(npz["bvh_features"]),
                'gesture_dim': npz["bvh_features"].shape[1],
                'audio_dim': npz["audio_features"].shape[1],
                'speaker_shape': npz["main_agent_id_one_hot"].shape
            }
            
            with open(os.path.join(output_path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save as memory-mapped arrays using float16
            np.save(os.path.join(output_path, 'bvh_features.npy'), 
                    npz["bvh_features"].astype(np.float16))
            np.save(os.path.join(output_path, 'audio_features.npy'), 
                    npz["audio_features"].astype(np.float16))
            np.save(os.path.join(output_path, 'speaker.npy'), 
                    np.array(npz["main_agent_id_one_hot"], dtype=np.float16))
    
    # Save dataset metadata
    with open(os.path.join(output_folder, 'dataset_meta.pkl'), 'wb') as f:
        pickle.dump({
            'precision': 'float16',
            'version': '1.0',
            'files': len(files)
        }, f)
    
    print(f"Conversion complete! Half-precision data saved to {output_folder}")

if __name__ == "__main__":
    convert_to_memmap_fp16(
        input_folder="dataset/genea2023_dataset/trn/main-agent/features",
        output_folder="dataset/genea2023_dataset/trn/main-agent/optimized_features_fp16"
    )