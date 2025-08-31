import numpy as np
import os
import csv
import pickle
from tqdm import tqdm
from utils.animation.processing.bvh_converter import BVHParser
from utils.audio_processing.extract_audio_features import extract_audio_features
import yaml
import argparse

class DataProcessor:
    def __init__(self, bvh_dir, wav_dir, metadata_file, output_dir, skeleton_config_file, normalization_meta_path=None):
        self.bvh_dir = bvh_dir
        self.wav_dir = wav_dir
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        self.features_dir = os.path.join(output_dir, "features")
        self.normalization_meta_path = normalization_meta_path  # Path to training meta data for normalization
        self.normalization_data = None  # Will hold normalization parameters

        # Ensure the directory exists and is writable
        try:
            os.makedirs(self.features_dir, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Unable to create directory '{self.features_dir}'. Check permissions.") from e
        
        # Load metadata and files
        self.metadata = self._load_metadata()
        self.bvh_files = sorted([f for f in os.listdir(bvh_dir) if f.endswith('.bvh')])
        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])

        # Load skeleton configuration
        self.skeleton_config = self._load_skeleton_config(skeleton_config_file)

        self.target_joints = self.skeleton_config['target_joints']
        self.bone_categories = self.skeleton_config['categories']
        
        # Skeleton definition
        self.skeleton = None

        # Load normalization parameters if provided
        if self.normalization_meta_path and os.path.exists(self.normalization_meta_path):
            self._load_normalization_parameters()
    
    def _load_normalization_parameters(self):
        """Load normalization parameters from training metadata file"""
        print(f"Loading normalization parameters from {self.normalization_meta_path}")
        try:
            with open(self.normalization_meta_path, 'rb') as f:
                self.normalization_data = pickle.load(f)
            print("Successfully loaded normalization parameters")
        except Exception as e:
            print(f"Error loading normalization parameters: {e}")
            self.normalization_data = None

    def _load_skeleton_config(self, skeleton_config_file):
        """Load skeleton configuration from YAML file"""
        try:
            with open(skeleton_config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Error loading skeleton config: {e}")

    def _load_metadata(self):
        """Load metadata from CSV file"""
        metadata = {}
        with open(self.metadata_file, 'r') as f:
            reader = csv.DictReader(f)
            metadata = {row['prefix']: row for row in reader}
        
        # Calculate num_speakers
        self.num_speakers = 0
        for key in metadata:
            self.num_speakers = max(self.num_speakers, int(metadata[key]['main-agent_id']))
            self.num_speakers = max(self.num_speakers, int(metadata[key]['interloctr_id']))
        
        return metadata
    
    def process_files(self):
        """Process all BVH and WAV files to extract features"""
        print(f"Processing {len(self.bvh_files)} BVH+WAV file pairs...")
        
        # Process each file pair
        for bvh_file, wav_file in tqdm(list(zip(self.bvh_files, self.wav_files)), desc="Extracting features"):
            prefix = os.path.splitext(bvh_file)[0].removesuffix("_main-agent")
            
            # Get metadata for this file
            file_metadata = self.metadata.get(prefix, {})
            main_agent_id = file_metadata.get('main-agent_id', '0')
            main_agent_has_fingers = file_metadata.get('main-agent_has_fingers', '0')
            interloctr_id = file_metadata.get('interloctr_id', '0')
            interloctr_has_fingers = file_metadata.get('interloctr_has_fingers', '0')
            
            # One-hot encode speaker IDs
            main_agent_id_one_hot = np.zeros(self.num_speakers, dtype=np.float16)
            main_agent_id_one_hot[int(main_agent_id) - 1] = 1
            interloctr_id_one_hot = np.zeros(self.num_speakers, dtype=np.float16)
            interloctr_id_one_hot[int(interloctr_id) - 1] = 1
            
            # Extract BVH features using BVHParser
            bvh_path = os.path.join(self.bvh_dir, bvh_file)
            parser = BVHParser(bvh_path, target_joints=self.target_joints, bone_categories=self.bone_categories)
            bvh_features = parser.to_features()
            
            # Cache skeleton info from first file
            if self.skeleton is None:
                self.skeleton = parser.skeleton
            
            # Extract audio features
            wav_path = os.path.join(self.wav_dir, wav_file)
            # mel_spec, mfcc, rms_energy, pitch, energy_derivatives, pitch_derivatives, onsets, wavlm_features = extract_audio_features(wav_path)
            mel_spec, mfcc, rms_energy, pitch, energy_derivatives, pitch_derivatives, onsets = extract_audio_features(wav_path)

            # Crop to minimum length (all audio features are the same length)
            min_length = min(bvh_features.shape[0], mel_spec.shape[0])

            bvh_features = bvh_features[:min_length, :]
            
            mel_spec = mel_spec[:min_length, :]
            mfcc = mfcc[:min_length, :]
            rms_energy = rms_energy[:min_length, :]
            pitch = pitch[:min_length, :]
            energy_derivatives = energy_derivatives[:min_length, :]
            pitch_derivatives = pitch_derivatives[:min_length, :]
            onsets = onsets[:min_length, :]
            # wavlm_features = wavlm_features[:min_length, :]
            
            # Convert to float16 to save space 
            bvh_features = bvh_features.astype(np.float16)
            mel_spec = mel_spec.cpu().numpy().astype(np.float16)
            mfcc = mfcc.cpu().numpy().astype(np.float16)
            rms_energy = rms_energy.cpu().numpy().astype(np.float16)
            pitch = pitch.cpu().numpy().astype(np.float16)
            energy_derivatives = energy_derivatives.cpu().numpy().astype(np.float16)
            pitch_derivatives = pitch_derivatives.cpu().numpy().astype(np.float16)
            onsets = onsets.cpu().numpy().astype(np.float16)
            # wavlm_features = wavlm_features.cpu().numpy().astype(np.float16)

            # Save features
            np.savez_compressed(
                os.path.join(self.features_dir, f"{prefix}.npz"),
                bvh_features=bvh_features,
                mel_spec_features=mel_spec,
                mfcc_features=mfcc,
                rms_energy_features=rms_energy,
                pitch_features=pitch,
                energy_derivatives_features=energy_derivatives,
                pitch_derivatives_features=pitch_derivatives,
                onsets_features=onsets,
                # wavlm_features=wavlm_features,
                prefix=prefix,
                main_agent_id_one_hot=main_agent_id_one_hot,
                main_agent_has_fingers=main_agent_has_fingers,
                interloctr_id_one_hot=interloctr_id_one_hot,
                interloctr_has_fingers=interloctr_has_fingers
            )
    
    def create_consolidated_data(self):
        """Create a single contiguous data file from all features"""
        print("Creating consolidated data file...")
        output_file = os.path.join(self.output_dir, "consolidated.npz")
        
        # Find all NPZ files
        feature_files = sorted([os.path.join(self.features_dir, f) 
                               for f in os.listdir(self.features_dir) 
                               if f.endswith('.npz')])
        
        if not feature_files:
            print("No feature files found!")
            return
            
        # Process first file to get dimensions
        with np.load(feature_files[0]) as npz:
            gesture_dim = npz["bvh_features"].shape[1]
            mel_spec_dim = npz["mel_spec_features"].shape[1]
            mfcc_dim = npz["mfcc_features"].shape[1]
            rms_energy_dim = npz["rms_energy_features"].shape[1]
            pitch_dim = npz["pitch_features"].shape[1]
            energy_derivatives_dim = npz["energy_derivatives_features"].shape[1]
            pitch_derivatives_dim = npz["pitch_derivatives_features"].shape[1]
            onsets_dim = npz["onsets_features"].shape[1]
            # wavlm_features_dim = npz["wavlm_features"].shape[1]
            speaker_shape = npz["main_agent_id_one_hot"].shape
        
        # First pass: calculate total frames and collect metadata
        total_frames = 0
        file_segments = []
        speaker_data = []
        
        for file_path in tqdm(feature_files, desc="Analyzing files"):
            with np.load(file_path) as npz:
                frames = len(npz["bvh_features"])
                
                if frames > 0:
                    file_segments.append({
                        'file': os.path.basename(file_path),
                        'start_idx': total_frames,
                        'end_idx': total_frames + frames,
                        'frames': frames
                    })
                    
                    speaker_data.append(np.array(npz["main_agent_id_one_hot"]))
                    total_frames += frames
        
        print(f"Total frames: {total_frames}")
        print(f"Gesture dim: {gesture_dim}")
        print(f"mel_spec dim: {mel_spec_dim}")
        print(f"mfcc dim: {mfcc_dim}")
        print(f"rms_energy dim: {rms_energy_dim}")
        print(f"pitch dim: {pitch_dim}")
        print(f"energy_derivatives dim: {energy_derivatives_dim}")
        print(f"pitch_derivatives dim: {pitch_derivatives_dim}")
        print(f"onsets dim: {onsets_dim}")
        # print(f"wavlm_features dim: {wavlm_features_dim}")
        print(f"Speaker shape: {speaker_shape}")
        
        # Create consolidated arrays
        gestures = np.zeros((total_frames, gesture_dim), dtype=np.float16)
        mel_spec = np.zeros((total_frames, mel_spec_dim), dtype=np.float16)
        mfcc = np.zeros((total_frames, mfcc_dim), dtype=np.float16)
        rms_energy = np.zeros((total_frames, rms_energy_dim), dtype=np.float16)
        pitch = np.zeros((total_frames, pitch_dim), dtype=np.float16)
        energy_derivatives = np.zeros((total_frames, energy_derivatives_dim), dtype=np.float16)
        pitch_derivatives = np.zeros((total_frames, pitch_derivatives_dim), dtype=np.float16)
        onsets = np.zeros((total_frames, onsets_dim), dtype=np.float16)
        # wavlm_features = np.zeros((total_frames, wavlm_features_dim), dtype=np.float16)

        # Second pass: fill the arrays
        for segment in tqdm(file_segments, desc="Consolidating data"):
            file_path = os.path.join(self.features_dir, segment['file'])
            start = segment['start_idx']
            end = segment['end_idx']
            
            with np.load(file_path) as npz:
                frames = segment['frames']
                gestures[start:end] = npz["bvh_features"][:frames]
                mel_spec[start:end] = npz["mel_spec_features"][:frames]
                mfcc[start:end] = npz["mfcc_features"][:frames]
                rms_energy[start:end] = npz["rms_energy_features"][:frames]
                pitch[start:end] = npz["pitch_features"][:frames]
                energy_derivatives[start:end] = npz["energy_derivatives_features"][:frames]
                pitch_derivatives[start:end] = npz["pitch_derivatives_features"][:frames]
                onsets[start:end] = npz["onsets_features"][:frames]
                # wavlm_features[start:end] = npz["wavlm_features"][:frames]

        print("Normalizing data...")

        # Apply log compression to mel_spec
        mel_spec = np.log1p(mel_spec)

        # We need to calculate world positions for the skeleton to be able to calculate the mean and std, which is used for FGD.
        world_pos_gestures = self.skeleton.calculate_world_positions(gestures).numpy()

        # If normalization parameters are provided, use them
        if self.normalization_data:
            print("Using pre-calculated normalization parameters from training data")
            mean_pose = self.normalization_data['mean_pose']
            std_pose = self.normalization_data['std_pose']
            world_pos_mean_pose = self.normalization_data.get('world_pos_mean_pose', None)
            world_pos_std_pose = self.normalization_data.get('world_pos_std_pose', None)
            mel_spec_min = self.normalization_data['mel_spec_min']
            mel_spec_max = self.normalization_data['mel_spec_max']
            mel_spec_range = self.normalization_data['mel_spec_range']
            mfcc_mean = self.normalization_data['mfcc_mean']
            mfcc_std = self.normalization_data['mfcc_std']
            rms_energy_mean = self.normalization_data['rms_energy_mean']
            rms_energy_std = self.normalization_data['rms_energy_std']
            pitch_mean = self.normalization_data['pitch_mean']
            pitch_std = self.normalization_data['pitch_std']
            energy_derivatives_mean = self.normalization_data['energy_derivatives_mean']
            energy_derivatives_std = self.normalization_data['energy_derivatives_std']
            pitch_derivatives_mean = self.normalization_data['pitch_derivatives_mean']
            pitch_derivatives_std = self.normalization_data['pitch_derivatives_std']
        else:
            print("Calculating mean and std for gestures")
            gestures_f64 = gestures.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            mean_pose = np.mean(gestures_f64, axis=0)
            std_pose = np.sqrt(np.mean((gestures_f64 - mean_pose)**2, axis=0, dtype=np.float64)) # This is supposed to be more numerically stable than std = np.std(gestures_f64, axis=0)
            std_pose[std_pose == 0] = 1.0

        
            world_pos_gestures_f64 = world_pos_gestures.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            world_pos_mean_pose = np.mean(world_pos_gestures_f64, axis=0)
            world_pos_std_pose = np.sqrt(np.mean((world_pos_gestures_f64 - world_pos_mean_pose)**2, axis=0, dtype=np.float64)) # This is supposed to be more numerically stable than std = np.std(gestures_f64, axis=0)
            world_pos_std_pose[world_pos_std_pose == 0] = 1.0

            # mel_spec uses log compression and then min-max normalization
            print("normalizing mel_spec")
            mel_spec_min = np.min(mel_spec, axis=0)
            mel_spec_max = np.max(mel_spec, axis=0)
            
            mel_spec_range = mel_spec_max - mel_spec_min
            mel_spec_range[mel_spec_range == 0] = 1.0

            print("normalizing mfcc")
            mfcc_f64 = mfcc.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            mfcc_mean = np.mean(mfcc_f64, axis=0)
            mfcc_std = np.std(mfcc_f64, axis=0)
            mfcc_std[mfcc_std == 0] = 1.0

            print("normalizing rms_energy")
            rms_energy_f64 = rms_energy.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            rms_energy_mean = np.mean(rms_energy_f64, axis=0)
            rms_energy_std = np.std(rms_energy_f64, axis=0)
            rms_energy_std[rms_energy_std == 0] = 1.0
        
            print("normalizing pitch")
            pitch_f64 = pitch.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            pitch_mean = np.mean(pitch_f64, axis=0)
            pitch_std = np.std(pitch_f64, axis=0)
            pitch_std[pitch_std == 0] = 1.0
        
            print("normalizing energy_derivatives")
            energy_derivatives_f64 = energy_derivatives.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            energy_derivatives_mean = np.mean(energy_derivatives_f64, axis=0)
            energy_derivatives_std = np.std(energy_derivatives_f64, axis=0)
            energy_derivatives_std[energy_derivatives_std == 0] = 1.0

            pitch_derivatives_f64 = pitch_derivatives.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
            pitch_derivatives_mean = np.mean(pitch_derivatives_f64, axis=0)
            pitch_derivatives_std = np.std(pitch_derivatives_f64, axis=0)
            pitch_derivatives_std[pitch_derivatives_std == 0] = 1.0

        self.skeleton.set_mean_std(mean_pose,std_pose, world_pos_mean_pose, world_pos_std_pose)

        mel_spec = (mel_spec - mel_spec_min) / mel_spec_range
        mel_spec = mel_spec * 3.0  # Scale to occupy similar space as other features

        # mfcc uses z-score normalization
        mfcc = ((mfcc - mfcc_mean) / mfcc_std).astype(np.float16) # Convert back to float16 for efficient storage / transfer

        # rms_energy uses z-score normalization
        rms_energy = ((rms_energy - rms_energy_mean) / rms_energy_std).astype(np.float16) # Convert back to float16 for efficient storage / transfer

        # pitch uses z-score normalization
        pitch = ((pitch - pitch_mean) / pitch_std).astype(np.float16) # Convert back to float16 for efficient storage / transfer

        # energy_derivatives uses z-score normalization
        energy_derivatives = ((energy_derivatives - energy_derivatives_mean) / energy_derivatives_std).astype(np.float16) # Convert back to float16 for efficient storage / transfer

        # pitch_derivatives uses z-score normalization
        print("normalizing pitch_derivatives")
        pitch_derivatives = ((pitch_derivatives - pitch_derivatives_mean) / pitch_derivatives_std).astype(np.float16) # Convert back to float16 for efficient storage / transfer

        # Onsets have no normalization applied, they are in 0,1 range

        # wavlm_features uses z-score normalization per dimension
        # print("normalizing wavlm_features")
        # wavlm_features_f64 = wavlm_features.astype(np.float64) # Convert to float64 to avoid overflow issues when calculating mean and std which happens on large datasets
        # wavlm_features_mean = np.mean(wavlm_features_f64, axis=0)
        # wavlm_features_std = np.std(wavlm_features_f64, axis=0)
        # wavlm_features_std[wavlm_features_std == 0] = 1.0
        # wavlm_features = ((wavlm_features - wavlm_features_mean) / wavlm_features_std).astype(np.float16) # Convert back to float16 for efficient storage / transfer


        # Concatenate all the non-wavlm features into a single tensor for convenience
        print("concatenating audio features")
        audio_features = np.concatenate([mel_spec, mfcc, rms_energy, pitch, energy_derivatives, pitch_derivatives, onsets], axis=1)

        # replace any near-zero values in std with 1.0 to avoid division by zero
        # Using np.isclose() to handle floating point precision issues
        zero_mask = np.isclose(std_pose, 0.0, atol=1e-8)
        std_pose[zero_mask] = 1.0
        
        # Save consolidated data
        print(f"Saving consolidated data to {output_file}")
        np.savez_compressed(
            output_file,
            gestures=gestures,
            world_pos_gestures=world_pos_gestures,
            audio=audio_features,
            speakers=np.array(speaker_data)
        )

        # Save metadata separately
        meta_file = os.path.join(self.output_dir, "consolidated_meta.pkl")
        with open(meta_file, 'wb') as f:
            metadata = {
                'total_frames': total_frames,
                'gesture_dim': gesture_dim,
                'mel_spec_dim': mel_spec_dim,
                'mfcc_dim': mfcc_dim,
                'rms_energy_dim': rms_energy_dim,
                'pitch_dim': pitch_dim,
                'energy_derivatives_dim': energy_derivatives_dim,
                'pitch_derivatives_dim': pitch_derivatives_dim,
                'onsets_dim': onsets_dim,
                # 'wavlm_features_dim': wavlm_features_dim,
                'audio_dim': audio_features.shape[1],
                'speaker_shape': speaker_shape,
                'file_segments': file_segments,
                'skeleton': self.skeleton,
                'mean_pose': mean_pose,
                'std_pose': std_pose,
                'mel_spec_min': mel_spec_min,
                'mel_spec_max': mel_spec_max,
                'mel_spec_range': mel_spec_range,
                'mfcc_mean': mfcc_mean,
                'mfcc_std': mfcc_std,
                'rms_energy_mean': rms_energy_mean,
                'rms_energy_std': rms_energy_std,
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'energy_derivatives_mean': energy_derivatives_mean,
                'energy_derivatives_std': energy_derivatives_std,
                'pitch_derivatives_mean': pitch_derivatives_mean,
                'pitch_derivatives_std': pitch_derivatives_std
                # 'wavlm_features_mean': wavlm_features_mean,
                # 'wavlm_features_std': wavlm_features_std
            }
            pickle.dump(metadata, f)
        
        print(f"Consolidated data created successfully!")
        print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

# Main execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process and consolidate dataset.")
    parser.add_argument("--dataset_type", choices=["trn", "val", "toy", "toy_val"], default="trn", help="Specify the dataset type to use: 'trn', 'toy', or 'test'. Default is 'trn'.")
    args = parser.parse_args()

    dataset_type = args.dataset_type
    base_dir = f'dataset/genea2023_dataset/{dataset_type}/main-agent'
    bvh_dir = os.path.join(base_dir, 'bvh')
    wav_dir = os.path.join(base_dir, 'wav')
    metadata_file = f'dataset/genea2023_dataset/{dataset_type}/metadata.csv'
    output_dir = base_dir
    skeleton_config_file = f'dataset/genea2023_dataset/skeleton_config.yaml'
    
     # Only use training normalization for non-training datasets
    normalization_meta_path = None
    if dataset_type != "trn":
        normalization_meta_path = 'dataset/genea2023_dataset/trn/main-agent/consolidated_meta.pkl'

    processor = DataProcessor(bvh_dir, wav_dir, metadata_file, output_dir, skeleton_config_file, normalization_meta_path)
    processor.process_files()
    processor.create_consolidated_data()