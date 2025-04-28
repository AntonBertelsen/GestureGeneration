import numpy as np
import os
import csv
import pickle
from tqdm import tqdm
from utils.bvh_processing.bvh_converter3 import OffsetBVHParser
from utils.audio_processing.extract_audio_features import extract_audio_features

class DataProcessor:
    def __init__(self, bvh_dir, wav_dir, metadata_file, output_dir):
        self.bvh_dir = bvh_dir
        self.wav_dir = wav_dir
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        self.features_dir = os.path.join(output_dir, "features")
        
        # Ensure the directory exists and is writable
        try:
            os.makedirs(self.features_dir, exist_ok=True)
        except PermissionError as e:
            raise PermissionError(f"Unable to create directory '{self.features_dir}'. Check permissions.") from e
        
        # Load metadata and files
        self.metadata = self._load_metadata()
        self.bvh_files = sorted([f for f in os.listdir(bvh_dir) if f.endswith('.bvh')])
        self.wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])

        self.target_joints = ['body_world', 'b_root', 'b_r_foot', 'b_l_foot', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 
                'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 
                'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 
                'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 
                'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 
                'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 
                'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 
                'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 
                'b_neck0', 'b_head']
        
        # Skeleton info cache
        self.skeleton_info = None
        
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
            
            # Extract BVH features using OffsetBVHParser
            bvh_path = os.path.join(self.bvh_dir, bvh_file)
            parser = OffsetBVHParser(bvh_path, target_joints=self.target_joints)
            bvh_features = parser.extract_channels()
            
            # Cache skeleton info from first file
            if self.skeleton_info is None:
                self.skeleton_info = {
                    'joint_names': parser.get_all_joints(),
                    'joint_channels': parser.joint_channels,
                    'joint_parents': parser.joint_parent,
                    'joint_offsets': parser.joint_offsets
                }
            
            # Extract audio features
            wav_path = os.path.join(self.wav_dir, wav_file)
            audio_features = extract_audio_features(wav_path).numpy()
            
            # Crop to minimum length
            min_length = min(bvh_features.shape[0], audio_features.shape[0])
            bvh_features = bvh_features[:min_length]
            audio_features = audio_features[:min_length]
            
            # Convert to float16 to save space 
            bvh_features = bvh_features.astype(np.float16)
            audio_features = audio_features.astype(np.float16)
            
            # print if gesture contains nan
            if np.isnan(bvh_features).any():
                print(f"Gesture contains NaN values: {prefix}")

            # Save features
            np.savez_compressed(
                os.path.join(self.features_dir, f"{prefix}.npz"),
                bvh_features=bvh_features,
                audio_features=audio_features,
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
            audio_dim = npz["audio_features"].shape[1]
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
        print(f"Audio dim: {audio_dim}")
        print(f"Speaker shape: {speaker_shape}")
        
        # Create consolidated arrays
        gestures = np.zeros((total_frames, gesture_dim), dtype=np.float16)
        audio = np.zeros((total_frames, audio_dim), dtype=np.float16)
        
        # Second pass: fill the arrays
        for segment in tqdm(file_segments, desc="Consolidating data"):
            file_path = os.path.join(self.features_dir, segment['file'])
            start = segment['start_idx']
            end = segment['end_idx']
            
            with np.load(file_path) as npz:
                frames = segment['frames']
                gestures[start:end] = npz["bvh_features"][:frames]
                audio[start:end] = npz["audio_features"][:frames]
        
        # Calculate statistics
        print("Computing statistics...")
        mean_pose = np.mean(gestures.astype(np.float64), axis=0)
        std_pose = np.std(gestures.astype(np.float64), axis=0)

        # replace any near-zero values in std with 1.0 to avoid division by zero
        # Using np.isclose() to handle floating point precision issues
        zero_mask = np.isclose(std_pose, 0.0, atol=1e-8)
        std_pose[zero_mask] = 1.0
        
        # Save consolidated data
        print(f"Saving consolidated data to {output_file}")
        np.savez_compressed(
            output_file,
            gestures=gestures,
            audio=audio,
            speakers=np.array(speaker_data),
            mean_pose=mean_pose,
            std_pose=std_pose
        )
        # Also save mean and std separately
        np.savez_compressed(
            os.path.join(self.output_dir, "statistics.npz"),
            mean_pose=mean_pose,
            std_pose=std_pose
        )
        
        # Save metadata separately
        meta_file = os.path.join(self.output_dir, "consolidated_meta.pkl")
        with open(meta_file, 'wb') as f:
            metadata = {
                'total_frames': total_frames,
                'gesture_dim': gesture_dim,
                'audio_dim': audio_dim,
                'speaker_shape': speaker_shape,
                'file_segments': file_segments,
                'skeleton_info': self.skeleton_info
            }
            pickle.dump(metadata, f)
        
        print(f"Consolidated data created successfully!")
        print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

# Main execution
if __name__ == "__main__":
    bvh_dir = 'dataset/genea2023_dataset/trn/main-agent/bvh'
    wav_dir = 'dataset/genea2023_dataset/trn/main-agent/wav'
    metadata_file = 'dataset/genea2023_dataset/trn/metadata.csv'
    output_dir = 'dataset/genea2023_dataset/trn/main-agent'
    
    processor = DataProcessor(bvh_dir, wav_dir, metadata_file, output_dir)
    processor.process_files()
    processor.create_consolidated_data()