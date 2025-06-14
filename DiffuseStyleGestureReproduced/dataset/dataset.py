import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from utils.animation.skeleton import Skeleton

class RAMDataset(Dataset):
    """Dataset that loads consolidated data into RAM and extracts windows on-the-fly"""
    def __init__(self, consolidated_file, seq_length=150, seed_length=8, 
                 batch_size=32, epoch_length=1000, return_audio_frame_index=False):
        self.seq_length = seq_length
        self.seed_length = seed_length
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.chunk_size = seq_length + seed_length
        self.return_audio_frame_index = return_audio_frame_index
        
        # Load metadata
        meta_file = consolidated_file.replace('.npz', '_meta.pkl')
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
            self.skeleton = self.metadata['skeleton']
            self.mean_pose = torch.from_numpy(self.metadata['mean_pose']).half()
            self.std_pose = torch.from_numpy(self.metadata['std_pose']).half()
        
        # Load the entire dataset into RAM
        data = np.load(consolidated_file)
        
        # Convert to torch tensors for faster access
        self.gestures = torch.from_numpy(data['gestures']).half()
        self.audio = torch.from_numpy(data['audio']).half()
        self.speakers = torch.from_numpy(data['speakers']).half()
        
        # Close numpy file to free file handles
        data.close()
        
        # Generate valid starting points
        self.valid_starts = self._find_valid_starting_points()
        print(f"Found {len(self.valid_starts)} valid starting points for windows")
        
        # Pre-generate batch indices (frame start points) for the epoch
        self.reshuffle()
    
    def _find_valid_starting_points(self):
        """Find all valid window starting points (excluding boundaries between files)"""
        valid_points = []
        total_frames = self.metadata['total_frames']
        
        # Process each file segment
        for segment in self.metadata['file_segments']:
            # Only consider segments long enough for a window
            if segment['frames'] >= self.chunk_size:
                # Valid start points are from segment start to (end - chunk_size)
                for start in range(segment['start_idx'], segment['end_idx'] - self.chunk_size + 1):
                    valid_points.append(start)
        
        return valid_points
    
    def reshuffle(self):
        # Generate all indices at once: shape = (epoch_length * batch_size,)
        all_indices = np.random.choice(self.valid_starts, self.epoch_length * self.batch_size, replace=True)
        
        # Reshape into (epoch_length, batch_size)
        self.batch_starts = all_indices.reshape(self.epoch_length, self.batch_size)
    
    def __len__(self):
        return self.epoch_length
    
    def __getitem__(self, idx):
        """Extract windows on-the-fly from the consolidated data"""
        start_points = self.batch_starts[idx]
        
        # Pre-allocate tensors for this batch
        gesture_batch = torch.zeros(
            (self.batch_size, self.seq_length, self.metadata['gesture_dim']), 
            dtype=torch.float16
        )
        seed_batch = torch.zeros(
            (self.batch_size, self.seed_length, self.metadata['gesture_dim']), 
            dtype=torch.float16
        )
        audio_batch = torch.zeros(
            (self.batch_size, self.seq_length, self.metadata['audio_dim']), 
            dtype=torch.float16
        )
        speaker_batch = torch.zeros(
            (self.batch_size, self.metadata['speaker_shape'][0]), 
            dtype=torch.float16
        )
        
        # For each item in the batch
        for i, start_frame in enumerate(start_points):
            # Find which file segment this frame belongs to
            segment_idx = next(
                (idx for idx, seg in enumerate(self.metadata['file_segments']) 
                 if seg['start_idx'] <= start_frame < seg['end_idx']), 
                None
            )
            
            if segment_idx is not None:
                # Extract windows directly from tensors
                seed_batch[i] = self.gestures[start_frame:start_frame + self.seed_length]
                gesture_batch[i] = self.gestures[start_frame + self.seed_length:start_frame + self.chunk_size]
                audio_batch[i] = self.audio[start_frame + self.seed_length:start_frame + self.chunk_size]
                speaker_batch[i] = self.speakers[segment_idx]

        # z-score normalization
        gesture_batch = (gesture_batch - self.mean_pose) / self.std_pose
        seed_batch = (seed_batch - self.mean_pose) / self.std_pose

        if self.return_audio_frame_index:
            # Return the start frame index for the audio snippet
            audio_frame_indices = start_points + self.seed_length
            return gesture_batch, seed_batch, audio_batch, speaker_batch, audio_frame_indices
        else:
            return gesture_batch, seed_batch, audio_batch, speaker_batch

class GPUDataset(Dataset):
    """Dataset that keeps data on GPU for maximum performance with vectorized operations"""
    def __init__(self, consolidated_file, seq_length=150, seed_length=0, 
                 batch_size=32, epoch_length=1000, return_audio_frame_index=False,
                 device=torch.device('cuda'), use_world_pos_gesture_features=False, loading_encoded_data=False):
        self.seq_length = seq_length
        self.seed_length = seed_length
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.chunk_size = seq_length + seed_length
        self.return_audio_frame_index = return_audio_frame_index
        self.device = device
        self.use_world_pos_gesture_features = use_world_pos_gesture_features
        self.loading_encoded_data = loading_encoded_data
        
        print(f"Initializing GPU-resident dataset on {device}")
        
        # Load metadata
        meta_file = consolidated_file.replace('.npz', '_meta.pkl')
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
            self.skeleton: Skeleton = self.metadata['skeleton']

            self.skeleton.set_device(device)
            
            # Move normalization stats directly to GPU
            self.mean_pose = torch.from_numpy(self.metadata['mean_pose']).half().to(device)
            self.std_pose = torch.from_numpy(self.metadata['std_pose']).half().to(device)
        
        print(f"Loading data from {consolidated_file} directly to GPU...")
        # Load the entire dataset into GPU memory
        data = np.load(consolidated_file)
        
        # Convert to torch tensors and move directly to GPU
        if use_world_pos_gesture_features:
            self.gestures = torch.from_numpy(data['world_pos_gestures']).half().to(device)
        else:
            self.gestures = torch.from_numpy(data['gestures']).half().to(device)

        self.audio = torch.from_numpy(data['audio']).half().to(device)
        self.speakers = torch.from_numpy(data['speakers']).half().to(device)
        
        # Close numpy file to free file handles
        data.close()
        
        print(f"Data loaded to GPU. Gesture shape: {self.gestures.shape}, Audio shape: {self.audio.shape}")
        
        # Create segment map
        self._create_segment_mapping()
        
        # Generate valid starting points
        self.valid_starts = self._find_valid_starting_points()
        print(f"Found {len(self.valid_starts)} valid starting points for windows")
        
        # Pre-generate offset tensors
        self._create_offset_tensors()
        
        # Pre-generate batch indices (frame start points) for the epoch
        self.reshuffle()
        
        print("Dataset initialization complete!")
    
    def _create_segment_mapping(self):
        """Create a tensor mapping each frame index to its segment index"""
        total_frames = self.metadata['total_frames']
        
        # Create tensor of -1s (invalid segment) of size total_frames
        self.segment_map = torch.full((total_frames,), -1, 
                                     device=self.device, 
                                     dtype=torch.long)
        
        # Fill in segment indices
        for i, segment in enumerate(self.metadata['file_segments']):
            start_idx = segment['start_idx']
            end_idx = segment['end_idx']
            self.segment_map[start_idx:end_idx] = i
    
    def _create_offset_tensors(self):
        # Seed sequence offsets: shape [1, seed_length]
        self.seed_offsets = torch.arange(0, self.seed_length, device=self.device).view(1, -1)
        
        # Gesture sequence offsets: shape [1, seq_length]
        self.gesture_offsets = torch.arange(0, self.seq_length, device=self.device).view(1, -1)
    
    def _find_valid_starting_points(self):
        valid_points = []
        
        # Process each file segment
        for segment in self.metadata['file_segments']:
            # Only consider segments long enough for a window
            if segment['frames'] >= self.chunk_size:
                # Generate all valid start points for this segment in one go
                segment_starts = torch.arange(
                    segment['start_idx'], 
                    segment['end_idx'] - self.chunk_size + 1,
                    device=self.device
                )
                valid_points.append(segment_starts)
        
        # Concatenate all valid points
        return torch.cat(valid_points)
    
    def reshuffle(self):
        # Generate random indices
        indices = torch.randint(
            0, len(self.valid_starts), 
            (self.epoch_length * self.batch_size,),
            device=self.device
        )
        # Get the actual start frames
        all_starts = self.valid_starts[indices]
        # Reshape into [epoch_length, batch_size]
        self.batch_starts = all_starts.reshape(self.epoch_length, self.batch_size)
    
    def __len__(self):
        return self.epoch_length
    
    def __getitem__(self, idx):
        # Get batch start points: shape [batch_size]
        start_points = self.batch_starts[idx]
        
        # Get segment indices for speaker data
        segment_indices = self.segment_map[start_points]
        
        # Create indices for seed frames: shape [batch_size, seed_length]
        # For each batch item, calculate [start, start+1, ..., start+seed_length-1]
        seed_indices = start_points.view(-1, 1) + self.seed_offsets
        
        # Create indices for gesture frames: shape [batch_size, seq_length]
        # For each batch item, calculate [start+seed_length, start+seed_length+1, ..., start+chunk_size-1]
        gesture_indices = (start_points.view(-1, 1) + self.seed_length) + self.gesture_offsets
        
        # Extract all data at once with advanced indexing - each operation is fully vectorized
        seed_batch = self.gestures[seed_indices]  # Shape: [batch_size, seed_length, gesture_dim]
        gesture_batch = self.gestures[gesture_indices]  # Shape: [batch_size, seq_length, gesture_dim]
        audio_batch = self.audio[gesture_indices]  # Shape: [batch_size, seq_length, audio_dim]
        speaker_batch = self.speakers[segment_indices]  # Shape: [batch_size, speaker_dim]
        
        if not self.loading_encoded_data:
            # Apply z-score normalization
            if self.use_world_pos_gesture_features:
                gesture_batch = self.skeleton.normalize_world_positions(gesture_batch)
                seed_batch = self.skeleton.normalize_world_positions(seed_batch)
            else:
                gesture_batch = self.skeleton.normalize_poses(gesture_batch)
                seed_batch = self.skeleton.normalize_poses(seed_batch)
        
        if self.return_audio_frame_index:
            # Return the start frame index for the audio snippet
            audio_frame_indices = start_points + self.seed_length
            return gesture_batch, seed_batch, audio_batch, speaker_batch, audio_frame_indices
        else:
            return gesture_batch, seed_batch, audio_batch, speaker_batch