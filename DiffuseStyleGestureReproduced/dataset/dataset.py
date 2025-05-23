import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import pickle

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

        # print("Mean pose is ", self.mean_pose)
        # print("Std pose is ", self.std_pose)
        
        # Generate valid starting points
        self.valid_starts = self._find_valid_starting_points()
        print(f"Found {len(self.valid_starts)} valid starting points for windows")
        
        # Pre-generate batch indices (frame start points) for the epoch
        self._reshuffle()
    
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
    
    def _reshuffle(self):
        """Generate new batch indices for the epoch, optimized for speed"""
        # Generate all indices at once: shape = (epoch_length * batch_size,)
        all_indices = np.random.choice(self.valid_starts, self.epoch_length * self.batch_size, replace=True)
        
        # Reshape into (epoch_length, batch_size)
        self.batch_starts = all_indices.reshape(self.epoch_length, self.batch_size)

    
    def reshuffle(self):
        """Public method to reshuffle data between epochs"""
        self._reshuffle()
        print("Reshuffled dataset with new random windows")
    
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
    def __init__(self, consolidated_file, seq_length=150, seed_length=8, 
                 batch_size=32, epoch_length=1000, return_audio_frame_index=False,
                 device=torch.device('cuda'), use_world_pos_gesture_features=False):
        self.seq_length = seq_length
        self.seed_length = seed_length
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.chunk_size = seq_length + seed_length
        self.return_audio_frame_index = return_audio_frame_index
        self.device = device
        self.use_world_pos_gesture_features = use_world_pos_gesture_features
        
        print(f"Initializing GPU-resident dataset on {device}")
        
        # Load metadata
        meta_file = consolidated_file.replace('.npz', '_meta.pkl')
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
            self.skeleton = self.metadata['skeleton']

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
        self._reshuffle()
        
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
        """Pre-create offset tensors for vectorized indexing"""
        # Seed sequence offsets: shape [1, seed_length]
        self.seed_offsets = torch.arange(0, self.seed_length, device=self.device).view(1, -1)
        
        # Gesture sequence offsets: shape [1, seq_length]
        self.gesture_offsets = torch.arange(0, self.seq_length, device=self.device).view(1, -1)
    
    def _find_valid_starting_points(self):
        """Find all valid window starting points directly on GPU"""
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
    
    def _reshuffle(self):
        """Generate new batch indices for the epoch, optimized for GPU"""
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
    
    def reshuffle(self):
        """Public method to reshuffle data between epochs"""
        self._reshuffle()
        print("Reshuffled dataset with new random windows")
    
    def __len__(self):
        return self.epoch_length
    
    def __getitem__(self, idx):
        """Fully vectorized extraction of windows from consolidated data"""
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
        
        # TODO: I think this should not be here, but instead be handled by the user after the data is retrieved
        if not self.use_world_pos_gesture_features:
            # Apply z-score normalization
            gesture_batch = (gesture_batch - self.mean_pose) / self.std_pose
            seed_batch = (seed_batch - self.mean_pose) / self.std_pose
        
        if self.return_audio_frame_index:
            # Return the start frame index for the audio snippet
            audio_frame_indices = start_points + self.seed_length
            return gesture_batch, seed_batch, audio_batch, speaker_batch, audio_frame_indices
        else:
            return gesture_batch, seed_batch, audio_batch, speaker_batch



class SingleSampleAnimationDataset(Dataset):
    def __init__(self, folder, seq_length_in_frames=150, seed_length_in_frames=8, 
                 batch_size=1, epoch_length=1):
        """
        A dataset that returns a single sample with a random starting frame and a random file.
        Useful for debugging. Returns data in the same format as ConsolidatedRAMDataset.
        
        Args:
            folder (str): Path to folder containing .npz files.
            seq_length_in_frames (int): Length of the gesture sequence in frames.
            seed_length_in_frames (int): Length of the seed sequence in frames.
        """
        self.folder = folder
        self.seq_length_in_frames = seq_length_in_frames
        self.seed_length_in_frames = seed_length_in_frames
        self.chunk_size = seq_length_in_frames + seed_length_in_frames
        self.epoch_length = epoch_length
        self.batch_size = batch_size
        
        # List all npz files
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        if not self.files:
            raise ValueError("No .npz files found in the provided folder.")
        
        # Load statistics for normalization. These are saved in a file called stats.npz one folder up from the dataset folder
        stats_file = os.path.join(os.path.dirname(folder), "statistics.npz")
        if os.path.exists(stats_file):
            with np.load(stats_file) as npz:
                print("Loaded statistics for normalization")
                self.mean_pose = npz["mean_pose"]
                self.std_pose = npz["std_pose"]
        
        self.mean_pose = torch.tensor(self.mean_pose, dtype=torch.float16)
        self.std_pose = torch.tensor(self.std_pose, dtype=torch.float16)

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # Randomly select a file
        file = random.choice(self.files)
        
        with np.load(file) as npz:
            total_frames = npz["bvh_features"].shape[0]
            if total_frames < self.chunk_size:
                raise ValueError(f"File {file} does not have enough frames.")
            
            # Choose a random start frame such that a chunk of self.chunk_size fits in the file
            start_frame = random.randint(0, total_frames - self.chunk_size)
            
            gesture_chunk = npz["bvh_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            seed_chunk = npz["bvh_features"][start_frame : start_frame + self.seed_length_in_frames]
            audio_chunk = npz["audio_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            speaker = npz["main_agent_id_one_hot"]
            full_audio_features = npz["audio_features"]
        
        # Convert to half precision for consistency with ConsolidatedRAMDataset
        gesture = torch.tensor(gesture_chunk, dtype=torch.float16)
        seed = torch.tensor(seed_chunk, dtype=torch.float16)
        audio = torch.tensor(audio_chunk, dtype=torch.float16)
        speaker = torch.tensor(speaker, dtype=torch.float16)

        # Full audio features
        full_audio_features_tensor = torch.tensor(full_audio_features, dtype=torch.float16)
        
        # Apply normalization to gestures
        gesture = (gesture - self.mean_pose) / self.std_pose
        seed = (seed - self.mean_pose) / self.std_pose

        gesture_batch = gesture.repeat(self.batch_size, 1, 1)
        seed_batch = seed.repeat(self.batch_size, 1, 1)
        audio_batch = audio.repeat(self.batch_size, 1, 1)
        speaker_batch = speaker.repeat(self.batch_size, 1)

        ##################################################################################
        # WAV FILE EXTRACTION
        ##################################################################################
        
        # Find the corresponding wav file in the 'wav' folder and extract the corresponding time
        # print(file)
        # wav_file = os.path.join(os.path.dirname(os.path.dirname(file)), 'wav', os.path.basename(file).replace('.npz', '_main-agent.wav'))
        # if not os.path.exists(wav_file):
        #     raise ValueError(f"Corresponding wav file {wav_file} not found.")

        # # Read the wav file
        # audio_data, samplerate = sf.read(wav_file)

        # # Calculate the start and end times in seconds
        # start_time = start_frame / 30.0
        # end_time = (start_frame + self.chunk_size) / 30.0

        # # Extract the corresponding audio snippet
        # start_sample = int(start_time * samplerate)
        # end_sample = int(end_time * samplerate)
        # audio_snippet = audio_data[start_sample:end_sample]

        # # Save the audio snippet one folder up from the dataset folder
        # output_wav_file = os.path.join(os.path.dirname(self.folder), 'audio_snippet.wav')
        # sf.write(output_wav_file, audio_snippet, samplerate)

        ##################################################################################
        # WAV FILE EXTRACTION
        ##################################################################################
        
        return gesture_batch, seed_batch, audio_batch, speaker_batch, full_audio_features_tensor, start_frame, file

class FixedSampleAnimationDataset(Dataset):
    def __init__(self, folder, seq_length_in_frames=150, seed_length_in_frames=8, 
                 batch_size=1, epoch_length=10000, start_frame=0):
        """
        A dataset that always returns the first snippet of the first file.
        Useful for debugging. Returns data in the same format as ConsolidatedRAMDataset.
        
        Args:
            folder (str): Path to folder containing .npz files.
            seq_length_in_frames (int): Length of the gesture sequence in frames.
            seed_length_in_frames (int): Length of the seed sequence in frames.
            batch_size (int): Size of the batch to return.
            epoch_length (int): Number of batches in an epoch.
        """
        self.folder = folder
        self.seq_length_in_frames = seq_length_in_frames
        self.seed_length_in_frames = seed_length_in_frames
        self.chunk_size = seq_length_in_frames + seed_length_in_frames
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        
        # List all npz files and select the first one
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        if not self.files:
            raise ValueError("No .npz files found in the provided folder.")
        
        self.file = self.files[0]  # Always use the first file
        
        with np.load(self.file) as npz:
            gesture_chunk = npz["bvh_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            seed_chunk = npz["bvh_features"][start_frame : start_frame + self.seed_length_in_frames]
            audio_chunk = npz["audio_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            speaker = npz["main_agent_id_one_hot"]
        
        # Load statistics for normalization. THese are saved in a file called stats.npz one folder up from the dataset folder
        stats_file = os.path.join(os.path.dirname(folder), "statistics.npz")
        if os.path.exists(stats_file):
            with np.load(stats_file) as npz:
                print("Loaded statistics for normalization")
                self.mean_pose = npz["mean_pose"]
                self.std_pose = npz["std_pose"]

        # Convert to half precision for consistency with ConsolidatedRAMDataset
        self.gesture = torch.tensor(gesture_chunk, dtype=torch.float16)
        self.seed = torch.tensor(seed_chunk, dtype=torch.float16)
        self.audio = torch.tensor(audio_chunk, dtype=torch.float16)
        self.speaker = torch.tensor(speaker, dtype=torch.float16)
        self.mean_pose = torch.tensor(self.mean_pose, dtype=torch.float16)
        self.std_pose = torch.tensor(self.std_pose, dtype=torch.float16)

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        """Return the same data for all indices in tuple format"""
        # Create batch by repeating the same data
        gesture_batch = self.gesture.repeat(self.batch_size, 1, 1)
        seed_batch = self.seed.repeat(self.batch_size, 1, 1)
        audio_batch = self.audio.repeat(self.batch_size, 1, 1)
        speaker_batch = self.speaker.repeat(self.batch_size, 1)

        # Apply normalization to gestures
        gesture_batch = (gesture_batch - self.mean_pose) / self.std_pose
        seed_batch = (seed_batch - self.mean_pose) / self.std_pose
        
        return gesture_batch, seed_batch, audio_batch, speaker_batch