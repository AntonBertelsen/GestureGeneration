import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class AnimationDataset(Dataset):
    def __init__(self, folder, seq_length=5, fps=30):
        """
        Args:
            folder (str): Path to folder containing .npz files.
            seq_length (int): Duration (in seconds) of the clip to load.
            fps (int): Frames per second in the animation.
        """
        self.folder = folder
        self.seq_length = seq_length
        self.fps = fps
        self.chunk_size = seq_length * fps  # number of frames per clip
        
        # List all npz files
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        
        # For each file, read only the shape from the 'bvh_features' array
        # (Assumes every file has a 'bvh_features' array)
        self.file_chunk_counts = []
        self.cum_counts = []
        total = 0
        for file in self.files:
            with np.load(file) as npz:
                total_frames = npz["bvh_features"].shape[0]
            # Calculate how many nonoverlapping chunks fit in the file.
            # If a file is too short, count is 0.
            count = max(0, (total_frames - self.chunk_size) // self.chunk_size + 1)
            self.file_chunk_counts.append(count)
            total += count
            self.cum_counts.append(total)
        self.total_chunks = total

        # Build a list of global indices and shuffle them for random sampling
        self.indices = list(range(self.total_chunks))
        random.shuffle(self.indices)

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        # Get the global chunk index from the shuffled list
        global_idx = self.indices[idx]

        # Find which file this global index falls into using cumulative counts
        file_idx = np.searchsorted(self.cum_counts, global_idx, side='right')
        # Compute the index within the chosen file
        start_in_file = global_idx - (self.cum_counts[file_idx - 1] if file_idx > 0 else 0)
        # Compute the starting frame of the chunk
        start_frame = start_in_file * self.chunk_size

        file = self.files[file_idx]
        with np.load(file) as npz:
            # Slice out the 5-second chunk from both bvh_features and audio_features
            bvh_chunk = npz["bvh_features"][start_frame : start_frame + self.chunk_size]
            audio_chunk = npz["audio_features"][start_frame : start_frame + self.chunk_size]

        # Convert to torch tensors (adjust dtype as needed)
        sample = {
            "bvh": torch.tensor(bvh_chunk, dtype=torch.float32),
            "audio": torch.tensor(audio_chunk, dtype=torch.float32)
        }
        return sample

    def on_epoch_end(self):
        """Call this method at the end of every epoch to reshuffle the data."""
        random.shuffle(self.indices)


class OverlapAnimationDataset(Dataset):
    def __init__(self, folder, seq_length_in_frames=150, seed_length_in_frames=10, epoch_length=10000):
        self.folder = folder
        self.seq_length_in_frames = seq_length_in_frames
        self.seed_length_in_frames = seed_length_in_frames
        self.chunk_size = seq_length_in_frames + seed_length_in_frames
        
        # List all npz files
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        
        # For each file, compute total frames (store this info)
        self.frames_per_file = {}
        for file in self.files:
            with np.load(file, allow_pickle = True) as npz:
                total_frames = npz["bvh_features"].shape[0]
            # Only consider files that are long enough
            if total_frames >= self.chunk_size:
                self.frames_per_file[file] = total_frames

        self.valid_files = list(self.frames_per_file.keys())
        self.epoch_length = epoch_length  # Fixed number of samples per epoch

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # Randomly select a file (each sample is independent)
        file = random.choice(self.valid_files)
        total_frames = self.frames_per_file[file]
        # Choose a random start such that a chunk of self.chunk_size fits in the file.
        start_frame = random.randint(0, total_frames - self.chunk_size)
        
        with np.load(file, mmap_mode='r') as npz:
            gesture_chunk = npz["bvh_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            seed_chunk = npz["bvh_features"][start_frame : start_frame + self.seed_length_in_frames]
            audio_chunk = npz["audio_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            speaker = npz["main_agent_id_one_hot"]

        sample = {
            "gesture": torch.tensor(gesture_chunk, dtype=torch.float32),
            "seed": torch.tensor(seed_chunk, dtype=torch.float32),
            "audio": torch.tensor(audio_chunk, dtype=torch.float32),
            "speaker": speaker
        }
        return sample
    
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

import pickle
import soundfile as sf
class ConsolidatedRAMDataset(Dataset):
    """Dataset that loads consolidated data into RAM and extracts windows on-the-fly"""
    def __init__(self, consolidated_file, seq_length=150, seed_length=8, 
                 batch_size=32, epoch_length=1000):
        self.seq_length = seq_length
        self.seed_length = seed_length
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.chunk_size = seq_length + seed_length
        
        # Load metadata
        meta_file = consolidated_file.replace('.npz', '_meta.pkl')
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
            self.skeleton_info = self.metadata['skeleton_info']
        
        # Load the entire dataset into RAM
        data = np.load(consolidated_file)
        
        # Convert to torch tensors for faster access
        self.gestures = torch.from_numpy(data['gestures']).half()
        self.audio = torch.from_numpy(data['audio']).half()
        self.speakers = torch.from_numpy(data['speakers']).half()
        self.mean_pose = torch.from_numpy(data['mean_pose']).half()
        self.std_pose = torch.from_numpy(data['std_pose']).half()
        
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
        """Generate new batch indices for the epoch"""
        self.batch_starts = []
        
        # For each batch
        for _ in range(self.epoch_length):
            # Random starting points for each item in the batch
            batch_indices = np.random.choice(self.valid_starts, self.batch_size, replace=True)
            self.batch_starts.append(batch_indices)
    
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
        gesture_batch = (gesture_batch - self.mean_pose) / self.std_pose / 10.0
        seed_batch = (seed_batch - self.mean_pose) / self.std_pose / 10.0

        return gesture_batch, seed_batch, audio_batch, speaker_batch