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
    
class FixedSampleAnimationDataset(Dataset):
    def __init__(self, folder, seq_length_in_frames=5, seed_length_in_frames=10, epoch_length=10000):
        """
        A dataset that always returns the first snippet of the first file.
        Useful for debugging.
        
        Args:
            folder (str): Path to folder containing .npz files.
            seq_length (int): Duration (in seconds) of the clip to load.
            fps (int): Frames per second in the animation.
        """
        self.folder = folder
        self.seq_length_in_frames = seq_length_in_frames
        self.seed_length_in_frames = seed_length_in_frames
        self.chunk_size = seq_length_in_frames + seed_length_in_frames
        self.epoch_length = epoch_length
        
        # List all npz files and select the first one
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
        if not self.files:
            raise ValueError("No .npz files found in the provided folder.")
        
        self.file = self.files[0]  # Always use the first file
        
        start_frame = 0
        
        with np.load(self.file) as npz:
            self.gesture_chunk = npz["bvh_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            self.seed_chunk = npz["bvh_features"][start_frame : start_frame + self.seed_length_in_frames]
            self.audio_chunk = npz["audio_features"][start_frame + self.seed_length_in_frames : start_frame + self.chunk_size]
            self.speaker = npz["main_agent_id_one_hot"]

        self.sample = {
            "gesture": torch.tensor(self.gesture_chunk, dtype=torch.float32),
            "seed": torch.tensor(self.seed_chunk, dtype=torch.float32),
            "audio": torch.tensor(self.audio_chunk, dtype=torch.float32),
            "speaker": self.speaker
        }

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        return self.sample
    

import time
import os
import pickle

class RAMResidentDataset(Dataset):
    """Dataset that keeps all data in RAM for maximum speed"""
    def __init__(self, folder, windows_file, batch_size=32, epoch_length=1000):
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        self.folder = folder

        # Create a log file
        self.log_dir = os.path.join(os.path.dirname(self.folder), "profiling_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"dataloader_profile_{time.strftime('%Y%m%d_%H%M%S')}.log")

        # Load metadata
        meta_file = windows_file.replace('.npz', '_meta.pkl')
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loading entire dataset into RAM from {windows_file}")
        start_time = time.time()
        
        # Load the ENTIRE dataset into RAM (not memory-mapped)
        data = np.load(windows_file)

        # Load the dataset into RAM using memory-mapped arrays
        # data = np.load(windows_file, mmap_mode='r')
        
        # Convert all arrays to torch tensors immediately
        self.gestures = torch.from_numpy(data['gestures']).half()
        self.seeds = torch.from_numpy(data['seeds']).half()
        self.audio = torch.from_numpy(data['audio']).half()
        self.speakers = torch.from_numpy(data['speakers']).half()
        
        # Close numpy file to free file handles
        data.close()
        
        self.num_windows = len(self.gestures)
        
        # Pre-generate batch indices once
        self.batch_indices = [torch.randperm(self.num_windows)[:self.batch_size] for _ in range(self.epoch_length)]
        
        with open(self.log_file, 'a') as f:
            f.write(f"Dataset loaded into RAM in {time.time() - start_time:.2f} seconds\n")
            f.write(f"Using {self.gestures.element_size() * self.gestures.nelement() / 1024**3:.2f} GB for gestures\n")
            f.write(f"Using {self.seeds.element_size() * self.seeds.nelement() / 1024**3:.2f} GB for seeds\n")
            f.write(f"Using {self.audio.element_size() * self.audio.nelement() / 1024**3:.2f} GB for audio\n")
            f.write(f"Total: {(self.gestures.element_size() * self.gestures.nelement() + self.seeds.element_size() * self.seeds.nelement() + self.audio.element_size() * self.audio.nelement()) / 1024**3:.2f} GB\n\n")
    
    def reshuffle(self):
        """Generate new random batches"""
        self.batch_indices = [torch.randperm(self.num_windows)[:self.batch_size] for _ in range(self.epoch_length)]
        
    def __len__(self):
        return self.epoch_length
        
    def __getitem__(self, idx):
        """Returns a complete batch without any disk access"""
        # Get pre-generated batch indices
        indices = self.batch_indices[idx]
        
        # Ultra-fast indexing of RAM-resident tensors
        gesture_batch = self.gestures[indices]
        seed_batch = self.seeds[indices]
        audio_batch = self.audio[indices]
        speaker_batch = self.speakers[indices]
        
        return gesture_batch, seed_batch, audio_batch, speaker_batch