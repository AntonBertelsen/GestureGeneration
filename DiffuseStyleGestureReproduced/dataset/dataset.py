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
            with np.load(file) as npz:
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
        
        with np.load(file) as npz:
            gesture_chunk = npz["bvh_features"][start_frame + self.seed_length_in_frames: start_frame + self.chunk_size]
            seed_chunk = npz["bvh_features"][start_frame: start_frame + self.seed_length_in_frames]
            audio_chunk = npz["audio_features"][start_frame: start_frame + self.chunk_size]
            speaker = npz["main_agent_id_one_hot"]

        sample = {
            "gesture": torch.tensor(gesture_chunk, dtype=torch.float32),
            "seed": torch.tensor(gesture_chunk, dtype=torch.float32),
            "audio": torch.tensor(audio_chunk, dtype=torch.float32),
            "speaker": speaker
        }
        return sample