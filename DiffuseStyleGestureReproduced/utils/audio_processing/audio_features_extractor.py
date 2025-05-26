import math
import torch
import pickle
import os
import numpy as np
from torchaudio.transforms import MelSpectrogram, MFCC
from utils.audio_processing.prosody_features import RMS, Pitch

class AudioFeaturesExtractor:
    def __init__(self, sample_rate=16000, context_frames=100, device=None,
                 normalization_file=None):
        # Set device
        self.device = device
            
        # Initialize parameters
        self.sample_rate = sample_rate
        self.frame_length = 2048
        self.hop_length = int(math.floor(sample_rate / 30))  # For 30fps output
        self.context_frames = context_frames
        
        # Initialize all transforms once
        self.mel_spec_transform = MelSpectrogram(
            sample_rate=self.sample_rate, 
            n_fft=self.frame_length, 
            hop_length=self.hop_length, 
            n_mels=16,
            power=1.0
        ).to(self.device)

        self.mfcc_transform = MFCC(
            sample_rate=self.sample_rate, 
            n_mfcc=16,
            melkwargs={
                "n_fft": self.frame_length, 
                "hop_length": self.hop_length, 
                "n_mels": 16
            }
        ).to(self.device)

        self.energy_transform = RMS(
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        ).to(self.device)
        
        self.pitch_transform = Pitch(
            sample_rate=self.sample_rate, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        ).to(self.device)
        
        # Audio buffer for processing
        self.buffer_size = self.frame_length + self.hop_length * 8 # I am not exactly sure why 8 is the right number, but it seems to work
        self.buffer = torch.zeros(1, self.buffer_size).to(self.device)
        self.prev_rms_energy = torch.zeros(1).to(self.device)
        self.prev_pitch = torch.zeros(1).to(self.device)
        self.max_energy_seen = 0.0  # For dynamic onset detection threshold
        
        # Feature dimensions
        self.mel_dim = 16
        self.mfcc_dim = 16
        
        # Pre-allocate feature buffers - newest data will always be on the right side
        self.mel_spec_buffer = torch.zeros((context_frames, self.mel_dim), device=self.device)
        self.mfcc_buffer = torch.zeros((context_frames, self.mfcc_dim), device=self.device)
        self.rms_energy_buffer = torch.zeros((context_frames, 1), device=self.device)
        self.pitch_buffer = torch.zeros((context_frames, 1), device=self.device)
        self.energy_derivatives_buffer = torch.zeros((context_frames, 1), device=self.device)
        self.pitch_derivatives_buffer = torch.zeros((context_frames, 1), device=self.device)
        self.onsets_buffer = torch.zeros((context_frames, 1), device=self.device)
        
        # Counter to track how many frames we've processed
        self.frames_processed = 0
        if normalization_file is not None and os.path.exists(normalization_file):
            self.normalize = True
            
            # Load normalization parameters from file
            with open(normalization_file, 'rb') as f:
                normalization_params = pickle.load(f)
            
            # Unpack normalization parameters
            self.mel_spec_min = torch.tensor(normalization_params['mel_spec_min'].astype(np.float32)).to(self.device)
            self.mel_spec_max = torch.tensor(normalization_params['mel_spec_max'].astype(np.float32)).to(self.device)
            self.mel_spec_range = torch.tensor(normalization_params['mel_spec_range'].astype(np.float32)).to(self.device)
            self.mfcc_mean = torch.tensor(normalization_params['mfcc_mean'].astype(np.float32)).to(self.device)
            self.mfcc_std = torch.tensor(normalization_params['mfcc_std'].astype(np.float32)).to(self.device)
            self.rms_energy_mean = torch.tensor(normalization_params['rms_energy_mean'].astype(np.float32)).to(self.device)
            self.rms_energy_std = torch.tensor(normalization_params['rms_energy_std'].astype(np.float32)).to(self.device)
            self.pitch_mean = torch.tensor(normalization_params['pitch_mean'].astype(np.float32)).to(self.device)
            self.pitch_std = torch.tensor(normalization_params['pitch_std'].astype(np.float32)).to(self.device)
            self.energy_derivatives_mean = torch.tensor(normalization_params['energy_derivatives_mean'].astype(np.float32)).to(self.device)
            self.energy_derivatives_std = torch.tensor(normalization_params['energy_derivatives_std'].astype(np.float32)).to(self.device)
            self.pitch_derivatives_mean = torch.tensor(normalization_params['pitch_derivatives_mean'].astype(np.float32)).to(self.device)
            self.pitch_derivatives_std = torch.tensor(normalization_params['pitch_derivatives_std'].astype(np.float32)).to(self.device)
        else:
            self.normalize = False

    def process_chunk(self, audio_chunk):
        # Ensure audio is on the correct device
        audio_chunk = audio_chunk.to(self.device)
        
        # Handle varying chunk sizes by standardizing to hop_length
        if (audio_chunk.shape[1] != self.hop_length):
            if (audio_chunk.shape[1] > self.hop_length):
                # If chunk is too large, take only what we need
                audio_chunk = audio_chunk[:, :self.hop_length]
            else:
                # If chunk is too small, pad with zeros
                padding = torch.zeros(1, self.hop_length - audio_chunk.shape[1]).to(self.device)
                audio_chunk = torch.cat([audio_chunk, padding], dim=1)
        
        # Shift buffer left by hop_length
        self.buffer = torch.roll(self.buffer, -self.hop_length, dims=1)
        
        # Add new audio chunk at the end
        self.buffer[:, -self.hop_length:] = audio_chunk
        
        # Apply transforms to the full buffer
        mel_spec = self.mel_spec_transform(self.buffer).squeeze(0).permute(1, 0)
        mfcc = self.mfcc_transform(self.buffer).squeeze(0).permute(1, 0)
        rms_energy = self.energy_transform(self.buffer).squeeze(0)
        pitch = self.pitch_transform(self.buffer).squeeze(0)
        
        # We only need the last frame of features
        if mel_spec.dim() > 1:
            mel_spec = mel_spec[-1:, :]
        if mfcc.dim() > 1:
            mfcc = mfcc[-1:, :]
        if rms_energy.dim() > 0:
            rms_energy = rms_energy[-1:]
        if pitch.dim() > 0:
            pitch = pitch[-1:]

        # Normalize features if normalization parameters are available
        if self.normalize:
            mel_spec = (mel_spec - self.mel_spec_min) / self.mel_spec_range
            mel_spec = mel_spec * 3.0 # Same as data_processsor
            mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
            rms_energy = (rms_energy - self.rms_energy_mean) / self.rms_energy_std
            pitch = (pitch - self.pitch_mean) / self.pitch_std
            
            # Calculate energy derivatives and pitch derivatives
            energy_derivatives = (rms_energy - self.prev_rms_energy) / self.rms_energy_std
            pitch_derivatives = (pitch - self.prev_pitch) / self.pitch_std
            
            # Normalize derivatives
            energy_derivatives = (energy_derivatives - self.energy_derivatives_mean) / self.energy_derivatives_std
            pitch_derivatives = (pitch_derivatives - self.pitch_derivatives_mean) / self.pitch_derivatives_std
        
        # Calculate derivatives using current and previous values
        energy_derivatives = rms_energy - self.prev_rms_energy
        pitch_derivatives = pitch - self.prev_pitch
        
        # Update previous values for next chunk
        self.prev_rms_energy = rms_energy
        self.prev_pitch = pitch
        
        # Update max energy for onset detection (with decay to adapt over time)
        self.max_energy_seen = max(self.max_energy_seen * 0.99, rms_energy.max().item())
        
        # Detect onsets
        onset_threshold = 0.5 * self.max_energy_seen
        onsets = torch.where(rms_energy > onset_threshold, 
                           torch.ones_like(rms_energy), 
                           torch.zeros_like(rms_energy))
        
        # Add feature dimensions to match expected shape (time_steps, feature_dim)
        rms_energy = rms_energy.unsqueeze(1)
        pitch = pitch.unsqueeze(1)
        energy_derivatives = energy_derivatives.unsqueeze(1)
        pitch_derivatives = pitch_derivatives.unsqueeze(1)
        onsets = onsets.unsqueeze(1)
        
        # Update feature buffers - shift all data left and add new frame at the end
        self.mel_spec_buffer = torch.roll(self.mel_spec_buffer, -1, dims=0)
        self.mfcc_buffer = torch.roll(self.mfcc_buffer, -1, dims=0)
        self.rms_energy_buffer = torch.roll(self.rms_energy_buffer, -1, dims=0)
        self.pitch_buffer = torch.roll(self.pitch_buffer, -1, dims=0)
        self.energy_derivatives_buffer = torch.roll(self.energy_derivatives_buffer, -1, dims=0)
        self.pitch_derivatives_buffer = torch.roll(self.pitch_derivatives_buffer, -1, dims=0)
        self.onsets_buffer = torch.roll(self.onsets_buffer, -1, dims=0)
        
        # Add the new frame at the end (rightmost position)
        self.mel_spec_buffer[-1:] = mel_spec
        self.mfcc_buffer[-1:] = mfcc
        self.rms_energy_buffer[-1:] = rms_energy
        self.pitch_buffer[-1:] = pitch
        self.energy_derivatives_buffer[-1:] = energy_derivatives
        self.pitch_derivatives_buffer[-1:] = pitch_derivatives
        self.onsets_buffer[-1:] = onsets
        
        # Increment frames counter
        self.frames_processed += 1
        
        # Return current frame features and number of valid frames
        current_features = (mel_spec, mfcc, rms_energy, pitch, energy_derivatives, pitch_derivatives, onsets)
        valid_frames = min(self.frames_processed, self.context_frames)
        
        return current_features, valid_frames
    
    def get_feature_buffers(self):
        # Determine how many frames are valid
        valid_frames = min(self.frames_processed, self.context_frames)
        
        if valid_frames == 0:
            return None
        
        # Return only the valid portion (rightmost frames are newest)
        start_idx = self.context_frames - valid_frames
        
        return {
            'mel_spec': self.mel_spec_buffer[start_idx:],
            'mfcc': self.mfcc_buffer[start_idx:],
            'rms_energy': self.rms_energy_buffer[start_idx:],
            'pitch': self.pitch_buffer[start_idx:],
            'energy_derivatives': self.energy_derivatives_buffer[start_idx:],
            'pitch_derivatives': self.pitch_derivatives_buffer[start_idx:],
            'onsets': self.onsets_buffer[start_idx:]
        }
    
    def get_concatenated_features(self):
        feature_dict = self.get_feature_buffers()
        if feature_dict is None:
            return None
        
        # Concatenate all features along feature dimension
        return torch.cat([
            feature_dict['mel_spec'],
            feature_dict['mfcc'],
            feature_dict['rms_energy'],
            feature_dict['pitch'],
            feature_dict['energy_derivatives'],
            feature_dict['pitch_derivatives'],
            feature_dict['onsets']
        ], dim=1)