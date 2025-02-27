import torch
import torchaudio
import torch.nn.functional as F

class RMS(torch.nn.Module):
    """
    Compute RMS energy over frames, with center padding to match MelSpectrogram.
    """
    def __init__(self, frame_length=2048, hop_length=512):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.eps = 1e-10  # for numerical stability

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (Tensor): shape (channels, time)

        Returns:
            Tensor: RMS energy with shape (channels, num_frames)
        """
        # Pad waveform: pad n_fft//2 on both sides (center padding)
        pad_amount = self.frame_length // 2
        waveform_padded = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')

        # Square the waveform
        waveform_squared = waveform_padded ** 2  # shape: (channels, time + pad)

        # Reshape to 4D for unfolding: (channels, 1, 1, time)
        x = waveform_squared.unsqueeze(1).unsqueeze(2)  # shape: (channels, 1, 1, time+pad)

        # Unfold: extract frames of length `frame_length` with stride `hop_length`
        patches = F.unfold(
            x,
            kernel_size=(1, self.frame_length),
            stride=(1, self.hop_length)
        )  # shape: (channels, frame_length, num_frames)

        channels = waveform.shape[0]
        num_frames = patches.shape[-1]
        patches = patches.view(channels, self.frame_length, num_frames)

        # Compute RMS: mean over frame_length, then sqrt
        rms = torch.sqrt(torch.mean(patches, dim=1) + self.eps)  # shape: (channels, num_frames)
        return rms
    


# This pitch calculation uses torchaudio's detect_pitch_frequency function.
# One problem is that it uses some kind of time thing that is not the same as the hop_length.
# This is a problem because we want to use the pitch as a feature for the same frames as the other features.
# The best I can do is to pad / crop the pitch tensor to match the other features.
# I tried to implement custom pitch calculation with help from chatgpt, but this seems like a very complex task.
# The results were not good when I visualized the pitch features, so I will stick with this for now.
class Pitch(torch.nn.Module):
    """
    Compute fundamental frequency (pitch) per frame.
    """
    def __init__(self, sample_rate, frame_length=2048, hop_length=512, win_length=30, freq_low=85, freq_high=3400):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_low = freq_low
        self.freq_high = freq_high

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Apply center padding: pad frame_length // 2 on both sides
        pad_amount = self.frame_length // 2
        waveform_padded = F.pad(waveform, (pad_amount, pad_amount), mode='reflect')

        # Compute pitch using torchaudio's detect_pitch_frequency
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform_padded,
            sample_rate=self.sample_rate,
            frame_time=self.hop_length / self.sample_rate,  # Align frame time with hop_length
            win_length=self.win_length,
            freq_low=self.freq_low,
            freq_high=self.freq_high
        )
        
        # Work out the length of the padding
        desired_length = waveform.shape[-1] // self.hop_length + 1

        # Crop or pad the pitch tensor to match the desired length
        if pitch.shape[-1] > desired_length:
            pitch = pitch[..., :desired_length]
        else:
            pad_amount = desired_length - pitch.shape[-1]
            pitch = F.pad(pitch, (0, pad_amount), mode='constant', value=0)

        return pitch