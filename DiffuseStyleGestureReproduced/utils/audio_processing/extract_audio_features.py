import math
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC
from utils.audio_processing.prosody_features import RMS, Pitch

from transformers import WavLMModel, Wav2Vec2FeatureExtractor

def extract_audio_features(audio_file):

    device = torch.device("cuda" if torch.cuda.is_available() else 
                        "mps" if torch.backends.mps.is_available() else 
                        "cpu")

    # Load audio file (with `torchaudio`)
    waveform, wav_sample_rate = torchaudio.load(audio_file, normalize=True)
    print(f"Sample rate: {wav_sample_rate}")

    # We have a problem here: wavlm expects audio at 16kHz, and we want our final features to be at 30fps.
    # For wavlm we unfortunately need to interpolate to 30fps from 50fps which is what wavlm outputs.
    # However, it also has implications for these other features. This is because for these we step through the audio
    # looking at small windows.
    # The sample rate is 16kHz, so 1 frame is 1/16000 seconds. We want to step through the audio at 30fps, so we need to step
    # 1/30 seconds each time. This means we need to step 16000/30 samples each time.
    # 16000/30 = 533.3333333333334 (Not an integer (extremely sad))

    # This means our final tensors will not have exactly the same number of frames as the wavlm embeddings.
    # I think the best thing we can do is chop off the last few fames of the other features to make them match the wavlm embeddings

    # If we could sample the audio 15kHz, then we could have a step size of 15kHz/30 = 500 samples, which would be perfect.
    sample_rate = 16000

    # Apply resampling to 16 kHz
    waveform = torchaudio.transforms.Resample(orig_freq=wav_sample_rate, new_freq=sample_rate)(waveform).to(device)

    # Extract features
    mel_spec, mfcc, rms_energy, pitch, energy_derivatives, pitch_derivatives, onsets = extract_simple_features(waveform, sample_rate, device)
    wavlm_features = extract_wavlm_features(waveform, device)

    print("Mel spectrogram shape:", mel_spec.shape)
    print("MFCC shape:", mfcc.shape)
    print("RMS energy shape:", rms_energy.shape)
    print("Pitch shape:", pitch.shape)
    print("Energy derivatives shape:", energy_derivatives.shape)
    print("Pitch derivatives shape:", pitch_derivatives.shape)
    print("Onsets shape:", onsets.shape)
    print("WavLM features shape:", wavlm_features.shape)




    # Because the features are not exactly the same length, we need to crop them to the same length. This is a little sketchy, but I think it should be fine
    # It is only 1 or 2 frames differnece typically.

    # Find the minimum length of the features
    min_length = min(mel_spec.shape[1], mfcc.shape[1], rms_energy.shape[0], pitch.shape[0], wavlm_features.shape[0])

    # Crop the features to the minimum length
    mel_spec = mel_spec[:, :min_length]
    mfcc = mfcc[:, :min_length]
    rms_energy = rms_energy[:min_length]
    energy_derivatives = energy_derivatives[:min_length]
    pitch = pitch[:min_length]
    pitch_derivatives = pitch_derivatives[:min_length]
    wavlm_features = wavlm_features[:min_length]

    # Concatenate features along the feature dimension
    audio_features = torch.cat([
        mel_spec.T,  # Transpose to match the shape (time_steps, features)
        mfcc.T,
        rms_energy.unsqueeze(1),  # Add a dimension to match the shape (time_steps, 1)
        energy_derivatives.unsqueeze(1),
        pitch.unsqueeze(1),  # Add a dimension to match the shape (time_steps, 1)
        pitch_derivatives.unsqueeze(1),
        wavlm_features
    ], dim=1)

    print(f"Audio features shape:", audio_features.shape)
    return audio_features.cpu()


def extract_simple_features(waveform, sample_rate, device):

    # The frame length is the number of samples that we consider for each calculation of the feature.
    frame_length = 2048

    # The hop length is the number of samples that we move forward each time we calculate the feature.
    hop_length = int(math.floor(sample_rate / 30)) # = 512

    mel_spec_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=frame_length, hop_length=hop_length, n_mels=16).to(device)

    mfcc_transform = MFCC(
        sample_rate=sample_rate, 
        n_mfcc=16,  # Reduce to a number <= n_mels
        melkwargs={"n_fft": frame_length, "hop_length": hop_length, "n_mels": 16}
    ).to(device)

    energy_transform = RMS(frame_length=frame_length, hop_length=hop_length).to(device)
    pitch_transform = Pitch(sample_rate=sample_rate, frame_length=frame_length, hop_length=hop_length).to(device)

    mel_spec = mel_spec_transform(waveform).squeeze(0)
    mfcc = mfcc_transform(waveform).squeeze(0)
    rms_energy = energy_transform(waveform).squeeze(0)
    pitch = pitch_transform(waveform).squeeze(0)

    # Derivative features. This is used by the orignal implementation, although they never mention it in the paper.
    energy_derivatives = torch.cat([torch.zeros(1).to(device), torch.diff(rms_energy)])
    pitch_derivatives = torch.cat([torch.zeros(1).to(device), torch.diff(pitch)])

    # Onsets
    # I am not sure this is a good way to detect onsets. It would be good to look into this more, maybe compare it with librosa onset detection to validate it.
    onset_threshold = 0.5 * rms_energy.max()
    onsets = torch.where(rms_energy > onset_threshold, torch.ones_like(rms_energy), torch.zeros_like(rms_energy))

    return mel_spec, mfcc, rms_energy, pitch, energy_derivatives, pitch_derivatives, onsets

def extract_wavlm_features(waveform, device):

    # Load the pretrained model and feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    model.to(device)

    # Load an audio file (assumed to be mono, 16kHz)
    # waveform, wav_sample_rate = torchaudio.load("dataset/genea2023_dataset/trn/main-agent/wav/trn_2023_v0_000_main-agent.wav")
    # waveform = torchaudio.transforms.Resample(orig_freq=wav_sample_rate, new_freq=16000)(waveform)

    sample_rate = 16000  # WavLM expects 16kHz input

    # Split the audio into 5-second chunks and collect them into a batch
    duration = 5  # seconds
    segment_length = sample_rate * duration 
    batch = []  # Store chunks

    for start in range(0, len(waveform[0]), segment_length):
        chunk = waveform[0][start:start + segment_length]
        if len(chunk) < segment_length:  # Pad if needed
            chunk = torch.nn.functional.pad(chunk, (0, segment_length - len(chunk)))
        batch.append(chunk)

    # Convert to tensor and process as batch
    batch_tensor = torch.stack(batch)  # Shape: [batch_size, time_steps]
    inputs = feature_extractor(batch_tensor, return_tensors="pt", sampling_rate=sample_rate)
    inputs["input_values"] = inputs["input_values"].squeeze() # Remove extra dimension which is added by the feature extractor

    # Batch processing does not work with WavLM because the attention mask is not automatically copied across batches when not supplied. 
    # I believe this might be a bug. This line is a hacky way to manually construct the attention mask for the input
    inputs["attention_mask"] = (inputs["input_values"] != 0)

    # Extract WavLM embeddings
    with torch.no_grad():
        inputs.to(device)
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state  # Shape: [batch, time_steps, hidden_dim]
    print(embeddings.shape)  # Example: [1, 500, 1024] (depends on audio length)

    # Unfortunately wavlm extracts embeddings at 20ms intervals (50 fps), 
    # so we need to downsample to match the sampling rate of the other features (30 fps)
    # (This is an area for possible performance improvement in the future, possibly it is possible to finetune wavlm to extract embeddings at 30fps)
    embeddings_downsampled = F.interpolate(embeddings.permute(0,2,1), size=(duration * 30,), mode="linear", align_corners=False).permute(0,2,1) # permutes are needed because interpolate always works on the last dimension (annoying). But performance impact should be negligible
    print(embeddings_downsampled.shape) # ([13, 150, 768])

    # Now I flatten the tensor to place each batch back one after each other. (How am I sure that it is being reconstructed in the right order?)
    embeddings_flattened = embeddings_downsampled.flatten(start_dim=0, end_dim=1)
    # print(embeddings_flattened.shape) # ([1950, 768])

    # The dimensions are not quite right. Let me copy the approach from pitch where I crop or pad the tensor to the right size
    # A little silly, but I don't think there is anything else to do
    # desired_length = waveform.shape[-1] // hop_length + 1

    # if embeddings_flattened.shape[0] < desired_length:
    #     embeddings_flattened = F.pad(embeddings_flattened, (0, 0, 0, desired_length - embeddings_flattened.shape[0]))
    # elif embeddings_flattened.shape[0] > desired_length:
    #     embeddings_flattened = embeddings_flattened[:desired_length]

    # print(embeddings_flattened.shape)
    return embeddings_flattened
