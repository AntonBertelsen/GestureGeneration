import torch
from torch.amp import autocast
import time
import matplotlib.pyplot as plt
from v1_model import ContinuousMotionModel
from dataset.dataset import *
import utils.utils as utils
import soundfile as sf
from IPython.display import clear_output


import utils.animation.visualisation.new.animation_visualisation as animation_visualisation


device = utils.get_device()

# Get the newest model from the directory. That means find the newest folder, and then the newest file in that folder.
# model_path = utils.get_latest_model_path("v1_models")
model_path = "v1_models/first_tests_2025-06-17_15-46-54_final/first_tests_2025-06-17_15-46-54_final_epoch_1000.pth"
print(f"Model path: {model_path}")

# Load the model
model: ContinuousMotionModel = ContinuousMotionModel.load_model(model_path,device)
model.condition_mask_probabilty = 0.0  # Disable condition mask probability for inference
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

print(animation_visualisation.init_visualization(display=False))

with autocast(device_type=device.type, dtype=torch.bfloat16):
    dataset = GPUDataset(
        consolidated_file="dataset/genea2023_dataset/val/main-agent/consolidated.npz",
        seq_length=100,
        seed_length=0,
        batch_size=1,
        epoch_length=1,  # Set to 1 for testing purposes
        return_audio_frame_index=True,  # Set to True to return the audio frame index
    )

    # Use no gradient calculation for inference
    with torch.no_grad():

        gesture_sequence, seed_gesture, _, main_agent_id_one_hot, start_frames = [
            item.to(device) for item in next(iter(dataset))
        ]
        full_audio_features = dataset.audio.to(device)
        start_frame = start_frames[0].item()  # Extract the first element from the tensor

        # Decode the input using the autoencoder model
        # encoded_gesture_seed = model.pose_encoder.encode(gesture_sequence)

        iteration_counter = 0
        
        # Generate pure noise as the initial input
        denoised_gesture_sequence = torch.randn((1, model.n_gesture_length, model.pose_features_per_frame), device=device)

        while True:
            actual_audio_features = full_audio_features[start_frame + iteration_counter * dataset.seq_length: start_frame + iteration_counter * dataset.seq_length + dataset.seq_length, :].unsqueeze(0)

            for timestep in range(model.diffusion.number_of_timesteps-1, -1, -1):

                # apply diffusion at the current timestep
                noisy_gesture_sequence = model.diffusion.forward(denoised_gesture_sequence, timestep)

                # Now we apply the model to denoise the gesture sequence
                timestep_tensor = torch.tensor([timestep], dtype=torch.int64, device=device)
                denoised_gesture_sequence = model.forward(
                    timestep=timestep_tensor,
                    one_hot_style=main_agent_id_one_hot,
                    audio_features=actual_audio_features,
                    noisy_gesture_sequence=noisy_gesture_sequence
                )
                animation_visualisation.send_debug_tensor(torch.cat((actual_audio_features.squeeze(0).to(torch.float32),noisy_gesture_sequence.squeeze(0).to(torch.float32)), dim=1), "full tensor")

            ########################################################################################################################################################################

            # Decode the output using the autoencoder model
            clear_output(wait=True)
        
            unencoded_denoised_gesture_sequence = model.pose_encoder.decode(denoised_gesture_sequence)

            denmormalized_unencoded_denoised_gesture_sequence = dataset.skeleton.denormalize_poses(unencoded_denoised_gesture_sequence).squeeze(0).squeeze(0)

            animation_visualisation.send_debug_tensor(torch.cat((actual_audio_features.squeeze(0).to(torch.float32),denoised_gesture_sequence.squeeze(0).to(torch.float32)), dim=1), "full tensor")

            for frame in denmormalized_unencoded_denoised_gesture_sequence:
                # Start time for the current frame
                frame_start_time = time.time()
                # Send each frame to the animation visualisation
                animation_visualisation.send_pose(frame.cpu(), dataset.skeleton)
                frame_end_time = time.time()
                # Print the time taken for the current frame
                # print(f"Frame {iteration_counter} processed in {frame_end_time - frame_start_time:.4f} seconds ({1/(frame_end_time - frame_start_time):.2f} FPS)")

                # Sleep for the remaining time in the 30 FPS frame
                time_to_sleep = max(0, (1/30) - (frame_end_time - frame_start_time) - 0.0005)  # 0.01 is a small buffer to account for processing time
                time.sleep(time_to_sleep)


            iteration_counter += 1
            if iteration_counter % 10 == 0:
                # Print the current iteration counter every 10 iterations
                print(f"Iteration: {iteration_counter}")