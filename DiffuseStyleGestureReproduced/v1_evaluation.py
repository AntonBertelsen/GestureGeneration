import time
import torch
from v1_model import ContinuousMotionModel
from torch.utils.data import DataLoader
from utils.animation.skeleton import Skeleton
from utils.evaluation.FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator
import utils.animation.visualisation.new.animation_visualisation as animation_visualisation
from v1_sliding_diffusion import SlidingDiffusion
from v1_normal_diffusion import NormalDiffusion
from torch.amp import autocast


def evaluate_frechet_gesture_distance_sliding_diffusion(
        model: ContinuousMotionModel, 
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        num_samples = 8192,
        calculate_raw_frechet_distance = True) -> None:
    with torch.no_grad():

        # first we need to create the embedding space evaluator
        embeddingSpaceEvaluator = EmbeddingSpaceEvaluator(
            embed_net_path  = f"utils/evaluation/FGD/models/fgd_model_{evaluation_length}.pth",
            n_frames        = evaluation_length,
            device          = device,
            pose_dim        = len(val_loader.dataset.skeleton.target_joints) * 3
        )

        # We need to retrieve sequences of a certain length from the dataloader
        # This is pretty hacky, but allows us to just reuse the validation dataloader
        # TODO: Find a better way to do this that is compatible with other datasets as well
        old_batch_size = val_loader.dataset.batch_size
        val_loader.dataset.batch_size = num_samples
        
        old_valid_starts = val_loader.dataset.valid_starts

        old_sequence_length = val_loader.dataset.seq_length
        val_loader.dataset.seq_length = (model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames) + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames + evaluation_length
        val_loader.dataset.chunk_size = val_loader.dataset.seq_length + val_loader.dataset.seed_length
        val_loader.dataset._create_offset_tensors()
        val_loader.dataset.valid_starts = val_loader.dataset._find_valid_starting_points()
        val_loader.dataset.reshuffle()
        
        skeleton: Skeleton = val_loader.dataset.skeleton

        pose_encoder = model.pose_encoder

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            # We load all the data we need. even when we have generated everything. 
            # That is we need the starting point (The "seed"), which gives us the starting clean gesture from which we continue our generation. (num_clean_frames)
            # We also need enough data to move past the the noising area (num_denoise_frames) + the initial noising area (num_denoise_frames)
            # We also need enough data to have n_frames of data generated when we are done. (evaluation_length)
            # i.e. sequence length retrieved by val_loader should be num_clean_frames + 2 * num_denoise_frames + num_clean_frames + n_frames
            full_gesture_sequence, gesture_seed, full_audio_features, main_agent_id_one_hot = [
                item.squeeze(0).to(device) for item in next(iter(val_loader))
            ]

            feature_dim = full_gesture_sequence.shape[2]
            encoded_feature_dim = pose_encoder.z_dim if pose_encoder is not None and not val_loader.dataset.loading_encoded_data else feature_dim

            # We prepare a result tensor to store the generated data
            # This is of shape (bs, num_timestep_frames + n_frames, feature_dim)
            result_tensor = torch.zeros(
                (num_samples, model.diffusion.num_denoise_frames + evaluation_length, encoded_feature_dim),
                device=device,
            )

            if not val_loader.dataset.loading_encoded_data and pose_encoder is not None:
                full_encoded_gesture_sequence = pose_encoder.encode(full_gesture_sequence)
            else:
                full_encoded_gesture_sequence = full_gesture_sequence

            # Now we need to cut the sequence to the right length.
            # We need to cut the sequence to the length of the pre-timestep frames + timestep frames + post-timestep frames
            # The audio features should be cut in the same way
            current_encoded_gesture_sequence = full_encoded_gesture_sequence[:, 0:model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames,:]

            print("Starting inference steps for sliding diffusion...")

            for i in range(model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames + evaluation_length):
                # We also need to cut out the relevant audio features. As these are not predicted by the model we cut them out of the full audio features
                # at every iteration
                current_audio_features = full_audio_features[:, i:i + model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames, :]

                current_encoded_gesture_sequence, _ = model.inference(
                    current_encoded_gesture_sequence, 
                    current_audio_features, 
                    main_agent_id_one_hot,
                    gesture_seed
                )
                # copy the newest predicted frame to the result tensor
                result_tensor[:, i, :] = current_encoded_gesture_sequence[:, model.diffusion.num_clean_frames, :]
                
                print("Step", i + 1, "of", model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames + evaluation_length, "done.")
            
            # The first num_clean_frames frames of the prediction had access to the original gesture sequence, so in a sense they 'cheated'. For this reason, we discard them.
            # The remaining frames (evaluation length) result tensor are the final generated gestures.
            result_tensor = result_tensor[:, model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames:, :]

            # Now we use the autoencoder to decode the result tensor from the low dimensional latent space to the original feature space.
            if pose_encoder is not None:
                output = pose_encoder.decode(result_tensor)
            else:
                output = result_tensor

            # We want to perform FGD calculations on the world pose space. AS such we need to calculate this.

            # First we need to denormalized the output tensor to recover the original feature space (the rotation matrices)
            denormalized_output = skeleton.denormalize_poses(output)

            # Send every frame in the original gesture sequence to the animation visualisation
            
            # animation_visualisation.send_debug_tensor(output[0].cpu(), "generated gesture sequence")
            # for frame in denormalized_output[0]:
            #     animation_visualisation.send_pose(frame.cpu(), skeleton)
            #     time.sleep( 1 / 30)

            # Now we want to calcualte world postions from the denormalized output tensor
            world_positions = skeleton.calculate_world_positions(denormalized_output)

            # Now we want to z-normalize the world positions to get the final output tensor
            normalized_world_positions = skeleton.normalize_world_positions(world_positions)

        embeddingSpaceEvaluator.push_generated_samples(normalized_world_positions.float())

        ########################################

        # We also need to acquire the original gesture sequence in world space
        # We need to cut out the area of the original gesture sequence that is equivalent to the area we generated
        # This is [num_presteps + num_steps: num_presteps + num_steps + n_frames]

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            original_gesture_sequence = full_gesture_sequence[:, model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames:model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + evaluation_length, :]
            
            if val_loader.dataset.loading_encoded_data:
                # If we are loading encoded data, we need to decode the original gesture sequence as well
                original_gesture_sequence = pose_encoder.decode(original_gesture_sequence)

            # We need to denormalize the original gesture sequence to recover the original feature space (the rotation matrices)
            denormalized_original_gesture_sequence = skeleton.denormalize_poses(original_gesture_sequence)
            
            # Now we want to calcualte world postions from the denormalized original gesture sequence
            world_positions = skeleton.calculate_world_positions(denormalized_original_gesture_sequence)
            # Now we want to z-normalize the world positions to get the final output tensor
            

            # Send every frame in the original gesture sequence to the animation visualisation
            # animation_visualisation.send_debug_tensor(original_gesture_sequence[0].cpu(), "generated gesture sequence")
            # for frame in denormalized_original_gesture_sequence[0]:
            #     animation_visualisation.send_pose(frame.cpu(), skeleton)
            #     time.sleep( 1 / 30)

            normalized_original_world_positions = skeleton.normalize_world_positions(world_positions)

        embeddingSpaceEvaluator.push_real_samples(normalized_original_world_positions.float())

        # Now we can calculate the Frechet distances
        frechet_distance_raw = embeddingSpaceEvaluator.get_fgd(use_feat_space=False) if calculate_raw_frechet_distance else None
        frechet_distance_feat_space = embeddingSpaceEvaluator.get_fgd(use_feat_space=True)
        
        # Now that we are done we need to reset the sequence length and batch size of the dataloader
        val_loader.dataset.seq_length = old_sequence_length
        val_loader.dataset.chunk_size = val_loader.dataset.seq_length + val_loader.dataset.seed_length
        val_loader.dataset._create_offset_tensors()
        val_loader.dataset.valid_starts = old_valid_starts
        val_loader.dataset.batch_size = old_batch_size

        # Set the model back to training mode
        model.train()

        print(f"Frechet distance in feature space: {frechet_distance_feat_space}, Frechet raw distance: {frechet_distance_raw}")

        return frechet_distance_feat_space, frechet_distance_raw
    
def evaluate_frechet_gesture_distance_normal_diffusion(
        model: ContinuousMotionModel, 
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        num_samples = 8192,
        calculate_raw_frechet_distance = True) -> None:
    with torch.no_grad():

        # first we need to create the embedding space evaluator
        embeddingSpaceEvaluator = EmbeddingSpaceEvaluator(
            embed_net_path  = f"utils/evaluation/FGD/models/fgd_model_{evaluation_length}.pth",
            n_frames        = evaluation_length,
            device          = device,
            pose_dim        = len(val_loader.dataset.skeleton.target_joints) * 3
        )

        old_batch_size = val_loader.dataset.batch_size
        val_loader.dataset.batch_size = num_samples

        val_loader.dataset.reshuffle()
        
        skeleton: Skeleton = val_loader.dataset.skeleton

        pose_encoder = model.pose_encoder

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            # We load all the data we need. even when we have generated everything. 
            # That is we need the starting point (The "seed"), which gives us the starting clean gesture from which we continue our generation. (num_presteps)
            # We also need enough data to move past the the noising area (num_of_timestep_frames) + the initial noising area (num_of_timestep_frames)
            # We also need enough data to have n_frames of data generated when we are done. (n_frames)
            # i.e. sequence length retrieved by val_loader should be num_presteps + 2 * num_of_timestep_frames + num_of_post_timestep_frames + n_frames
            gesture_sequence, gesture_seed, audio_features, main_agent_id_one_hot = [
                item.squeeze(0).to(device) for item in next(iter(val_loader))
            ]

            # feature_dim = gesture_sequence.shape[2]
            # encoded_feature_dim = pose_encoder.z_dim if pose_encoder is not None and not val_loader.dataset.loading_encoded_data else feature_dim
            
            # if not val_loader.dataset.loading_encoded_data and pose_encoder is not None:
            #     encoded_gesture_sequence = pose_encoder.encode(gesture_sequence)
            # else:
            #     encoded_gesture_sequence = gesture_sequence

            # Generate pure noise as the initial input
            denoised_gesture_sequence = torch.randn((num_samples, model.gesture_length, model.pose_features_per_frame), device=device)
            
            for timestep in range(model.diffusion.number_of_timesteps-1, -1, -1):
                # apply diffusion at the current timestep
                noisy_gesture_sequence = model.diffusion.forward(denoised_gesture_sequence, timestep)

                # Now we apply the model to denoise the gesture sequence
                timestep_tensor = torch.tensor([timestep], dtype=torch.int64, device=device).expand(num_samples)
                denoised_gesture_sequence = model.forward(
                    timestep=timestep_tensor,
                    one_hot_style=main_agent_id_one_hot,
                    audio_features=audio_features,
                    noisy_gesture_sequence=noisy_gesture_sequence,
                    seed_gesture_sequence=gesture_seed
                )
            
            # Extract the last evaluation_length frames from the denoised gesture sequence. The reason we grab the last frames
            # is to avoid frames which are heavily influenced by the seed. We do the same for sliding diffusion.
            result_tensor = denoised_gesture_sequence[:, -evaluation_length:, :]

            # Now we use the autoencoder to decode the result tensor from the low dimensional latent space to the original feature space.
            if pose_encoder is not None:
                output = pose_encoder.decode(result_tensor)
            else:
                output = result_tensor

            # We want to perform FGD calculations on the world pose space. AS such we need to calculate this.

            # First we need to denormalized the output tensor to recover the original feature space (the rotation matrices)
            denormalized_output = skeleton.denormalize_poses(output)

            # Send every frame in the original gesture sequence to the animation visualisation
            
            # animation_visualisation.send_debug_tensor(output[0].cpu(), "generated gesture sequence")
            # for frame in denormalized_output[0]:
            #     animation_visualisation.send_pose(frame.cpu(), skeleton)
            #     time.sleep( 1 / 30)

            # Now we want to calcualte world postions from the denormalized output tensor
            world_positions = skeleton.calculate_world_positions(denormalized_output)

            # Now we want to z-normalize the world positions to get the final output tensor
            normalized_world_positions = skeleton.normalize_world_positions(world_positions)

        embeddingSpaceEvaluator.push_generated_samples(normalized_world_positions.float())

        ########################################

        # We also need to acquire the original gesture sequence in world space
        # We need to cut out the area of the original gesture sequence that is equivalent to the area we generated
        # This is [num_presteps + num_steps: num_presteps + num_steps + n_frames]

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            original_gesture_sequence = gesture_sequence[:, -evaluation_length:, :]
            
            if val_loader.dataset.loading_encoded_data:
                # If we are loading encoded data, we need to decode the original gesture sequence as well
                original_gesture_sequence = pose_encoder.decode(original_gesture_sequence)

            # We need to denormalize the original gesture sequence to recover the original feature space (the rotation matrices)
            denormalized_original_gesture_sequence = skeleton.denormalize_poses(original_gesture_sequence)
            
            # Now we want to calcualte world postions from the denormalized original gesture sequence
            world_positions = skeleton.calculate_world_positions(denormalized_original_gesture_sequence)
            # Now we want to z-normalize the world positions to get the final output tensor
            
            # Send every frame in the original gesture sequence to the animation visualisation
            # animation_visualisation.send_debug_tensor(original_gesture_sequence[0].cpu(), "generated gesture sequence")
            # for frame in denormalized_original_gesture_sequence[0]:
            #     animation_visualisation.send_pose(frame.cpu(), skeleton)
            #     time.sleep( 1 / 30)

            normalized_original_world_positions = skeleton.normalize_world_positions(world_positions)

        embeddingSpaceEvaluator.push_real_samples(normalized_original_world_positions.float())

        # Now we can calculate the Frechet distances
        frechet_distance_raw = embeddingSpaceEvaluator.get_fgd(use_feat_space=False) if calculate_raw_frechet_distance else None
        frechet_distance_feat_space = embeddingSpaceEvaluator.get_fgd(use_feat_space=True)


        val_loader.dataset.batch_size = old_batch_size

        # Set the model back to training mode
        model.train()

        print(f"Frechet distance in feature space: {frechet_distance_feat_space}, Frechet raw distance: {frechet_distance_raw}")

        return frechet_distance_feat_space, frechet_distance_raw
    

def evaluate_frechet_gesture_distance(
        model: ContinuousMotionModel, 
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        num_samples = 8192,
        calculate_raw_frechet_distance = True) -> None:

    if isinstance(model.diffusion, SlidingDiffusion):
        return evaluate_frechet_gesture_distance_sliding_diffusion(
            model, 
            val_loader, 
            device, 
            evaluation_length, 
            num_samples, 
            calculate_raw_frechet_distance
        )
    elif isinstance(model.diffusion, NormalDiffusion):
        return evaluate_frechet_gesture_distance_normal_diffusion(
            model, 
            val_loader, 
            device, 
            evaluation_length, 
            num_samples, 
            calculate_raw_frechet_distance
        )