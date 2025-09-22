import time
import torch
import numpy as np
from tqdm import tqdm
from model import ContinuousMotionModel
from torch.utils.data import DataLoader
from utils.animation.skeleton import Skeleton
from utils.evaluation.FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator
import utils.animation.visualisation.new.animation_visualisation as animation_visualisation
from diffusion_process_sliding import SlidingDiffusion
from diffusion_process_normal import NormalDiffusion
from diffusion_process_outpaint import OutpaintDiffusion
from torch.amp import autocast


def evaluate_frechet_gesture_distance_sliding_diffusion(
        model: ContinuousMotionModel,
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        calculate_raw_frechet_distance = True,
        samples_per_iteration = 8192,
        num_iterations = 1,  # Number of iterations to pool samples
        bootstrap_samples = 0  # Number of bootstrap samples for confidence intervals (0 disables)
        ) -> None:
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
        old_batch_size = val_loader.dataset.batch_size
        
        old_valid_starts = val_loader.dataset.valid_starts

        old_sequence_length = val_loader.dataset.seq_length
        val_loader.dataset.seq_length = (model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames) + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames + evaluation_length
        val_loader.dataset.chunk_size = val_loader.dataset.seq_length + val_loader.dataset.seed_length
        val_loader.dataset._create_offset_tensors()
        val_loader.dataset.valid_starts = val_loader.dataset._find_valid_starting_points()
        
        skeleton: Skeleton = val_loader.dataset.skeleton
        pose_encoder = model.pose_encoder

        # Storage for bootstrap resampling
        all_gen_samples = []
        all_real_samples = []

        for iteration in tqdm(range(num_iterations), desc="Collecting FGD samples"):
            val_loader.dataset.batch_size = samples_per_iteration
            val_loader.dataset.reshuffle()
            
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # We load all the data we need. even when we have generated everything. 
                # That is we need the starting point (The "seed"), which gives us the starting clean gesture from which we continue our generation. (num_presteps)
                # We also need enough data to move past the the noising area (num_of_timestep_frames) + the initial noising area (num_of_timestep_frames)
                # We also need enough data to have n_frames of data generated when we are done. (n_frames)
                # i.e. sequence length retrieved by val_loader should be num_presteps + 2 * num_of_timestep_frames + num_of_post_timestep_frames + n_frames
                full_gesture_sequence, gesture_seed, full_audio_features, main_agent_id_one_hot, finger_availability = [
                    item.squeeze(0).to(device) for item in next(iter(val_loader))
                ]

                feature_dim = full_gesture_sequence.shape[2]
                encoded_feature_dim = pose_encoder.z_dim if pose_encoder is not None and not val_loader.dataset.loading_encoded_data else feature_dim

                # We prepare a result tensor to store the generated data
                # This is of shape (bs, num_timestep_frames + n_frames, feature_dim)
                result_tensor = torch.zeros(
                    (samples_per_iteration, model.diffusion.num_denoise_frames + evaluation_length, encoded_feature_dim),
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

                # Generate frames sequentially
                for i in range(model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames + evaluation_length):
                    # We also need to cut out the relevant audio features. As these are not predicted by the model we cut them out of the full audio features
                    # at every iteration
                    current_audio_features = full_audio_features[:, i:i + model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames, :]

                    current_encoded_gesture_sequence, _ = model.inference(
                        current_encoded_gesture_sequence, 
                        current_audio_features, 
                        main_agent_id_one_hot,
                        finger_availability,
                        gesture_seed
                    )
                    # copy the newest predicted frame to the result tensor
                    result_tensor[:, i, :] = current_encoded_gesture_sequence[:, model.diffusion.num_clean_frames, :]
                
                # The first num_presteps frames of the prediction had access to the original gesture sequence, so in a sense they 'cheated'. For this reason, we discard them.
                # The remaining n_frames result tensor are the final generated gestures.
                result_tensor = result_tensor[:, model.diffusion.num_denoise_frames + model.diffusion.num_noise_frames:, :]

                # Extract only the original feature dimensions if richer features were used
                result_tensor = result_tensor[..., :model.original_pose_features_per_frame]

                # Decode if needed
                if pose_encoder is not None:
                    output = pose_encoder.decode(result_tensor)
                else:
                    output = result_tensor

                # We want to perform FGD calculations on the world pose space. As such we need to calculate this.
                denormalized_output = skeleton.denormalize_poses(output)
                # Now we want to calcualte world postions from the denormalized output tensor
                world_positions = skeleton.calculate_world_positions(denormalized_output)
                # Now we want to z-normalize the world positions to get the final output tensor
                normalized_world_positions = skeleton.normalize_world_positions(world_positions)

            # Push generated samples for this iteration
            embeddingSpaceEvaluator.push_generated_samples(normalized_world_positions.float())
        
            # For bootstrap resampling
            if bootstrap_samples > 0:
                all_gen_samples.append(normalized_world_positions.float().cpu())

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # We also need to acquire the original gesture sequence in world space
                # We need to cut out the area of the original gesture sequence that is equivalent to the area we generated
                # This is [num_presteps + num_steps: num_presteps + num_steps + n_frames]

                original_gesture_sequence = full_gesture_sequence[:, model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames:model.diffusion.num_clean_frames + model.diffusion.num_denoise_frames + evaluation_length, :]
                
                # Extract only the original feature dimensions if richer features were used
                original_gesture_sequence = original_gesture_sequence[..., :model.original_pose_features_per_frame]

                if val_loader.dataset.loading_encoded_data:
                    original_gesture_sequence = pose_encoder.decode(original_gesture_sequence)

                denormalized_original_gesture_sequence = skeleton.denormalize_poses(original_gesture_sequence)
                world_positions = skeleton.calculate_world_positions(denormalized_original_gesture_sequence)
                normalized_original_world_positions = skeleton.normalize_world_positions(world_positions)

            # Push real samples for this iteration
            embeddingSpaceEvaluator.push_real_samples(normalized_original_world_positions.float())
            
            # For bootstrap resampling
            if bootstrap_samples > 0:
                all_real_samples.append(normalized_original_world_positions.float().cpu())
        
        # Calculate the main FGD score using all pooled samples
        frechet_distance_raw = embeddingSpaceEvaluator.get_fgd(use_feat_space=False) if calculate_raw_frechet_distance else None
        frechet_distance_feat_space = embeddingSpaceEvaluator.get_fgd(use_feat_space=True)
        
        # Perform bootstrap resampling if requested
        bootstrap_results = None
        if bootstrap_samples > 0:
            bootstrap_results = perform_bootstrap_resampling(
                all_gen_samples, 
                all_real_samples, 
                embeddingSpaceEvaluator.net, 
                bootstrap_samples,
                device
            )
            
            print(f"Bootstrap results: Mean FGD = {bootstrap_results['mean']:.4f}, 95% CI: [{bootstrap_results['ci_low']:.4f}, {bootstrap_results['ci_high']:.4f}]")
        
        # Restore dataloader settings
        val_loader.dataset.seq_length = old_sequence_length
        val_loader.dataset.chunk_size = val_loader.dataset.seq_length + val_loader.dataset.seed_length
        val_loader.dataset._create_offset_tensors()
        val_loader.dataset.valid_starts = old_valid_starts
        val_loader.dataset.batch_size = old_batch_size

        # Set the model back to training mode
        model.train()

        print(f"Frechet distance in feature space: {frechet_distance_feat_space}, Frechet raw distance: {frechet_distance_raw}")

        if bootstrap_results:
            return frechet_distance_feat_space, frechet_distance_raw, bootstrap_results
        else:
            return frechet_distance_feat_space, frechet_distance_raw
    
def evaluate_frechet_gesture_distance_normal_diffusion(
        model: ContinuousMotionModel,
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        calculate_raw_frechet_distance = True,
        samples_per_iteration = 8192,
        num_iterations = 1,  # Number of iterations to pool samples
        bootstrap_samples = 0  # Number of bootstrap samples for confidence intervals (0 disables)
        ) -> None:
    with torch.no_grad():

        # first we need to create the embedding space evaluator
        embeddingSpaceEvaluator = EmbeddingSpaceEvaluator(
            embed_net_path  = f"utils/evaluation/FGD/models/fgd_model_{evaluation_length}.pth",
            n_frames        = evaluation_length,
            device          = device,
            pose_dim        = len(val_loader.dataset.skeleton.target_joints) * 3
        )
            
        old_batch_size = val_loader.dataset.batch_size
        
        skeleton: Skeleton = val_loader.dataset.skeleton
        pose_encoder = model.pose_encoder
        
        # Storage for bootstrap resampling
        all_gen_samples = []
        all_real_samples = []

        for iteration in range(num_iterations):
            val_loader.dataset.batch_size = samples_per_iteration
            val_loader.dataset.reshuffle()
            
            print(f"Processing iteration {iteration+1}/{num_iterations}")

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Get data from dataloader
                gesture_sequence, gesture_seed, audio_features, main_agent_id_one_hot, finger_availability = [
                    item.squeeze(0).to(device) for item in next(iter(val_loader))
                ]
                
                # Generate from pure noise
                denoised_gesture_sequence = torch.randn((samples_per_iteration, model.gesture_length, model.pose_features_per_frame), device=device)
                
                # Run diffusion process
                for timestep in range(model.diffusion.number_of_timesteps-1, -1, -1):
                    noisy_gesture_sequence = model.diffusion.forward(denoised_gesture_sequence, timestep)
                    timestep_tensor = torch.tensor([timestep], dtype=torch.int64, device=device).expand(samples_per_iteration)
                    
                    denoised_gesture_sequence = model.forward(
                        timestep=timestep_tensor,
                        one_hot_style=main_agent_id_one_hot,
                        audio_features=audio_features,
                        noisy_gesture_sequence=noisy_gesture_sequence,
                        seed_gesture_sequence=gesture_seed,
                        finger_availability=finger_availability
                    )
                
                # Extract evaluation frames
                result_tensor = denoised_gesture_sequence[:, -evaluation_length:, :]

                # Extract only the original feature dimensions if richer features were used
                result_tensor = result_tensor[:, :, :model.original_pose_features_per_frame]
                
                # Decode if needed
                if pose_encoder is not None:
                    output = pose_encoder.decode(result_tensor)
                else:
                    output = result_tensor
                    
                # Transform to world positions
                denormalized_output = skeleton.denormalize_poses(output)
                world_positions = skeleton.calculate_world_positions(denormalized_output)
                normalized_world_positions = skeleton.normalize_world_positions(world_positions)

            # Push generated samples for this iteration
            embeddingSpaceEvaluator.push_generated_samples(normalized_world_positions.float())
                
            # For bootstrap resampling
            if bootstrap_samples > 0:
                all_gen_samples.append(normalized_world_positions.float().cpu())

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Process real samples for this iteration
                original_gesture_sequence = gesture_sequence[:, -evaluation_length:, :]

                # Extract only the original feature dimensions if richer features were used
                original_gesture_sequence = original_gesture_sequence[:, :, :model.original_pose_features_per_frame]
                
                if val_loader.dataset.loading_encoded_data:
                    original_gesture_sequence = pose_encoder.decode(original_gesture_sequence)

                denormalized_original_gesture_sequence = skeleton.denormalize_poses(original_gesture_sequence)
                world_positions = skeleton.calculate_world_positions(denormalized_original_gesture_sequence)
                normalized_original_world_positions = skeleton.normalize_world_positions(world_positions)

            # Push real samples for this iteration
            embeddingSpaceEvaluator.push_real_samples(normalized_original_world_positions.float())
            
            # For bootstrap resampling
            if bootstrap_samples > 0:
                all_real_samples.append(normalized_original_world_positions.float().cpu())

        # Calculate the main FGD score using all pooled samples
        frechet_distance_raw = embeddingSpaceEvaluator.get_fgd(use_feat_space=False) if calculate_raw_frechet_distance else None
        frechet_distance_feat_space = embeddingSpaceEvaluator.get_fgd(use_feat_space=True)
        
        # Perform bootstrap resampling if requested
        bootstrap_results = None
        if bootstrap_samples > 0:
            bootstrap_results = perform_bootstrap_resampling(
                all_gen_samples, 
                all_real_samples, 
                embeddingSpaceEvaluator.net, 
                bootstrap_samples,
                device
            )
            
            print(f"Bootstrap results: Mean FGD = {bootstrap_results['mean']:.4f}, 95% CI: [{bootstrap_results['ci_low']:.4f}, {bootstrap_results['ci_high']:.4f}]")
        
        # Restore original batch size
        val_loader.dataset.batch_size = old_batch_size

        # Set the model back to training mode
        model.train()

        print(f"Frechet distance in feature space: {frechet_distance_feat_space}, Frechet raw distance: {frechet_distance_raw}")

        if bootstrap_results:
            return frechet_distance_feat_space, frechet_distance_raw, bootstrap_results
        else:
            return frechet_distance_feat_space, frechet_distance_raw
    
def perform_bootstrap_resampling(gen_samples_list, real_samples_list, embed_net, n_bootstrap, device):
    print("Performing bootstrap resampling...")
    
    # Concatenate all samples
    all_gen = torch.cat(gen_samples_list, dim=0)
    all_real = torch.cat(real_samples_list, dim=0)
    
    total_samples = all_gen.shape[0]
    bootstrap_fgds = []
    
    for i in tqdm(range(n_bootstrap)):
        # Create a temporary evaluator for this bootstrap sample
        temp_evaluator = EmbeddingSpaceEvaluator(
            embed_net_path = None,  # We'll set the network directly
            n_frames = all_gen.shape[1],
            device = device,
            pose_dim = all_gen.shape[2] // 3
        )
        temp_evaluator.net = embed_net  # Use the same network
        
        # Sample with replacement
        indices = torch.randint(0, total_samples, (total_samples,))
        bootstrap_gen = all_gen[indices].to(device)
        bootstrap_real = all_real[indices].to(device)
        
        # Push samples and calculate FGD
        temp_evaluator.push_generated_samples(bootstrap_gen)
        temp_evaluator.push_real_samples(bootstrap_real)
        bootstrap_fgd = temp_evaluator.get_fgd(use_feat_space=True)
        bootstrap_fgds.append(bootstrap_fgd)
    
    # Calculate statistics
    bootstrap_fgds = np.array(bootstrap_fgds)
    mean = np.mean(bootstrap_fgds)
    std = np.std(bootstrap_fgds)
    ci_low, ci_high = np.percentile(bootstrap_fgds, [2.5, 97.5])  # 95% CI
    
    return {
        "bootstrap_fgds": bootstrap_fgds.tolist(),
        "mean": float(mean),
        "std": float(std),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }

def evaluate_frechet_gesture_distance(
        model: ContinuousMotionModel,
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        calculate_raw_frechet_distance = True,
        samples_per_iteration = 8192,
        num_iterations = 1,  # Number of iterations to pool samples
        bootstrap_samples = 0  # Number of bootstrap samples for confidence intervals (0 disables)
        ):

    if isinstance(model.diffusion, SlidingDiffusion):
        return evaluate_frechet_gesture_distance_sliding_diffusion(
            model, 
            val_loader, 
            device, 
            evaluation_length, 
            calculate_raw_frechet_distance,
            samples_per_iteration, 
            num_iterations,
            bootstrap_samples
        )
    elif isinstance(model.diffusion, NormalDiffusion) or isinstance(model.diffusion, OutpaintDiffusion):
        return evaluate_frechet_gesture_distance_normal_diffusion(
            model, 
            val_loader, 
            device, 
            evaluation_length, 
            calculate_raw_frechet_distance,
            samples_per_iteration, 
            num_iterations,
            bootstrap_samples
        )