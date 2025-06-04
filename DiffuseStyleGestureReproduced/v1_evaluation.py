import torch
from v1_model import ContinuousMotionModel
from torch.utils.data import DataLoader
from utils.animation.skeleton import Skeleton
from utils.evaluation.FGD.embedding_space_evaluator import EmbeddingSpaceEvaluator

def evaluate_frechet_gesture_distance(
        model: ContinuousMotionModel, 
        val_loader: DataLoader, 
        device: torch.device,
        evaluation_length = 30, 
        num_samples = 100,
        calculate_raw_frechet_distance = True) -> None:
    with torch.no_grad():

        n_frames = evaluation_length

        # first we need to create the embedding space evaluator
        embeddingSpaceEvaluator = EmbeddingSpaceEvaluator(
            embed_net_path  = f"utils/evaluation/FGD/models/fgd_model_{evaluation_length}.pth",
            n_frames        = evaluation_length,
            device          = device,
            pose_dim        = len(val_loader.dataset.skeleton.target_joints) * 3
        )

        batch_size = num_samples

        # We need to retrieve sequences of a certain length from the dataloader
        # This is pretty hacky, but allows us to just reuse the validation dataloader
        # TODO: Find a better way to do this that is compatible with other datasets as well
        old_batch_size = val_loader.dataset.batch_size
        val_loader.dataset.batch_size = batch_size
        
        old_valid_starts = val_loader.dataset.valid_starts

        old_sequence_length = val_loader.dataset.seq_length
        val_loader.dataset.seq_length = model.num_of_pre_timestep_frames + 2 * model.num_of_timestep_frames + model.num_of_post_timestep_frames + n_frames
        val_loader.dataset.chunk_size = val_loader.dataset.seq_length + val_loader.dataset.seed_length
        val_loader.dataset._create_offset_tensors()
        val_loader.dataset.valid_starts = val_loader.dataset._find_valid_starting_points()
        val_loader.dataset.reshuffle()
        
        skeleton: Skeleton = val_loader.dataset.skeleton

        pose_encoder = model.pose_encoder

        # We load all the data we need. even when we have generated everything. 
        # That is we need the starting point (The "seed"), which gives us the starting clean gesture from which we continue our generation. (num_presteps)
        # We also need enough data to move past the the noising area (num_of_timestep_frames) + the initial noising area (num_of_timestep_frames)
        # We also need enough data to have n_frames of data generated when we are done. (n_frames)
        # i.e. sequence length retrieved by val_loader should be num_presteps + 2 * num_of_timestep_frames + num_of_post_timestep_frames + n_frames
        full_gesture_sequence, _, full_audio_features, main_agent_id_one_hot = [
            item.squeeze(0).to(device) for item in next(iter(val_loader)) # TODO: Maybe this always returns the same item? Hopefully not
        ]

        feature_dim = full_gesture_sequence.shape[2]
        encoded_feature_dim = pose_encoder.z_dim if pose_encoder is not None else feature_dim

        # We prepare a result tensor to store the generated data
        # This is of shape (bs, num_timestep_frames + n_frames, feature_dim)
        result_tensor = torch.zeros(
            (batch_size, model.num_of_timestep_frames + n_frames, encoded_feature_dim),
            device=device,
        )

        # Cast to float precision (TODO: Why are we doing this?)
        full_gesture_sequence = full_gesture_sequence.float()

        if pose_encoder is not None:
            full_encoded_gesture_sequence = pose_encoder.encode(full_gesture_sequence)
        else:
            full_encoded_gesture_sequence = full_gesture_sequence

        # Now we need to cut the sequence to the right length.
        # We need to cut the sequence to the length of the pre-timestep frames + timestep frames + post-timestep frames
        # The audio features should be cut in the same way
        current_encoded_gesture_sequence = full_encoded_gesture_sequence[:, 0:model.num_of_pre_timestep_frames + model.num_of_timestep_frames + model.num_of_post_timestep_frames,:]

        for i in range(n_frames + model.num_of_timestep_frames):
            # We also need to cut out the relevant audio features. As these are not predicted by the model we cut them out of the full audio features
            # at every iteration
            current_audio_features = full_audio_features[:, i:i + model.num_of_pre_timestep_frames + model.num_of_timestep_frames + model.num_of_post_timestep_frames, :]

            for stacking_level in range(model.max_timestep_stacking_level):                
                # We apply noise to the gesture sequence at every iteration because we predict the clean image at every step.
                starting_point_encoded_gesture_sequence = current_encoded_gesture_sequence
                
                noisy_gesture_sequence = model.diffusion.forward(current_encoded_gesture_sequence, stacking_level)

                current_encoded_gesture_sequence = model(
                    time_step_stacking_level    = stacking_level,
                    one_hot_style               = main_agent_id_one_hot,
                    audio_features              = current_audio_features, 
                    noisy_gesture_sequence      = noisy_gesture_sequence
                )

            if model.predict_full_duration:
                # The model predicts the whole sequence, but only the last frames were noised to begin with. We are essentially doing infill-diffusion.
                # We need to copy the original data to the result tensor, so that we can use it as a starting point for the next iteration.
                # We copy the original real data on top of the area that was not denoised, in order to avoid the model degenerating over time.
                current_encoded_gesture_sequence[:, :model.num_of_pre_timestep_frames, :] = starting_point_encoded_gesture_sequence[:, :model.num_of_pre_timestep_frames, :]

            copy_index = model.num_of_timestep_frames if model.predict_full_duration else 0 # TODO: Make sure this is the correct frame being copied. I am slighty worried we are copying one frame to far forward (i.e. still not fully denoised)

            # copy the newest predicted frame to the result tensor
            result_tensor[:, i, :] = current_encoded_gesture_sequence[:, copy_index, :]

            # TODO: Does this actually belong at the top of the loop? I am not sure
            # Shift the gesture_sequence by one frame
            current_encoded_gesture_sequence = torch.roll(noisy_gesture_sequence, shifts=-1, dims=1)
            # Clear the last frame (this will be filled with noise in the next iteration by the diffusion model)
            current_encoded_gesture_sequence[:, -1] = torch.zeros_like(current_encoded_gesture_sequence[:, -1])
        
        # The first num_presteps frames of the prediction had access to the original gesture sequence, so in a sense they 'cheated'. For this reason, we discard them.
        # The remaining n_frames result tensor are the final generated gestures.
        result_tensor = result_tensor[:, model.num_of_pre_timestep_frames:, :]

        # Now we use the autoencoder to decode the result tensor from the low dimensional latent space to the original feature space.
        if pose_encoder is not None:
            output = pose_encoder.decode(result_tensor)
        else:
            output = result_tensor

        # We want to perform FGD calculations on the world pose space. AS such we need to calculate this.

        # First we need to denormalized the output tensor to recover the original feature space (the rotation matrices)
        denormalized_output = skeleton.denormalize_poses(output)

        # Now we want to calcualte world postions from the denormalized output tensor
        world_positions = skeleton.calculate_world_positions(denormalized_output)

        # Now we want to z-normalize the world positions to get the final output tensor
        normalized_world_positions = skeleton.normalize_world_positions(world_positions)

        embeddingSpaceEvaluator.push_generated_samples(normalized_world_positions)

        ########################################

        # We also need to acquire the original gesture sequence in world space
        # We need to cut out the area of the original gesture sequence that is equivalent to the area we generated
        # This is [num_presteps + num_steps: num_presteps + num_steps + n_frames]
        original_gesture_sequence = full_gesture_sequence[:, model.num_of_pre_timestep_frames + model.num_of_timestep_frames:model.num_of_pre_timestep_frames + model.num_of_timestep_frames + n_frames, :]
        
        # We need to denormalize the original gesture sequence to recover the original feature space (the rotation matrices)
        denormalized_original_gesture_sequence = skeleton.denormalize_poses(original_gesture_sequence)
        
        # Now we want to calcualte world postions from the denormalized original gesture sequence
        world_positions = skeleton.calculate_world_positions(denormalized_original_gesture_sequence)
        # Now we want to z-normalize the world positions to get the final output tensor
        
        normalized_original_world_positions = skeleton.normalize_world_positions(world_positions)

        embeddingSpaceEvaluator.push_real_samples(normalized_original_world_positions)

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