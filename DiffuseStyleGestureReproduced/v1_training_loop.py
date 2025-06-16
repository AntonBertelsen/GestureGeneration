import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast
from tqdm import tqdm
import time
import v1_evaluation
from IPython.display import clear_output
from pose_encoder.advanced_pose_encoder import AdvancedPoseEncoder
from v1_model import ContinuousMotionModel
from torch.utils.data import DataLoader
from utils.utils import get_latest_model_path
import wandb
import yaml
import utils.animation.visualisation.new.animation_visualisation as animation_visualisation

def train(
        experiment_collection_name: str, # Name of the gruope of experiments, this run is a part of
        device: torch.device,
        model: ContinuousMotionModel,
        training_loader: DataLoader,
        val_loader: DataLoader, 
        num_epochs: int,
        run = None, # A wandb.run object to log the training process
        wandb_config: dict = None, # A wandb.config from a yml or dict. This is logged hyper parameters for wandb used for hyperparameter tuning / sweeps. Given to be attached to the run, and extended with given, but not sweeping hyperparams
        model_checkpoint_dir: str = None, # dir of the model checkpoint to load from
        upload_model_check_point: bool = False, # should upload the model checkpoint to wandb
        learning_rate = 0.0003,
        reconstruction_loss_weight = 1.0, # Weight for the reconstruction loss
        variance_loss_weight = 0.1,
        velocity_loss_weight = 0.1,
        acceleration_loss_weight = 0.1,
        jerk_loss_weight = 0.0,
        latent_space_loss_weight = 2.0,
        category_weighting: dict[str, float] = None,
        frame_weighting_segments_info: list[(float, float, float)] = [], # This is a list of tuples, where each tuple contains the start and end of a segment, and the end frame of the segment. This is used to create a frame weighting vector that weights the loss for each frame. Since we are essentially doing infill diffusion, we want to bias the the loss for the frames that are not masked out.
        visualize_step: int = 200, # How often to print profiling stats
        continue_from_checkpoint: str = None, # Path to a checkpoint to continue training from. If provided, the model will be loaded from this checkpoint and training will continue from there.
        should_visualize_training_progress = True, # Whether to display the training progress in a Jupyter notebook
    ):

    current_model_name = f"{experiment_collection_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    # dictionaries to keep track of the losses during training and validation so we can plot them later.
    # The first element of the tuple is all losses, the second element is the averaged loss over the last visualize_step.
    train_loss_rec = {'loss':                       ([],[]),
                      'reconstruction_loss':        ([],[]), 
                      'encoded_latent_space_loss':  ([],[]), 
                      'variance_loss':              ([],[]), 
                      'velocity_loss':              ([],[]), 
                      'acceleration_loss':          ([],[]),
                      'jerk_loss':                  ([],[]),
                      'epoch_loss':                 ([],[])}
    
    val_loss_rec = {'loss':                         ([],[]),
                    'reconstruction_loss':          ([],[]),
                    'encoded_latent_space_loss':    ([],[]), 
                    'variance_loss':                ([],[]), 
                    'velocity_loss':                ([],[]), 
                    'acceleration_loss':            ([],[]),
                    'jerk_loss':                    ([],[]),}
                    
    frechet_distance_rec = {'encoded':              [], 
                            'raw':                  []}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    torch.set_float32_matmul_precision('high')
    
    hyper_parameters = {
        "experiment_collection_name": experiment_collection_name,
        "device": device.type,
        "model_name": current_model_name,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "reconstruction_loss_weight": reconstruction_loss_weight,
        "variance_loss_weight": variance_loss_weight,
        "velocity_loss_weight": velocity_loss_weight,
        "acceleration_loss_weight": acceleration_loss_weight,
        "latent_space_loss_weight": latent_space_loss_weight,
        "category_weighting": category_weighting,
        "frame_weighting_segments_info": frame_weighting_segments_info,
        "batch_size": training_loader.dataset.batch_size,
        "dataset_type": training_loader.dataset.__class__.__name__,
        "model": model.get_WnB_config_specs()
    }
    # If the wandb_config is provided, I update it with the hyper parameters of the model and training process.
    # This is to ensure that the hyper parameters are logged to wandb (W&B) for tracking and visualization.
    # Doing it in this way also allows us to use configs for hyperparameter sweeps.
    if wandb_config is not None:
        wandb_config.update(hyper_parameters, allow_val_change=True)  # <- Important: allows adding/updating keys

    # category_weighting data used to weight bones differently in the loss function based on assigned category weightings.
    if category_weighting is not None:
        if training_loader.dataset.loading_encoded_data:
            # If we are training using encoded data, we cannot weight individual bones, since we dont have access to the full pose, only the encoded latent space.
            if isinstance(model.pose_encoder, AdvancedPoseEncoder):
                # If we are training using an advanced pose encoder, we construct the bone weighting vector we weight according to component specifications from the advanced pose encoder.
                category_weighting_vector = model.pose_encoder.construct_component_weighting_vector(category_weighting)
        else:
            # If we are not loading encoded data we can weight individual bones based on the category weighting.
            category_weighting_vector = training_loader.dataset.skeleton.construct_bone_weighting_vector(category_weighting)
    else:
        category_weighting_vector = None
    
    # frame weighting vector used to weight the loss for each frame based on the segments info provided.
    frame_weighting = construct_frame_weighting_vector(frame_weighting_segments_info) if frame_weighting_segments_info else None

    # Initialize start_epoch to 0 (will be updated if loading from checkpoint)
    start_epoch = 0

    # If a model checkpoint directory is provided we load the model from the checkpoint.
    if model_checkpoint_dir is not None and continue_from_checkpoint is not None:
        
        # If the continue_from_checkpoint is "latest", we find the latest checkpoint in the model_checkpoint_dir (otherwise we use the provided path).
        if continue_from_checkpoint == "latest":
            continue_from_checkpoint, checkpoint_folder = get_latest_model_path(model_checkpoint_dir, return_folder = True)

        print (f"Continuing training from checkpoint: {continue_from_checkpoint}")

        # Load the model from the checkpoint
        model, reached_epoch = resume_training_from_checkpoint(
            optimizer               = optimizer,
            train_loss_rec          = train_loss_rec,
            val_loss_rec            = val_loss_rec,
            frechet_distance_rec    = frechet_distance_rec,
            checkpoint_path         = continue_from_checkpoint,
            device                  = device
        )

        start_epoch = reached_epoch + 1  # Start from the next epoch after the checkpoint
        print(f"Resuming training from epoch {start_epoch}...")

        # Update the current model name to reflect that we are continuing from a checkpoint
        current_model_name = checkpoint_folder

        # Update the hyper parameters with the model's configuration and the checkpoint path.
        hyper_parameters['continue_from_checkpoint'] = continue_from_checkpoint
        hyper_parameters['model'] = model.get_WnB_config_specs()

    # I then move the model to the device that is being used and put in traning mode
    model = model.to(device)

    # We compile the model using cudagraphs for performance optimization.
    # This is a PyTorch feature that allows us to optimize the model for faster training.
    # Note: cudagraphs is only available on CUDA devices, so we check if the device is CUDA.
    # There are other backends available, but cudagraphs is the only one I could get working on Windows.
    # In Andrej Kaparthy's gpt video he uses a different backend.
    if device.type == 'cuda':
        model = torch.compile(model)

    epoch_length = len(training_loader)

    # display_url = animation_visualisation.init_visualization(display=False)  # Initialize the animation visualization, but don't display it yet

    # Main training loop, where we iterate over the number of epochs and the training data.
    for epoch in range(start_epoch, num_epochs):
        # I set the model to training mode
        model.train()

        # reshuffle the dataset from the dataloaders at the start of each epoch.
        training_loader.dataset.reshuffle()
        val_loader.dataset.reshuffle()

        # During the epoch, all the data items are iterated over.        
        progress_bar = tqdm(training_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        for i, batch_data in enumerate(progress_bar):
            
            # Zero gradients before forward pass
            optimizer.zero_grad()

            # We are using our own batching mechanism in the dataset to to avoid having to use a collate function.
            # As such, each item contains a full batch, but we need to handle the extra dimension 
            # from batch_size=1 from the dataloader.
            gesture_sequence, _, audio_features, main_agent_id_one_hot = [
                item.squeeze(0).to(device) for item in batch_data
            ]

            output, encoded_gesture_sequence, encoded_output, noisy_gesture_sequence = model.generate(
                gesture_sequence            = gesture_sequence,
                audio_features              = audio_features,
                main_agent_id_one_hot       = main_agent_id_one_hot,
                gesture_sequence_is_encoded = training_loader.dataset.loading_encoded_data
            )

            # Perform loss calculations.
            total_loss = calculate_loss(
                pred                        = output,
                gt                          = gesture_sequence,
                encoded_pred                = encoded_output,
                encoded_gt                  = encoded_gesture_sequence,
                reconstruction_loss_weight  = reconstruction_loss_weight,
                latent_space_loss_weight    = latent_space_loss_weight,
                variance_loss_weight        = variance_loss_weight,
                velocity_loss_weight        = velocity_loss_weight,
                acceleration_loss_weight    = acceleration_loss_weight,
                jerk_loss_weight            = jerk_loss_weight,
                bone_weighting_vector       = category_weighting_vector,
                frame_weighting_vector      = frame_weighting,
                loss_recorder               = train_loss_rec
            )

            # Backward pass to compute gradients                        
            total_loss.backward()

            # Take an optimization step
            optimizer.step()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({f"\033[91mTraining loss": f"{total_loss.item()}\033[0m", 
                                      f"\033[92mLast epoch loss": f"{train_loss_rec['epoch_loss'][0][-1] if train_loss_rec['epoch_loss'][0] else 0}\033[0m",
                                      f"\033[94mFrechet distance": f"{frechet_distance_rec['encoded'][-1] if frechet_distance_rec['encoded'] else 0}\033[0m"})
            
            # log the loss to wandb (W&B)
            # if run is not None: 
            #     step = i + epoch * epoch_length
            #     run.log({"total_loss": total_loss.item()}, step=step)

            # Visualization
            if i % visualize_step == 0:
                
                # Calculate averaged losses over the last visualize_step and store in the loss records.
                for key in train_loss_rec.keys():
                    train_loss_rec[key][1].append(np.mean(train_loss_rec[key][0][-visualize_step:]))
                
                # Log the results to wandb (W&B) if a run is provided.
                if run is not None:
                    run.log({
                        "total_loss": total_loss.item(),
                        "reconstruction_loss": train_loss_rec['reconstruction_loss'][1][-1],
                        "encoded_latent_space_loss": train_loss_rec['encoded_latent_space_loss'][1][-1],
                        "variance_loss": train_loss_rec['variance_loss'][1][-1],
                        "velocity_loss": train_loss_rec['velocity_loss'][1][-1],
                        "acceleration_loss": train_loss_rec['acceleration_loss'][1][-1],
                        "jerk_loss": train_loss_rec['jerk_loss'][1][-1]
                    }, step = i + epoch * epoch_length)

                if not is_running_on_slurm() and should_visualize_training_progress:
                    visualize_training_progress(
                        full_gesture_sequence               = gesture_sequence,
                        full_denoised_gesture_sequence      = output,
                        encoded_gesture_sequence            = encoded_gesture_sequence,
                        encoded_denoised_gesture_sequence   = encoded_output,
                        noisy_gesture_sequence              = noisy_gesture_sequence,
                        using_pose_encoder                  = model.pose_encoder is not None and not training_loader.dataset.loading_encoded_data,
                        train_loss_rec                      = train_loss_rec,
                        val_loss_rec                        = val_loss_rec,
                        frechet_distance_rec                = frechet_distance_rec,
                        visualize_step                      = visualize_step,
                        frame_weighting                     = frame_weighting
                    )
                    # print("Visualize results at", display_url)  # Display the animation visualization URL
                
        ########################################################################
        # End of epoch
        ########################################################################

        if epoch % 10 == 0:
            save_model_checkpoint(
                model                   = model,
                checkpoint_dir          = model_checkpoint_dir,
                model_name              = current_model_name,
                epoch                   = epoch,
                hyper_parameters        = hyper_parameters,
                optimizer               = optimizer,
                train_loss_rec          = train_loss_rec,
                val_loss_rec            = val_loss_rec,
                frechet_distance_rec    = frechet_distance_rec,
                upload                  = upload_model_check_point,
                run                     = run
            )

        train_loss_rec['epoch_loss'][0].append(np.mean(train_loss_rec['loss'][0][-epoch_length:]))

        # At the end of the epoch, we evaluate the model on the validation set.
        model.eval()
        
        # We calculate the Frechet distance between the generated and true gestures.
        # This is a measure of how similar the distribution of the generated gestures is to the distribution of the true gestures.
        frechet_distance, _, = v1_evaluation.evaluate_frechet_gesture_distance(
            model             = model,
            val_loader        = val_loader,
            device            = device,
            evaluation_length = 30,
            num_samples       = 8192,
            calculate_raw_frechet_distance = False
        )

        # Log the Frechet distance so we can see how it changes over time.
        frechet_distance_rec['encoded'].append(frechet_distance)

        # We run the model on the validation set to calculate the validation loss.
        with torch.no_grad():
            for val_batch in val_loader:
                # We are using our own batching mechanism in the dataset to to avoid having to use a collate function.
                gesture_sequence, _, audio_features, main_agent_id_one_hot = [
                    item.squeeze(0).to(device) for item in val_batch
                ]
                
                output, encoded_gesture_sequence, encoded_output, noisy_gesture_sequence = model.generate(
                    gesture_sequence            = gesture_sequence,
                    audio_features              = audio_features,
                    main_agent_id_one_hot       = main_agent_id_one_hot,
                    gesture_sequence_is_encoded = val_loader.dataset.loading_encoded_data
                )

                # Perform loss calculations.
                total_loss = calculate_loss(
                    pred                        = output,
                    gt                          = gesture_sequence,
                    encoded_pred                = encoded_output,
                    encoded_gt                  = encoded_gesture_sequence,
                    reconstruction_loss_weight  = reconstruction_loss_weight,
                    latent_space_loss_weight    = latent_space_loss_weight,
                    variance_loss_weight        = variance_loss_weight,
                    velocity_loss_weight        = velocity_loss_weight,
                    acceleration_loss_weight    = acceleration_loss_weight,
                    jerk_loss_weight            = jerk_loss_weight,
                    bone_weighting_vector       = category_weighting_vector,
                    frame_weighting_vector      = frame_weighting,
                    loss_recorder               = val_loss_rec
                )
        
        # Calculate averaged losses over the last validation 
        for key in val_loss_rec.keys():
            val_loss_rec[key][1].append(np.mean(train_loss_rec[key][0]))
            # clear the losses for the next epoch
            val_loss_rec[key][0].clear()
        
        # Log the validation losses and frechet distance to wandb at the end of each epoch
        if run is not None:
            # Log all validation loss components
            print("LOGGING VALUDATION!!! TO W&B")
            run.log({
                "validation/total_loss": val_loss_rec['loss'][1][-1] if val_loss_rec['loss'][1] else 0,
                "validation/reconstruction_loss": val_loss_rec['reconstruction_loss'][1][-1] if val_loss_rec['reconstruction_loss'][1] else 0,
                "validation/encoded_latent_space_loss": val_loss_rec['encoded_latent_space_loss'][1][-1] if val_loss_rec['encoded_latent_space_loss'][1] else 0,
                "validation/variance_loss": val_loss_rec['variance_loss'][1][-1] if val_loss_rec['variance_loss'][1] else 0,
                "validation/velocity_loss": val_loss_rec['velocity_loss'][1][-1] if val_loss_rec['velocity_loss'][1] else 0,
                "validation/acceleration_loss": val_loss_rec['acceleration_loss'][1][-1] if val_loss_rec['acceleration_loss'][1] else 0,
                "validation/jerk_loss": val_loss_rec['jerk_loss'][1][-1] if val_loss_rec['jerk_loss'][1] else 0,
                "validation/frechet_distance": frechet_distance_rec['encoded'][-1] if frechet_distance_rec['encoded'] else 0,
            }, step=(epoch+1) * epoch_length)

    # close the wandb (W&B) run
    if run is not None: 
        run.finish() 
    
    # Save the final model checkpoint
    save_model_checkpoint( 
        model                   = model,
        checkpoint_dir          = model_checkpoint_dir,
        model_name              = current_model_name + "_final",
        epoch                   = num_epochs-1,
        hyper_parameters        = hyper_parameters,
        optimizer               = optimizer,
        train_loss_rec          = train_loss_rec,
        val_loss_rec            = val_loss_rec,
        frechet_distance_rec    = frechet_distance_rec,
        upload                  = upload_model_check_point,
        run                     = run
    )

    # Return the trained model
    return model

# Full loss function that combines all the individual losses.
def calculate_loss(pred, gt, 
         encoded_pred, encoded_gt,
         reconstruction_loss_weight,
         latent_space_loss_weight, 
         variance_loss_weight, 
         velocity_loss_weight, 
         acceleration_loss_weight,
         jerk_loss_weight,
         bone_weighting_vector, # This is a vector that weights the loss for each bone category.
         frame_weighting_vector, # This is a vector that weights the loss for each frame. Since we are essentially doing infill diffusion, we want to bias the the loss for the frames that are not masked out.
         loss_recorder = None # This is a dictionary that keeps track of the losses during training and validation so we can plot them later.
    ):
    device = pred.device
    with autocast(device_type=device.type, dtype=torch.bfloat16):

        # Use the casted target for all loss calculations
        recon_l = nn.HuberLoss(reduction="none")(pred, gt) * reconstruction_loss_weight

        encoded_latent_space_l = nn.HuberLoss(reduction="none")(encoded_gt, encoded_pred) * latent_space_loss_weight
        variance_l = variance_loss(pred, gt) * variance_loss_weight 
        velocity_l = velocity_loss(pred, gt) * velocity_loss_weight 
        acceleration_l = acceleration_loss(pred, gt) * acceleration_loss_weight
        jerk_l = jerk_loss(pred, gt) * jerk_loss_weight

        # Apply the bone category weighting (Can't be applied to encoded latent space loss, since it is not a tensor of the same shape as the other losses)
        if bone_weighting_vector is not None:
            recon_l = bone_weighting_vector * recon_l
            variance_l = bone_weighting_vector * variance_l
            velocity_l = bone_weighting_vector * velocity_l
            acceleration_l = bone_weighting_vector * acceleration_l
            jerk_l = bone_weighting_vector * jerk_l

        # Apply the frame weighting vector (Can't be applied to variance loss, since variance is calculated over time, and not per frame)
        if frame_weighting_vector is not None:
            # Unsqueeze the frame weighting vector to match the shape of the losses
            frame_weighting_vector = frame_weighting_vector.to(device).unsqueeze(0).unsqueeze(2)  # Shape: (1, num_frames, 1)
            
            recon_l = frame_weighting_vector * recon_l
            encoded_latent_space_l = frame_weighting_vector * encoded_latent_space_l
            velocity_l = frame_weighting_vector[:,:-1,:] * velocity_l # Velocity used finite difference to determine the derivative, so we need to remove the last frame for lengths to match
            acceleration_l = frame_weighting_vector[:,:-2,:] * acceleration_l # Acceleration used finite difference to determine the second derivative, so we need to remove the last two frames for lengths to match
            jerk_l = frame_weighting_vector[:,:-3,:] * jerk_l # Jerk used finite difference to determine the third derivative, so we need to remove the last three frames for lengths to match

        # Now we find the mean over the batch and time dimensions
        recon_l = recon_l.mean()
        encoded_latent_space_l = encoded_latent_space_l.mean()
        variance_l = variance_l.mean()
        velocity_l = velocity_l.mean()
        acceleration_l = acceleration_l.mean()
        jerk_l = jerk_l.mean()

        # Combine all losses
        total_loss = recon_l + encoded_latent_space_l + variance_l + velocity_l + acceleration_l + jerk_l

        # If a loss_recorder is provided, we will record the losses in it.
        if loss_recorder is not None:
            loss_recorder['loss'][0].append(total_loss.item())
            loss_recorder['reconstruction_loss'][0].append(recon_l.item())
            loss_recorder['encoded_latent_space_loss'][0].append(encoded_latent_space_l.item())
            loss_recorder['variance_loss'][0].append(variance_l.item())
            loss_recorder['velocity_loss'][0].append(velocity_l.item())
            loss_recorder['acceleration_loss'][0].append(acceleration_l.item())
            loss_recorder['jerk_loss'][0].append(jerk_l.item())

        return total_loss

# Variance loss function to penalize lower variance in the denoised gesture compared to the true gesture.
# This is to help the model not end up prefering motions where it is mostly standing still
def variance_loss(denoised_gesture, true_gesture):
    var_pred = torch.var(denoised_gesture, dim=1, unbiased=False)  # Variance over time
    var_true = torch.var(true_gesture, dim=1, unbiased=False)

    return torch.relu(var_true - var_pred)  # Penalize lower variance only with relu

# Velocity loss function to penalize the difference in velocity between the predicted and true gesture.
def velocity_loss(pred, gt):
    vel_pred = pred[:, 1:] - pred[:, :-1]  # First-order difference
    vel_gt = gt[:, 1:] - gt[:, :-1]
    return (vel_pred - vel_gt) ** 2

# Acceleration loss function to penalize the difference in acceleration between the predicted and true gesture.
# This is to prevent jittery in the motion, and to help the model learn smoother motions, 
# since jittery motions will cause acceleration to be high. (constantly changing direction)
def acceleration_loss(pred, gt):
    vel_pred = pred[:, 1:] - pred[:, :-1]  # First-order difference
    vel_gt = gt[:, 1:] - gt[:, :-1]
    acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]  # Second-order difference
    acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
    return (acc_pred - acc_gt) ** 2

# Jerk loss function to penalize the difference in jerk between the predicted and true gesture.
# Jerk is the third derivative of the motion, and is used to prevent sudden changes in acceleration.
def jerk_loss(pred, gt):
    vel_pred = pred[:, 1:] - pred[:, :-1]  # First-order difference
    vel_gt = gt[:, 1:] - gt[:, :-1]
    acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]  # Second-order difference
    acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
    jerk_pred = acc_pred[:, 1:] - acc_pred[:, :-1]  # Third-order difference
    jerk_gt = acc_gt[:, 1:] - acc_gt[:, :-1]
    return (jerk_pred - jerk_gt) ** 2

# Encoded latent space loss function to penalize the difference between the encoded latent space and the predicted encoded latent space.
# This helps the model learn a good representation of the data in latent space. Possibly this is all that is needed, since the model is trained to predict the encoded latent space.
def encoded_latent_space_loss(pred, gt):
    # calculate the loss between the encoded latent space and the predicted encoded latent space
    return (pred - gt) ** 2

def construct_frame_weighting_vector(segments_info: list[(float, float, float)]):
    segments = []
    current_frame = 0
    for idx, (start, end, end_frame) in enumerate(segments_info):
        count = end_frame - current_frame + 1
        assert count > 0, f"End frame cannot be less than the end frame of the previous segment"
        segment = torch.linspace(start, end, count)
        segment = segment[1:]  # avoid duplicate of endpoint
        segments.append(segment)
        current_frame = end_frame

    frame_weighting_vector = torch.cat(segments)
    # Normalize to average each number of frames to 1, so that the loss is not biased by the number of frames.
    frame_weighting_vector = (frame_weighting_vector / frame_weighting_vector.sum()) * len(frame_weighting_vector)

    return frame_weighting_vector

def visualize_training_progress(
        full_gesture_sequence: torch.Tensor,
        full_denoised_gesture_sequence: torch.Tensor,
        encoded_gesture_sequence: torch.Tensor,
        encoded_denoised_gesture_sequence: torch.Tensor,
        noisy_gesture_sequence: torch.Tensor,
        using_pose_encoder: bool,
        train_loss_rec: dict,
        val_loss_rec: dict,
        frechet_distance_rec: dict,
        visualize_step: int,
        frame_weighting: torch.Tensor = None
    ):
    
    clear_output(wait=True)

    # Create a single figure with GridSpec to manage all plots
    # Height ratios for each row

    full_gesture_sequence_height = full_gesture_sequence.shape[1]
    
    if using_pose_encoder:
        height_ratios = [4, 2, full_gesture_sequence_height * 0.02]  # Loss plots, Latent space viz, Gesture viz
    else:
        height_ratios = [4, 0, full_gesture_sequence_height * 0.02]  # Loss plots, Gesture viz
    
    # Create figure with appropriate height
    fig = plt.figure(figsize=(30, 9 * 2 + full_gesture_sequence_height * 0.01))
    gs = fig.add_gridspec(3, 4, height_ratios=height_ratios)
    
    # Add overall title
    fig.suptitle("Training Progress Visualization", fontsize=24)
    
    # Row 1: Training Losses
    ax_train = fig.add_subplot(gs[0, 0])
    ax_train.plot(train_loss_rec['loss'][1], label='Total Loss', color='red')
    ax_train.plot(train_loss_rec['reconstruction_loss'][1], label='Reconstruction Loss', color='blue')
    ax_train.plot(train_loss_rec['encoded_latent_space_loss'][1], label='Encoded Latent Space Loss', color='green')
    ax_train.plot(train_loss_rec['variance_loss'][1], label='Variance Loss', color='orange')
    ax_train.plot(train_loss_rec['velocity_loss'][1], label='Velocity Loss', color='purple')
    ax_train.plot(train_loss_rec['acceleration_loss'][1], label='Acceleration Loss', color='brown')
    ax_train.plot(train_loss_rec['jerk_loss'][1], label='Jerk Loss', color='pink')
    ax_train.set_title('Losses over Training Steps', fontsize=20)
    ax_train.set_xlabel('Step')
    ax_train.set_ylabel('Loss')
    ax_train.grid(True)
    ax_train.legend()

    # Validation losses
    ax_val = fig.add_subplot(gs[0, 1])
    ax_val.plot(val_loss_rec['loss'][1], label='Total Loss', color='red')
    ax_val.plot(val_loss_rec['reconstruction_loss'][1], label='Reconstruction Loss', color='blue')
    ax_val.plot(val_loss_rec['encoded_latent_space_loss'][1], label='Encoded Latent Space Loss', color='green')
    ax_val.plot(val_loss_rec['variance_loss'][1], label='Variance Loss', color='orange')
    ax_val.plot(val_loss_rec['velocity_loss'][1], label='Velocity Loss', color='purple')
    ax_val.plot(val_loss_rec['acceleration_loss'][1], label='Acceleration Loss', color='brown')
    ax_val.plot(val_loss_rec['jerk_loss'][1], label='Jerk Loss', color='pink')
    ax_val.set_title('Validation Losses over Training Steps', fontsize=20)
    ax_val.set_xlabel('Step')
    ax_val.set_ylabel('Loss')
    ax_val.grid(True)
    ax_val.legend()

    # Train vs Val loss
    ax_comp = fig.add_subplot(gs[0, 2])
    ax_comp.plot(train_loss_rec['epoch_loss'][0], label='Training Loss', color='blue')
    ax_comp.plot(val_loss_rec['loss'][1], label='Validation Loss', color='orange')
    ax_comp.set_title('Training vs Validation Loss', fontsize=20)
    ax_comp.set_xlabel('Step')
    ax_comp.set_ylabel('Loss')
    ax_comp.grid(True)
    ax_comp.legend()

    # Frechet distance plot
    ax_frechet = fig.add_subplot(gs[0, 3])
    ax_frechet.plot(frechet_distance_rec['encoded'], label='Encoded Frechet Distance', color='blue')
    ax_frechet.set_title('Frechet Distance over Training Steps', fontsize=20)
    ax_frechet.set_xlabel('Step')
    ax_frechet.set_ylabel('Frechet Distance')
    ax_frechet.grid(True)
    ax_frechet.legend()

    cmap = 'viridis'
    vmin = -2.5
    vmax = 2.5

    # Row 2 (optional): Latent space visualization
    if using_pose_encoder:
        ax_encoded = fig.add_subplot(gs[1, 0])
        ax_encoded.set_title("Encoded Gesture", fontsize=20)
        ax_encoded.imshow(encoded_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded.set_xlabel("Time")
        ax_encoded.set_ylabel("feature")
        ax_encoded.grid(False)
        ax_encoded.axis('off')
        
        ax_encoded_diff = fig.add_subplot(gs[1, 1])
        ax_encoded_diff.set_title("Encoded Diffused Gesture", fontsize=20)
        ax_encoded_diff.imshow(noisy_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded_diff.set_xlabel("Time")
        ax_encoded_diff.set_ylabel("feature")
        ax_encoded_diff.grid(False)
        ax_encoded_diff.axis('off')
        if frame_weighting is not None:
            # Draw frame weighting as a graph on top of the encoded diffused gesture
            ax_encoded_diff.plot(48.0 - frame_weighting.cpu().numpy() * 24.0, color='red', linewidth=2, label='Frame Weighting')
            ax_encoded_diff.legend(loc='upper left')
        
        ax_encoded_denoised = fig.add_subplot(gs[1, 2])
        ax_encoded_denoised.set_title("Encoded Denoised Gesture", fontsize=20)
        ax_encoded_denoised.imshow(encoded_denoised_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded_denoised.set_xlabel("Time")
        ax_encoded_denoised.set_ylabel("feature")
        ax_encoded_denoised.grid(False)
        ax_encoded_denoised.axis('off')
        
        ax_encoded_diff_actual = fig.add_subplot(gs[1, 3])
        ax_encoded_diff_actual.set_title("Difference (Encoded Actual - Encoded Denoised)", fontsize=20)
        ax_encoded_diff_actual.imshow((encoded_gesture_sequence - encoded_denoised_gesture_sequence).to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded_diff_actual.set_xlabel("Time")
        ax_encoded_diff_actual.set_ylabel("feature")
        ax_encoded_diff_actual.grid(False)
        ax_encoded_diff_actual.axis('off')

    # Row 3: Gesture visualizations
    ax_actual = fig.add_subplot(gs[2, 0])
    ax_actual.set_title("Actual Gesture", fontsize=20)
    ax_actual.imshow(full_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax_actual.set_xlabel("Time")
    ax_actual.set_ylabel("feature")
    ax_actual.grid(False)
    ax_actual.axis('off')
    
    ax_diffused = fig.add_subplot(gs[2, 1])
    ax_diffused.set_title("Diffused Gesture", fontsize=20)
    if not using_pose_encoder:
        ax_diffused.imshow(noisy_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax_diffused.text(0.5, 0.5, "Diffused Gesture is not available\n" \
                                      "when using pose encoder.\n\n" \
                                      "The latent space of the pose encoder\n" \
                                      "is diffused instead.", 
                                      horizontalalignment='center', verticalalignment='center', transform=ax_diffused.transAxes, fontsize=16, color='red')
    
    ax_diffused.set_xlabel("Time")
    ax_diffused.set_ylabel("feature")
    ax_diffused.grid(False)
    ax_diffused.axis('off')
    
    ax_denoised = fig.add_subplot(gs[2, 2])
    ax_denoised.set_title("Denoised Gesture", fontsize=20)
    ax_denoised.imshow(full_denoised_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax_denoised.set_xlabel("Time")
    ax_denoised.set_ylabel("feature")
    ax_denoised.grid(False)
    ax_denoised.axis('off')
    
    ax_diff = fig.add_subplot(gs[2, 3])
    ax_diff.set_title("Difference (Actual - Denoised)", fontsize=20)
    ax_diff.imshow((full_gesture_sequence - full_denoised_gesture_sequence).to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax_diff.set_xlabel("Time")
    ax_diff.set_ylabel("feature")
    ax_diff.grid(False)
    ax_diff.axis('off')

    # Adjust spacing between subplots
    fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])  # rect adjusts for the suptitle
    plt.show()

def save_model_checkpoint(
        model: ContinuousMotionModel, 
        checkpoint_dir: str, 
        model_name: str, 
        epoch: int,
        hyper_parameters: dict, # Hyper parameters to save in the checkpoint
        optimizer: torch.optim.Optimizer,
        train_loss_rec: dict,
        val_loss_rec: dict,
        frechet_distance_rec: dict,
        upload: bool = False,
        run = None
    ):
    
    checkpoint_path = f"{checkpoint_dir}/{model_name}/{model_name}_epoch_{epoch + 1}.pth"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Write the hyper parameters in a pretty indented YAML file
    if hyper_parameters is not None:
        with open(f"{checkpoint_dir}/{model_name}/hyper_parameters.yaml", 'w') as f:
            yaml.dump(hyper_parameters, f, default_flow_style=False, sort_keys=False, indent=2)

    model_state = model.get_model_state()

    # Save the model state
    torch.save({
        'model_state': model_state,
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'train_loss_rec': train_loss_rec,
        'val_loss_rec': val_loss_rec,
        'frechet_distance_rec': frechet_distance_rec
    }, checkpoint_path)
    
    if upload and run is not None:
        artifact = wandb.Artifact(model_name, type='model')
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)

def resume_training_from_checkpoint(
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer, # Optional optimizer to load state,
        train_loss_rec: dict, # Optional training loss recorder to load state,
        val_loss_rec: dict, # Optional validation loss recorder to load state
        frechet_distance_rec, # Optional frechet distance recorder to load state
        device: torch.device = torch.device('cpu') # Device to load the model on
    ):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint['model_state']

    model = ContinuousMotionModel.load_model(model_state, device)
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load loss records
    train_loss_rec.update(checkpoint['train_loss_rec'])
    val_loss_rec.update(checkpoint['val_loss_rec'])
    frechet_distance_rec.update(checkpoint['frechet_distance_rec'])

    return model, checkpoint['epoch'] if 'epoch' in checkpoint else 0

def is_running_on_slurm():
    return "SLURM_JOB_ID" in os.environ
