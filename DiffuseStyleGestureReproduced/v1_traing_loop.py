import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast
from tqdm import tqdm
import time
import v1_evaluation
import os
from IPython.display import clear_output
from v1_sliding_diffusion import Diffusion
from v1_model import ContinuousMotionModel
from torch.utils.data import DataLoader
from variational_autoencoder import VAE
import wandb

def train(
        experiment_collection_name: str, # Name of the gruope of experiments, this run is a part of
        device: torch.device,
        model: ContinuousMotionModel,
        training_loader: DataLoader,
        val_loader: DataLoader, 
        num_epochs: int, 
        autoencoder_model: VAE = None,
        run = None, # A wandb.run object to log the training process
        wandb_config: dict = None, # A wandb.config from a yml or dict. This is logged hyper parameters for wandb used for hyperparameter tuning / sweeps. Given to be attached to the run, and extended with given, but not sweeping hyperparams
        model_checkpoint_dir: str = None, # dir of the model checkpoint to load from
        upload_model_check_point: bool = False, # should upload the model checkpoint to wandb
        condition_mask_probabilty = 0.1,  # TODO: should be in model, as hyper param, not here
        learning_rate = 0.0003,
        reconstruction_loss_weight = 1.0, # Weight for the reconstruction loss
        variance_loss_weight = 0.1,
        velocity_loss_weight = 0.1,
        acceleration_loss_weight = 0.1,
        latent_space_loss_weight = 2.0,
        category_weighting: dict[str, float] = {},
        visualize_step: int = 200, # How often to print profiling stats,
        save_step: int = 1000 # How often to save the model checkpoint
    ):

    current_model_name = f"{experiment_collection_name}_started_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    diffusion: Diffusion = model.diffusion_noise_scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    torch.set_float32_matmul_precision('high')
    
    # If the wandb_config is provided, I update it with the hyper parameters of the model and training process.
    # This is to ensure that the hyper parameters are logged to wandb (W&B) for tracking and visualization.
    # Doing it in this way also allows us to use configs for hyperparameter sweeps.
    if wandb_config is not None:
        wandb_config.update({
            # Training hyper parameters
            "batch_size": training_loader.batch_size,
            "epochs": num_epochs,
            "optimizer": optimizer.__class__.__name__,
            "velocity_loss_weight": velocity_loss_weight,
            "acceleration_loss_weight": acceleration_loss_weight,

            # Data hyper params:
            "dataset_type": training_loader.__class__.__name__,

            # Noising hyper parameters
            **model.diffusion_noise_scheduler.get_WnB_config_specs(),

            # training_loader hyper parameters
            **model.get_WnB_config_specs(),
        }, allow_val_change=True)  # <- Important: allows adding/updating keys


    # I then move the model to the device that is being used and put in traning mode
    model = model.to(device)

    # We compile the model using cudagraphs for performance optimization.
    # This is a PyTorch feature that allows us to optimize the model for faster training.
    # Note: cudagraphs is only available on CUDA devices, so we check if the device is CUDA.
    # There are other backends available, but cudagraphs is the only one I could get working on windows.
    # In Andrej Kaparthy's gpt video he uses a different backend.
    if device.type == 'cuda':
        model = torch.compile(model, backend="cudagraphs")
    
    # dictionaries to keep track of the losses during training and validation so we can plot them later.
    # The first element of the tuple is all losses, the second element is the averaged loss over the last visualize_step.
    train_loss_rec = {'loss' :                      ([],[]),
                      'reconstruction_loss' :       ([],[]), 
                      'encoded_latent_space_loss' : ([],[]), 
                      'variance_loss' :             ([],[]), 
                      'velocity_loss' :             ([],[]), 
                      'acceleration_loss' :         ([],[]),
                      'epoch_loss' :                ([],[])}
    
    val_loss_rec = {'loss' :                        ([],[]),
                    'reconstruction_loss' :         ([],[]),
                    'encoded_latent_space_loss' :   ([],[]), 
                    'variance_loss' :               ([],[]), 
                    'velocity_loss' :               ([],[]), 
                    'acceleration_loss' :           ([],[])}
                    
    frechet_distance_rec = {'encoded':              [], 
                            'raw':                  []}

    # bone_category_weigthing data used to weight bones differently in the loss function based on assigned category weightings.
    bone_category_weighting = training_loader.dataset.skeleton.construct_bone_weighting_vector(category_weighting)
    
    # If an autoencoder model is provided, we will use it to encode the gesture sequence before passing it to the main model.
    # This is extremely effective for reducing the dimensionality of the data. Moreover, it constructs a latent space
    # that is highly gaussian, which should be beneficial for the diffusion model.
    # If the autoencoder model is not provided, we will simply use the gesture sequence as is.
    if autoencoder_model is not None:
        # Lock the weights of the autoencoder model, since we are simply using it to reduce the dimensionality of the data, 
        # and it is already trained. We dont want to change it.
        for param in autoencoder_model.parameters():
            param.requires_grad = False
        
        autoencoder_model = autoencoder_model.to(device)
        autoencoder_model.eval()

    # Main training loop, where we iterate over the number of epochs and the training data.
    for epoch in range(num_epochs):
        # I set the model to training mode
        model.train()

        # reshuffle the dataset from the dataloaders at the start of each epoch.
        training_loader.dataset._reshuffle()
        val_loader.dataset._reshuffle()

        # During the epoch, all the data items are iterated over.        
        progress_bar = tqdm(training_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
        for i, batch_data in enumerate(progress_bar):
            
            # We are using our own batching mechanism in the dataset to to avoid having to use a collate function.
            # As such, each item contains a full batch, but we need to handle the extra dimension 
            # from batch_size=1 from the dataloader.
            gesture_sequence, _, audio_features, main_agent_id_one_hot = [
                item.squeeze(0).to(device) for item in batch_data
            ]

            # If the autoencoder model is provided, we use it to encode the gesture sequence to a lower
            # dimensional latent space which is also more gaussian.
            # This is to help the diffusion model learn a better representation of the data.
            encoded_gesture_sequence = encode_gesture_sequence(autoencoder_model,gesture_sequence)

            # We diffuse the encoded gesture sequence to create a noisy starting point for the diffusion model.
            # TODO: Currently the same time step stacking level is used for all the sequences in the batch. This might be bad? Not sure
            time_step_stacking_level = torch.randint(0, diffusion.num_of_timestep_stackings, ())
            noisy_gesture_sequence = diffusion.forward(encoded_gesture_sequence, time_step_stacking_level)

            # Zero gradients before forward pass
            optimizer.zero_grad()

            # Apply the model to denoise the noisy gesture sequence.
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                encoded_output = model(
                    time_step_stacking_level    = time_step_stacking_level.item(),
                    one_hot_style               = main_agent_id_one_hot,
                    audio_features              = audio_features, 
                    noisy_gesture_sequence      = noisy_gesture_sequence,
                    condition_mask_probabilty   = condition_mask_probabilty,
                )

            # If the autoencoder model is provided, we use it to decode the output
            output = decode_gesture_sequence(autoencoder_model, encoded_output)

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
                bone_weighting_vector       = bone_category_weighting,
                frame_weighting_vector      = None,
                loss_recorder               = train_loss_rec
            )

            # Backward pass to compute gradients                        
            total_loss.backward()

            # Take an optimization step
            optimizer.step()

            # Update the progress bar with the current loss
            progress_bar.set_postfix({'training loss': total_loss.item()})

            # log the loss to wandb (W&B)
            if run is not None: 
                step = i + epoch * len(training_loader)
                run.log({"total_loss": total_loss.item()}, step=step)

            # Visualization
            if i % visualize_step == 0 and not is_running_on_slurm():
                visualize_training_progress(
                    full_gesture_sequence               = gesture_sequence,
                    full_denoised_gesture_sequence      = output,
                    encoded_gesture_sequence            = encoded_gesture_sequence,
                    encoded_denoised_gesture_sequence   = encoded_output,
                    noisy_gesture_sequence              = noisy_gesture_sequence,
                    using_autoencoder                   = autoencoder_model is not None,
                    train_loss_rec                      = train_loss_rec,
                    val_loss_rec                        = val_loss_rec,
                    frechet_distance_rec                = frechet_distance_rec,
                    visualize_step                      = visualize_step
                )

            # Saving
            if i % save_step == 0:
                save_model_checkpoint(
                    model           = model,
                    checkpoint_dir  = model_checkpoint_dir,
                    model_name      = current_model_name,
                    epoch           = epoch,
                    step            = i,
                    upload          = upload_model_check_point,
                    run             = run
                )
        
        ########################################################################
        # End of epoch
        ########################################################################

        epoch_length = len(training_loader)
        train_loss_rec['epoch_loss'][0].append(np.mean(train_loss_rec['loss'][0][-epoch_length:]))

        # At the end of the epoch, we evaluate the model on the validation set.
        model.eval()
        
        val_loss = 0

        # We calculate the Frechet distance between the generated and true gestures.
        # This is a measure of how similar the distribution of the generated gestures is to the distribution of the true gestures.
        frechet_distance, frechet_distance_raw_features, = v1_evaluation.evaluate_frechet_gesture_distance(
            model             = model,
            val_loader        = val_loader,
            device            = device,
            autoencoder_model = autoencoder_model,
            evaluation_length = 30,
            num_samples       = 100
        )

        # Log the Frechet distance so we can see how it changes over time.
        frechet_distance_rec['encoded'].append(frechet_distance)
        frechet_distance_rec['raw'].append(frechet_distance_raw_features)

        # We run the model on the validation set to calculate the validation loss.
        with torch.no_grad():
            for val_batch in val_loader:
                # We are using our own batching mechanism in the dataset to to avoid having to use a collate function.
                gesture_sequence, _, audio_features, main_agent_id_one_hot = [
                    item.squeeze(0).to(device) for item in val_batch
                ]
                # If the autoencoder model is provided, we use it to encode the gesture sequence to a lower
                # dimensional latent space which is also more gaussian.
                encoded_gesture_sequence = encode_gesture_sequence(autoencoder_model, gesture_sequence)

                # We diffuse the encoded gesture sequence to create a noisy starting point for the diffusion model.
                time_step_stacking_level = torch.randint(0, diffusion.num_of_timestep_stackings, (1,))
                noisy_gesture_sequence = diffusion.forward(encoded_gesture_sequence, time_step_stacking_level)

                # Apply the model to denoise the noisy gesture sequence.
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    encoded_output = model(
                        time_step_stacking_level    = time_step_stacking_level.item(),
                        one_hot_style               = main_agent_id_one_hot,
                        audio_features              = audio_features, 
                        noisy_gesture_sequence      = noisy_gesture_sequence,
                        condition_mask_probabilty   = condition_mask_probabilty,
                    )

                # If the autoencoder model is provided, we use it to decode the output
                output = decode_gesture_sequence(autoencoder_model, encoded_output)

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
                    bone_weighting_vector       = bone_category_weighting,
                    frame_weighting_vector      = None,
                    loss_recorder               = val_loss_rec
                )
            val_loss += total_loss.item()
        # Log the average validation loss for the epoch.
        
        # Calculate averaged losses over the last validation 
        for key in val_loss_rec.keys():
            val_loss_rec[key][1].append(np.mean(train_loss_rec[key][0]))
            # clear the losses for the next epoch
            val_loss_rec[key][0].clear()
        
        # Log the losses to wandb (W&B) if a run is provided.
        if run is not None: 
            step = i + epoch * len(training_loader)
            run.log({"validation loss": val_loss / len(val_loader)}, step=step)

    # close the wandb (W&B) run
    if run is not None: 
        run.finish() 
    
    # Save the final model checkpoint
    save_model_checkpoint(
        model=model,
        checkpoint_dir=model_checkpoint_dir,
        current_model_name=current_model_name + "_final",
        epoch=num_epochs-1,
        step=len(training_loader),
        upload_model_check_point=upload_model_check_point,
        run=run
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
         bone_weighting_vector, # This is a vector that weights the loss for each bone category.
         frame_weighting_vector, # This is a vector that weights the loss for each frame. Since we are essentially doing infill diffusion, we want to bias the the loss for the frames that are not masked out.
         loss_recorder=None # This is a dictionary that keeps track of the losses during training and validation so we can plot them later.
         ):
    device = pred.device
    with autocast(device_type=device.type, dtype=torch.bfloat16):
                
        # Use the casted target for all loss calculations
        recon_l = nn.HuberLoss(reduction="none")(pred, gt) * reconstruction_loss_weight

        encoded_latent_space_l = encoded_latent_space_loss(encoded_gt, encoded_pred) * latent_space_loss_weight
        variance_l = variance_loss(pred, gt) * variance_loss_weight 
        velocity_l = velocity_loss(pred, gt) * velocity_loss_weight 
        acceleration_l = acceleration_loss(pred, gt) * acceleration_loss_weight

        # Apply the bone category weighting (Can't be applied to encoded latent space loss, since it is not a tensor of the same shape as the other losses)
        recon_l = bone_weighting_vector * recon_l
        variance_l = bone_weighting_vector * variance_l
        velocity_l = bone_weighting_vector * velocity_l
        acceleration_l = bone_weighting_vector * acceleration_l
        
        # Apply the frame weighting vector
        if frame_weighting_vector is not None:
            recon_l = frame_weighting_vector * recon_l
            encoded_latent_space_l = frame_weighting_vector * encoded_latent_space_l
            variance_l = frame_weighting_vector * variance_l
            velocity_l = frame_weighting_vector * velocity_l
            acceleration_l = frame_weighting_vector * acceleration_l

        # Now we find the mean over the batch and time dimensions
        recon_l = recon_l.mean()
        encoded_latent_space_l = encoded_latent_space_l.mean()
        variance_l = variance_l.mean()
        velocity_l = velocity_l.mean()
        acceleration_l = acceleration_l.mean()

        # Combine all losses
        total_loss = recon_l + encoded_latent_space_l + variance_l + velocity_l + acceleration_l

        # If a loss_recorder is provided, we will record the losses in it.
        if loss_recorder is not None:
            loss_recorder['loss'][0].append(total_loss.item())
            loss_recorder['reconstruction_loss'][0].append(recon_l.item())
            loss_recorder['encoded_latent_space_loss'][0].append(encoded_latent_space_l.item())
            loss_recorder['variance_loss'][0].append(variance_l.item())
            loss_recorder['velocity_loss'][0].append(velocity_l.item())
            loss_recorder['acceleration_loss'][0].append(acceleration_l.item())

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
    acc_pred = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # Second-order difference
    acc_gt = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
    return (acc_pred - acc_gt) ** 2

# Encoded latent space loss function to penalize the difference between the encoded latent space and the predicted encoded latent space.
# This helps the model learn a good representation of the data in latent space. Possibly this is all that is needed, since the model is trained to predict the encoded latent space.
def encoded_latent_space_loss(pred, gt):
    # calculate the loss between the encoded latent space and the predicted encoded latent space
    return (pred - gt) ** 2

def encode_gesture_sequence(autoencoder_model, gesture_sequence):
    device = gesture_sequence.device
    if autoencoder_model is not None:
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            encoded_gesture_sequence, _ = autoencoder_model.encode(gesture_sequence)
        return encoded_gesture_sequence
    else:
        return gesture_sequence
    
def decode_gesture_sequence(autoencoder_model, encoded_gesture_sequence):
    device = encoded_gesture_sequence.device
    if autoencoder_model is not None:
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            output = autoencoder_model.decode(encoded_gesture_sequence)
        return output
    else:
        return encoded_gesture_sequence

def visualize_training_progress(
        full_gesture_sequence: torch.Tensor,
        full_denoised_gesture_sequence: torch.Tensor,
        encoded_gesture_sequence: torch.Tensor,
        encoded_denoised_gesture_sequence: torch.Tensor,
        noisy_gesture_sequence: torch.Tensor,
        using_autoencoder: bool,
        train_loss_rec: dict,
        val_loss_rec: dict,
        frechet_distance_rec: dict,
        visualize_step: int,
    ):
    
    clear_output(wait=True)
    visualisation_start = time.time()

    # Calculate averaged losses over the last visualize_step and store in the loss records.
    for key in train_loss_rec.keys():
        train_loss_rec[key][1].append(np.mean(train_loss_rec[key][0][-visualize_step:]))

    # Create a single figure with GridSpec to manage all plots
    # Calculate total rows needed: 3 rows + 1 if using autoencoder
    total_rows = 3 + (1 if using_autoencoder else 0)
    
    # Height ratios for each row
    if using_autoencoder:
        height_ratios = [4, 2, 8, 2]  # Loss plots, Frechet, Gesture viz, Latent space viz
    else:
        height_ratios = [4, 2, 8]  # Loss plots, Frechet, Gesture viz
    
    # Create figure with appropriate height
    fig = plt.figure(figsize=(30, 9 * total_rows))
    gs = fig.add_gridspec(total_rows, 4, height_ratios=height_ratios)
    
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
    ax_train.set_title('Losses over Training Steps')
    ax_train.set_xlabel('Step')
    ax_train.set_ylabel('Loss')
    # ax_train.set_yscale('log')
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
    ax_val.set_title('Validation Losses over Training Steps')
    ax_val.set_xlabel('Step')
    ax_val.set_ylabel('Loss')
    # ax_val.set_yscale('log')
    ax_val.grid(True)
    ax_val.legend()

    # Train vs Val loss
    ax_comp = fig.add_subplot(gs[0, 2:4])
    ax_comp.plot(train_loss_rec['loss'][1], label='Training Loss', color='blue')
    ax_comp.plot(val_loss_rec['loss'][1], label='Validation Loss', color='orange')
    ax_comp.set_title('Training vs Validation Loss')
    ax_comp.set_xlabel('Step')
    ax_comp.set_ylabel('Loss')
    # ax_comp.set_yscale('log')
    ax_comp.grid(True)
    ax_comp.legend()

    # Row 2: Frechet distance
    ax_frechet1 = fig.add_subplot(gs[1, 0:2])
    ax_frechet1.plot(frechet_distance_rec['encoded'], label='Encoded Frechet Distance', color='blue')
    ax_frechet1.set_title('Frechet Distance over Training Steps')
    ax_frechet1.set_xlabel('Step')
    ax_frechet1.set_ylabel('Frechet Distance')
    # ax_frechet1.set_yscale('log')
    ax_frechet1.grid(True)
    ax_frechet1.legend()
    
    ax_frechet2 = fig.add_subplot(gs[1, 2:4])
    ax_frechet2.plot(frechet_distance_rec['raw'], label='Raw Frechet Distance', color='orange')
    ax_frechet2.set_title('Raw Frechet Distance over Training Steps')
    ax_frechet2.set_xlabel('Step')
    ax_frechet2.set_ylabel('Frechet Distance')
    ax_frechet2.set_yscale('log')
    ax_frechet2.grid(True)
    ax_frechet2.legend()

    # Row 3: Gesture visualization
    cmap = 'viridis'
    vmin = -2.5
    vmax = 2.5

    ax_actual = fig.add_subplot(gs[2, 0])
    ax_actual.set_title("Actual Gesture", fontsize=20)
    ax_actual.imshow(full_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax_actual.set_xlabel("Time")
    ax_actual.set_ylabel("feature")
    ax_actual.grid(False)
    ax_actual.axis('off')
    
    ax_diffused = fig.add_subplot(gs[2, 1])
    ax_diffused.set_title("Diffused Gesture", fontsize=20)
    if not using_autoencoder:
        ax_diffused.imshow(noisy_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax_diffused.text(0.5, 0.5, "Diffused Gesture is not available\n" \
                                      "when using an autoencoder.\n\n" \
                                      "The latent space of the autoencoder\n" \
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

    # Row 4 (optional): Latent space visualization
    if using_autoencoder:
        ax_encoded = fig.add_subplot(gs[3, 0])
        ax_encoded.set_title("Encoded Gesture", fontsize=20)
        ax_encoded.imshow(encoded_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded.set_xlabel("Time")
        ax_encoded.set_ylabel("feature")
        ax_encoded.grid(False)
        ax_encoded.axis('off')
        
        ax_encoded_diff = fig.add_subplot(gs[3, 1])
        ax_encoded_diff.set_title("Encoded Diffused Gesture", fontsize=20)
        ax_encoded_diff.imshow(noisy_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded_diff.set_xlabel("Time")
        ax_encoded_diff.set_ylabel("feature")
        ax_encoded_diff.grid(False)
        ax_encoded_diff.axis('off')
        
        ax_encoded_denoised = fig.add_subplot(gs[3, 2])
        ax_encoded_denoised.set_title("Encoded Denoised Gesture", fontsize=20)
        ax_encoded_denoised.imshow(encoded_denoised_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded_denoised.set_xlabel("Time")
        ax_encoded_denoised.set_ylabel("feature")
        ax_encoded_denoised.grid(False)
        ax_encoded_denoised.axis('off')
        
        ax_encoded_diff_actual = fig.add_subplot(gs[3, 3])
        ax_encoded_diff_actual.set_title("Difference (Encoded Actual - Encoded Denoised)", fontsize=20)
        ax_encoded_diff_actual.imshow((encoded_gesture_sequence - encoded_denoised_gesture_sequence).to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax_encoded_diff_actual.set_xlabel("Time")
        ax_encoded_diff_actual.set_ylabel("feature")
        ax_encoded_diff_actual.grid(False)
        ax_encoded_diff_actual.axis('off')

    # Adjust spacing between subplots
    fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.97])  # rect adjusts for the suptitle
    plt.show()

    # Print the averaged losses
    print("Averaged Training Losses over the last visualize_step:")
    for key, value in train_loss_rec.items():
        print(f"{key}: {value[0][-1]:.4f}")

    if len(val_loss_rec['loss'][0]) > 0:
        print("\nAveraged Validation Losses over the last visualize_step:")
        for key, value in val_loss_rec.items():
            print(f"{key}: {value[0][-1]:.4f}")

    if len(frechet_distance_rec['encoded']) > 0:
        print("\nFrechet Distance over the last visualize_step:")
        print(f"Encoded: {frechet_distance_rec['encoded'][-1]:.4f}, Raw: {frechet_distance_rec['raw'][-1]:.4f}")
    
    visualisation_time = time.time() - visualisation_start
    print(f"Visualisation time: {visualisation_time:.2f} s")

def save_model_checkpoint(
        model, 
        checkpoint_dir: str, 
        model_name: str, 
        epoch: int, 
        step: int,
        upload: bool = False,
        run = None
    ):
    # Save the model checkpoint
    checkpoint_path = f"{checkpoint_dir}/{model_name}/{model_name}_epoch_{epoch}_step_{step}.pth"
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    
    if upload and run is not None:
        artifact = wandb.Artifact(model_name, type='model')
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)


def is_running_on_slurm():
    return "SLURM_JOB_ID" in os.environ
