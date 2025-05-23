import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast

from tqdm import tqdm
from IPython.display import clear_output
import time
from collections import defaultdict
import v1_evaluation

import os

import wandb

def variance_loss(denoised_gesture, true_gesture):
    var_pred = torch.var(denoised_gesture, dim=1, unbiased=False)  # Variance over time
    var_true = torch.var(true_gesture, dim=1, unbiased=False)

    loss = torch.mean(torch.relu(var_true - var_pred))  # Penalize lower variance only
    return loss

def velocity_loss(pred, gt):
    vel_pred = pred[:, 1:] - pred[:, :-1]  # First-order difference
    vel_gt = gt[:, 1:] - gt[:, :-1]
    return torch.mean((vel_pred - vel_gt) ** 2)

def acceleration_loss(pred, gt):
    acc_pred = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # Second-order difference
    acc_gt = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
    return torch.mean((acc_pred - acc_gt) ** 2)

def encoded_latent_space_loss(pred, gt):
    # calculate the loss between the encoded latent space and the predicted encoded latent space
    return torch.mean((pred - gt) ** 2)

def train(
        experiment_collection_name: str, # Name of the gruope of experiments, this run is a part ofmodel, 
        device,
        model,
        training_loader,
        val_loader, 
        num_epochs: int, 
        autoencoder_model = None,
        run = None, # A wandb.run object to log the training process
        wandb_config: dict = None, # A wandb.config from a yml or dict. This is logged hyper parameters for wandb used for hyperparameter tuning / sweeps. Given to be attached to the run, and extended with given, but not sweeping hyperparams
        model_checkpoint_dir: str = None, # dir of the model checkpoint to load from
        model_check_point_interval_in_epochs: int = 2, # how often to save the model checkpoint
        upload_model_check_point: bool = False, # should upload the model checkpoint to wandb
        condition_mask_probabilty = 0.1,  # TODO: should be in model, as hyper param, not here
        lr = 0.0003,
        variance_loss_weight = 0.1,
        velocity_loss_weight = 0.1,
        acceleration_loss_weight = 0.1,
        latent_space_loss_weight = 2.0,
        category_weighting: dict[str, float] = {}):

    diffusion = model.diffusion_noise_scheduler
    current_model_name = f"{experiment_collection_name}_started_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    visualize_step = 200  # How often to print profiling stats
    save_step = 1000  # How often to save the model checkpoint

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    torch.set_float32_matmul_precision('high')
    
    if wandb_config is not None:
        wandb_config.update({
            # Training hyper parameters
            "batch_size": training_loader.batch_size,
            "epochs": num_epochs,
            "optimizer": optimizer.__class__.__name__,
            "velocity_loss_weight": velocity_loss_weight,
            "acceleration_loss_weight": acceleration_loss_weight,

            # Data hyper params:
            "dataloader": "RAMResidentDataset",
            "float_precision_or_type": "Halvs",

            # Noising hyper parameters
            **model.deffsion_noise_scheduler.get_WnB_config_specs(),

            # training_loader hyper parameters
            **model.get_WnB_config_specs(),
        }, allow_val_change=True)  # <- Important: allows adding/updating keys


    # I then move the model to the device that is being used and put in traning mode
    model = model.to(device)
    # model = torch.compile(model, backend="cudagraphs")
    model.train()
    
    # I then define a map of lists used for tracking the training and validation loss for each epoch. 
    # I'll later use these two sets to plot the progress of the model training.
    loss_rec = {'train' : [], 'val' : [], 'train_plot': [], 'frechet_distance_feat_space': [], 'frechet_distance_raw': []}
    best_loss_rec = {'train' : np.inf, 'val' : np.inf}

    # Skeleton data used to weight bones differently in the loss function.
    # Initialize a tensor of zeros with the same length as the number of bones
    skeleton = training_loader.dataset.skeleton
    bone_index_weighted_by_category_vector = skeleton.construct_bone_weighting_vector(category_weighting)
    
    if autoencoder_model is not None:
        # Lock the weights of the autoencoder model, since we are simply using it to reduce the dimensionality of the data, and it is already trained. We dont want to change it.
        for param in autoencoder_model.parameters():
            param.requires_grad = False
        # Move the autoencoder model to the same device as the main model
        autoencoder_model = autoencoder_model.to(device)
        # Set the autoencoder model to evaluation mode
        autoencoder_model.eval()

    # This is the main training loop that goes through the entire dataset and trains the model on it for each epoch.
    for epoch in range(num_epochs):
        
        progress_bar = tqdm(training_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

        # reshuffle the dataset from the dataloaders at the start of each epoch.
        training_loader.dataset._reshuffle()
        val_loader.dataset._reshuffle()

        # During the epoch, all the data items are iterated over.
        for i, batch_data in enumerate(progress_bar):
            # IMPORTANT CHANGE: Handle pre-batched data from Consolidated Ram Dataset
            # Each item contains a full batch already, so we just need to unpack the tuple
            # and handle the extra dimension from batch_size=1
            gesture_sequence, seed_gesture, audio_features, main_agent_id_one_hot = [
                item.squeeze(0) for item in batch_data
            ]

            # If the autoencoder model is provided, we use it to encode the gesture sequence
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                if autoencoder_model is not None:
                    # Encode the gesture sequence using the autoencoder model
                    encoded_gesture_sequence, _ = autoencoder_model.encode(gesture_sequence)
                else:
                    encoded_gesture_sequence = gesture_sequence

            time_step_stacking_level = torch.randint(0, diffusion.num_of_timestep_stackings, (1,))
            noisy_gesture_sequence = diffusion.forward(
                sequence_tensor=encoded_gesture_sequence,
                stacking_step=time_step_stacking_level,
            )

            # Zero gradients before forward pass
            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Convert all inputs to same precision as autocast context
                encoded_output = model(
                    current_time_step_stacking_level = time_step_stacking_level.item(),
                    one_hot_style = main_agent_id_one_hot,
                    audio_features = audio_features, 
                    noisy_gesture_sequence = noisy_gesture_sequence,
                    condition_mask_probabilty = condition_mask_probabilty,
                )

            # If the autoencoder model is provided, we use it to decode the output
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                if autoencoder_model is not None:
                    # Decode the output using the autoencoder model
                    output = autoencoder_model.decode(encoded_output)
                else:
                    output = encoded_output

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Use the casted target for all loss calculations
                loss = nn.HuberLoss(reduction="none")(output, gesture_sequence)

                # Applying the bone category weighting
                # loss = bone_index_weighted_by_category_vector * loss
                # Now we find the mean over the batch and time dimensions
                loss = loss.mean()
                
                loss += encoded_latent_space_loss(encoded_output, encoded_gesture_sequence) * latent_space_loss_weight
                loss += variance_loss(output, gesture_sequence) * variance_loss_weight 
                loss += velocity_loss(output, gesture_sequence) * velocity_loss_weight 
                loss += acceleration_loss(output, gesture_sequence) * acceleration_loss_weight
            loss_val = loss.item()
            progress_bar.set_postfix({'loss': loss_val})
            loss_rec['train'].append(loss_val)
                        
            loss.backward()

            optimizer.step()
            # Visualization
            if i % visualize_step == 0:
                
                # log the loss to wandb (W&B)
                step = i + epoch * len(training_loader)
                if run is not None: run.log({"train/loss": loss.item()}, step=step)
                
                clear_output(wait=True)
                visualisation_start = time.time()

                # add the averaged loss over hte last visualize_step to the loss_rec['train_plot']
                loss_rec['train_plot'].append(np.mean(loss_rec['train'][-visualize_step:]))
                
                # Visualization code remains unchanged
                fig, axs = plt.subplots(2, 5, figsize=(30, 24))

                cmap = 'viridis'
                vmin = -1
                vmax = 1

                axs[0,0].imshow(output.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[0,0].set_title("Output tensor")
                # axs[0,0].text(10, 190, f"Max: {torch.max(output):.4f}", color="black")
                # axs[0,0].text(10, 200, f"Min: {torch.min(output):.4f}", color="black")
                # axs[0,0].text(10, 210, f"Mean: {torch.mean(output):.4f}", color="black")

                axs[0,1].imshow(gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[0,1].set_title("Actual gesture")
                # axs[0,1].text(10, 190, f"Max: {torch.max(gesture_sequence):.4f}", color="black")
                # axs[0,1].text(10, 200, f"Min: {torch.min(gesture_sequence):.4f}", color="black")
                # axs[0,1].text(10, 210, f"Mean: {torch.mean(gesture_sequence):.4f}", color="black")

                axs[0,2].imshow(noisy_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[0,2].set_title("Noisy gesture starting point")
                # axs[0,2].text(10, 190, f"Max: {torch.max(noisy_gesture_sequence):.4f}", color="black")
                # axs[0,2].text(10, 200, f"Min: {torch.min(noisy_gesture_sequence):.4f}", color="black")
                # axs[0,2].text(10, 210, f"Mean: {torch.mean(noisy_gesture_sequence):.4f}", color="black")

                axs[0,3].imshow((gesture_sequence - output).to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[0,3].set_title("Difference between output and actual gesture")
                # axs[0,3].text(10, 190, f"Max: {torch.max(gesture_sequence - output):.4f}", color="black")

                axs[0,4].plot(loss_rec['train_plot'], label='Training Loss', color='blue')
                if len(loss_rec['val']) > 0:
                    val_plot = np.interp(
                        np.linspace(0, len(loss_rec['val'])-1, len(loss_rec['train_plot'])),
                        np.arange(len(loss_rec['val'])),
                        loss_rec['val']
                    )
                    axs[0,4].plot(val_plot, label='Validation Loss', color='orange')
                axs[0,4].set_title('Training & Validation Loss')
                axs[0,4].set_xlabel('Step')
                axs[0,4].set_ylabel('Loss')
                axs[0,4].set_yscale('log')
                axs[0,4].grid(True)
                axs[0,4].legend()

                if autoencoder_model is not None:
                    # draw the encoded_gesture_sequence
                    axs[1,0].imshow(encoded_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                    axs[1,0].set_title("Encoded gesture sequence")
                    # axs[1,0].text(10, 190, f"Max: {torch.max(encoded_gesture_sequence):.4f}", color="black")
                    # axs[1,0].text(10, 200, f"Min: {torch.min(encoded_gesture_sequence):.4f}", color="black")
                    # axs[1,0].text(10, 210, f"Mean: {torch.mean(encoded_gesture_sequence):.4f}", color="black")

                    # Draw the encoded output
                    axs[1,1].imshow(encoded_output.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                    axs[1,1].set_title("Encoded output")
                    # axs[1,1].text(10, 190, f"Max: {torch.max(encoded_output):.4f}", color="black")
                    # axs[1,1].text(10, 200, f"Min: {torch.min(encoded_output):.4f}", color="black")
                    # axs[1,1].text(10, 210, f"Mean: {torch.mean(encoded_output):.4f}", color="black")

                # Add a graph with FGD distance. Because the values are very different we will plot them in two different graphs
                axs[1,2].plot(loss_rec['frechet_distance_feat_space'], label='Frechet distance feat space', color='green')
                axs[1,2].set_title('Frechet distance feat space')
                axs[1,2].set_xlabel('Step')
                axs[1,2].set_ylabel('Frechet distance')
                axs[1,2].set_yscale('log')
                axs[1,2].grid(True)

                # Write the latest frechet distance to the graph
                if len(loss_rec['frechet_distance_feat_space']) > 0:
                    axs[1,2].text(0.5,0.5, f"Latest: {loss_rec['frechet_distance_feat_space'][-1]:.4f}", color="black", transform=axs[1,2].transAxes)

                axs[1,3].plot(loss_rec['frechet_distance_raw'], label='Frechet distance raw', color='red')
                axs[1,3].set_title('Frechet distance raw')
                axs[1,3].set_xlabel('Step')
                axs[1,3].set_ylabel('Frechet distance')
                axs[1,3].set_yscale('log')
                axs[1,3].grid(True)

                # Write the latest frechet distance to the graph
                if len(loss_rec['frechet_distance_raw']) > 0:
                    axs[1,3].text(0.5,0.5, f"Latest: {loss_rec['frechet_distance_raw'][-1]:.4f}", color="black", transform=axs[1,3].transAxes)
                
                
                if run is not None: run.log({"train/predoction_illutration": wandb.Image(plt.gcf())}, step=step) # Log the figure to W&B
                plt.show()

                visualisation_time = time.time() - visualisation_start
                print(f"Visualisation time: {visualisation_time:.2f} s")

            if i % save_step == 0:
                # Save the model checkpoint
                checkpoint_path = f"{model_checkpoint_dir}/{current_model_name}/{current_model_name}_epoch_{epoch}_step_{i}.pth"
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path} under the name {checkpoint_path} at loss: {loss.item()}")
                
                if upload_model_check_point:
                    artifact = wandb.Artifact('current_model_name', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

        # Calculate validation loss after each epoch
        model.eval()
        val_loss = 0

        frechet_distance_feat_space, frechet_distance = v1_evaluation.evaluate_frechet_gesture_distance(
            model                   = model,
            val_loader              = val_loader,
            device                  = device,
            autoencoder_model       = autoencoder_model,
            evaluation_length       = 30,
            num_samples             = 100
        )

        loss_rec['frechet_distance_feat_space'].append(frechet_distance_feat_space)
        loss_rec['frechet_distance_raw'].append(frechet_distance)

        with torch.no_grad():
            for val_batch in val_loader:
                gesture_sequence, seed_gesture, audio_features, main_agent_id_one_hot = [
                    item.squeeze(0).to(device) for item in val_batch
                ]
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    if autoencoder_model is not None:
                        encoded_gesture_sequence, _ = autoencoder_model.encode(gesture_sequence)
                    else:
                        encoded_gesture_sequence = gesture_sequence

                    time_step_stacking_level = torch.randint(0, diffusion.num_of_timestep_stackings, (1,))
                    noisy_gesture_sequence = diffusion.forward(
                        sequence_tensor=encoded_gesture_sequence,
                        stacking_step=time_step_stacking_level,
                    )

                    encoded_output = model(
                        current_time_step_stacking_level = time_step_stacking_level.item(),
                        one_hot_style = main_agent_id_one_hot,
                        audio_features = audio_features, 
                        noisy_gesture_sequence = noisy_gesture_sequence,
                        condition_mask_probabilty = condition_mask_probabilty,
                    )

                    if autoencoder_model is not None:
                        output = autoencoder_model.decode(encoded_output)
                    else:
                        output = encoded_output

                    loss = nn.HuberLoss(reduction="none")(output, gesture_sequence)
                    loss = bone_index_weighted_by_category_vector * loss
                    loss = loss.mean()
                    loss += encoded_latent_space_loss(encoded_output, encoded_gesture_sequence) * latent_space_loss_weight
                    loss += variance_loss(output, gesture_sequence) * variance_loss_weight 
                    loss += velocity_loss(output, gesture_sequence) * velocity_loss_weight 
                    loss += acceleration_loss(output, gesture_sequence) * acceleration_loss_weight

                val_loss += loss.item()
        loss_rec['val'].append(val_loss / len(val_loader))
        if run is not None: run.log({"val/loss": val_loss / len(val_loader)}, step=step)
        model.train()

    # When all of the epochs are over, the entire list of training loss and validation loss are returned.
    if run is not None: run.finish() # close the wandb (W&B) run
    return model # loss_rec['train'] #, loss_rec['val']

