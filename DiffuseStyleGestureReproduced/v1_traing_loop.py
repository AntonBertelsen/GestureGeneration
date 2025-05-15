import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from io import BytesIO
from PIL import Image
from moviepy import ImageSequenceClip
from datetime import datetime
from torch.amp import autocast

from tqdm import tqdm
from IPython.display import clear_output
import time
from collections import defaultdict

import os

import wandb



# First the loss function is defined, as Huber Loss.
# This is the same loss function that was used in the original paper.
loss_f = nn.HuberLoss()

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

def train(
        experiment_collection_name: str, # Name of the gruope of experiments, this run is a part ofmodel, 
        device,
        training_loader,
        val_loader, 
        num_epochs: int, 
        run = None, # A wandb.run object to log the training process
        wandb_config: dict = None, # A wandb.config from a yml or dict. This is logged hyper parameters for wandb used for hyperparameter tuning / sweeps. Given to be attached to the run, and extended with given, but not sweeping hyperparams
        model_checkpoint_dir: str = None, # dir of the model checkpoint to load from
        model_check_point_interval_in_epocs: int = 2, # how often to save the model checkpoint
        uplaod_model_check_point: bool = False, # should upload the model checkpoint to wandb
        condition_mask_probabilty=0.1,  # TODO: should be in model, as hyper param, not here
        lr=0.0003,
        variance_loss_weight=0.1,
        velocity_loss_weight=0.1,
        acceleration_loss_weight=0.1,
        category_weighting: dict[str, float] = {}):


    diffusion = model.deffsion_noise_scheduler
    current_model_name = f"{experiment_collection_name}_started_{time.strftime('%Y-%m-%d_%H-%M-%S')}"


    # Add profiling data structures
    profiling = defaultdict(list)
    visalize_step = 50  # How often to print profiling stats
    save_Step = 1000  # How often to save the model checkpoint

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
    # print("Model compling to cudagraphs runtime")
    model.train()
    
    # I then define a map of lists used for tracking the training and validation loss for each epoch. 
    # I'll later use these two sets to plot the progress of the model training.
    loss_rec = {'train' : [], 'val' : [], 'train_plot': []}
    best_loss_rec = {'train' : np.inf, 'val' : np.inf}

    # Skeleton data used to weight bones differently in the loss function
    # Initialize a tensor of zeros with the same length as the number of bones
    skeleton_info = training_loader.dataset.skeleton_info
    num_features = skeleton_info['number_of_features']
    bone_index_weighted_by_category_vector = torch.ones(num_features)

    # Assign weights based on the categories
    for category, weight in category_weighting.items():
        # Check if the category exists in the skeleton info
        if category not in skeleton_info['bone_categories']:
            print(f"Warning!!! Category '{category}' not found in skeleton info. Skipping.")
            continue
        for bone_name in skeleton_info['bone_categories'][category]:
            bone_indices = skeleton_info['bone_to_indices'][bone_name]
            # Check if bone exists in the skeleton info
            if bone_indices is None:
                print(f"Warning!!! Bone '{bone_name}' not found in skeleton info. Skipping.")
                continue
            for index in bone_indices:
                bone_index_weighted_by_category_vector[index] = weight
    
    # This is the main training loop that goes through the entire dataset and trains the model on it for each epoch.
    for epoch in range(num_epochs):
        progress_bar = tqdm(training_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)

        # I reset the training loss for each epoch.
        epoch_train_loss = 0
        
        # Start batch timer
        batch_start_time = time.time()

        # reshuffle the training data at the start of each epoch
        training_loader.dataset._reshuffle()


        # During the epoch, all the data items are iterated over.
        for i, batch_data in enumerate(progress_bar):
            # Data loading time
            data_load_time = time.time() - batch_start_time
            profiling["data_loading"].append(data_load_time)
            

            reshape_time = time.time()

            # IMPORTANT CHANGE: Handle pre-batched data from RAMResidentDataset
            # Each item contains a full batch already, so we just need to unpack the tuple
            # and handle the extra dimension from batch_size=1
            gesture_sequence, seed_gesture, audio_features, main_agent_id_one_hot = [
                item.squeeze(0).to(device) for item in batch_data
            ]
            
            # gesture_sequence = gesture_sequence.reshape(total_batch_size, *gesture_sequence.shape[2:])
            # seed_gesture = seed_gesture.reshape(total_batch_size, *seed_gesture.shape[2:])
            # audio_features = audio_features.reshape(total_batch_size, *audio_features.shape[2:])
            # main_agent_id_one_hot = main_agent_id_one_hot.reshape(total_batch_size, *main_agent_id_one_hot.shape[2:])

            reshape_time = time.time() - reshape_time
            profiling["data_reshaping"].append(reshape_time)

            # Device transfer time
            device_start = time.time()

            # Transfer to device
            # gesture_sequence = gesture_sequence.to(device)
            # seed_gesture = seed_gesture.to(device)
            # audio_features = audio_features.to(device)
            # main_agent_id_one_hot = main_agent_id_one_hot.to(device)
            
            device_time = time.time() - device_start
            profiling["device_transfer"].append(device_time)

            # Diffusion time
            diffusion_start = time.time()
            time_step_stakking_level = torch.randint(0, diffusion.num_of_timestap_stackings, (1,))
            noisy_gesture_sequence = diffusion.forward(
                seqence_tensor=gesture_sequence,
                stacking_step=time_step_stakking_level,
            )
            diffusion_time = time.time() - diffusion_start
            profiling["diffusion_forward"].append(diffusion_time)

            # Zero gradients before forward pass
            optimizer.zero_grad()

            # Model forward time
            forward_start = time.time()
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Convert all inputs to same precision as autocast context
                output = model(
                    current_time_step_stacking_level = time_step_stakking_level.item(),
                    one_hot_style = main_agent_id_one_hot,
                    audio_features = audio_features, 
                    noisy_gesture_sequence = noisy_gesture_sequence,
                    condition_mask_probabilty = condition_mask_probabilty,
                )
            forward_time = time.time() - forward_start
            profiling["model_forward"].append(forward_time)

            # Loss calculation time
            loss_start = time.time()
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                # Use the casted target for all loss calculations
                loss = nn.HuberLoss(reduction="none")(output, gesture_sequence)
                print(f"Loss shape before weighting: {loss.shape}")
                # Applying the bone category weighting
                loss = bone_index_weighted_by_category_vector * loss
                # Now we find the mean over the batch and time dimensions
                print(f"Loss shape after weighting: {loss.shape}")
                loss = loss.mean()
                print(f"Loss shape after meaning: {loss.shape}")
                loss += variance_loss(output, gesture_sequence) * variance_loss_weight 
                loss += velocity_loss(output, gesture_sequence) * velocity_loss_weight 
                loss += acceleration_loss(output, gesture_sequence) * acceleration_loss_weight

            loss_time = time.time() - loss_start
            profiling["loss_calculation"].append(loss_time)

            epoch_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            loss_rec['train'].append(loss.item())


            # update the best loss record and maybe save the model checkpoint
            # if epoch == 0 or epoch == 1:
            #     next_checkpount_epoch = model_check_point_interval_in_epocs

            # if loss.item() < best_loss_rec['train']:
            #     best_loss_rec['train'] = loss.item()

            # Saveing the model checkpoint
            # if epoch >= next_checkpount_epoch == 0 and model_checkpoint_dir is not None:
            #     next_checkpount_epoch += model_check_point_interval_in_epocs
            #     checkpoint_path = f"{model_checkpoint_dir}/{current_model_name}/{current_model_name}_epoch_{epoch}.pth"
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print(f"Model checkpoint saved at {checkpoint_path} under the name {checkpoint_path} at loss: {loss.item()}")

            # Backward pass time
            backward_start = time.time()
            # Use the scaler to handle the backward pass with mixed precision
                        
            loss.backward()

            backward_time = time.time() - backward_start
            profiling["backward"].append(backward_time)

            # Optimizer step time
            optimizer_start = time.time()
            # Use the scaler for optimizer step
            optimizer.step()
            optimizer_time = time.time() - optimizer_start
            profiling["optimizer"].append(optimizer_time)

            # Visualization
            if i % visalize_step == 0:
                
                # log the loss to wandb (W&B)
                step = i + epoch * len(training_loader)
                if run is not None: run.log({"train/loss": loss.item()}, step=step)
                
                clear_output(wait=True)
                visualisation_start = time.time()

                # add the averaged loss over hte last visalize_step to the loss_rec['train_plot']
                loss_rec['train_plot'].append(np.mean(loss_rec['train'][-visalize_step:]))
                
                # Visualization code remains unchanged
                fig, axs = plt.subplots(1, 5, figsize=(30, 12))

                cmap = 'viridis'
                vmin = -1
                vmax = 1

                axs[0].imshow(output.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[0].set_title("Output tensor")
                axs[0].text(10, 190, f"Max: {torch.max(output):.4f}", color="black")
                axs[0].text(10, 200, f"Min: {torch.min(output):.4f}", color="black")
                axs[0].text(10, 210, f"Mean: {torch.mean(output):.4f}", color="black")

                axs[1].imshow(gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[1].set_title("Actual gesture")
                axs[1].text(10, 190, f"Max: {torch.max(gesture_sequence):.4f}", color="black")
                axs[1].text(10, 200, f"Min: {torch.min(gesture_sequence):.4f}", color="black")
                axs[1].text(10, 210, f"Mean: {torch.mean(gesture_sequence):.4f}", color="black")

                axs[2].imshow(noisy_gesture_sequence.to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[2].set_title("Noisy gesture starting point")
                axs[2].text(10, 190, f"Max: {torch.max(noisy_gesture_sequence):.4f}", color="black")
                axs[2].text(10, 200, f"Min: {torch.min(noisy_gesture_sequence):.4f}", color="black")
                axs[2].text(10, 210, f"Mean: {torch.mean(noisy_gesture_sequence):.4f}", color="black")

                axs[3].imshow((gesture_sequence - output).to(torch.float32).permute(0, 2, 1)[0, :, :].cpu().detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
                axs[3].set_title("Difference between output and actual gesture")
                axs[3].text(10, 190, f"Max: {torch.max(gesture_sequence - output):.4f}", color="black")

                axs[4].plot(loss_rec['train_plot'])
                axs[4].set_title('Training Loss')
                axs[4].set_xlabel('Epoch')
                axs[4].set_ylabel('Loss')
                axs[4].set_yscale('log')
                axs[4].grid(True)
                
                # Add text with profiling data to the loss plot
                avg_data_time = np.mean(profiling["data_loading"][-visalize_step:]) * 1000
                avg_reshape_time = np.mean(profiling["data_reshaping"][-visalize_step:]) * 1000
                avg_device_time = np.mean(profiling["device_transfer"][-visalize_step:]) * 1000
                avg_diffusion_time = np.mean(profiling["diffusion_forward"][-visalize_step:]) * 1000
                avg_forward_time = np.mean(profiling["model_forward"][-visalize_step:]) * 1000
                avg_loss_time = np.mean(profiling["loss_calculation"][-visalize_step:]) * 1000
                avg_backward_time = np.mean(profiling["backward"][-visalize_step:]) if profiling["backward"] else 0
                avg_optimizer_time = np.mean(profiling["optimizer"][-visalize_step:]) if profiling["optimizer"] else 0
                
                profiling_text = (
                    f"PROFILING (ms/batch):\n"
                    f"Data loading: {avg_data_time:.2f}\n"
                    f"Data reshaping: {avg_reshape_time:.2f}\n"
                    f"Device transfer: {avg_device_time:.2f}\n"
                    f"Diffusion forward: {avg_diffusion_time:.2f}\n"
                    f"Model forward: {avg_forward_time:.2f}\n"
                    f"Loss calculation: {avg_loss_time:.2f}\n"
                    f"Backward: {avg_backward_time:.2f}\n"
                    f"Optimizer: {avg_optimizer_time:.2f}"
                )
                axs[4].text(0.02, 0.98, profiling_text, transform=axs[4].transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                if run is not None: run.log({"train/predoction_illutration": wandb.Image(plt.gcf())}, step=step) # Log the figure to W&B
                plt.show()

                visualisation_time = time.time() - visualisation_start
                print(f"Visualisation time: {visualisation_time:.2f} s")

            if i % save_Step == 0:
                # Save the model checkpoint
                checkpoint_path = f"{model_checkpoint_dir}/{current_model_name}/{current_model_name}_epoch_{epoch}_step_{i}.pth"
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model checkpoint saved at {checkpoint_path} under the name {checkpoint_path} at loss: {loss.item()}")
                
                if uplaod_model_check_point:
                    artifact = wandb.Artifact('current_model_name', type='model')
                    artifact.add_file(checkpoint_path)
                    wandb.log_artifact(artifact)

            # Start timing for next batch
            batch_start_time = time.time()

        # After the training loop, the launch on the validation set is calculated in the same way as on 
        # the training set, but without the gradient descent and transformation of the model premises.
        
        epoch_val_loss = 0
        # model.eval()
        # for images_v, labels_v in loaders['validation']:
        #     if torch.cuda.is_available():
        #        images_v, labels_v = images_v.cuda(), labels_v.cuda()
        #     output = model(images_v)
        #     loss_v = loss_f(output, labels_v)
        #     epoch_val_loss += loss_v.item()
        
        # I then take the average of the training loss and validation loss.
        train_loss = epoch_train_loss / len(training_loader)
        # val_loss = epoch_val_loss / len(loaders["validation"])
        
        # And format them using print statements and output them after the epoch is finished.
        # print(f'Epoch {epoch+1}')
        # print(f'Training Loss: {train_loss}')
        # print(f'Validation Loss: ???') # {val_loss}')
        # print('-------------------')
        
        # Finally, I record the loss and append them to the list used for plotting the loss later.
        # loss_rec['train'].append(train_loss)
        # loss_rec['val'].append(val_loss)
        
        # If the validation loss is smaller than the previously best validation loss, the model is saved to a 
        # separate file in the same folder as this, or given in the save function. 
        # if train_loss < current_min_val_loss:
        #     print(f'train Loss Decreased({current_min_val_loss}--->{train_loss}) \t Saving The Model')
        #     current_min_val_loss = train_loss
        #     # Saving State Dict
        #     torch.save(model.state_dict(), 'saved_model.pth')

        # draw the training loss
        # print(len(loss_rec['train']))
        # clear_output(wait=True)
    
    # When all of the epochs are over, the entire list of training loss and validation loss are returned.
    if run is not None: run.finish() # close the wandb (W&B) run
    return model # loss_rec['train'] #, loss_rec['val']

