import os
import matplotlib.pyplot as plt
import torch
from v1_model import ContinuousMotionModel
from torch.utils.data import DataLoader
from dataset.dataset import *
from datetime import datetime


if __name__ == "__main__":
    # Assuming val_loader and model are defined elsewhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuousMotionModel().to(device)
    val_loader = DataLoader(
        GPUDataset(
            consolidated_file = "dataset/genea2023_dataset/val/main-agent/advanced_encoder/consolidated.npz",
            seq_length = 100,
            seed_length = 8,
            batch_size = 64,
            epoch_length = 30,
            loading_encoded_data = True,
            include_vel_acc_features = False,
            device = device
        ),
        batch_size = 1,
        num_workers = 0,
        pin_memory = False
    )

    avgerage_absolute_error_experiemnt(val_loader, model, device)



def avgerage_absolute_error_experiemnt(val_loader: DataLoader, model: ContinuousMotionModel, device: torch.device):
    all_outputs = []
    all_ground_truths = []

    with torch.no_grad():
        for val_batch in val_loader:
            gesture_sequence, gesture_seed, audio_features, main_agent_id_one_hot = [
                item.squeeze(0).to(device) for item in val_batch
            ]
            
            output, _, _, _ = model.generate(
                gesture_sequence            = gesture_sequence,
                audio_features              = audio_features,
                main_agent_id_one_hot       = main_agent_id_one_hot,
                gesture_seed                = gesture_seed,
                gesture_sequence_is_encoded = val_loader.dataset.loading_encoded_data
            )
            
            all_outputs.append(output.cpu())
            all_ground_truths.append(gesture_sequence.cpu())

    # Stack all batches
    all_outputs = torch.cat(all_outputs, dim=0)                 # Shape: (total_samples, *dims)
    all_ground_truths = torch.cat(all_ground_truths, dim=0)     # Same shape

    # Compute the averaged error matrix
    avg_error_matrix = compute_average_error_matrix(all_outputs, all_ground_truths)

    print(avg_error_matrix.shape)
    visualize_and_save_error_matrix(
        avg_error_matrix, 
        save_dir="/avg_abs_error_diagram_results", 
        filename=f"average_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png", 
        cmap="hot", 
    )


def compute_average_error_matrix(predictions, ground_truths):
    
    assert predictions.shape == ground_truths.shape, "Shape mismatch between predictions and ground truths"
    
    # Compute absolute difference per element
    abs_diff = torch.abs(predictions - ground_truths)  # Shape: (batch_size, *dims)
    
    # Average over the batch dimension
    avg_error_matrix = abs_diff.mean(dim=0)  # Shape: (*dims)
    
    return avg_error_matrix

def visualize_and_save_error_matrix(error_matrix, save_dir, filename="average_error.png", cmap="hot"):
    if isinstance(error_matrix, torch.Tensor):
        error_matrix = error_matrix.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(6, 6))
    plt.imshow(error_matrix, cmap=cmap, interpolation='nearest')
    plt.title("Average Per-Pixel Absolute Error")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved error visualization to {save_path}")