import torch
import os
import glob

def get_latest_model_path(directory: str, return_folder=False) -> str:
    # Find the newest folder
    folders = sorted(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)
    if not folders:
        raise FileNotFoundError("No model folders found in the specified directory.")

    latest_folder = folders[-1]

    # Find the newest file in that folder
    files = sorted(glob.glob(os.path.join(latest_folder, '*.pth')), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("No model files found in the latest folder.")

    if return_folder:
        return files[-1], latest_folder.split(os.path.sep)[-1]
    else:
        return files[-1]
    
def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )