# Sliding Diffusion
A Streaming Framework for Real-Time Co-Speech Gesture Synthesis

This repository contains the implementation of Sliding Diffusion, developed as part of the Master’s Thesis by Anton Dalsgaard Bertelsen and Peter Berndt Fuchs at the IT University of Copenhagen.

Sliding Diffusion builds upon diffusion models for gesture generation and introduces a streaming-friendly variant that supports low-latency, real-time inference driven by audio input.

---

## Repository Structure

### Core Model & Diffusion Process
- `model.py`
  - Implements the baseline model: a reproduction of [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture) by Sicheng Yang, Zhiyong Wu, Minglei Li, Zhensong Zhang, Lei Hao, Weihong Bao, Ming Cheng, Long Xiao and as discribed here: https://arxiv.org/abs/2305.04919
  - Provides the architecture used for both Normal Diffusion and Sliding Diffusion.
  - The difference lies only in the inference function and the specific diffusion sub-type passed during initialization.

- `diffusion_process_super.py` / `diffusion_process_sliding.py` / `diffusion_process_normal.py`
  - `Diffusion` (abstract super-class) defines the core forward diffusion process (adding noise to gesture tensors).
  - `SlidingDiffusion` (in `diffusion_process_sliding.py`)
  - `NormalDiffusion` (in `diffusion_process_normal.py`)
  - Key differences are in their hyperparameters:
    - Sliding Diffusion: stacking levels, clean/denoising section lengths
    - Normal Diffusion: number of diffusion timesteps

### Training
- `model_training_loop.py`
  - Unified training loop used for both Normal and Sliding Diffusion.
  - Model-agnostic: noise injection is determined by the chosen diffusion subclass.
  - Designed to make hyperparameter configuration clear and centralized for experimentation.

### Dataset Processing
- `dataset/data_processor.py`
  - Handles the full preprocessing pipeline:
    - Feature Extraction: Extracts motion features from `.bvh` and audio features from `.wav`.
    - Consolidation: Packs features into `.npz` + metadata for efficient loading.
    - Normalization: Applies normalization for stable training.

- `dataset/dataset.py`
  - Contains two PyTorch dataset classes:
    - `RAMDataset`: Loads dataset into RAM. This may not be fully functional.
    - `GPUDataset`: Loads dataset directly to GPU for maximum performance with large datasets.

### Utilities
- `utils/`
  - Helper functions for data loading, evaluation, and visualization.
  - Includes a custom BVH viewer for real-time rendering of 3D animations inside Jupyter notebooks.
  - Located under: `utils/animation/visualisation/new`.

---

## Getting Started

### Environment Setup
with Conda (Anaconda or Miniconda)
```bash
conda create -n SlidingDiffusion python=3.11.6
conda activate SlidingDiffusion
pip install -r requirements.txt
````

---

### Dataset Setup

This project uses the GENEA 2023 dataset. Download and unpack it. (On Windows you will have to manually unzip it. The folder structure should be eg. `dataset/genea2023_dataset/val/...`)

**Validation Dataset:**
```bash
wget -O sliding_diffusion_project/dataset/genea2023_val_dataset.zip "https://zenodo.org/records/8199133/files/genea2023_val.zip?download=1" \
&& unzip sliding_diffusion_project/dataset/genea2023_val_dataset.zip -d sliding_diffusion_project/dataset/
```

**Training Dataset** (Large)
```bash
wget -O sliding_diffusion_project/dataset/genea2023_trn_dataset.zip "https://zenodo.org/records/8199133/files/genea2023_trn.zip?download=1" \
&& unzip sliding_diffusion_project/dataset/genea2023_trn_dataset.zip -d sliding_diffusion_project/dataset/
```

Steps performed:

1. Downloads the dataset archive from [Zenodo](https://zenodo.org/records/8199133)
2. Saves it as `genea2023_dataset.zip` inside `sliding_diffusion_project/dataset/`
3. Extracts the archive to `dataset/`
4. (Optional) Remove `.zip` to save space

Finally, copy the skeleton config into the dataset folder:

```bash
cp sliding_diffusion_project/utils/skeleton_config.yaml sliding_diffusion_project/dataset/genea2023_dataset/
```

> **Note:** The validation dataset is normalized relative to the traning dataset. This means that if you choose to only download the validation dataset you need to download the precompiled normalization data for the training dataset. 
> 
> This is avaliable [here](https://drive.google.com/file/d/1DirJSyIoNL4J6o_YIoSO3MAIvaw0rPZW/view?usp=sharing).
> Place the file in `dataset\genea2023_dataset\trn\main-agent\consolidated_meta.pkl`



---

### Preprocess the Dataset

Run:

```bash
cd sliding_diffusion_project
python -m dataset.data_processor --dataset_type **trn / val**
```

## Ready to go


### To try training a model:
Open the `quick_start_training_your_own.ipynb` Jupyter notebook to try the following

1. Load `GPUDataset`
2. Adjust hyperparameters as needed
3. Run the training loop
4. Test inference mode

### To try generating animation with a pre-trained model:

Download a pre-trained model from [here](https://drive.google.com/file/d/1fohAsiM5CShuEul0RqNc0r-oFJOpOME8/view?usp=sharing) and place it in the `trained_models` folder.
Then go to `quick_start_inference_with_pretrained.ipynb` to try generating animations with the pre-trained model and our bvh viewer.

Alternatively go to `microphone_inference.ipynb` to generate gestures based on your microphone. Note that this works very poorly if the microphone loudness or quality is outside the distribution of the dataset.