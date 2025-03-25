import numpy as np
import os
import csv
from utils.bvh_processing.bvh_converter2 import OptimizedBVHProcessor
from utils.audio_processing.extract_audio_features import extract_audio_features
from tqdm import tqdm

bvh_dir = 'dataset/genea2023_dataset/trn/main-agent/bvh'
wav_dir = 'dataset/genea2023_dataset/trn/main-agent/wav'
metadata_file = 'dataset/genea2023_dataset/trn/metadata.csv'
stats_file = 'dataset/genea2023_dataset/trn/main-agent/stats.npz'

# Find all the bvh files in the dataset directory for the main agent

bvh_files = sorted([f for f in os.listdir(bvh_dir) if f.endswith('.bvh')])
wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])

# Load the csv metadata file
metadata = {}
with open(metadata_file, 'r') as f:
    reader = csv.DictReader(f)
    metadata = {row['prefix']: row for row in reader}

# Find number of speakers in the dataset
num_speakers = 0
for key in metadata:
    num_speakers = max(num_speakers, int(metadata[key]['main-agent_id']))
    num_speakers = max(num_speakers, int(metadata[key]['interloctr_id']))

# calculate mean and std of the dataset. This will be used to normalize the data to improve training
print("Calculating dataset statistics")
# if means file exists, load it
if os.path.exists(stats_file):
    means, stds, skeleton, original_skeleton = OptimizedBVHProcessor.load_statistics(stats_file)
else:
    means, stds, skeleton, original_skeleton = OptimizedBVHProcessor.calculate_dataset_statistics(bvh_files, bvh_dir, stats_file=stats_file)

print("Extracting features")
# Import tqdm for progress bar

# Now I want to loop over each file pair and extract the joint angles from the bvh file and the audio features from the wav file
for bvh_file, wav_file in tqdm(list(zip(bvh_files, wav_files)), desc="(2/2) Processing files"):
    prefix = os.path.splitext(bvh_file)[0].removesuffix("_main-agent")  # Adjust if necessary

    file_metadata = metadata.get(prefix, {})
    
    # extract metadata features prefix,main-agent_id,main-agent_has_fingers,interloctr_id,interloctr_has_fingers
    prefix = file_metadata.get('prefix', prefix)
    main_agent_id = file_metadata.get('main-agent_id', '0')
    main_agent_has_fingers = file_metadata.get('main-agent_has_fingers', '0')
    interloctr_id = file_metadata.get('interloctr_id', '0')
    interloctr_has_fingers = file_metadata.get('interloctr_has_fingers', '0')

    # agent id should be one-hot encoded in a vector
    main_agent_id_one_hot = np.zeros(num_speakers)
    main_agent_id_one_hot[int(main_agent_id) - 1] = 1
    interloctr_id_one_hot = np.zeros(num_speakers)
    interloctr_id_one_hot[int(interloctr_id) - 1] = 1

    # Extract joint angles from the bvh file
    bvh_features = OptimizedBVHProcessor.bvh_to_features(os.path.join(bvh_dir, bvh_file))
    # Convert to float16
    bvh_features = bvh_features.astype(np.float16)
 
    # Extract audio features from the wav file
    audio_features = extract_audio_features(os.path.join(wav_dir, wav_file))
    # Convert to float16
    audio_features = audio_features.half()

    # Crop the features to the minimum length
    min_length = min(bvh_features.shape[0], audio_features.shape[0])
    bvh_features = bvh_features[:min_length]
    audio_features = audio_features[:min_length]

    # Convert one-hot vectors to float16
    main_agent_id_one_hot = main_agent_id_one_hot.astype(np.float16)
    interloctr_id_one_hot = interloctr_id_one_hot.astype(np.float16)

    # make features directory if it does not exist
    os.makedirs('dataset/genea2023_dataset/trn/main-agent/features', exist_ok=True)

    # Construct a npz file with the features
    np.savez_compressed(f'dataset/genea2023_dataset/trn/main-agent/features/{prefix}.npz', 
                        bvh_features=bvh_features, 
                        audio_features=audio_features, 
                        prefix=prefix, 
                        main_agent_id_one_hot=main_agent_id_one_hot, 
                        main_agent_has_fingers=main_agent_has_fingers, 
                        interloctr_id_one_hot=interloctr_id_one_hot, 
                        interloctr_has_fingers=interloctr_has_fingers
                        )