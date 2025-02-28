import numpy as np
import os
import csv
from utils.bvh_processing.bvh_converter import BVHConverter
from utils.audio_processing.extract_audio_features import extract_audio_features

bvh_dir = 'dataset/genea2023_dataset/trn/main-agent/bvh'
wav_dir = 'dataset/genea2023_dataset/trn/main-agent/wav'
metadata_file = 'dataset/genea2023_dataset/trn/metadata.csv'


# Find all the bvh files in the dataset directory for the main agent
bvh_files = sorted(os.listdir(bvh_dir))
wav_files = sorted(os.listdir(wav_dir))

# The bvh converter is an object because some information needs to be retained between conversions to and from features. 
# One reason is that not all bones are present in the features, so when we convert back, we need to add them back. This information is kept track off by
# the bvh converter.
bvh_converter = BVHConverter()

# Load the csv metadata file
metadata = {}
with open(metadata_file, 'r') as f:
    reader = csv.DictReader(f)
    metadata = {row['prefix']: row for row in reader}

# Find number of speakers in the dataset
speakers = set()
for key in metadata:
    speakers.add(metadata[key]['main-agent_id'])
    speakers.add(metadata[key]['interloctr_id'])
num_speakers = len(speakers)

# Now I want to loop over each file pair and extract the joint angles from the bvh file and the audio features from the wav file
for bvh_file, wav_file in zip(bvh_files, wav_files):
    print(bvh_file, wav_file)

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
    main_agent_id_one_hot[int(main_agent_id)] = 1
    interloctr_id_one_hot = np.zeros(num_speakers)
    interloctr_id_one_hot[int(interloctr_id)] = 1

    # Extract joint angles from the bvh file
    bvh_features = bvh_converter.to_features(os.path.join(bvh_dir, bvh_file))

    # Extract audio features from the wav file
    audio_features = extract_audio_features(os.path.join(wav_dir, wav_file))

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