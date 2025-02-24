# This code was written by Simon Alexanderson
# and is released here: https://github.com/simonalexanderson/PyMO

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from libraries.pymo.parsers import BVHParser
from libraries.pymo.data import Joint, MocapData
from libraries.pymo.preprocessing import *
from libraries.pymo.viz_tools import *
from libraries.pymo.writers import *

from argparse import ArgumentParser

import joblib as jl

def feat2bvh(feat_file, bvh_file):

    features = np.load(feat_file, allow_pickle=True)['clips']
    print("Original features shape: ", features.shape)

    # shorten sequence length for visualization
    features = features[:3000]
    print("Shortened features shape: ", features.shape)

    # transform the data back to it's original shape
    # note: in a real scenario this is usually done with predicted data   
    # note: some transformations (such as transforming to joint positions) are not inversible
    bvh_data = pipeline.inverse_transform([features])

    # ensure correct body orientation
    bvh_data[0].values["body_world_Xrotation"] = 0
    bvh_data[0].values["body_world_Yrotation"] = 0
    bvh_data[0].values["body_world_Zrotation"] = 0

    # Test to write some of it to file for visualization in blender or motion builder
    writer = BVHWriter()
    with open(bvh_file,'w') as f:
        writer.write(bvh_data[0], f)

if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--feat_dir', '-feat', required=True,
                                   help="Path where motion features are stored")
    parser.add_argument('--bvh_dir', '-bvh', required=True,
                                   help="Path where produced motion files (in BVH format) will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline is be stored")

    params = parser.parse_args()


    # load data pipeline
    pipeline = jl.load(params.pipeline_dir + 'data_pipe.sav')

    # convert a file
    feat2bvh(params.feat_dir, params.bvh_dir)