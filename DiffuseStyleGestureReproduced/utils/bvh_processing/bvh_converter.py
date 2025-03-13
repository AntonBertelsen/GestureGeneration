import sys
import os
from sklearn.pipeline import Pipeline
from utils.bvh_processing.pymo.parsers import BVHParser
from utils.bvh_processing.pymo.writers import BVHWriter
from utils.bvh_processing.pymo.preprocessing import *
import os
import joblib as jl

target_joints = ['body_world', 'b_root', 'b_r_foot', 'b_l_foot', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

class BVHConverter:

    def to_features(cls, path, avg_pose):
        p = BVHParser()
        data = p.parse(path)
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=30,  keep_all=False)),
            ('jtsel', JointSelector(target_joints, include_root=False)),
            ('pose_relativizer', PoseRelativizer(avg_pose)),
            ('exp', MocapParameterizer('expmap')),
            ('posscale', PositionScaler(scale=0.1)),
            ('np', Numpyfier())
        ])
        out_data = data_pipe.fit_transform([data])
        jl.dump(data_pipe, os.path.join('./utils/data_pipe.sav'))
        return out_data[0]
    
    def to_bvh(cls, features, bvh_file):
        # Apply the inverse of the pipeline
        data_pipe = jl.load('./utils/data_pipe.sav')
        
        bvh_data = data_pipe.inverse_transform([features])

        # ensure correct body orientation
        bvh_data[0].values["body_world_Xrotation"] = 0
        bvh_data[0].values["body_world_Yrotation"] = 0
        bvh_data[0].values["body_world_Zrotation"] = 0

        # Test to write some of it to file for visualization in blender or motion builder
        writer = BVHWriter()
        with open(bvh_file,'w') as f:
            writer.write(bvh_data[0], f)


    # This function calculates the average pose of a set of bvh files. This is useful for normalizing the data relative to this which will
    # hopefully help the model learn more effectively, and mean we can use less noise in the diffusion process as the data is contained within a smaller range.
    # At least this is the idea. Right now it weights every bvh file equally, but not all bvh files are the same length, so this could be improved.
    def calculate_average_pose(cls, bvh_files, bvh_dir):
        # Load the data
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=30,  keep_all=False)),
            ('jtsel', JointSelector(target_joints, include_root=False)),
            ('posscale', PositionScaler(scale=0.1))
        ])
        all_data = None
        i = 0
        for bvh_file in bvh_files:
            i = i + 1
            if i % 5 == 0:
                break
            print("appending", bvh_file)
            p = BVHParser()
            print("extracting features from", bvh_file)
            bvh_data = p.parse(os.path.join(bvh_dir, bvh_file))
            data = data_pipe.fit_transform([bvh_data])

            print(data[0].values)
            if(all_data is None):
                all_data = {key: 0.0 for key in data[0].values}
            else:
                all_data = {key: all_data[key] + np.mean(data[0].values[key] / 5, axis=0) for key in data[0].values}
        
        print("Calculating average pose")
        avg_pose = all_data
        print("Done calculating average pose:")
        print(avg_pose)
        return avg_pose

        