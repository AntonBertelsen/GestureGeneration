from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser
from pymo.preprocessing import *
import os
import joblib as jl
from pymo.writers import BVHWriter

target_joints = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_r_upleg', 'b_r_leg', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_l_shoulder', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_r_shoulder', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_neck0', 'b_head']

class BVHConverter:

    def to_features(cls, path):
        p = BVHParser()
        data = p.parse(path)
        data_pipe = Pipeline([
            ('dwnsampl', DownSampler(tgt_fps=30,  keep_all=False)),
            ('jtsel', JointSelector(target_joints, include_root=False)),
            ('exp', MocapParameterizer('expmap')),
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
        