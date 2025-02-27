import torch

class JointSelector:
    def __init__(self, target_joints, include_root=True):
        self.target_joints = target_joints
        self.include_root = include_root

    def __call__(self, tracks):
        Q = []

        for track in tracks:
            new_track = track.clone()
            new_track.skeleton = [joint for joint in track.skeleton if joint in self.target_joints]
            if self.include_root:
                new_track.skeleton = [track.root_name] + new_track.skeleton
            new_track.values = {key: value for key, value in track.values.items() if key in new_track.skeleton}
            Q.append(new_track)

        return Q

class DownSampler:
    def __init__(self, tgt_fps, keep_all=True):
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all

    def __call__(self, tracks):
        Q = []
        
        for track in tracks:
            orig_fps = round(1.0 / track.framerate)
            rate = orig_fps // self.tgt_fps
            if orig_fps % self.tgt_fps != 0:
                raise ValueError(f"Error: orig_fps ({orig_fps}) is not divisible by tgt_fps ({self.tgt_fps})")
            else:
                print(f"Downsampling with rate: {rate}")

            for ii in range(rate):  # Loop for staggered subsampling
                new_track = track.clone()
                new_track.values = track.values[ii:-1:rate].copy()
                new_track.framerate = 1.0 / self.tgt_fps
                Q.append(new_track)
                
                if not self.keep_all:
                    break  # Only keep the first downsampled track if keep_all=False

        return Q

class ReverseTime:
    def __init__(self, append=True):
        self.append = append

    def __call__(self, tracks):
        Q = list(tracks) if self.append else []  # Keep original tracks if append=True
        
        for track in tracks:
            new_track = track.clone()
            new_track.values = track.values[::-1]  # Reverse the motion sequence
            Q.append(new_track)

        return Q
    def reverse(self, tracks):
        return self.__call__(tracks)

class Mirror:
    def __init__(self, axis="X", append=True):
        """
        Mirrors the motion capture data along the given axis.
        """
        self.axis = axis
        self.append = append

    def __call__(self, tracks):
        print("Mirror: " + self.axis)
        Q = list(tracks) if self.append else []

        # Define the axis transformation signs
        signs = {
            "X": torch.tensor([1, -1, -1]),
            "Y": torch.tensor([-1, 1, -1]),
            "Z": torch.tensor([-1, -1, 1])
        }[self.axis]

        for track in tracks:
            # Clone the track to avoid modifying the original
            new_track = track.clone()

            # Mirror root position (assuming root position is part of the track values)
            root_pos = [f"{track.root_name}_Xposition", f"{track.root_name}_Yposition", f"{track.root_name}_Zposition"]
            for i, pos in enumerate(root_pos):
                new_track.values[pos] = -signs[i] * track.values[pos]

            # Mirror rotation for left-right joints
            lft_joints = [joint for joint in track.skeleton if '_l_' in joint and 'Nub' not in joint]
            rgt_joints = [joint for joint in track.skeleton if '_r_' in joint and 'Nub' not in joint]

            for lft_joint in lft_joints:
                rgt_joint = lft_joint.replace('_l_', '_r_')
                for axis in ["X", "Y", "Z"]:
                    new_track.values[f"{lft_joint}_{axis}rotation"] = signs["XYZ".index(axis)] * track.values[f"{rgt_joint}_{axis}rotation"]
                    new_track.values[f"{rgt_joint}_{axis}rotation"] = signs["XYZ".index(axis)] * track.values[f"{lft_joint}_{axis}rotation"]

            # Mirror trunk joints (non-left-right joints)
            trunk_joints = [joint for joint in track.skeleton if '_l_' not in joint and '_r_' not in joint and 'Nub' not in joint]
            for joint in trunk_joints:
                for axis in ["X", "Y", "Z"]:
                    new_track.values[f"{joint}_{axis}rotation"] = signs["XYZ".index(axis)] * track.values[f"{joint}_{axis}rotation"]

            Q.append(new_track)

        return Q
    
    def reverse(self, tracks):
        return self.__call__(tracks)