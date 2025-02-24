import numpy as np
import re
from enum import Enum

# Enum for states to improve readability
class State(Enum):
    DEFINITION = 1
    BODY = 2

class BVHParser:
    def __init__(self, filename):
        self.filename = filename
        self.channelmap = {'Xrotation': 'x', 'Yrotation': 'y', 'Zrotation': 'z'}
        self.names = []
        self.offsets = []
        self.parents = []
        self.rotations = []
        self.positions = []
        self.frametime = 1.0 / 60.0
        self.channels = 0
        self.state = State.DEFINITION

    def parse(self, start=None, end=None, order=None):
        """Main function to parse the BVH file."""
        with open(self.filename, 'r') as f:
            active = -1
            end_site = False
            frame_data = []
            frame_count = 0  # Initialize frame count

            # First pass to determine frame count and array size
            for line in f:
                if self.state == State.DEFINITION:
                    self.state = self._parse_definition(line, active, end_site, order)
                elif self.state == State.BODY:
                    frame_count += 1  # Count the number of frames

            # Preallocate arrays based on the frame count and number of joints
            self.frame_data = np.zeros((frame_count, len(self.parents), 3), dtype=np.float32)

            # Second pass to fill the arrays with parsed data
            f.seek(0)  # Reset file pointer to beginning
            for line in f:
                if self.state == State.DEFINITION:
                    self.state = self._parse_definition(line, active, end_site, order)
                elif self.state == State.BODY:
                    self._parse_body(line, start, end)

        return self._generate_data()

    def _parse_definition(self, line, active, end_site, order):
        """Parse the definition section of the BVH file."""
        if "HIERARCHY" in line or "MOTION" in line:
            return State.DEFINITION

        # Handle ROOT or JOINT
        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            self.names.append(rmatch.group(1))
            self.offsets.append(np.array([0, 0, 0], dtype=np.float32))
            self.parents.append(active)
            active = len(self.parents) - 1
            return State.DEFINITION

        # Handle CHANNELS
        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            self.channels = int(chanmatch.group(1))
            if order is None:
                order = 'zyx' if self.channels == 3 else 'xyz'  # Example default
            return State.DEFINITION

        # Handle joint names and offsets
        jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
        if jmatch:
            self.names.append(jmatch.group(1))
            self.offsets.append(np.array([0, 0, 0], dtype=np.float32))
            self.parents.append(active)
            active = len(self.parents) - 1
            return State.DEFINITION

        # Handle end site
        if "End Site" in line:
            end_site = True
            return State.DEFINITION

        # Handle frame count and time
        fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
        if fmatch:
            frame_count = int(fmatch.group(1))
            return State.DEFINITION

        # Handle frame time
        ftime_match = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
        if ftime_match:
            self.frametime = float(ftime_match.group(1))
            return State.BODY

        return self.state

    def _parse_body(self, line, start, end):
        """Parse the body section of the BVH file and fill preallocated arrays."""
        data = line.strip().split()
        if data:
            frame_idx = len(self.frame_data)
            if start and end and (frame_idx < start or frame_idx >= end):
                return  # Skip this frame

            data_block = np.asarray(list(map(float, data)))
            if self.channels == 3:
                positions = data_block[:3]
                rotations = data_block[3:].reshape([len(self.parents), 3])
            elif self.channels == 6:
                data_block = data_block.reshape([len(self.parents), 6])
                positions = data_block[:, :3]
                rotations = data_block[:, 3:6]
            elif self.channels == 9:
                positions = data_block[:3]
                data_block = data_block[3:].reshape([len(self.parents)-1, 9])
                rotations = data_block[:, 3:6]
                positions[1:] += data_block[:, :3] * data_block[:, 6:9]
            else:
                raise ValueError(f"Unsupported channel count: {self.channels}")

            # Fill preallocated arrays
            self.frame_data[frame_idx] = (positions, rotations)

    def _generate_data(self):
        """Convert lists to numpy arrays and return the final parsed data."""
        return {
            'rotations': np.array([frame[1] for frame in self.frame_data]),
            'positions': np.array([frame[0] for frame in self.frame_data]),
            'offsets': np.array(self.offsets),
            'parents': np.array(self.parents),
            'names': self.names,
            'frametime': self.frametime
        }

    def save(self, filename, translations=False):
        """Save the parsed data into a BVH file."""
        with open(filename, 'w') as f:
            t = ""
            f.write(f"{t}HIERARCHY\n")
            f.write(f"{t}ROOT {self.names[0]}\n")
            f.write(f"{t}{{\n")
            t += '\t'

            # Write offsets and channels
            f.write(f"{t}OFFSET {self.offsets[0][0]} {self.offsets[0][1]} {self.offsets[0][2]}\n")
            f.write(f"{t}CHANNELS 6 Xposition Yposition Zposition {self.channelmap[self.order[0]]} {self.channelmap[self.order[1]]} {self.channelmap[self.order[2]]}\n")

            # Write joints
            jseq = [0]
            for i in range(len(self.parents)):
                if self.parents[i] == 0:
                    t, jseq = self._save_joint(f, t, i, jseq, translations)

            t = t[:-1]
            f.write(f"{t}}}\n")
            f.write("MOTION\n")
            f.write(f"Frames: {len(self.rotations)}\n")
            f.write(f"Frame Time: {self.frametime}\n")

            # Write motion data
            for i in range(len(self.rotations)):
                for j in jseq:
                    if translations or j == 0:
                        f.write(f"{self.positions[i, j, 0]} {self.positions[i, j, 1]} {self.positions[i, j, 2]} "
                                f"{self.rotations[i, j, 0]} {self.rotations[i, j, 1]} {self.rotations[i, j, 2]} ")
                    else:
                        f.write(f"{self.rotations[i, j, 0]} {self.rotations[i, j, 1]} {self.rotations[i, j, 2]} ")
                f.write("\n")

    def _save_joint(self, f, t, i, jseq, translations=False):
        """Helper function to save joint data."""
        jseq.append(i)
        f.write(f"{t}JOINT {self.names[i]}\n")
        f.write(f"{t}{{\n")
        t += '\t'
        f.write(f"{t}OFFSET {self.offsets[i][0]} {self.offsets[i][1]} {self.offsets[i][2]}\n")

        # Write channels
        if translations:
            f.write(f"{t}CHANNELS 6 Xposition Yposition Zposition {self.channelmap[self.order[0]]} {self.channelmap[self.order[1]]} {self.channelmap[self.order[2]]}\n")
        else:
            f.write(f"{t}CHANNELS 3 {self.channelmap[self.order[0]]} {self.channelmap[self.order[1]]} {self.channelmap[self.order[2]]}\n")

        end_site = True
        for j in range(len(self.parents)):
            if self.parents[j] == i:
                t, jseq = self._save_joint(f, t, j, jseq, translations)
                end_site = False

        if end_site:
            f.write(f"{t}End Site\n")
            f.write(f"{t}{{\n")
            t += '\t'
            f.write(f"{t}OFFSET 0.0 0.0 0.0\n")
            t = t[:-1]
            f.write(f"{t}}}\n")

        t = t[:-1]
        f.write(f"{t}}}\n")
        return t, jseq

# Example usage
bvh_parser = BVHParser('your_file.bvh')
data = bvh_parser.parse(start=0, end=10)
bvh_parser.save('output_file.bvh', translations=True)