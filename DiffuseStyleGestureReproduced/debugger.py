import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from local_attention import transformer
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
import matplotlib.pyplot as plt
from typing import Union
from io import BytesIO
from PIL import Image
from moviepy import ImageSequenceClip
from datetime import datetime

class Show:
        SHAPE = 0
        SHAPE_AND_CONTENTS = 1
        IMAGE = 2
        MAX_MIN = 3


class Debugger: 
    def __init__(self, on: bool = False, keys_for_printing_while_running: Union[str, list[str]] = "ALL"):
        self.on = on
        self.keys_for_printing_while_running = keys_for_printing_while_running

        if not isinstance(keys_for_printing_while_running, list):
            self.keys_for_printing_while_running = [keys_for_printing_while_running]

        self.debugger_dict = {
                "ALL": []
        }


    def capture(self, 
        tensor_name_pairs: Union[tuple[str, torch.Tensor], list[tuple[str, torch.Tensor]]], 
        displays: Union[Show, list[Show]], 
        keys: Union[str, list[str]]
    ):
        if not self.on: return
        
        if not isinstance(displays, list): displays = [displays]
        if not isinstance(keys, list): keys = [keys]
        if not isinstance(tensor_name_pairs, list): tensor_name_pairs = [tensor_name_pairs]

        self.debugger_dict["ALL"].append((tensor_name_pairs, displays))
        for key in keys:
            if key not in self.debugger_dict:
                self.debugger_dict[key] = []
            self.debugger_dict[key].append((tensor_name_pairs, displays))

        if any(k in self.keys_for_printing_while_running for k in keys):
            self.write(tensor_name_pairs, displays)
        

    def write(self, 
        tensor_name_pairs: Union[tuple[str, torch.Tensor], list[tuple[str, torch.Tensor]]], 
        displays: Union[Show, list[Show]]
    ):  
        if isinstance(displays, Show): displays = [displays]
        if not isinstance(tensor_name_pairs, list): tensor_name_pairs = [tensor_name_pairs]
        
        if self.on:
            for tensor_name, tensor in tensor_name_pairs:
                for display in displays:
                    if display == Show.SHAPE:
                        print(tensor_name, " shape: ", tensor.shape)
                    elif display == Show.SHAPE_AND_CONTENTS:
                        print(tensor_name, " shape: ", tensor.shape)
                        print(tensor_name, " contents: ", tensor)
                    elif display == Show.IMAGE:

                        if tensor.dim() == 3:
                            plt.imshow(tensor.permute(2,1,0)[:,:,0].cpu().to(torch.float32).detach().numpy(), vmin=-2.5, vmax=2.5)
                        elif tensor.dim() == 2:
                            plt.imshow(tensor.permute(1, 0).cpu().to(torch.float32).detach().numpy(), vmin=-2.5, vmax=2.5)
                        
                        plt.title(tensor_name)
                        plt.show()
                    elif display == Show.MAX_MIN:
                        print(tensor_name, " max value: ", torch.max(tensor))
                        print(tensor_name, " min value: ", torch.min(tensor))


    def get_available_keys(self):
        return self.debugger_dict.keys()
    
    def print_from_key(self, keys: Union[str, list[str]]):
        if isinstance(keys, str): keys = [keys]
        for key in keys:
            for tensor_name_pairs, displays in self.debugger_dict[key]:
                self.write(tensor_name_pairs, displays)
    
    def make_video_from_tensors(
            self, 
            tensors: list[torch.Tensor],
            tensor_name: str,
            destination_file_name: str = f"tensor_debug_video",
            add_timestamp: bool = True
        ):

        if add_timestamp:
            destination_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{destination_file_name}"

        frames = []

        for tensor in tensors:
            
            # prepare the tensor for display
            fig, ax = plt.subplots()

            if tensor.dim() == 3:
                ax.imshow(tensor.permute(2, 1, 0).cpu().detach().numpy())
            elif tensor.dim() == 2:
                ax.imshow(tensor.cpu().detach().numpy())

            ax.set_title(tensor_name)
            
            # Saving the figure - use byteIO
            buf = BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)  # close the figure to free memory - otherwise it will hang around until the end of the program
            
            # Convert the image buffer to a numpy array that moviepy can use as a frame for the filnal video.
            img = Image.open(buf)
            frames.append(np.array(img))

        clip = ImageSequenceClip(frames, fps=30)

        # create folder if it does not exist
        import os
        if not os.path.exists("./debug_files"):
            os.makedirs("./debug_files")

        # We can change the bitrate to get a better quality video, but this might be too much
        clip.write_videofile(f"./debug_files/{destination_file_name}.mp4", codec="libx264", bitrate="50M")
        