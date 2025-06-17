import torch
import torch.nn as nn
from torch.amp import autocast
from local_attention import transformer
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from typing import Union
from utils.WnB_trackable import WnBTrackable
from diffusion import Diffusion
from v1_sliding_diffusion import SlidingDiffusion
from v1_normal_diffusion import NormalDiffusion
from pose_encoder.pose_encoder import PoseEncoder
from pose_encoder.vae_pose_encoder import VAEPoseEncoder
from pose_encoder.advanced_pose_encoder import AdvancedPoseEncoder
from utils.debugger import Debugger, Show
import utils.utils as utils

class ContinuousMotionModel(nn.Module, WnBTrackable):
    def __init__(self, 
                n_gesture_length: int,                      # Length of the sequence snippets to generate. We geneate in autoregressive manner, where we are constantly generating small chunks continously
                audio_features_per_frame: int,              # Number of audio features per frame. This is a mixture of prosodic features, onsets, wavlm, etc.
                pose_features_per_frame: int,               # Number of pose features per frame. These are the rotations / translations of the bones in the character skeleton. We may not pay attention to every channel for every bone, or every bone. 
                number_of_styles: int,                      # Number of unique styles. In this context this is the number of speakers, since we treat each speaker as a style 
                diffusion: Diffusion,                       # Diffusion model to use for the model. This is used to generate the sequence snippets in an autoregressive manner.
                condition_mask_probabilty = 0.1,            # Probability of masking the style and audio features. This is used to condition the model on the style and audio features. The original paper uses 0.1, but we can experiment with this.
                number_of_attention_heads: int = 8,         # Number of attention heads to use in the local attention layer. The original paper uses 8, but we can experiment with this.
                pose_encoder: PoseEncoder = None,           # Autoencoder model to use for encoding the gesture sequence to a lower dimensional space. This is used to reduce the dimensionality of the input sequence.
                predict_full_duration: bool = True,         # If True, the model will predict the full duration of the gesture sequence. If False, it will predict only the noised frames at the end of the sequence.
                debugger: Debugger = Debugger(False),       # Debugger to use for debugging the model. This is used to capture and display information about the model during training and inference.
                device = utils.get_device()
            ):
        super().__init__()

        self.hyperparameter_dict_to_WnB_tracking = {
            "n_gesture_length": n_gesture_length,
            "audio_features_per_frame": audio_features_per_frame,
            "pose_features_per_frame": pose_features_per_frame,
            "number_of_styles": number_of_styles,
            "diffusion": diffusion.get_WnB_config_specs(),
            "condition_mask_probabilty": condition_mask_probabilty,
            "number_of_attention_heads": number_of_attention_heads,
            "predict_full_duration": predict_full_duration,
            "pose_encoder": pose_encoder.get_WnB_config_specs() if pose_encoder is not None else None,
            "device": device,
        }

        self.n_gesture_length = n_gesture_length
        self.audio_features_per_frame = audio_features_per_frame
        self.pose_features_per_frame = pose_features_per_frame
        self.number_of_styles = number_of_styles
        self.diffusion = diffusion
        self.condition_mask_probabilty = condition_mask_probabilty
        self.number_of_attention_heads = number_of_attention_heads
        self.pose_encoder = pose_encoder
        self.predict_full_duration = predict_full_duration
        self.debugger = debugger
        self.device = device
        self.pose_features_per_frame = pose_features_per_frame

        # If a pose encoder model is provided, we will use it to encode the gesture sequence before passing it to the main model.
        # If the pose encoder model is not provided, we will simply use the gesture sequence as is.
        if pose_encoder is not None:
            # Lock the weights of the pose encoder model since it is already trained.
            for param in pose_encoder.parameters():
                param.requires_grad = False
            pose_encoder.eval()

        # Maybe this serves as a learned position encoding as described in Vaswani et al? It's from the original paper, but we are not sure why it is needed.
        self.timestep_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )

        # Style linear layer - for dimensionality expansion from a one-hot encoded (number_of_styles) to (64) shape
        # We move from a one hot encoded format to a 64 dimensional vector. Instead of working with individual styles 
        # we can extract features of the style. For instance, 2 speakers might share the same general waviness in their 
        # gestures, and this can be encoded as a feature which is shared between the two speakers. This is a way to make 
        # the model more general, and to make it easier to generalize to new speakers.
        self.style_linear = nn.Linear(
            in_features=number_of_styles, 
            out_features=64
        )
        
        # Audio feature linear layer per frame - for dimensionality reduction
        self.audio_linear = nn.Linear(
            in_features=audio_features_per_frame, 
            out_features=64
        )

        # Noisy gesture sequence linear layer - for dimensionality reduction (TODO: Actually dimensionality expansion if we use pose encoder)
        self.noisy_gesture_linear = nn.Linear(
            in_features=pose_features_per_frame, 
            out_features=256
        )

        self.pre_local_attention_linear = nn.Linear(
            in_features=64 + 64 + 256, 
            out_features=256
        )

        # Local Attention. The idea here is to pay attention only to local features.
        self.multi_head_local_attention = transformer.LocalMHA(
            dim = 256,
            window_size=16, # Was 15 in the original paper
            dim_head=32,
            heads=8,
            dropout = 0.1,
            causal=True, # TODO: remember that this is something we can experiment with
            look_backward = 1,
            look_forward = 0
        )

        # We re-apply the relative positional embeddings before the transformer encoder
        # This is done in the original implementation, but we do it in a slightly different way.
        # They transform the input tensor in such a way that they can apply relative positional embeddings that will be identical
        # when the input tensor is sliced up into attention heads. We do not do this based on the intuition that each head
        # works off the entire input tensor in a linear projection. We don't think that there is any reason to repeat
        # positional embeddings for every head_dim values in the input tensor, since the heads are not constructed by slicing
        # the input tensor, but by projecting it into a lower dimension.
        # It may be worth investigating this further, and see if we can find any reason to do it the way they do it.

        self.relative_positional_embedding_funtion = SinusoidalEmbeddings(256)        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Final transformation, creating the output
        self.final_linear = nn.Linear(256, pose_features_per_frame)

    def display_debug_info(self, display_debug_info: bool, filter_keys: Union[str, list[str]]):
        self.debugger = Debugger(on=display_debug_info, keys=filter_keys)


    def forward(self, 
                timestep,
                one_hot_style, 
                audio_features, 
                noisy_gesture_sequence):

        self.debugger.capture([("one_hot_style", one_hot_style),("audio_features", audio_features),("noisy_gesture_sequence", noisy_gesture_sequence)],[Show.MAX_MIN,], keys=["input_analysis"])
        
        # Prepare the diffusion time steps t
        sequence_timesteps = self.diffusion.get_sequence_timesteps(timestep)

        bs, frames = sequence_timesteps.shape
        t_after_timestep_mlp = self.timestep_mlp(sequence_timesteps.reshape(-1, 1)).reshape(bs, frames, -1)
        self.debugger.capture(("t_after_timestep_mlp", t_after_timestep_mlp), [Show.MAX_MIN, Show.IMAGE, Show.SHAPE], keys=["t_after_timestep_mlp"])

        # Apply a linear layer to get a tensor of shape (bs, 1, 64)
        style = self.style_linear(one_hot_style)
        self.debugger.capture(("style AFTER style_linear", style), [Show.MAX_MIN, Show.IMAGE, Show.SHAPE], keys="style")

        # Mask the style to condition on style (as per the original paper) A little questionable, and not how I understood conditioning for classifier free guidance.
        if self.training:
            style_mask = torch.bernoulli(torch.full_like(style, 1 - self.condition_mask_probabilty))
            style *= style_mask
            self.debugger.capture(("style AFTER style_mask", style), [Show.MAX_MIN, Show.IMAGE], keys="style")

        # Reshape the style to be of shape (bs, sequence_length, 64) for broadcasting
        style = style.unsqueeze(1).repeat(1, frames, 1)  # (bs, N, 64)

        # Combine the style as timestep tensors using element vise addition
        style_with_t = t_after_timestep_mlp + style  # (bs, 1, 64) # TODO: Is this a good idea? Should we not maintain a fully separate timestep embedding to help the new diffusion approach?
        self.debugger.capture([("style_with_t", style_with_t)], [Show.MAX_MIN, Show.IMAGE], keys=["style"])

        # 1.4 - Prepare the audio features tensor
        #       Apply a linear layer to get a tensor of shape (bs, N, 64) - every column is the features for that frame

        audio_features = self.audio_linear(audio_features)
        self.debugger.capture(("audio_features AFTER audio_linear", audio_features), [Show.MAX_MIN, Show.IMAGE], keys="audio_features")
        
        # 1.5 - Prepare the noisy gesture sequence tensor (bs, N, 1141)
        #       Apply a linear layer to get a tensor of shape (bs, N, 256) - every column is the features for that frame
        #       The linear layer is applied per frame
        noisy_gesture_sequence = self.noisy_gesture_linear(noisy_gesture_sequence)
        self.debugger.capture(("noisy_gesture_sequence AFTER noisy_gesture_linear", noisy_gesture_sequence), [Show.MAX_MIN, Show.IMAGE], keys="noisy_gesture_sequence")

        # 2 - Combine input tensors to get the input tensor for the model

        # 2.2 - Concatenate the audio features tensor and Noisy gesture sequence tensor
        #       To get a tensor of shape (320, N)
        audio_noisy_gesture = torch.cat([audio_features, noisy_gesture_sequence], dim=-1)  # (320, N)
        self.debugger.capture(("audio_noisy_gesture", audio_noisy_gesture), [Show.MAX_MIN, Show.IMAGE], keys="audio_noisy_gesture")

    
        # 2.4 - Concatenate the seed_style_t and the audio_features_noisy_gesture_sequence tensor
        #       To get a tensor of shape (576, N) (Could be nicer)
        #       This gives us the 'input' for the model

        full_data_tensor = torch.cat([style_with_t, audio_noisy_gesture], dim=-1)
        self.debugger.capture(("The final combied tensor of all the data", full_data_tensor), [Show.MAX_MIN, Show.IMAGE], keys="full_data_tensor")

        # 2.5 - Srink and mix the full_data_tensor to get a more compressed, optimsied tensor for the attention layers
        #       We apply a linear layer to get a tensor of shape (bs, N, 256)

        input = self.pre_local_attention_linear(full_data_tensor)
        self.debugger.capture(("input after pre_local_attention_linear", input), [Show.MAX_MIN, Show.IMAGE], keys=["attention_input", "pre_local_attention_linear"])

        # 3 - The Attention layers

        # We apply attention. This involves applying both local attention and self attention.
        # The idea is to first pay attention to local features, and then to pay attention to the global features.

        # 3.1 - Add RPE (Relative Positional Encoding) to the input
        # This is actually done in the local attention mechanism, so we do not need to do it here
        # This is a difference with their modified implementation of cross-local attention, where they add RPE before the local attention
        # We will use the original implementation, and add RPE in the local attention layer

        # 3.2 - apply local attention to the input tensor
        local_attention_output = self.multi_head_local_attention(input)
        self.debugger.capture(("local_attention_output", local_attention_output), [Show.MAX_MIN, Show.IMAGE], keys="local_attention_output")
        
        # 3.3 - Apply a self attention layer to the tensor of shape (256, N+1)
        #       We now apply full self attention. The paper and illustration makes it look as though we are applying a single
        #       self attention layer, but the code seems to actually apply a full 8 layer encoder transformer model.
        #       This is a little surprising to us. 
        
        #       Note that because we have appended an extra "frame" with the seed_style_t at the beginning of the sequence
        #       We will have N+1 frames in the sequence. We will ignore the first frame output ([:,1:]) from the transformer, 
        #       since it is the seed_style_t frame. This is in accordance with the implementation from the original paper.

        relative_positional_embedding, scale = self.relative_positional_embedding_funtion(local_attention_output)
        
        local_attention_output, _ = apply_rotary_pos_emb(local_attention_output, local_attention_output, relative_positional_embedding, scale)
        self.debugger.capture(("combined_tensor with RPE", local_attention_output), [Show.MAX_MIN, Show.IMAGE], keys="combined_tensor")
        
        transformer_encoder_output = self.transformer_encoder(local_attention_output)
        self.debugger.capture(("transformer_encoder_output", transformer_encoder_output), [Show.MAX_MIN, Show.IMAGE], keys="transformer_encoder_output")
        # 3.4 - Pass the output of the self attention layer to a linear layer,
        #       to get a tensor of shape (1141, N)
        output_tensor = self.final_linear(transformer_encoder_output)
        self.debugger.capture(("output_tensor", output_tensor), [Show.MAX_MIN, Show.IMAGE], keys="output_tensor")

        # 4 - Return the output of the liniear layer
        return output_tensor

    def generate(self, gesture_sequence, audio_features, main_agent_id_one_hot, gesture_sequence_is_encoded: bool = False):
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):

            # If the autoencoder model is provided, we use it to encode the gesture sequence to a lower dimensional latent space
            encoded_gesture_sequence = self.pose_encoder.encode(gesture_sequence) if self.pose_encoder is not None and not gesture_sequence_is_encoded else gesture_sequence

            # We diffuse the encoded gesture sequence to create a noisy starting point for the diffusion model.
            # generate a random timestep for each sequence in the batch
            per_sequence_timestep = torch.randint(0, self.diffusion.number_of_timesteps, (encoded_gesture_sequence.shape[0],), device=self.device)
            noisy_gesture_sequence = self.diffusion.forward(encoded_gesture_sequence, per_sequence_timestep)

            # Apply the model to denoise the noisy gesture sequence.
            encoded_output = self.forward(
                timestep                    = per_sequence_timestep,
                one_hot_style               = main_agent_id_one_hot,
                audio_features              = audio_features, 
                noisy_gesture_sequence      = noisy_gesture_sequence
            )
            
            if not self.predict_full_duration:
                # Ensure the replace_frames does not exceed the length of the gesture sequence
                encoded_output[:, :self.diffusion.clean_frame_index] = encoded_gesture_sequence[:, :self.diffusion.clean_frame_index]

            # If the autoencoder model is provided, we use it to decode the output
            output = self.pose_encoder.decode(encoded_output) if self.pose_encoder is not None and not gesture_sequence_is_encoded else encoded_output

            return output, encoded_gesture_sequence, encoded_output, noisy_gesture_sequence

    def inference(self, starting_point, audio_features, main_agent_id_one_hot):
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # Shift the gesture_sequence by one frame
            shifted_gesture_sequence = torch.roll(starting_point, shifts=-1, dims=1)
            
            # Replace the last frame with pure noise
            shifted_gesture_sequence[0,-1] = torch.zeros_like(shifted_gesture_sequence[0,-1])
            gesture_sequence = shifted_gesture_sequence

            for stacking_level in range(self.diffusion.number_of_timesteps):                
                # We apply noise to the gesture sequence at every iteration because we predict the clean image at every step.                
                noisy_gesture_sequence = self.diffusion.forward(gesture_sequence, stacking_level)

                gesture_sequence = self.forward(
                    time_step_stacking_level    = stacking_level,
                    one_hot_style               = main_agent_id_one_hot,
                    audio_features              = audio_features, 
                    noisy_gesture_sequence      = noisy_gesture_sequence
                )

                if not self.predict_full_duration:
                    # Ensure the replace_frames does not exceed the length of the gesture sequence
                    gesture_sequence[:, :self.diffusion.clean_frame_index] = shifted_gesture_sequence[:, :self.diffusion.clean_frame_index]

            return gesture_sequence, noisy_gesture_sequence

    # Functions for Weights & Biases tracking
    def add_hyperparameters_to_WnB_tracking(self, hyperparameter_dict: dict):
        self.hyperparameter_dict_to_WnB_tracking.update(hyperparameter_dict)

    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking

    def get_model_state(self):        
        return {
            'state_dict': self.state_dict(),
            'config': {
                **self.hyperparameter_dict_to_WnB_tracking,
                'diffusion': self.diffusion.get_WnB_config_specs(),
                'pose_encoder': self.pose_encoder.get_WnB_config_specs() if self.pose_encoder else None,
            }
        }

    @staticmethod
    def load_model(model_checkpoint, device=None):
        
        # If a path is provided, load the model checkpoint from the file
        if isinstance(model_checkpoint, str):
            model_checkpoint = torch.load(model_checkpoint, map_location='cpu', weights_only=False)['model_state']

        # Extract configuration
        config = model_checkpoint['config']

        model = ContinuousMotionModel.construct_model(config, device)
        
        # Fix state_dict keys if compiled with torch.compile
        state_dict = model_checkpoint['state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    # This constructs a new model based on a wandb configuration dictionary, which holds all the hyper parameters for the model.
    def construct_model(config: dict, device):
        # Get diffusion noise schedule
        schedule_name = config['diffusion']['name']
        schedule_func = getattr(Diffusion, f"{schedule_name}", None)
        noise_schedule = schedule_func(
            config['diffusion']['beta_min'],
            config['diffusion']['beta_max']
        )
        
        diffusion_type = config['diffusion'].get('type', 'normal_diffusion') # TODO: FIX!!!
        if diffusion_type == 'normal_diffusion':
            # Create normal diffusion model
            diffusion = NormalDiffusion(
                num_timesteps               = config['diffusion']['num_timesteps'],
                sequence_length             = config['n_gesture_length'],
                noise_schedule              = noise_schedule,
                device                      = device
            )
        elif diffusion_type == 'sliding_diffusion':
            # Create sliding diffusion model
            diffusion = SlidingDiffusion(
                num_clean_frames            = config['diffusion']['num_clean_frames'],
                num_denoise_frames          = config['diffusion']['num_denoise_frames'],
                num_noise_frames            = config['diffusion']['num_noise_frames'],
                num_timestep_stackings      = config['diffusion']['num_timestep_stackings'],
                noise_schedule              = noise_schedule,
                device                      = device
            )
        
        # Create pose encoder if it was used
        pose_encoder = None
        if config['pose_encoder']:
            pose_encoder_type = config['pose_encoder']['type']
            pose_encoder_checkpoint_name = config['pose_encoder']['checkpoint_name']

            if pose_encoder_type == 'vae_pose_encoder':
                pose_encoder = VAEPoseEncoder.load_from_checkpoint(pose_encoder_checkpoint_name, device)
            elif pose_encoder_type == 'advanced_pose_encoder':
                pose_encoder = AdvancedPoseEncoder.load_from_checkpoint(pose_encoder_checkpoint_name, device)
        
        # Create the main model
        model = ContinuousMotionModel(
            n_gesture_length            = config['n_gesture_length'],
            audio_features_per_frame    = config['audio_features_per_frame'],
            pose_features_per_frame     = config['pose_features_per_frame'],
            number_of_styles            = config['number_of_styles'],
            diffusion                   = diffusion,
            condition_mask_probabilty   = config['condition_mask_probabilty'],
            number_of_attention_heads   = config['number_of_attention_heads'],
            predict_full_duration       = config['predict_full_duration'],
            pose_encoder                = pose_encoder,
            device                      = device
        ).to(device)

        return model