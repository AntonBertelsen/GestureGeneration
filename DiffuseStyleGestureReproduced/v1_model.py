import torch
import torch.nn as nn
from local_attention import transformer
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from typing import Union
import numpy as np

from v1_sliding_diffusion import Diffusion
from debugger import Debugger, Show


class ContinuousMotionModel(nn.Module):
    def __init__(self, 
                device,
                n_gesture_length: int,                      # Length of the sequence snippets to generate. We geneate in autoregressive manner, where we are constantly generating small chunks continously
                diffusion_noise_scheduler: Diffusion,
                number_of_styles: int,                      # Number of unique styles. In this context this is the number of speakers, since we treat each speaker as a style 
                audio_features_per_frame: int,              # Number of audio features per frame. This is a mixture of prosodic features, onsets, wavlm, etc.
                pose_features_per_frame: int, 
                number_of_attention_heads: int = 8,
                 
                debugger: Debugger = Debugger(False)):      # Number of pose features per frame. These are the rotations / translations of the bones in the character skeleton. We may not pay attention to every channel for every bone, or every bone. 
        super().__init__()

        self.hyperparameter_dict_to_WnB_tracking = {
            "n_gesture_length": n_gesture_length,
            "number_of_styles": number_of_styles,
            "audio_features_per_frame": audio_features_per_frame,
            "pose_features_per_frame": pose_features_per_frame,
            "number_of_attention_heads": number_of_attention_heads,
        }

        self.device = device
        self.debugger = debugger
        self.diffusion_noise_scheduler = diffusion_noise_scheduler

        # Parameter from the diffusion model:
        self.max_timestep_stacking_level = diffusion_noise_scheduler.num_of_timestep_stackings

        self.num_of_pre_timestep_frames = diffusion_noise_scheduler.num_of_pre_timestep_frames
        self.num_of_timestep_frames = diffusion_noise_scheduler.num_of_timestep_frames
        self.num_of_post_timestep_frames = diffusion_noise_scheduler.num_of_post_timestep_frames

        self.n_gesture_length = n_gesture_length

        # TODO: This seems wrong, can't we just apply the error message to assert?
        try:
            assert (self.num_of_pre_timestep_frames +
                    self.num_of_timestep_frames +
                    self.num_of_post_timestep_frames == self.n_gesture_length)
        except AssertionError:
            print("Assertion failed!")
            print("Pre:", self.num_of_pre_timestep_frames,
                "Timestep:", self.num_of_timestep_frames,
                "Post:", self.num_of_post_timestep_frames,
                "Total Expected:", self.n_gesture_length)
            raise
        
        self.number_of_attention_heads = number_of_attention_heads

        # Implemetation of the DiffuseStyleGestureModel based on the paper by YoungSeng et al.
        # We instantiate all learned model layers needed below.

        # We precompute the pre- and post timestep vectores - that is timestep 0 and max_number_of_time_steps * self.max_timestep_stacking_level
        
        # We create tensors for the timestep linear layers
        self.timestep_0 = torch.tensor([0.0]).to(self.device) # TODO: should we send to device here or in the forward pass?
        self.timestep_max = torch.tensor([float(self.num_of_timestep_frames * self.max_timestep_stacking_level)]).to(self.device)  # TODO: should we send to device here or in the forward pass?
        
        # We rescale them to be between 0 and 1 to make it easier for the MLP positional embedding to learn
        # self.timestep_0 /= (self.num_of_timestep_frames * self.max_timestep_stacking_level)
        # self.timestep_max /= (self.num_of_timestep_frames * self.max_timestep_stacking_level)

        # Then we make the timestep vectores each of the timestep frames in the sequence - rescaling them to be between 0 and 1 # TODO: maybe
        self.timesteps_for_frames_at_stacking_level = {}
        for time_step_stacking_level in range(self.max_timestep_stacking_level):
            timesteps = [(t + 1) * self.max_timestep_stacking_level - (time_step_stacking_level % self.max_timestep_stacking_level) for t in range(self.num_of_timestep_frames)]
            # timesteps /= [t / (self.num_of_timestep_frames * self.max_timestep_stacking_level) for t in timesteps]
            timesteps = torch.tensor(timesteps, dtype=torch.bfloat16).unsqueeze(-1).to(self.device) # TODO: should we send to device here or in the forward pass?
            # print("!!!! timesteps", timesteps, "shape", timesteps.shape)
            self.timesteps_for_frames_at_stacking_level[time_step_stacking_level] = timesteps # 1 is added, since stakkings start at 1, not 0. The Stakking level multiplies the number of defusion steps each timestep undergoes, so if the stakking level is 0 - there is no diffusion steps. 

        # The time step encoding MLP. Our best guess is that this is actually a learned position encoding as described in Vaswani et al. 
        # Maybe it could be interesting to investigate using sinosoidal positional encoding? That would be one way to reduce the number 
        # of weights and maybe it would run faster? Sinosoidal position embeddings seem to work very well, and in Andrej Kaparthy's llm
        # video series he even investigates the gpt2 weights, and claim that they are not fully converged yet, because they are spiky.
        self.time_step_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )
        self.add_hyperparameters_to_WnB_tracking({"timestep_encoding_mlp_layers": 2, "time_step_mlp_output_dim": 64})

        # Style linear layer - for dimensionality expansion from a one-hot encoded (number_of_styles) to (64) shape
        # We move from a one hot encoded format to a 64 dimensional vector. My best guess is that instead of working 
        # with individual styles we sort of extract features of the style. For instance, 2 speakers might share the 
        # same general waviness in their gestures, and this can be encoded as a feature which is shared between the 
        # two speakers. This is a way to make the model more general, and to make it easier to generalize to new speakers.
        self.style_linear = nn.Linear(
            in_features=number_of_styles, 
            out_features=64
        )
        self.add_hyperparameters_to_WnB_tracking({"style_linear_output_dim": 64})
        
        # Audio feature linear layer per frame - for dimensionality reduction
        # The idea is to reduce the number of audio features per frame to a much smaller number. I know that we use wavlm 
        # which produces a huge number of embeddings. From the illustration, it is not clear if only wavlm is passed through 
        # the layer, or other audio featuers as well. Something to investigate.
        # TODO: think about transpose, I think we should do it directly in forward pass code stuff
        self.audio_linear = nn.Linear(
            in_features=audio_features_per_frame, 
            out_features=64
        )
        self.add_hyperparameters_to_WnB_tracking({"audio_linear_output_dim": 64})

        # Noisy gesture sequence linear layer - for dimensionality reduction
        # Must be applied to each frame vector in the sequence tensor, individually
        # Here we go from pose feature dimension (1141) to 256. I guess we are representing the pose in a more general way. 
        # Of course the vast majority of combinations of rotations of limbs are highly unlikely, so it makes some sense that 
        # we compress it a lot. Based on the idea of "nice" and "ugly" numbers which we got from Andrej Kaparthy, we are thinking 
        # that it might be good to change the number 1141. If we can get rid of / add some extra featuers to pay attention to it 
        # might make it faster.
        self.noisy_gesture_linear = nn.Linear(
            in_features=pose_features_per_frame, 
            out_features=256
        )
        self.add_hyperparameters_to_WnB_tracking({"noisy_gesture_linear_output_dim": 256})
        # Attention layers

        # embed_dim=256

        # The paper uses Local Attention. The idea here seems to be to pay attention only to local features.
        # For some reason they refer to this as cross-local attention. I think the reason for this name is that
        # they are attending to several modalities, but I dont think it is strictly correct to refer to this as
        # cross local attention. The original paper uses the implementation we have imported with some modifications.
        # (mostly simplifications)
        # We make do with the original implementation.
        # self.local_attention = LocalAttention(
        #     dim=576,            # This is the dimension of each head
        #                         # In this case we use a single head, so the dimension should match the input tensor (576)
        #                         # We are experimenting with multiheaded attention below, here we want 8 attention heads, 
        #                         # and since the input tensor is of shape (576, N) we need to have 72 as the dim (should be 
        #                         # handled automatically)
        #     
        #     window_size=15,     # This is the window size: The number of frames that are joined togehter in small windows.
        #                         # The idea is to only pay attention to local features (groups of 15 frames in this case)
        #                         # However, we can also look forward and look back to surrounding windows. This is still
        #                         # faster than full attention
        #
        #     causal=True,        # auto-regressive or not. If causal is true, we cannot attend to the future.
        #                         # TODO: This is interesting and worth exploring. Since we are using diffusion, there is no need to
        #                         # not attend to the future. We can imagine that gestures are informed by things that will occur
        #                         # in the near future. - for instance waving before saying "hello"
        #                         # The original paper uses causal=True, and they claim the model performs better
        #                         # when this is the case. 
        # 
        #     look_backward=1,    # Number of windows prior to the current window that we look at. Since this is 1, and the window size is 15,
        #                         # the total receptive field for the attention will be 30. (current window + the previous one)
        # 
        #     look_forward=0,     # In our case we will not attend to the future since we are generating in an causal manner
        #                         # Again, this is worth investigating - we suspect it might be better to look forward a bit, but the
        #                         # paper finds that it produces worse results. At least on their dataset of a single speaker.
        # 
        #     dropout=0.1,        # post-attention dropout
        # )

        self.pre_local_attention_linear = nn.Linear(
            in_features=64 + 64 + 256, 
            out_features=256
        )
        self.add_hyperparameters_to_WnB_tracking({"pre_local_attention_linear_output_dim": 256})

        # The original paper uses single headed local attention (above). However, the implementation supports multiheaded attention
        # We experiment with this to see if it can improve performance.
        self.multi_head_local_attention = transformer.LocalMHA(
            dim = 256,
            window_size=16, # Was 15 in the original paper
            dim_head=32,
            heads=8,
            dropout = 0.1,
            causal=True,    # TODO: remember that this is something we can experiment with
            look_backward = 1,
            look_forward = 0
        )
        self.add_hyperparameters_to_WnB_tracking({
            "local_attention_window_size": 16, 
            "local_attention_heads": 8, 
            "local_attention_dim_head": 32,
            "local_attention_look_backward": 1,
            "local_attention_causal": True,
        })

        # We reapply the relative positional embeddings before the transformer encoder
        # This is done in the original implementation, but we do it in a slightly different way.
        # They transform the input tensor in such a way that they can apply relative positional embeddings that will be identical
        # when the input tensor is sliced up into attention heads. We do not do this based on the intuition that each head
        # works off the entire input tensor in a linear projection. We don't think that there is any reason to repeat
        # positional embeddings for every head_dim values in the input tensor, since the heads are not constructed by slicing
        # the input tensor, but by projecting it into a lower dimension.
        # It may be worth investigating this further, and see if we can find any reason to do it the way they do it.

        self.relative_positional_embedding_funtion = SinusoidalEmbeddings(256)
        self.add_hyperparameters_to_WnB_tracking({
            "relative_positional_embedding_funtion_post_local_attention": True,
        })
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.add_hyperparameters_to_WnB_tracking({
            "transformer_encoder_layers": 6
        })

        # Final transformation, creating the output
        self.final_linear = nn.Linear(256, pose_features_per_frame)
    

    def display_debug_info(self, display_debug_info: bool, filter_keys: Union[str, list[str]]):
        self.debugger = Debugger(on=display_debug_info, keys=filter_keys)


    def forward(self, 
                current_time_step_stacking_level: int,
                one_hot_style, 
                audio_features, 
                noisy_gesture_sequence,
                condition_mask_probabilty = 0.1):

        self.debugger.capture([  
            ("one_hot_style", one_hot_style), 
            ("audio_features", audio_features), 
            ("noisy_gesture_sequence", noisy_gesture_sequence)], 
            [Show.MAX_MIN,], 
            keys=["input_analysis"])

        # Ensure all inputs have the same dtype as the model weights
        weight_dtype = self.audio_linear.weight.dtype
        one_hot_style = one_hot_style.to(weight_dtype)
        audio_features = audio_features.to(weight_dtype)
        noisy_gesture_sequence = noisy_gesture_sequence.to(weight_dtype)

        # 1.1 - Prepare the diffusion time steps t, associated with eatch frame in the sequence

        # TODO: This might be pre computable, by using signalsoidal positional encoding, 
        # or by using a allready trained mapping in infurrence. I might still need to be trained, as it interacts with a lot of other
        # positional encodings, and is in a slightly different context. It the learned positional encoding is not fully converged, we should
        # be able to see wether it resemples a signalsoidal positional encoding or not.

        # 1.1.1 - Add positional encoding to the t (current time step) using a MLP
        # This is our best guess as to what is happening. It is a bit unclear from the paper / code. The paper mentions that positional encoding is added
        # in the same manner as Vaswani et al (attention is all you need) but in the paper they primarily use / discuss sinonoidal positional encoding.
        # We know from Andrej Kaparthy that gpt2 used learned positional encoding, and that it was not fully converged. 

        #       Producing a tensor of shape (bs, number_of_frames, 192)

        # 1.1.2 - first we retreave the position embedding for timestep 0 and max_number_of_time_steps + 1, and 
        

        # 1.1.3 - Then we make the positional embedding each of the timestep frames in the sequence

        # t_for_each_timestep_frame = torch.arange(float(self.num_of_timestep_frames)).unsqueeze(-1).to(self.device)  # (bs, t_for_each_timestep_frame, 1)


        # For the post de-noising stpes
        # We divide the timestep by the max number of time steps, to make it easier for the MLP to learn
        # This gives values between 0 and 1, which is what the MLP is trained on.
        timestep_0_pos_embedding = self.time_step_mlp(self.timestep_0 / self.timestep_max)  # (bs, 1, 1) -> (bs, 1, 64)
        
        # For the pre de-noising stpes
        # We divide the timestep by the max number of time steps, to make it easier for the MLP to learn
        # This gives values between 0 and 1, which is what the MLP is trained on.
        timestep_max_pos_embedding = self.time_step_mlp(self.timestep_max / self.timestep_max)  # (bs, 1, 1) -> (bs, 1, 64)
        
        # For the de-nosing steps
        # print("!!!!, current_time_step_stacking_level: ", current_time_step_stacking_level)
        timesteps_for_frames = self.timesteps_for_frames_at_stacking_level[current_time_step_stacking_level]  # (bs, t_for_each_timestep_frame, 1) -> (bs, t_for_each_timestep_frame, 1)
        # print("!!!!, current_time_step_stacking_level: ", current_time_step_stacking_level,"  timesteps_for_frames IN THE FORWARD PASS: ", timesteps_for_frames, " shape: ", timesteps_for_frames.shape)
        t_with_pos_embedding_for_each_timestep_frame = self.time_step_mlp(
            timesteps_for_frames / self.timestep_max
        )  # (bs, t_for_each_timestep_frame, 64)

        self.debugger.capture(("t_with_pos_embedding_for_each_timestep_frame", t_with_pos_embedding_for_each_timestep_frame), 
            [Show.MAX_MIN, Show.IMAGE, Show.SHAPE],
            keys=["t_with_pos_embedding_for_each_timestep_frame", "timesteps_pos_embedding"])


        # 1.2 - Prepare the style tensor

        # 1.2.1 - Apply a linear layer to get a tensor of shape (bs, 1, 64)

        self.debugger.capture(("one_hot_style BEFORE style_linear", one_hot_style), [Show.MAX_MIN, Show.IMAGE, Show.SHAPE], keys="style")
        
        style = self.style_linear(one_hot_style)

        self.debugger.capture(("style after style_linear", style), [Show.MAX_MIN, Show.IMAGE, Show.SHAPE], keys="style")

        # 1.2.2 - Mask the style if apply_random_mask_to_style is True
        style_mask = torch.bernoulli(torch.full_like(style, 1 - condition_mask_probabilty))
        style *= style_mask

        self.debugger.capture(("style after style_mask", style), [Show.MAX_MIN, Show.IMAGE], keys="style")


        # 1.3 - Combine the style som timestep tensors using element vise addition

        style_plus_t_0_pos_embedding = style + timestep_0_pos_embedding
        style_plus_t_max_pos_embedding = style + timestep_max_pos_embedding

        pre_timesteps_style_vectors = style_plus_t_0_pos_embedding.unsqueeze(1).expand(style_plus_t_0_pos_embedding.shape[0], self.num_of_pre_timestep_frames, style_plus_t_0_pos_embedding.shape[1])
        post_timesteps_style_vectors = style_plus_t_max_pos_embedding.unsqueeze(1).expand(style_plus_t_max_pos_embedding.shape[0], self.num_of_post_timestep_frames, style_plus_t_max_pos_embedding.shape[1])

        style_plus_t_for_each_timestep_frame = t_with_pos_embedding_for_each_timestep_frame + style.unsqueeze(1).expand(style.shape[0], self.num_of_timestep_frames, style.shape[1])

        self.debugger.capture([("style_plus_t_0_pos_embedding", style_plus_t_0_pos_embedding), 
                               ("style_plus_t_max_pos_embedding", style_plus_t_max_pos_embedding),
                               ("style_plus_t_for_each_timestep_frame", style_plus_t_for_each_timestep_frame)], 
                               [Show.MAX_MIN, Show.IMAGE], 
                               keys=["style_plus_t_0_pos_embedding", "style", "timesteps_pos_embedding"])

        # 1.3.2 - concatenate the pre-timesteps, the style, and the post-timesteps
        style_t_frames = torch.cat([pre_timesteps_style_vectors, style_plus_t_for_each_timestep_frame, post_timesteps_style_vectors], dim=1)

        self.debugger.capture(("style_t_frames", style_t_frames), [Show.MAX_MIN, Show.IMAGE], keys=["style_t_frames", "style", "timesteps_pos_embedding"])    

        # 1.4 - Prepare the audio features tensor
        #       Apply a linear layer to get a tensor of shape (bs, N, 64) - every column is the features for that frame
        #       TODO: Consider making this 128 to make the final tenser of shape (bs, N, 640) (Nice number)

        self.debugger.capture(("audio_features BEFORE audio_linear", audio_features), [Show.MAX_MIN, Show.IMAGE], keys="audio_features")
        
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

        full_data_tensor = torch.cat([style_t_frames, audio_noisy_gesture], dim=-1)

        self.debugger.capture(("The final combied tensor of all the data", full_data_tensor), [Show.MAX_MIN, Show.IMAGE], keys="full_data_tensor")

        # 2.5 - Srink and mix the full_data_tensor to get a more compressed, optimsied tensor for the attention layers
        #       We apply a linear layer to get a tensor of shape (bs, N, 256)

        input = self.pre_local_attention_linear(full_data_tensor)

        self.debugger.capture(("input after pre_local_attention_linear", input), [Show.MAX_MIN, Show.IMAGE], keys=["attention_input", "pre_local_attention_linear"])

        # TODO: consider normaliseing the data

        # 3 - The Attention layers

        # We apply attention. This involves applying both local attention and self attention.
        # The idea is to first pay attention to local features, and then to pay attention to the global features.

        # 3.1 - Add RPE (Relative Positional Encoding) to the input
        # This is actually done in the local attention mechanism, so we do not need to do it here
        # This is a difference with their modified implementation of cross-local attention, where they add RPE before the local attention
        # We will use the original implementation, and add RPE in the local attention layer

        # 3.2 - apply local attention to the input tensor
        # local_attention_output = self.local_attention(
        #     q = input, 
        #     k = input, 
        #     v = input
        # )
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
    
    def forward_with_profiling(self, 
            current_time_step_stacking_level: int,
            one_hot_style, 
            audio_features, 
            noisy_gesture_sequence,
            condition_mask_probabilty = 0.1):
    
        from torch.profiler import profile, record_function, ProfilerActivity
        
        profiling_results = {}
        
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True) as prof:
            # Input preprocessing
            with record_function("input_prep"):
                # Ensure all inputs have the same dtype as the model weights
                weight_dtype = self.audio_linear.weight.dtype
                one_hot_style = one_hot_style.to(weight_dtype)
                audio_features = audio_features.to(weight_dtype)
                noisy_gesture_sequence = noisy_gesture_sequence.to(weight_dtype)
            
            # Time step embedding
            with record_function("timestep_embedding"):
                timestep_0_pos_embedding = self.time_step_mlp(self.timestep_0 / self.timestep_max)
                timestep_max_pos_embedding = self.time_step_mlp(self.timestep_max / self.timestep_max)
                timesteps_for_frames = self.timesteps_for_frames_at_stacking_level[current_time_step_stacking_level]
                t_with_pos_embedding_for_each_timestep_frame = self.time_step_mlp(
                    timesteps_for_frames / self.timestep_max
                )
            
            # Style processing
            with record_function("style_processing"):
                style = self.style_linear(one_hot_style)
                style_mask = torch.bernoulli(torch.full_like(style, 1 - condition_mask_probabilty))
                style *= style_mask
                
                style_plus_t_0_pos_embedding = style + timestep_0_pos_embedding
                style_plus_t_max_pos_embedding = style + timestep_max_pos_embedding
                
                pre_timesteps_style_vectors = style_plus_t_0_pos_embedding.unsqueeze(1).expand(
                    style_plus_t_0_pos_embedding.shape[0], self.num_of_pre_timestep_frames, style_plus_t_0_pos_embedding.shape[1]
                )
                post_timesteps_style_vectors = style_plus_t_max_pos_embedding.unsqueeze(1).expand(
                    style_plus_t_max_pos_embedding.shape[0], self.num_of_post_timestep_frames, style_plus_t_max_pos_embedding.shape[1]
                )
                
                style_plus_t_for_each_timestep_frame = t_with_pos_embedding_for_each_timestep_frame + style.unsqueeze(1).expand(
                    style.shape[0], self.num_of_timestep_frames, style.shape[1]
                )
                
                style_t_frames = torch.cat([pre_timesteps_style_vectors, style_plus_t_for_each_timestep_frame, post_timesteps_style_vectors], dim=1)
            
            # Feature processing
            with record_function("feature_processing"):
                audio_features = self.audio_linear(audio_features)
                noisy_gesture_sequence = self.noisy_gesture_linear(noisy_gesture_sequence)
                audio_noisy_gesture = torch.cat([audio_features, noisy_gesture_sequence], dim=-1)
                full_data_tensor = torch.cat([style_t_frames, audio_noisy_gesture], dim=-1)
                input_tensor = self.pre_local_attention_linear(full_data_tensor)
            
            # Attention layers
            with record_function("local_attention"):
                local_attention_output = self.multi_head_local_attention(input_tensor)
            
            with record_function("rotary_pos_emb"):
                relative_positional_embedding, scale = self.relative_positional_embedding_funtion(local_attention_output)
                local_attention_output, _ = apply_rotary_pos_emb(
                    local_attention_output, local_attention_output, relative_positional_embedding, scale
                )
            
            with record_function("transformer_encoder"):
                transformer_encoder_output = self.transformer_encoder(local_attention_output)
            
            # Final output
            with record_function("final_linear"):
                output_tensor = self.final_linear(transformer_encoder_output)
        
        profiling_results = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(profiling_results)
        
        # You might also want to save the profiling results to a file
        prof.export_chrome_trace("forward_trace.json")
        
        return output_tensor

    # Functions for Weights & Biases tracking
    def add_hyperparameters_to_WnB_tracking(self, hyperparameter_dict: dict):
        self.hyperparameter_dict_to_WnB_tracking.update(hyperparameter_dict)


    def get_WnB_config_specs(self):
        return self.hyperparameter_dict_to_WnB_tracking
