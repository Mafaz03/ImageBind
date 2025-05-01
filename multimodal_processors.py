import torch
from torch import nn
from typing import Callable, List, Optional, Tuple
import math
from timm.models.layers import trunc_normal_
import numpy as np

class VerboseNNModule(nn.Module):
    
    @staticmethod
    def get_readable_tensor_representation(name: str, tensor: torch.Tensor):
        st = (
            "(" + name + "): " + "tensor(" + str(tuple(tensor[1].shape)) + ", requires_grad=" + str(tensor[1].requires_grad) + ")\n"
            )
        return st
    
    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update(p[0])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name in named_modules:
                string_repr += self.get_readable_tensor_representation(name, p)
        
        for p in self.named_buffers():
            name = p[0].split(".")[0]
            string_repr += self.get_readable_tensor_representation(name, p)
        
        return string_repr

class MyLayer(VerboseNNModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)
        self.register_buffer("running_mean", torch.zeros(5))

def build_causal_attention_mask(context_length):
    mask = torch.empty(context_length, context_length, requires_grad=False)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask

class TextPreprocessor(VerboseNNModule):
    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, causual_mask: bool, 
                 supply_seq_len_to_head: bool = True, init_param_style: str = "openclip"):
        """
        `vocab_size`: Number of tokens in your vocabulary.                 the number of words in your text, so we can map nn.Embedding
	    `context_length`: Maximum number of tokens per input sequence.     usually: 77
	    `embed_dim`: Dimensionality of each token embedding.               usually: 768
        """

        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.causual_mask = causual_mask
        self.embed_dim = embed_dim
        self.supply_seq_len_to_head = supply_seq_len_to_head
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.empty(1, context_length, embed_dim)
        )
        if causual_mask:
            mask = build_causal_attention_mask(context_length)
            self.register_buffer("mask", mask) # register the mask as a buffer so it can be moved to the right device
        
        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style = "openclip"):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # I did'nt use init_param_style as I was too lazy to implment [CLS]
    
    def forward(self, text):
        token_text = self.token_embedding(text)
        token_text = token_text + self.pos_embed
        
        return_dict = {
            "trunk": {
                "tokens": token_text
            },
            "head": {},
        }

        if self.supply_seq_len_to_head:
            text_lengths = text.argmax(dim = -1)
            #  hacky and non-standard way of getting the sequence length.
            return_dict["head"] = {
                "seq_len": text_lengths,
            }
        if self.causual_mask:
            return_dict["trunk"].update({"attn_mask": self.mask})
        
        return return_dict

class Im2Video(VerboseNNModule):
    """ Converts image to video (Just adding T dimension lol)"""
    def __init__(self, time_dim = 2):
        super().__init__()
        self.time_dim = time_dim
    
    def forward(self, x: torch.Tensor):
        if x.ndim == 5:
            # Already includes T dimension
            return x
        if x.ndim == 4:
            # Convert (B, C, H, W) -> (B, C, T, H, W)
            return x.unsqueeze(dim = self.time_dim)
        raise ValueError(f"Dimension incorrect {x.shape}")

class PadIm2Video(Im2Video):
    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__(time_dim=time_dim)
        assert ntimes > 0
        assert pad_type in ["zero", "repeat"]
        self.ntimes = ntimes
        self.pad_type = pad_type
    
    def forward(self, x: torch.Tensor):
        x = super().forward(x) # (B, C, H, W) -> (B, C, T, H, W)
        if x.shape[self.time_dim] == 1:
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
                return x
            elif self.pad_type == "zero":
                raise NotImplemented(f"Todo: Need to implement this in the future")
        else: return x 
class PatchEmbedGeneric(nn.Module):
    def __init__(self, proj_stem, norm_layer: Optional[Callable] = None):
        super().__init__()

        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]
        self.norm_layer = norm_layer
    
    def get_patch_layout(self, image_size):
        with torch.no_grad():
            dummy_img = torch.zeros([1,] + image_size)      # 1, C, (T), H, W
            dummy_out = self.proj(dummy_img)
        # print(dummy_out.shape)
        embed_dim = dummy_out.shape[1]                    # `embed_dim`    = C        
        patch_layout = tuple(dummy_out.shape[2:])         # `patch_layout` = (T), H, W       
        num_patches = np.prod(patch_layout)               # `num_patches`  = (T) * H * W       
        return embed_dim, patch_layout, num_patches
    
    def forward(self, x: torch.Tensor):
        x = x.flatten(2)                                  # B, C, (T), H, W -> B, C, (T)*H*W
        x = x.transpose(1, 2)                             # B, C, (T)*H*W   -> B, (T)*H*W, C
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def interpolate_pos_encoding(npatch_per_image, pos_embed, first_patch_idx: int = 1): 
    # If CLS present first_patch_idx = 1

    assert first_patch_idx == 0 or first_patch_idx == 1, "CLS can be either present or not present"
    # assert 
    N = pos_embed.shape[1] - first_patch_idx             # If CLS is present tokens from the 1: to rest are actual stuff
    
    if npatch_per_image == N:
        return pos_embed

    class_emb = pos_embed[:, :first_patch_idx]
    pos_embed = pos_embed[:, first_patch_idx:]

    return torch.cat((class_emb, pos_embed), dim=1)

def _get_pos_embedding(npatch_per_image, pos_embed, first_patch_idx: int = 1):
    return interpolate_pos_encoding(npatch_per_image, pos_embed, first_patch_idx)

class SpatioTemporal_posEmbeddingHelper(VerboseNNModule):
    def __init__(self, num_patches: int, num_cls_tokens: int, embed_dim: int, learnable: bool):
        super().__init__()
        self.num_patches = num_patches
        self.num_cls_tokens = num_cls_tokens
        self.embed_dim = embed_dim
        self.learnable = learnable

        self.num_tokens = num_patches + num_cls_tokens

        if learnable:
            self.pos_embed = nn.Parameter(
                                torch.zeros(1, self.num_tokens, embed_dim)
                            )
            trunc_normal_(self.pos_embed, std=0.02)

        else: self.register_buffer(
            "pos_embed", get_sinusoid_encoding_table(n_position = self.num_tokens, d_hid = embed_dim)
            )
    
    def get_pos_embedding(self, all_vision_tokens):
        pos_embed = _get_pos_embedding(
            npatch_per_image = all_vision_tokens.size(1) - self.num_cls_tokens,
            pos_embed=self.pos_embed,
            first_patch_idx=self.num_cls_tokens,
        )
        return pos_embed

class RGBTProcessor(VerboseNNModule):
    def __init__(self, rgbt_stem: PatchEmbedGeneric, img_size: Tuple = [3, 224, 224],
                 num_cls_token: int = 1, pos_embed_fn: SpatioTemporal_posEmbeddingHelper = None, 
                 use_type_embed: bool = False, init_param_style: str = "openclip"):
        super().__init__()

        self.embed_dim, self.patches_layout, self.num_patches = rgbt_stem.get_patch_layout(img_size)
        self.num_cls_token = num_cls_token
        self.use_type_embed = use_type_embed
        self.init_param_style = init_param_style
        self.use_pos_embed = pos_embed_fn is not None
        self.rgbt_stem = rgbt_stem

        if self.use_pos_embed:
            self.pos_embed_helper = pos_embed_fn(
                num_patches = self.num_patches,
                num_cls_tokens = self.num_cls_token,
                embed_dim = self.embed_dim,
                learnable = True
            )
        
        if num_cls_token > 0:
            self.cls_tokens = nn.Parameter(
                torch.zeros(1, self.num_cls_token, self.embed_dim)
            )
        if self.use_type_embed: # The model learns to adjust type_embed so that it provides differentiation for different modalities
            self.type_embed = nn.Parameter(
                torch.zeros(1, 1, self.embed_dim)
            )
        
        self.init_parameters(init_param_style)
    
    @torch.no_grad()
    def init_parameters(self, parameter_style):
        if parameter_style == "openclip":
            # OpenCLIP style initialization
            scale = self.embed_dim ** -0.5
        
            if self.use_type_embed:
                nn.init.normal_(self.pos_embed_helper.pos_embed)
                self.pos_embed_helper.pos_embed *= scale
            
            if self.num_cls_token > 0:
                nn.init.normal_(self.cls_tokens)
                self.cls_tokens *= scale
        
        elif parameter_style == "vit":
            self.cls_tokens.data.fill_(0)
        
        else:
            raise ValueError(f"Unknown init {parameter_style}")
        
        if self.use_type_embed:
            nn.init.normal_(self.type_embed)
        
    def tokenize_input_and_cls_pos(self, input, stem):
        tokens = stem(input)
        assert tokens.ndim == 3
        assert tokens.shape[-1] == self.embed_dim

        B = tokens.shape[0] # batch size
        
        if self.num_cls_token > 0:
            cls_tokens = self.cls_tokens.expand(B, -1, -1)   # Making sure Batches are matching or shape mismatch might occur
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        if self.use_pos_embed:
            pos_embed = self.pos_embed_helper.get_pos_embedding(all_vision_tokens = tokens)
            tokens = tokens + pos_embed
        
        if self.use_type_embed:
            tokens = tokens + self.type_embed.expand(B, -1, -1)
        
        return tokens
    
    def forward(self, vision = None):
        vision_tokens = self.tokenize_input_and_cls_pos(input = vision, stem = self.rgbt_stem)
        return_dict = {
                        "trunk": {
                            "tokens": vision_tokens
                        },
                        "head": {}
                    }
        return return_dict

