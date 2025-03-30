# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Tuple, Dict, Any, List, Set, Optional, Union, Callable, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
import lightning as L

from hvit.model.blocks import *
from hvit.model.transformation import *


WO_SELF_ATT = False # without self attention
_NUM_CROSS_ATT = -1
ndims = 3 # H,W,D

class Attention(nn.Module):
    """
    Attention module for hierarchical vision transformer.

    This module implements both local and global attention mechanisms.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        patch_size: Union[int, List[int]],
        attention_type: str = "local",
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ) -> None:
        """
        Initialize the Attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            patch_size (Union[int, List[int]]): Size of the patches.
            attention_type (str): Type of attention mechanism ("local" or "global").
            qkv_bias (bool): Whether to use bias in query, key, value projections.
            qk_scale (Optional[float]): Scale factor for query-key dot product.
            attn_drop (float): Dropout rate for attention matrix.
            proj_drop (float): Dropout rate for output projection.
        """
        super().__init__()

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.patch_size: List[int] = [patch_size] * ndims if isinstance(patch_size, int) else patch_size
        self.attention_type: str = attention_type
        
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.head_dim: int = dim // num_heads
        self.scale: float = qk_scale or self.head_dim ** -0.5

        # Skip initialization if using local attention without self-attention
        if self.attention_type == "local" and WO_SELF_ATT:
            return

        # Initialize query, key, value projections based on attention type
        if attention_type == "local":
            self.qkv: nn.Linear = nn.Linear(dim, dim * 3, bias=qkv_bias)
        elif attention_type == "global":
            self.qkv: nn.Linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop: nn.Dropout = nn.Dropout(attn_drop)
        self.proj: nn.Linear = nn.Linear(dim, dim)
        self.proj_drop: nn.Dropout = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, q_ms: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (Tensor): Input tensor.
            q_ms (Optional[Tensor]): Query tensor for global attention.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        B_, N, C = x.size()

        # Return input if using local attention without self-attention
        if self.attention_type == "local" and WO_SELF_ATT:
            return x

        if self.attention_type == "local":
            qkv: Tensor = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
        else:
            B: int = q_ms.size()[0]
            kv: Tensor = self.qkv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            q: Tensor = self._process_global_query(q_ms, B, B_, N, C)

        # Compute attention scores and apply attention
        attn: Tensor = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x: Tensor = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def _process_global_query(self, q_ms: Tensor, B: int, B_: int, N: int, C: int) -> Tensor:
        """
        Process the global query tensor.

        Args:
            q_ms (Tensor): Global query tensor.
            B (int): Batch size of q_ms.
            B_ (int): Batch size of input tensor.
            N (int): Number of patches.
            C (int): Channel dimension.

        Returns:
            Tensor: Processed global query tensor.
        """
        q_tmp: Tensor = q_ms.reshape(B, self.num_heads, N, C // self.num_heads)
        div_, rem_ = divmod(B_, B)
        q_tmp = q_tmp.repeat(div_, 1, 1, 1)
        q_tmp = q_tmp.reshape(B * div_, self.num_heads, N, C // self.num_heads)
        
        q: Tensor = torch.zeros(B_, self.num_heads, N, C // self.num_heads, device=q_ms.device)
        q[:B*div_] = q_tmp
        if rem_ > 0:
            q[B*div_:] = q_tmp[:rem_]
        
        return q * self.scale


def get_patches(x: Tensor, patch_size: int) -> Tuple[Tensor, int, int, int]:
    """
    Divide the input tensor into patches and reshape them for processing.

    Args:
        x (Tensor): Input tensor of shape (B, H, W, D, C).
        patch_size (int): Size of each patch.

    Returns:
        Tuple[Tensor, int, int, int]: A tuple containing:
            - windows: Reshaped tensor of patches.
            - H, W, D: Updated dimensions of the input tensor.
    """
    B, H, W, D, C = x.size()
    nh: float = H / patch_size
    nw: float = W / patch_size
    nd: float = D / patch_size

    # Check if downsampling is required
    down_req: float = (nh - int(nh)) + (nw - int(nw)) + (nd - int(nd))
    if down_req > 0:
        # Downsample the input tensor to fit patch size
        new_dims: List[int] = [int(nh) * patch_size, int(nw) * patch_size, int(nd) * patch_size]
        x = downsampler_fn(x.permute(0, 4, 1, 2, 3), new_dims).permute(0, 2, 3, 4, 1)
        B, H, W, D, C = x.size()

    # Reshape the tensor into patches
    x = x.view(B, H // patch_size, patch_size,
               W // patch_size, patch_size,
               D // patch_size, patch_size,
               C)
    
    # Rearrange dimensions and flatten patches
    windows: Tensor = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, patch_size, patch_size, patch_size, C)
    
    return windows, H, W, D


def get_image(windows: Tensor, patch_size: int, Hatt: int, Watt: int, Datt: int, H: int, W: int, D: int) -> Tensor:
    """
    Reconstruct the image from windows (patches).

    Args:
        windows (Tensor): Input tensor containing the windows.
        patch_size (int): Size of each patch.
        Hatt, Watt, Datt (int): Dimensions of the attention space.
        H, W, D (int): Original dimensions of the image.

    Returns:
        Tensor: Reconstructed image.
    """
    # Calculate batch size
    B: int = int(windows.size(0) / ((Hatt * Watt * Datt) // (patch_size ** 3)))
    
    # Reshape windows into image
    x: Tensor = windows.view(B, 
                    Hatt // patch_size,
                    Watt // patch_size,
                    Datt // patch_size,
                    patch_size, patch_size, patch_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hatt, Watt, Datt, -1)

    # Downsample if necessary
    if H != Hatt or W != Watt or D != Datt:
        x = downsampler_fn(x.permute(0, 4, 1, 2, 3), [H, W, D

