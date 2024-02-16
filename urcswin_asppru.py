import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from torchinfo import summary

class MaxPoolLayer(nn.Sequential):
    def __init__(self, kernel_size=3, dilation=1, stride=1):
        super(MaxPoolLayer, self).__init__(
            nn.MaxPool2d(kernel_size=kernel_size, dilation=dilation, stride=stride,
                         padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class AvgPoolLayer(nn.Sequential):
    def __init__(self, kernel_size=3, stride=1):
        super(AvgPoolLayer, self).__init__(
            nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                         padding=(kernel_size-1)//2)
        )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class TransposeConvBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, norm_layer=nn.BatchNorm2d):
        super(TransposeConvBNReLu, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            norm_layer(out_channels),
            nn.ReLU()
        )


class TransposeConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, norm_layer=nn.BatchNorm2d):
        super(TransposeConvBN, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            norm_layer(out_channels)
        )


class TransposeConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(TransposeConv, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )


class PyramidPool(nn.Sequential):
    def __init__(self, in_channels, out_channels, pool_size=1, norm_layer=nn.BatchNorm2d):
        super(PyramidPool, self).__init__(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # print(self.relative_position_bias_table.shape)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # print(coords_h, coords_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # print(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # print(coords_flatten)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # print(relative_coords[0,7,:])
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # print(relative_coords[:, :, 0], relative_coords[:, :, 1])
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(B_,N,C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # print(attn.shape)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # print(relative_position_bias.unsqueeze(0))
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained models,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.apply(self._init_weights)

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        # print('patch_embed', x.size())

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                # print('layer{} out size {}'.format(i, out.size()))

        return tuple(outs)

    def train(self, mode=True):
        """Convert the models into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


def l2_norm(x):   # Q的形状 torch.Size([1, 48, 1024])  后面: [1, 48]
    # print('-------显示结构-----------')
    # print('x进来的结构',x.shape)
    # print(torch.norm(x, p=2, dim=-2).shape)
    # print('------------------------')
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


# class SharedSpatialAttention(nn.Module):
#     def __init__(self, in_places, eps=1e-6):
#         super(SharedSpatialAttention, self).__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.in_places = in_places
#         self.l2_norm = l2_norm
#         self.eps = eps
#
#         self.query_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)
#
#     def forward(self, x):
#         # Apply the feature map to the queries and keys
#         batch_size, chnnels, width, height = x.shape
#         # print("x的形状",x.shape)
#         Q = self.query_conv(x).view(batch_size, -1, width * height)
#         K = self.key_conv(x).view(batch_size, -1, width * height)
#         V = self.value_conv(x).view(batch_size, -1, width * height)
#         # print("Q的形状",Q.shape)
#         # print("k的形状", K.shape)
#         # print("v的形状", V.shape)
#
#         Q = self.l2_norm(Q).permute(-3, -1, -2)
#         # print('q', Q.shape)
#         K = self.l2_norm(K)
#         # print('k', K.shape)
#         # print("torch.sum(K, dim=-1)",torch.sum(K, dim=-1).shape)
#
#         tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
#         # print('tailor_sum', tailor_sum.shape)
#         value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
#         # print('value_sum', value_sum.shape)
#         value_sum = value_sum.expand(-1, chnnels, width * height)
#         # print('value_sum', value_sum.shape)
#
#         matrix = torch.einsum('bmn, bcn->bmc', K, V)
#         # print('matrix',matrix.shape)
#         matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
#         # print('matrix_sum', matrix_sum.shape)
#
#         weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
#         # print('weight_value', weight_value.shape)
#         # 2023.4.3修改
#         # weight_value = weight_value.view(batch_size, chnnels, height, width)
#         weight_value = weight_value.view(batch_size, chnnels, width,height)
#         # print('----------------v----------------------------------------------------------------')
#         # print("weight_value",weight_value.shape)
#         # print("x", x.shape)
#         # weight_value = weight_value
#         return (x + self.gamma * weight_value).contiguous()


# class SharedChannelAttention(nn.Module):
#     def __init__(self, eps=1e-6):
#         super(SharedChannelAttention, self).__init__()
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.l2_norm = l2_norm
#         self.eps = eps
#
#     def forward(self, x):
#         batch_size, chnnels, width, height = x.shape
#         Q = x.view(batch_size, chnnels, -1)
#         K = x.view(batch_size, chnnels, -1)
#         V = x.view(batch_size, chnnels, -1)
#
#         Q = self.l2_norm(Q)
#         K = self.l2_norm(K).permute(-3, -1, -2)
#
#         tailor_sum = 1 / (width * height + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))
#         value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
#         value_sum = value_sum.expand(-1, chnnels, width * height)
#         matrix = torch.einsum('bcn, bnm->bcm', V, K)
#         matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)
#
#         weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
#         # 2023.4.3修改
#         # weight_value = weight_value.view(batch_size, chnnels, height, width)
#         weight_value = weight_value.view(batch_size, chnnels, width, height)
#
#         return (x + self.gamma * weight_value).contiguous()


class DownConnection(nn.Module):
    def __init__(self, inplanes, planes, stride=2):
        super(DownConnection, self).__init__()
        self.convbn1 = ConvBN(inplanes, planes, kernel_size=3, stride=1)
        self.convbn2 = ConvBN(planes, planes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = ConvBN(inplanes, planes, stride=stride)

    def forward(self, x):   #x2
        residual = x
        x = self.convbn1(x)
        x = self.relu(x)
        x = self.convbn2(x)
        x = x + self.downsample(residual)
        x = self.relu(x)

        return x


class DCFAM(nn.Module):
    def __init__(self, encoder_channels=(96, 192, 384, 768), atrous_rates=(6, 12)):

        super(DCFAM, self).__init__()
        rate_1, rate_2 = tuple(atrous_rates)
        self.conv4 = Conv(encoder_channels[3], encoder_channels[3], kernel_size=1)
        self.conv1 = Conv(encoder_channels[0], encoder_channels[0], kernel_size=1)

        self.lf4 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-1], encoder_channels[-2], dilation=rate_1),
                                 nn.UpsamplingNearest2d(scale_factor=2),
                                 SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))
        self.lf3 = nn.Sequential(SeparableConvBNReLU(encoder_channels[-2], encoder_channels[-3], dilation=rate_1),
                                 nn.UpsamplingNearest2d(scale_factor=2),
                                 SeparableConvBNReLU(encoder_channels[-3], encoder_channels[-4], dilation=rate_2),
                                 nn.UpsamplingNearest2d(scale_factor=2))


        self.down12 = DownConnection(encoder_channels[0], encoder_channels[1])
        self.down231 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down232 = DownConnection(encoder_channels[1], encoder_channels[2])
        self.down34 = DownConnection(encoder_channels[2], encoder_channels[3])

        self.ASPPRU = ASPPRU(oup_channels=384,group_num=384,gate_treshold=0.5)
        # self.CCRU = CCRU(op_channel=192)

    def forward(self, x1, x2, x3, x4):
        # print("x1",x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("self.pa(x3)", self.pa(x3).shape)
        # print("self.down232(x2)", self.down232(x2).shape)
        # out4 = self.conv4(x4) + self.down34(self.pa(self.down232(x2)))
        # idea1
        # out4 = self.conv4(x4) + self.down34(self.pa(self.down232(x2)))
        out4 = self.conv4(x4) + self.down34(self.ASPPRU(self.down232(x2)))
        # print("-----out4---------",out4.shape)
        # idea1
        # out3 = self.pa(x3) + self.down231(self.ca(self.down12(x1)))

        # print("--------xxx------------------")
        # print('self.down12(x1)),',self.down12(x1).shape)
        # print("self.ca(self.down12(x1))",self.CCRU(self.down12(x1)).shape)

        out3 = self.ASPPRU(x3) + self.down231(self.down12(x1))
        # print("------out3--------",out3.shape)

        # print('self.CCRU(x2),', self.CCRU(x2).shape)
        # print("------------------------------")
        # print("self.lf4(out4)",self.lf4(out4).shape)
        # out2 = self.CCRU(x2) + self.lf4(out4)

        out2 =  self.lf4(out4)
        # print("out2", out2.shape)

        del out4
        out1 = self.lf3(out3) + self.conv1(x1)
        del out3

        return out1, out2
    # 模块分析
    # def forward(self, x1):
    #     out2 = self.lf4(x1)
    #     return out2

# idea1:
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta

# idea1:
class ASPPRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()
        self.ASPP= ASPP()

    def forward(self, x):
        # print("x进入SRU",x.shape)
        gn_x = self.gn(x)
        # print("self.gn(x) ",gn_x.shape)
        # print("self.gn.gamma ", self.gn.gamma)
        # print("self.gn.gamma.shape ", self.gn.gamma.shape)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        # print(" self.gn.gamma / sum(self.gn.gamma) ", w_gamma.shape)
        reweigts = self.sigomid(gn_x * w_gamma)
        # print("self.sigomid(gn_x * w_gamma)", reweigts .shape)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        # print("info_mask ", info_mask)
        # print("noninfo_mask ", noninfo_mask)

        x_1 = info_mask * x
        # print("self.info_mask * x",  x_1.shape)
        x_2 = noninfo_mask * x
        # print("self.noninfo_mask * x", x_2.shape)
        x = self.reconstruct(x_1, x_2)
        # print("self.reconstruct(x_1, x_2)", x.shape)
        x = self.ASPP(x)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

# without BN version

# idea2:
class CCRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

        self.pconv1 = nn.Conv2d(op_channel, op_channel, kernel_size=1, bias=False)
        self.pconv2 = nn.Conv2d(op_channel * 2, op_channel * 2, kernel_size=1, bias=False)

    def forward(self, x):
        # Split
        # print("-----x------",x.shape)
        p1 = self.pconv1(x)
        # print("--------------------",p1.shape)
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out

        outs = self.pconv2(out)
        # print("---------outs----------", outs.shape)
        out1, out2 = torch.split(outs, outs.size(1) // 2, dim=1)
        # print("OUT1",out1.shape)
        # print("OUT2",out.shape)
        outm = out1 + out2
        # print("--------outm---------",outm.shape)
        outall = p1 + outm
        # print("--------outall---------",outall.shape)
        return outall

class ASPP(nn.Module):
    def __init__(self, in_channel=384, out_channel=384):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 num_classes=6):
        super(Decoder, self).__init__()
        self.dcfam = DCFAM(encoder_channels, atrous_rates)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(encoder_channels[0], encoder_channels[0]),
            Conv(encoder_channels[0], num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4))
        self.up = nn.Sequential(
            ConvBNReLU(encoder_channels[1], encoder_channels[0]),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

        self.init_weight()

    def forward(self, x1, x2, x3, x4):
        out1, out2 = self.dcfam(x1, x2, x3, x4)
        x = out1 + self.up(out2)
        x = self.dropout(x)
        x = self.segmentation_head(x)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class URCSwin(nn.Module):
    def __init__(self,
                 encoder_channels=(96, 192, 384, 768),
                 # encoder_channels=(64, 128, 256, 512),
                 dropout=0.05,
                 atrous_rates=(6, 12),
                 num_classes=6,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 frozen_stages=2):
        super(URCSwin, self).__init__()
        self.backbone = SwinTransformer(embed_dim=embed_dim, depths=depths, num_heads=num_heads, frozen_stages=frozen_stages)
        self.decoder = Decoder(encoder_channels, dropout, atrous_rates, num_classes)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        x = self.decoder(x1, x2, x3, x4)
        return x




def urcswin_base(pretrained=True, num_classes=4, weight_path='pretrain_weights/stseg_base.pth'):
    # pretrained weights are load from official repo of Swin Transformer
    model = URCSwin(encoder_channels=(128, 256, 512, 1024),
                   num_classes=num_classes,
                   embed_dim=128,
                   depths=(2, 2, 18, 2),
                   num_heads=(4, 8, 16, 32),
                   frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def urcswin_small(pretrained=True, num_classes=4, weight_path='pretrain_weights/stseg_small.pth'):
    model = URCSwin(encoder_channels=(96, 192, 384, 768),
                   num_classes=num_classes,
                   embed_dim=96,
                   depths=(2, 2, 18, 2),
                   num_heads=(3, 6, 12, 24),
                   frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def urcswin_tiny(pretrained=True, num_classes=4, weight_path='pretrain_weights/stseg_tiny.pth'):
    model = URCSwin(encoder_channels=(96, 192, 384, 768),
                   num_classes=num_classes,
                   embed_dim=96,
                   depths=(2, 2, 6, 2),
                   num_heads=(3, 6, 12, 24),
                   frozen_stages=2)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)['state_dict']
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model



if __name__ == '__main__':
    model = urcswin_tiny(pretrained=False)
    # model = SpatialAttention(64)
    # print(model)

    test_data = torch.randn(1, 3, 512, 512)
    x = model(test_data)
    print("x.shape   ",x[0].shape)

    # 参数统计
    from torchstat import stat
    # import torch


    # net = DCFAM()
    # stat(net, (3, 1024, 1024))
    # y = net(torch.randn(1, 3, 512, 512))
    # print(y[0].size())
    # summary(net, (1, 768, 16, 16))
    #
    # from thop import profile
    #
    # model = dcswin_small()
    # test = torch.randn(1, 3, 512, 512)
    # flops, params = profile(model, (test,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    # model = ASPPRU(384)
    # x = torch.randn(1, 384, 16, 16)
    # print(model(x).shape)


    # x = torch.randn(1, 192, 64, 64)
    # model = CCRU(192)
    # print(model(x).shape)

    # model = ASPP()
    # x = torch.randn(1, 384, 16, 16)
    # print(model(x).shape)