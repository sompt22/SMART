"""
MPS / CPU-compatible Deformable Convolution v2 (DCN).

Drop-in replacement for the compiled DCNv2 CUDA extension, implemented with
torchvision.ops.deform_conv2d which supports CUDA, MPS (Apple Silicon), and CPU.

Usage:
    This module is imported automatically when the compiled _ext module is
    unavailable (e.g., on Apple Silicon or CPU-only environments).
"""
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

try:
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
    _TORCHVISION_DCN_AVAILABLE = True
except ImportError:
    _TORCHVISION_DCN_AVAILABLE = False


class DCNv2(nn.Module):
    """Deformable Convolution v2 implemented via torchvision.ops.deform_conv2d.

    Matches the interface of the original compiled DCNv2:
        forward(input, offset, mask) -> output

    Args match the original: in_channels, out_channels, kernel_size, stride,
    padding, dilation, deformable_groups.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__()
        assert _TORCHVISION_DCN_AVAILABLE, (
            "torchvision is required for MPS/CPU DCN. "
            "Install with: pip install torchvision"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return _tv_deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class DCN(DCNv2):
    """DCN with built-in offset & mask prediction (same interface as compiled DCN).

    forward(input) -> output  (offsets and mask predicted internally)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups
        )
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self._init_offset()

    def _init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return _tv_deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
