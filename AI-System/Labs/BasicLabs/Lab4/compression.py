# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch
import numpy as np


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor

class Bit16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed

class BIT8Compressor(Compressor):
    """Compress all floating point gradients to 8-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 8-bit."""
        # print("### compress::tensor = "  str(tensor))
        if tensor.dtype != torch.float32:
            print("WARNING: BIT8Compressor taking non-float32")
            # Only allow compression from other floating point types
            tensor = tensor.type(torch.float32)
        
        tensor_np = tensor.numpy()
        tensor_np_fp8 = fp32_conv_fp8(tensor_np)

        tensor_compressed = torch.tensor(tensor_np_fp8, dtype=torch.int8)
        # print("### compress::tensor_compressed = "  str(tensor_compressed))


        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed

class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor


frac_len_fp32 = 23
exp_len_fp32 = 8
len_fp32 = 32
frac_len_fp8 = 3
exp_len_fp8 = 4
len_fp8 = 8
# frac_len_fp8 = 5
# exp_len_fp8 = 2
# len_fp8 = 8

def fp32_conv_fp8(tensor_np):
   tensor_np = tensor_np.view(np.int32)
   tensor_frac_mask = np.empty(tensor_np.shape, dtype=np.int32)
   tensor_exp_mask8 = np.empty(tensor_np.shape, dtype=np.int32)
   tensor_exp_mask8nosign = np.empty(tensor_np.shape, dtype=np.int32)
   tensor_exp_mask32 = np.empty(tensor_np.shape, dtype=np.int32)
   tensor_exp_offset32 = np.empty(tensor_np.shape, dtype=np.int32)
   tensor_frac_mask.fill(2**frac_len_fp8-1)
   tensor_exp_mask8.fill(2**exp_len_fp8 -1)
   tensor_exp_mask8nosign.fill(2 ** (exp_len_fp8-1) - 1)
   tensor_exp_mask32.fill(2 ** exp_len_fp32 - 1)
   tensor_exp_offset32.fill(2 ** (exp_len_fp32 - 1) - 1)
   output_frac = np.right_shift(tensor_np, frac_len_fp32 - frac_len_fp8) & tensor_frac_mask
   output_exp = np.left_shift(((np.right_shift(tensor_np, frac_len_fp32) & tensor_exp_mask32) - tensor_exp_offset32) & tensor_exp_mask8nosign, frac_len_fp8)
   output_sign = np.left_shift(np.right_shift(tensor_np, len_fp32 - 2), len_fp8 - 2)
   output_fp8 = output_frac | output_exp | output_sign
   tensor_result_int8 = output_fp8.astype(np.int8)
   return tensor_result_int8