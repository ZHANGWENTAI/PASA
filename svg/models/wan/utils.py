"""Mask Mod for Image2Video"""

import math
from functools import lru_cache
from math import ceil
from typing import Tuple

import flashinfer
import numpy as np
import sympy as sp
import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
)

from ...logger import logger


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def generate_temporal_head_mask_mod(
    context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2
):

    def round_to_multiple(idx):
        return ceil(idx / 128) * 128

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = torch.abs(q_idx - kv_idx) <= two_frame

        # return temporal_head_mask
        first_frame_mask = kv_idx < token_per_frame
        video_mask = first_frame_mask | temporal_head_mask
        return video_mask

    return temporal_mask_mod


def generate_dense_mask_mod():
    def dense_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= 0  # True

    return dense_mask_mod


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size):
    """
    Generate the emulated attention mask. Used for online profiling.
    """

    from termcolor import colored

    allocated = torch.cuda.memory_allocated() / 1e9
    print(colored(f"Allocated Memory: {allocated:.2f} GB", "yellow"))

    attention_mask = torch.zeros(
        (context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu"
    )

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask = pixel_attn_mask
    else:
        pixel_attn_mask = torch.zeros_like(attention_mask, dtype=torch.bool, device="cpu")

        pixel_attn_mask[:, :frame_size] = 1  # First Frame Sink

        block_size, block_thres = 128, frame_size * 2
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1

        pixel_attn_mask = (
            pixel_attn_mask.reshape(frame_size, num_frame, frame_size, num_frame)
            .permute(1, 0, 3, 2)
            .reshape(frame_size * num_frame, frame_size * num_frame)
        )
        attention_mask = pixel_attn_mask

    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


def get_factor(num_frames: int, num_tokens_per_frame: int) -> int:
    """
    Get the factor of num_frames * num_tokens_per_frame.
    Find the maximum factor that is less than 128.
    """
    # factors = sp.divisors(num_frames * num_tokens_per_frame)
    factors = sp.divisors(num_tokens_per_frame)
    # sort it, from large to small
    factors.sort(reverse=True)

    for factor in factors:
        if factor < 256:
            return factor

    raise ValueError(f"No factor found for {num_frames} * {num_tokens_per_frame}")


def flashinfer_sparse_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    temporal_mask_metadata: Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]],
) -> torch.Tensor:
    """
    Forward pass for spatial attention head.
    Separate calculation of text attention sink and video sparse attention

    Args:
        q,k,v: [seq_len, num_heads, head_dim]
        temporal_mask_metadata: metadata for temporal attention head

    Returns:
        torch.Tensor: output tensor
    """
    row_indices, column_indices, block_size = temporal_mask_metadata

    cfg, num_heads, seq_len, head_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

    q = q.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
    k = k.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
    v = v.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)

    # Assert the dimension of video modality
    assert q.shape[0] % block_size[0] == 0, f"Query length {q.shape[0]} % block_size {block_size[0]} != 0"
    assert k.shape[0] % block_size[1] == 0, f"Key length {k.shape[0]} % block_size {block_size[1]} != 0"
    assert k.shape[0] == v.shape[0], f"Key length {k.shape[0]} != Value length {v.shape[0]}"

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)

    bsr_wrapper.plan(
        row_indices,
        column_indices,
        q.shape[0],  # video length
        k.shape[0],  # video length
        block_size[0],
        block_size[1],
        q.shape[1],  # num_qo_heads
        k.shape[1],  # num_kv_heads
        q.shape[2],  # head_dim
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    o_image = bsr_wrapper.run(q, k, v, return_lse=False)

    o_image = o_image.reshape(seq_len, cfg, num_heads, head_dim).permute(1, 2, 0, 3).contiguous()

    return o_image

rescaled_factors = {
    952: 0.9751290602409638,
    946: 0.9700656385542168,
    940: 0.9650698795180721,
    934: 0.9601417831325302,
    927: 0.9544778554216867,
    920: 0.9489060240963856,
    913: 0.9434262891566265,
    906: 0.9380386506024097,
    898: 0.9319941204819276,
    890: 0.9260698795180723,
    882: 0.9202659277108434,
    873: 0.913880265060241,
    863: 0.9069636385542168,
    854: 0.9008993734939759,
    843: 0.8936942409638554,
    833: 0.887341469879518,
    821: 0.8799662409638555,
    809: 0.8728616626506024,
    796: 0.8654704578313253,
    783: 0.8583968915662651,
    768: 0.85062978313253,
    753: 0.8432855662650602,
    737: 0.8359178554216867,
    720: 0.8286168674698794,
    701: 0.821099734939759,
    681: 0.8139199759036144,
    660: 0.8071903614457832,
    636: 0.800514313253012,
    611: 0.7947113012048193,
    584: 0.789763469879518,
    555: 0.7859753012048193,
    522: 0.7835873734939759,
    487: 0.7832913493975904,
    448: 0.7856736385542169,
    405: 0.7916138554216866,
    356: 0.8026193734939759,
    302: 0.819974843373494,
    241: 2.2507440386715194,
    172: 2.7757292234192645,
    92: 3.0025936743632604,
}