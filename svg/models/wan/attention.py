import json
import os
import sys
import warnings
import triton
from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from torch.nn.attention.flex_attention import (
    flex_attention,
)
from svg.kernels.triton.group_pisa_attention import pisa_pg_attention

from ...kernels.triton.rmsnorm import triton_rmsnorm_forward

from ...logger import logger
from ...timer import time_logging_decorator
from ...utils.misc import Color

from .utils import (
    create_block_mask_cached,
    flashinfer_sparse_attn_forward,
    gen_temporal_mask,
    generate_temporal_head_mask_mod,
    rescaled_factors,
)

try:
    # raise ImportError  # TODO: Remove this
    sys.path.append("svg/kernels/build/")
    import _kernels

    def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, freqs: torch.Tensor):
        freqs_real, freqs_imag = freqs
        _kernels.apply_qk_rope_inplace_cossin_complex(query, key, freqs_real, freqs_imag, 0)  # len_text_prompt = 0
        return query, key

    ENABLE_FAST_KERNEL = True

    logger.info(f"{Color.green}Using Fast CUDA and Triton Kernels{Color.reset}")


except ImportError:
    warnings.warn("Could not import RoPE / Norm kernels! Falling back to PyTorch implementation.")

    def apply_rotary_emb(query: torch.Tensor, key: torch.Tensor, freqs: torch.Tensor):
        def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
            x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
            x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
            return x_out.type_as(hidden_states)

        query = _apply_rotary_emb(query, freqs)
        key = _apply_rotary_emb(key, freqs)
        return query, key

    ENABLE_FAST_KERNEL = False

    logger.info(f"{Color.red}Disable Fast CUDA and Triton Kernels{Color.reset}")

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
torch._dynamo.config.cache_size_limit = 192 * 3
torch._dynamo.config.accumulated_cache_size_limit = 192 * 3


def _ensure_triton_allocator():
    """Set Triton global scratch allocator to PyTorch CUDA allocator if not set.
    Kernels using tl.make_tensor_descriptor (e.g. fused_chunk_reduce) need runtime
    memory for global scratch; without this, Triton raises RuntimeError."""
    try:
        allocator = getattr(triton.runtime._allocation, "_allocator", None)
        from triton.runtime._allocation import NullAllocator
        if allocator is None or isinstance(allocator, NullAllocator):
            def _torch_allocator(size: int, alignment: int, stream):
                if size <= 0:
                    return torch.empty(0, dtype=torch.uint8, device="cuda")
                total = size + alignment - 1
                buf = torch.empty(total, dtype=torch.uint8, device="cuda")
                ptr = buf.data_ptr()
                aligned_ptr = (ptr + alignment - 1) // alignment * alignment
                offset = aligned_ptr - ptr
                return buf[offset : offset + size]

            triton.set_allocator(_torch_allocator)
    except Exception:
        pass

# Ensure Triton has an allocator when this module is used (e.g. PISA attention on CUDA)
_ensure_triton_allocator()

class WanAttn_SVGAttn_Processor2_0:
    version = None
    context_length = 0
    num_frame = 0
    frame_size = 0

    first_layers_fp = 0
    first_times_fp = 0

    num_sampled_rows = 32
    attention_masks = None
    sparsity = 0

    block_mask = None
    temporal_mask_metadata = None

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    @time_logging_decorator("Level 2 - qkv")
    def get_qkv(self, attn, hidden_states, encoder_hidden_states):
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        return query, key, value

    @time_logging_decorator("Level 2 - qk_norm")
    def get_qk_norm(self, attn, query, key):
        if attn.norm_q is not None:
            if isinstance(attn.norm_q, torch.nn.RMSNorm) or isinstance(attn.norm_q, DiffusersRMSNorm):
                # query = attn.norm_q(query)
                query = triton_rmsnorm_forward(query, attn.norm_q.weight, attn.norm_q.eps)
            else:
                raise ValueError(f"Unsupported norm type: {type(attn.norm_q)}")

        if attn.norm_k is not None:
            if isinstance(attn.norm_k, torch.nn.RMSNorm) or isinstance(attn.norm_k, DiffusersRMSNorm):
                # key = attn.norm_k(key)
                key = triton_rmsnorm_forward(key, attn.norm_k.weight, attn.norm_k.eps)
            else:
                raise ValueError(f"Unsupported norm type: {type(attn.norm_k)}")
        return query, key

    @time_logging_decorator("Level 2 - transpose")
    def get_transpose_qkv(self, attn, query, key, value):
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        return query, key, value

    @time_logging_decorator("Level 2 - rotary_emb")
    def get_rotary_emb(self, query, key, rotary_emb):

        if rotary_emb is not None:
            query, key = apply_rotary_emb(query, key, rotary_emb)

        return query, key

    @time_logging_decorator("Level 2 - output")
    def get_o(self, attn, query, hidden_states, hidden_states_img):
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        _l1_log_disable = os.environ.get("WAN_HIDDEN_L1_ENABLE", "1").strip().lower() in ("0", "false", "no")
        if not _l1_log_disable:
            hidden_states_pre_attn = hidden_states.detach()

        query, key, value = self.get_qkv(attn, hidden_states, encoder_hidden_states)

        query, key = self.get_qk_norm(attn, query, key)

        query, key, value = self.get_transpose_qkv(attn, query, key, value)

        query, key = self.get_rotary_emb(query, key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # # ============================== Save QKV ==============================
        # save_flag = timestep[0] % 4 == 0 and self.layer_idx % 4 == 0
        # print(f"save_flag: {save_flag}, timestep: {timestep[0]}, layer_idx: {self.layer_idx}")
        # save_dir = f"assets/svg_tensors"
        # if save_flag:
        #     save_qkvx(query, key, value, hidden_states, save_dir, self.layer_idx, timestep[0].item())

        # ========================================================================
        if timestep is None:  # Cross Attention in Wan
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:  # The main attention
            hidden_states = self.attention_core_logic(query, key, value, timestep)
        # ========================================================================

        # L1 distance: hidden_states before attention vs after attention (flattened to [B, S, D]).
        if not _l1_log_disable:
            hs_post = hidden_states.transpose(1, 2).flatten(2, 3)
            if hs_post.shape == hidden_states_pre_attn.shape:
                l1_mean = float(
                    (hidden_states_pre_attn.float() - hs_post.float()).abs().mean().cpu()
                )
            else:
                l1_mean = float("nan")
            log_path = os.environ.get(
                "WAN_HIDDEN_L1_LOG", os.path.join("result", "wan_attention_hidden_l1.jsonl")
            )
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            ts = (
                None
                if timestep is None
                else int(timestep[0].item())
                if torch.is_tensor(timestep)
                else int(timestep)
            )
            prompt_id = int(os.environ.get("WAN_HIDDEN_L1_PROMPT_ID", "0"))
            with open(log_path, "a") as _hf:
                _hf.write(
                    json.dumps(
                        {
                            "prompt_id": prompt_id,
                            "layer": self.layer_idx,
                            "timestep": ts,
                            "l1_mean": l1_mean,
                        }
                    )
                    + "\n"
                )

        hidden_states = self.get_o(attn, query, hidden_states, hidden_states_img)

        return hidden_states

    @time_logging_decorator("Level 3 - sample mse")
    def sample_mse(self, query, key, value):
        assert len(self.attention_masks) == 2

        cfg, num_heads, seq_len, dim = query.size()
        num_sampled_rows = min(self.num_sampled_rows, seq_len)
        sampled_rows = torch.randint(low=0, high=self.sample_mse_max_row, size=(num_sampled_rows,))
        sampled_q = query[:, :, sampled_rows, :]
        sampled_qk_scores = torch.matmul(sampled_q, key.transpose(-2, -1)) / (dim**0.5)

        sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
        sampled_golden_hidden_states = torch.matmul(sampled_attn_weights, value)  # (1, seq_len, dim)

        sampled_mses = torch.zeros(len(self.attention_masks), cfg, num_heads, device=query.device, dtype=query.dtype)

        # Only have Tri-diagonal and Striped
        for mask_idx, attn_mask in enumerate(self.attention_masks):
            sampled_attention_mask = attn_mask[sampled_rows, :]
            sampled_attention_scores = sampled_qk_scores.masked_fill(sampled_attention_mask == 0, float("-inf"))
            sampled_attn_weights = F.softmax(sampled_attention_scores, dim=-1)
            sampled_hidden_states = torch.matmul(sampled_attn_weights, value)
            mse = torch.mean((sampled_hidden_states - sampled_golden_hidden_states) ** 2, dim=(2, 3))
            sampled_mses[mask_idx] = mse

        return sampled_mses

    @time_logging_decorator("Level 3 - sparse flex attention")
    def sparse_flex_attention(self, query, key, value, block_mask):
        return flex_attention(query, key, value, block_mask=block_mask)

    @time_logging_decorator("Level 3 - sparse flashinfer attention")
    def sparse_flashinfer_attention(self, query, key, value, temporal_mask_metadata):
        return flashinfer_sparse_attn_forward(query, key, value, temporal_mask_metadata)

    @time_logging_decorator("Level 3 - Dense Flash Attention")
    def flash_attention(self, query, key, value):
        output_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        return output_hidden_states


def prepare_flexattention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    seq_len = context_length + num_frame * frame_size
    query, key, value = [
        torch.zeros((cfg_size, num_head, seq_len, head_dim), dtype=dtype, device=device) for _ in range(3)
    ]

    mask_mod = generate_temporal_head_mask_mod(context_length, prompt_length, num_frame, frame_size, mul=multiplier)
    block_mask = create_block_mask_cached(mask_mod, None, None, seq_len, seq_len, device=device, _compile=True)
    _ = flex_attention(query, key, value, block_mask=block_mask)

    return block_mask


def prepare_flashinfer_attention(
    cfg_size,
    num_head,
    head_dim,
    dtype,
    device,
    context_length,
    prompt_length,
    num_frame,
    frame_size,
    diag_width=1,
    multiplier=2,
):
    assert diag_width == multiplier, f"{diag_width} is not equivalent to {multiplier}"

    temporal_mask_metadata = gen_temporal_mask(num_frame, frame_size, multiplier)

    return temporal_mask_metadata

class Wan_PASA_Processor(WanAttn_SVGAttn_Processor2_0):
    def __init__(self, layer_idx):
        super().__init__(layer_idx)
        self.layer_idx = layer_idx

    def attention_core_logic(self, query, key, value, timestep):
        cfg, num_heads, seq_len, dim = query.size()
        assert cfg == 1, "Batch size must be 1 for kmeans block sparse attention"
        
        context_length, num_frame, frame_size = self.context_length, self.num_frame, self.frame_size
        assert (
            seq_len == context_length + num_frame * frame_size
        ), f"Query Shape: {seq_len} is not equivalent to {context_length} + {num_frame} * {frame_size}"

        full_attention_flag = False

        if self.layer_idx < self.first_layers_fp:
            full_attention_flag = True
        if timestep[0] > self.first_times_fp:
            full_attention_flag = True

        # json 里 timestep 键为 str；这里用与推理一致的整数再转 str
        if not full_attention_flag:
            ts_key = int(timestep[0].item())
            density = self.rescaled_density.get(ts_key)

            assert density is not None, f"Density is not found for timestep {timestep[0].item()}"

        if full_attention_flag:
            output_hidden_states = self.flash_attention(query, key, value)
            # output_hidden_states = self.flashinfer_attention(query, key, value)
            return output_hidden_states.reshape(cfg, num_heads, seq_len, dim)

        else:
            output = piecewise_sparse_attention(
                query, key, value,
                density=density,
                block_size=64,
            )

            return output.reshape(cfg, num_heads, seq_len, dim)
