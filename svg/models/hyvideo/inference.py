import os

import torch

from ...logger import logger
from .attention import (
    Hunyuan_PASA_Processor,
    HunyuanVideoAttnProcessor2_0_FlashAttention,
)
from .custom_models import replace_sparse_forward
from .utils import rescaled_factor


def replace_hyvideo_flashattention(pipe):
    """
    Replace the FSDP + masked attention with flash attention + varlen. Crucial for inference efficiency.
    """
    for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
        self_attn = m.attn
        self_attn.processor = HunyuanVideoAttnProcessor2_0_FlashAttention(layer_idx=layer_idx)
        print(f"Replaced FlashAttention implementation in double stream transformer block {layer_idx}")

    for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
        self_attn = m.attn
        self_attn.processor = HunyuanVideoAttnProcessor2_0_FlashAttention(
            layer_idx=layer_idx + len(pipe.transformer.transformer_blocks)
        )
        print(f"Replaced FlashAttention implementation in single stream transformer block {layer_idx}")


def replace_hyvideo_attention(
    pipe,
    height,
    width,
    num_frames,
    prompt_length,
    first_layers_fp,
    first_times_fp,
    pattern="PASA",
    logging_file=None,
    base_density=0.15,
    use_dynamic=True,
    use_group=True,
    use_random=True,
):
    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame = 256, 1 + num_frames // 4  # TODO: Make it more formal
    frame_size = height * width // 256  # TODO: Make it more formal

    if pattern == "PASA":
        logger.info(
            f"Configuring PASA (Piecewise Sparse) attention with base_density: {base_density}, use_dynamic: {use_dynamic}, use_group: {use_group}, use_random: {use_random}"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")
                
        AttnModule = Hunyuan_PASA_Processor

        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file
        AttnModule.base_density = base_density
        AttnModule.use_group = use_group
        AttnModule.use_random = use_random

        rescaled_density = {}
        if use_dynamic:
            for ts, factor in rescaled_factor.items():
                rescaled_density[ts] = base_density * factor
        else:
            for ts, factor in rescaled_factor.items():
                rescaled_density[ts] = base_density
        AttnModule.rescaled_density = rescaled_density

        replace_sparse_forward()

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))

    else:
        assert pattern == "dense", f"Invalid pattern: {pattern}"
