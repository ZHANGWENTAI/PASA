import json
import math
import os

from ...logger import logger
from .attention import (
    Wan_PASA_Processor,
)
from .custom_models import replace_sparse_forward


def replace_wan_attention(
    pipe,
    height,
    width,
    num_frames,
    first_layers_fp,
    first_times_fp,
    pattern="PASA",
    logging_file=None,
    base_density=0.15,
):

    context_length = 0  # This seems to be 0 for I2V in SVG
    num_frame_patches = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_patches_one_frame = int(height // mod_value) * int(width // mod_value)

    replace_sparse_forward()  # Assuming this is a general patch; if not, it might need to be conditional.

    num_layers = len(pipe.transformer.blocks)

    if pattern == "PASA":
        logger.info(
            f"Configuring PASA attention"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")
        
        AttnModule = Wan_PASA_Processor
        
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file
        
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame
        AttnModule.num_layers = num_layers

        AttnModule.base_density = base_density
        rescaled_density = {}
        for step, rescale_factor in rescaled_factors.items():
            rescaled_density[step] = base_density * rescale_factor
        AttnModule.rescaled_density = rescaled_density

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                m.attn1.set_processor(current_processor)

    else:  # dense or other patterns
        raise ValueError(f"Pattern '{pattern}' not supported")

    # Common logic for processors that were set
    # The loop for m.attn1.processor.layer_idx = layer_idx was integrated into SVG specific part.
    # For KMEANS_BLOCK, layer_idx is passed during instantiation.
    # The generic loop below might be redundant if all processors handle layer_idx internally or via init.
    # Ensure Attn2 (cross-attention) is not affected if it uses a different processor type or no processor.
    # The original code iterated all Attention modules, let's refine to target only self-attention (attn1)

    print(f"Attention processors replaced with {pattern} pattern.")
