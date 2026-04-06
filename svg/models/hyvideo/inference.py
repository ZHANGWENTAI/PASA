import os

import torch

from ...logger import logger
from .attention import (
    Hunyuan_PASA_Processor,
    Hunyuan_PISA_Processor,
    Hunyuan_SAPAttn_Processor2_0,
    Hunyuan_SAPAttn_Processor2_0_SR,
    Hunyuan_SVGAttn_Processor2_0,
    HunyuanVideoAttnProcessor2_0_FlashAttention,
    prepare_flexattention,
)
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width, rescaled_factor


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
    pattern="SVG",  # Default to SVG for backward compatibility
    # SVG specific, but provide defaults for general call signature
    num_sampled_rows=64,
    sample_mse_max_row=10000,
    sparsity=0.25,
    # Pattern dispatcher and KMEANS_BLOCK specific args
    num_q_centroids=None,
    num_k_centroids=None,
    top_p_kmeans=None,
    min_kc_ratio=0,
    logging_file=None,
    q_kmeans_iter_init=0,
    k_kmeans_iter_init=0,
    q_kmeans_iter_step=0,
    k_kmeans_iter_step=0,
    q_kmeans_init_method="random",
    k_kmeans_init_method="random",
    zero_step_kmeans_init=False,
    base_density=0.15,
    use_dynamic=True,
    use_group=True,
    use_random=True,
):

    cfg_size, num_head, head_dim, dtype, device = 1, 24, 128, torch.bfloat16, "cuda"
    context_length, num_frame = 256, 1 + num_frames // 4  # TODO: Make it more formal
    frame_size = height * width // 256  # TODO: Make it more formal

    if pattern == "SVG":
        masks = ["spatial", "temporal"]

        # Calculation
        spatial_width = temporal_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

        print(f"Spatial_width: {spatial_width}, Temporal_width: {temporal_width}. Sparsity: {sparsity}")

        AttnModule = Hunyuan_SVGAttn_Processor2_0

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.attention_masks = [
            get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        block_mask = prepare_flexattention(
            cfg_size,
            num_head,
            head_dim,
            dtype,
            device,
            context_length,
            prompt_length,
            num_frame,
            frame_size,
            diag_width=spatial_width,
            multiplier=temporal_width,
        )
        AttnModule.block_mask = block_mask
        replace_sparse_forward()

        logger.info("Flexattn block_mask prepared.")
        logger.info(block_mask)

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
            print(f"Replaced Sparse VideoGen block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            print(
                f"Replaced Sparse VideoGen block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    elif pattern in ["SAP"]:

        # Pass K-means specific parameters to the processor's constructor or set them as attributes
        # The processor itself will handle the K-means logic internally
        logger.info(
            f"Configuring KMEANS_BLOCK attention with QC: {num_q_centroids}, KC: {num_k_centroids}, P: {top_p_kmeans}, min_kc_ratio: {min_kc_ratio}"
        )

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = Hunyuan_SAPAttn_Processor2_0

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.q_kmeans_iter_init = q_kmeans_iter_init
        AttnModule.k_kmeans_iter_init = k_kmeans_iter_init
        AttnModule.q_kmeans_iter_step = q_kmeans_iter_step
        AttnModule.k_kmeans_iter_step = k_kmeans_iter_step
        AttnModule.q_kmeans_init_method = q_kmeans_init_method
        AttnModule.k_kmeans_init_method = k_kmeans_init_method
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init

        replace_sparse_forward()

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
            print(f"Replaced Semantic Aware Permutation block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            print(
                f"Replaced Semantic Aware Permutation block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )

    elif pattern == "SAP_SR":
        logger.info(
            f"Configuring KMEANS_BLOCK attention with QC: {num_q_centroids}, KC: {num_k_centroids}, P: {top_p_kmeans}, min_kc_ratio: {min_kc_ratio}"
        )

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = Hunyuan_SAPAttn_Processor2_0_SR

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.q_kmeans_iter_init = q_kmeans_iter_init
        AttnModule.k_kmeans_iter_init = k_kmeans_iter_init
        AttnModule.q_kmeans_iter_step = q_kmeans_iter_step
        AttnModule.k_kmeans_iter_step = k_kmeans_iter_step
        AttnModule.q_kmeans_init_method = q_kmeans_init_method
        AttnModule.k_kmeans_init_method = k_kmeans_init_method
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init

        replace_sparse_forward()

        for layer_idx, m in enumerate(pipe.transformer.transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx)
            print(f"Replaced Semantic Aware Permutation SR block for Double Stream Transformer at layer {layer_idx}")

        for layer_idx, m in enumerate(pipe.transformer.single_transformer_blocks):
            self_attn = m.attn
            self_attn.processor = AttnModule(layer_idx=layer_idx + len(pipe.transformer.transformer_blocks))
            print(
                f"Replaced Semantic Aware Permutation SR block for Single Stream Transformer at layer {layer_idx + len(pipe.transformer.transformer_blocks)}"
            )
    elif pattern == "PISA":
        logger.info(
            f"Configuring PISA (Piecewise Sparse) attention with num_k_centroids: {num_k_centroids}, "
            f"k_kmeans_iter_init: {k_kmeans_iter_init}, k_kmeans_iter_step: {k_kmeans_iter_step}"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")
        
        AttnModule = Hunyuan_PISA_Processor

        AttnModule.prompt_length = prompt_length
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame
        AttnModule.frame_size = frame_size

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file
        AttnModule.base_density = base_density
        AttnModule.use_random = use_random
        AttnModule.use_group = use_group

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

    elif pattern == "PASA":
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
