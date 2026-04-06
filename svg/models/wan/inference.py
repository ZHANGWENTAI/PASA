import json
import math
import os

import torch

from ...logger import logger
from ..utils import visualize_sparse_bsr
from .attention import (
    Wan_GROUP_PISA_Processor,
    Wan_PISA_Processor,
    WanAttn_PISAttn_Processor,
    WanAttn_SAPAttn_Processor,
    WanAttn_SAPAttn_Processor_SR,
    WanAttn_SVGAttn_Processor2_0,
    prepare_flashinfer_attention,
    prepare_flexattention,
    Wan_MIX_PISA_Processor,
)
from .stability_analysis import ClusterStabilityTracker
from .custom_models import replace_sparse_forward
from .utils import get_attention_mask, sparsity_to_width


def replace_wan_attention(
    pipe,
    height,
    width,
    num_frames,
    first_layers_fp,
    first_times_fp,
    attention_backend="flexattn",
    pattern="SVG",  # Default to SVG for backward compatibility
    # SVG specific, but provide defaults for general call signature
    num_sampled_rows=64,
    sample_mse_max_row=10000,
    sparsity=0.25,
    # Pattern dispatcher and KMEANS_BLOCK specific args
    num_q_centroids=None,
    num_k_centroids=None,
    num_x_centroids=None,
    top_p_kmeans=None,
    r_kmeans=None,
    min_kc_ratio=0,
    logging_file=None,
    q_kmeans_iter_init=0,
    k_kmeans_iter_init=0,
    q_kmeans_iter_step=0,
    k_kmeans_iter_step=0,
    q_kmeans_init_method="random",
    k_kmeans_init_method="random",
    x_kmeans_iter_init=0,
    x_kmeans_iter_step=0,
    x_kmeans_init_method="random",
    density=0.15,
    zero_step_kmeans_init=False,
    kmeans_step_type="full",  # "full" or "key_only"
    stability_analysis_dir=None,  # 如果提供，则启用稳定性分析并保存到该目录
    window_size=2,  # SAP_AdaKmeans 前若干 step 的序列窗口大小
):

    context_length = 0  # This seems to be 0 for I2V in SVG
    num_frame_patches = 1 + num_frames // (pipe.vae_scale_factor_temporal * pipe.transformer.config.patch_size[0])
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    frame_patches_one_frame = int(height // mod_value) * int(width // mod_value)

    dtype = torch.bfloat16  # Or pipe.dtype
    device = pipe.device

    replace_sparse_forward()  # Assuming this is a general patch; if not, it might need to be conditional.

    num_layers = len(pipe.transformer.blocks)

    if pattern == "SVG":
        AttnModule = WanAttn_SVGAttn_Processor2_0
        AttnModule.num_sampled_rows = num_sampled_rows
        AttnModule.sample_mse_max_row = sample_mse_max_row
        AttnModule.sparsity = sparsity

        masks = ["spatial", "temporal"]
        AttnModule.attention_masks = [
            get_attention_mask(
                mask_name, sample_mse_max_row, context_length, num_frame_patches, frame_patches_one_frame
            )
            for mask_name in masks
        ]
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp

        multiplier = diag_width = sparsity_to_width(
            sparsity, context_length, num_frame_patches, frame_patches_one_frame
        )

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        if attention_backend == "flexattn":
            block_mask = prepare_flexattention(
                1,
                pipe.transformer.num_attention_heads,
                pipe.transformer.attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.block_mask = block_mask

            logger.info("Flexattn block_mask prepared.")
            logger.info(block_mask)

        elif attention_backend == "flashinfer":
            temporal_mask_metadata = prepare_flashinfer_attention(
                1,
                pipe.transformer.num_attention_heads,
                pipe.transformer.attention_head_dim,
                dtype,
                device,
                context_length,
                context_length,
                num_frame_patches,
                frame_patches_one_frame,
                diag_width,
                multiplier,
            )
            AttnModule.temporal_mask_metadata = temporal_mask_metadata

            print(
                visualize_sparse_bsr(
                    temporal_mask_metadata[0],
                    temporal_mask_metadata[1],
                    temporal_mask_metadata[2],
                )
            )

            logger.info("Flashinfer temporal_mask_metadata prepared.")
        else:
            raise ValueError(f"Attention backend {attention_backend} not supported")

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):  # Check if processor exists
                # Ensure layer_idx is set for SVG processor logic if it relies on it being an instance property after init
                current_processor = AttnModule(layer_idx=layer_idx)  # Instantiate with layer_idx
                current_processor.num_layers = num_layers
                # Other SVG specific properties already set on AttnModule class can be used or copied if needed
                m.attn1.set_processor(current_processor)

    elif pattern == "SAP":

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

        AttnModule = WanAttn_SAPAttn_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.num_layers = num_layers
        AttnModule.q_kmeans_iter_init = q_kmeans_iter_init
        AttnModule.k_kmeans_iter_init = k_kmeans_iter_init
        AttnModule.q_kmeans_iter_step = q_kmeans_iter_step
        AttnModule.k_kmeans_iter_step = k_kmeans_iter_step
        AttnModule.q_kmeans_init_method = q_kmeans_init_method
        AttnModule.k_kmeans_init_method = k_kmeans_init_method
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init
        AttnModule.kmeans_step_type = kmeans_step_type

        # 初始化稳定性分析跟踪器（如果启用）
        stability_tracker = None
        if stability_analysis_dir is not None:
            stability_tracker = ClusterStabilityTracker(output_dir=stability_analysis_dir)
            AttnModule.stability_tracker = stability_tracker
            logger.info(f"Stability analysis enabled, output directory: {stability_analysis_dir}")

        # KMEANS_BLOCK specific params for each instance, passed at init
        # replace_sparse_forward() was called earlier, assuming it's general.

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):  # Check if processor exists
                # Instantiate KMEANS_BLOCK processor with its specific parameters
                current_processor = AttnModule(
                    layer_idx=layer_idx,
                )
                m.attn1.set_processor(current_processor)

    elif pattern == "SAP_AdaKmeans":
        logger.info(
            f"Configuring SAP_AdaKmeans attention with QC: {num_q_centroids}, KC: {num_k_centroids}, P: {top_p_kmeans}, min_kc_ratio: {min_kc_ratio}, window_size: {window_size}"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = WanAttn_SAPAttn_Processor_AdaKmeans
        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame
        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.num_layers = num_layers
        AttnModule.q_kmeans_iter_init = q_kmeans_iter_init
        AttnModule.k_kmeans_iter_init = k_kmeans_iter_init
        AttnModule.q_kmeans_iter_step = q_kmeans_iter_step
        AttnModule.k_kmeans_iter_step = k_kmeans_iter_step
        AttnModule.q_kmeans_init_method = q_kmeans_init_method
        AttnModule.k_kmeans_init_method = k_kmeans_init_method
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init
        AttnModule.kmeans_step_type = kmeans_step_type
        AttnModule.window_size = window_size

        stability_tracker = None
        if stability_analysis_dir is not None:
            stability_tracker = ClusterStabilityTracker(output_dir=stability_analysis_dir)
            AttnModule.stability_tracker = stability_tracker
            logger.info(f"Stability analysis enabled, output directory: {stability_analysis_dir}")

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                m.attn1.set_processor(current_processor)

    elif pattern == "SAP_SR":
        logger.info(
            f"Configuring KMEANS_BLOCK attention with QC: {num_q_centroids}, KC: {num_k_centroids}, P: {top_p_kmeans}, min_kc_ratio: {min_kc_ratio}"
        )

        # Make dir and clear the logging file
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = WanAttn_SAPAttn_Processor_SR

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        # These might be needed by the processor if it has to adapt to sequence dimensions
        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.top_p_kmeans = top_p_kmeans
        AttnModule.r_kmeans = r_kmeans
        AttnModule.min_kc_ratio = min_kc_ratio
        AttnModule.num_layers = num_layers
        AttnModule.q_kmeans_iter_init = q_kmeans_iter_init
        AttnModule.k_kmeans_iter_init = k_kmeans_iter_init
        AttnModule.q_kmeans_iter_step = q_kmeans_iter_step
        AttnModule.k_kmeans_iter_step = k_kmeans_iter_step
        AttnModule.q_kmeans_init_method = q_kmeans_init_method
        AttnModule.k_kmeans_init_method = k_kmeans_init_method
        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init
        AttnModule.kmeans_step_type = kmeans_step_type

        # KMEANS_BLOCK specific params for each instance, passed at init
        # replace_sparse_forward() was called earlier, assuming it's general.

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):  # Check if processor exists
                # Instantiate KMEANS_BLOCK processor with its specific parameters
                current_processor = AttnModule(
                    layer_idx=layer_idx,
                )
                m.attn1.set_processor(current_processor)
    
    elif pattern == "PISA":
        logger.info(
            f"Configuring PISA (Piecewise Sparse) attention with num_k_centroids: {num_k_centroids}, "
            f"k_kmeans_iter_init: {k_kmeans_iter_init}, k_kmeans_iter_step: {k_kmeans_iter_step}"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")

        AttnModule = WanAttn_PISAttn_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_q_centroids = num_q_centroids
        AttnModule.num_k_centroids = num_k_centroids
        AttnModule.q_kmeans_iter_init = q_kmeans_iter_init
        AttnModule.k_kmeans_iter_init = k_kmeans_iter_init
        AttnModule.q_kmeans_iter_step = q_kmeans_iter_step
        AttnModule.k_kmeans_iter_step = k_kmeans_iter_step
        AttnModule.q_kmeans_init_method = q_kmeans_init_method
        AttnModule.k_kmeans_init_method = k_kmeans_init_method

        AttnModule.zero_step_kmeans_init = zero_step_kmeans_init
        AttnModule.kmeans_step_type = "full"
        AttnModule.num_layers = num_layers

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                m.attn1.set_processor(current_processor)
    elif pattern == "PURE_PISA":
        logger.info(
            f"Configuring PURE_PISA (Piecewise Sparse) attention"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")
        
        AttnModule = Wan_PISA_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_layers = num_layers

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                m.attn1.set_processor(current_processor)

    elif pattern == "MIX_PISA":
        logger.info(
            f"Configuring MIX_PISA (Multi-Level Piecewise Sparse) attention"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")
        
        AttnModule = Wan_MIX_PISA_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_layers = num_layers

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                m.attn1.set_processor(current_processor)
    
    elif pattern == "GROUP_PISA":
        logger.info(
            f"Configuring GROUP_PISA (Group-Level Piecewise Sparse) attention"
        )
        if logging_file is not None:
            os.makedirs(os.path.dirname(logging_file), exist_ok=True)
            with open(logging_file, "w") as f:
                f.write("")
        
        AttnModule = Wan_GROUP_PISA_Processor

        AttnModule.first_layers_fp = first_layers_fp
        AttnModule.first_times_fp = first_times_fp
        AttnModule.logging_file = logging_file

        AttnModule.context_length = context_length
        AttnModule.num_frame = num_frame_patches
        AttnModule.frame_size = frame_patches_one_frame

        AttnModule.num_layers = num_layers

        for layer_idx, m in enumerate(pipe.transformer.blocks):
            if hasattr(m.attn1, "processor"):
                current_processor = AttnModule(layer_idx=layer_idx)
                current_processor.rescaled_density = _rescale_density(
                    layer_idx, density, data=l1_layer_timestep
                )
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


def _rescale_density(layer_idx, density, data=None):
    # 读取 result/profiling/wan_attention_hidden_l1_layer_timestep.json：layer -> {str(timestep): factor}
    if data is None:
        with open("result/profiling/wan_attention_hidden_l1_layer_timestep.json", "r") as f:
            data = json.load(f)
    # json.load 的 dict 键均为 str，与 layer_idx 对齐
    rescale_factor = data[str(layer_idx)]
    rescaled_density = {}
    for timestep, factor in rescale_factor.items():
        if factor <= 0:
            rescaled_density[timestep] = density * math.exp(factor)
        else:
            rescaled_density[timestep] = max(density + factor, 0.85)

    return rescaled_density
