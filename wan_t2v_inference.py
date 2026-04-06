import argparse
import json
import os
import math
from copy import deepcopy

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from termcolor import colored

from dataloader import load_prompt_or_image
from svg.models.wan.inference import replace_wan_attention
from svg.utils.seed import seed_everything
from svg.timer import print_operator_log_data

from svg.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video from text prompt using Wan-Diffuser")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers", help="Model ID to use for generation")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for video generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative text prompt to avoid certain features")

    parser.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Wan_VBench", "T2V_Xingyang_VBench"], help="Source of the prompt")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of the prompt")

    parser.add_argument("--height", type=int, default=720, help="Height of the generated video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the generated video")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames in the generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps in the generated video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file name")
    parser.add_argument("--logging_file", type=str, default=None, help="Path to the logging file.")
    parser.add_argument("--stability_analysis_dir", type=str, default=None, help="Directory to save stability analysis visualization. If provided, enables stability analysis.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generating existing output files")

    parser.add_argument(
        "--pattern",
        type=str,
        default="dense",
        choices=["SVG", "dense", "SAP", "SAP_SR", "SAP_AdaKmeans", "PISA", "PURE_PISA", "MIX_PISA", "GROUP_PISA"],
    )
    parser.add_argument("--first_layers_fp", type=float, default=0.025, help="Only works for best config. Leave the 0, 1, 2, 40, 41 layers in FP")
    parser.add_argument("--first_times_fp", type=float, default=0.075, help="Only works for best config. Leave the first 10% timestep in FP")
    
    # SVG related
    parser.add_argument("--num_sampled_rows", type=int, default=64, help="The number of sampled rows")
    parser.add_argument("--sample_mse_max_row", type=int, default=10000, help="The maximum number of rows in attention mask. Prevent OOM.")
    parser.add_argument("--sparsity", type=float, default=0.25, help="The sparsity of the striped attention pattern. Accepts one or two float values.")

    # SAP related
    parser.add_argument("--num_q_centroids", "--qc", type=int, default=50, help="Number of query centroids for KMEANS_BLOCK.")
    parser.add_argument("--num_k_centroids", "--kc", type=int, default=200, help="Number of key centroids for KMEANS_BLOCK.")
    parser.add_argument("--top_p_kmeans", type=float, default=0.9, help="Top-p threshold for block selection in KMEANS_BLOCK.")
    parser.add_argument("--r_kmeans", type=float, default=0.05, help="R threshold for block selection in KMEANS_BLOCK for SR attention.")
    parser.add_argument("--min_kc_ratio", type=float, default=0, help="At least this proportion of key blocks to keep per query block in KMEANS_BLOCK.")
    parser.add_argument("--q_kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for query initialization in KMEANS_BLOCK.")
    parser.add_argument("--k_kmeans_iter_init", type=int, default=0, help="Number of KMeans iterations for key initialization in KMEANS_BLOCK.")
    parser.add_argument("--q_kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for query in other diffusion steps in KMEANS_BLOCK.")
    parser.add_argument("--k_kmeans_iter_step", type=int, default=0, help="Number of KMeans iterations for key in other diffusion steps in KMEANS_BLOCK.")
    parser.add_argument("--q_kmeans_init_method", type=str, default="random", choices=["random", "uniform", "cache"], help="Initialization method for query KMeans centroids.")
    parser.add_argument("--k_kmeans_init_method", type=str, default="random", choices=["random", "uniform"], help="Initialization method for key KMeans centroids.")
    parser.add_argument("--kmeans_step_type", type=str, default="full", choices=["full", "key_only"], help="KMeans step: 'full' (Q+K) or 'key_only' (only K further clustering).")
    parser.add_argument("--window_size", type=int, default=2, help="SAP_AdaKmeans: window size for first N steps (sequence downsampling).")
    # Backward compatibility: if old args are provided, use them for both q and k
    parser.add_argument("--kmeans_iter_init", type=int, default=None, help="[Deprecated] Use --q_kmeans_iter_init and --k_kmeans_iter_init instead.")
    parser.add_argument("--kmeans_iter_step", type=int, default=None, help="[Deprecated] Use --q_kmeans_iter_step and --k_kmeans_iter_step instead.")

    # X_Kmeans_SAP related
    parser.add_argument("--num_x_centroids", type=int, default=1200, help="Number of x centroids for X_Kmeans_SAP.")

    # GROUP_PISA related
    parser.add_argument("--density", type=float, default=0.15, help="Density for GROUP_PISA.")

    args = parser.parse_args()

    seed_everything(args.seed)

    # In some cases it will raise RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR
    torch.backends.cuda.preferred_linalg_library(backend="magma")
    
    if args.skip_existing:
        if os.path.exists(args.output_file):
            logger.info(f"Output file {args.output_file} already exists. Skipping generation.")
            exit(0)

    #########################################################
    # Load the model
    #########################################################
    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.scheduler = scheduler
    pipe.to("cuda")
        
    config = pipe.transformer.config
    
    #########################################################
    # Translate the percentage of warmup of layers and timesteps to the actual layers and timesteps
    #########################################################
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    ref_timesteps = ref_scheduler.timesteps
    
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * config.num_layers)
    if num_fp_timesteps > 0:
        args.first_times_fp = ref_scheduler.timesteps[num_fp_timesteps - 1] - 1
    else:
        args.first_times_fp = 1001 # 1000 is the first timestep
    args.first_layers_fp = num_fp_layers
    
    logger.info(f"Warmup of Timesteps: {num_fp_timesteps} / {args.num_inference_steps} || {args.first_times_fp} / 1000 use FP")
    logger.info(f"Warmup of Layers: {num_fp_layers} / {config.num_layers} use FP")
    
    #########################################################
    # Load the prompt
    #########################################################
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)

    if args.prompt is None:
        logger.info(colored("Using default prompt", "red"))
        args.prompt = "A cat walks on the grass, realistic"

    if args.negative_prompt is None:
        args.negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    print("=" * 20 + " Prompts " + "=" * 20)
    print(f"Prompt: {args.prompt}\n\n" + f"Negative Prompt: {args.negative_prompt}")

    #########################################################
    # Replace the attention & time logger
    #########################################################
    if args.pattern == "SVG":
        replace_wan_attention(
            pipe, 
            args.height, 
            args.width, 
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SVG specific
            num_sampled_rows=args.num_sampled_rows,
            sample_mse_max_row=args.sample_mse_max_row,
            sparsity=args.sparsity,
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "SAP" or args.pattern == "SAP_SR":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # SAP specific
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            r_kmeans=args.r_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            q_kmeans_iter_init=args.q_kmeans_iter_init if args.kmeans_iter_init is None else args.kmeans_iter_init,
            k_kmeans_iter_init=args.k_kmeans_iter_init if args.kmeans_iter_init is None else args.kmeans_iter_init,
            q_kmeans_iter_step=args.q_kmeans_iter_step if args.kmeans_iter_step is None else args.kmeans_iter_step,
            k_kmeans_iter_step=args.k_kmeans_iter_step if args.kmeans_iter_step is None else args.kmeans_iter_step,
            q_kmeans_init_method=args.q_kmeans_init_method,
            k_kmeans_init_method=args.k_kmeans_init_method,
            kmeans_step_type=args.kmeans_step_type,
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "SAP_AdaKmeans":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            q_kmeans_iter_init=args.q_kmeans_iter_init if args.kmeans_iter_init is None else args.kmeans_iter_init,
            k_kmeans_iter_init=args.k_kmeans_iter_init if args.kmeans_iter_init is None else args.kmeans_iter_init,
            q_kmeans_iter_step=args.q_kmeans_iter_step if args.kmeans_iter_step is None else args.kmeans_iter_step,
            k_kmeans_iter_step=args.k_kmeans_iter_step if args.kmeans_iter_step is None else args.kmeans_iter_step,
            q_kmeans_init_method=args.q_kmeans_init_method,
            k_kmeans_init_method=args.k_kmeans_init_method,
            kmeans_step_type=args.kmeans_step_type,
            window_size=args.window_size,
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "X_Kmeans_SAP":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            # X_Kmeans_SAP specific
            num_x_centroids=args.num_x_centroids,
            top_p_kmeans=args.top_p_kmeans,
            min_kc_ratio=args.min_kc_ratio,
            logging_file=args.logging_file,
            x_kmeans_iter_init=args.kmeans_iter_init,
            x_kmeans_iter_step=args.kmeans_iter_step,
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "PISA":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            num_q_centroids=args.num_q_centroids,
            num_k_centroids=args.num_k_centroids,
            logging_file=args.logging_file,
            q_kmeans_iter_init=args.q_kmeans_iter_init if args.kmeans_iter_init is None else args.kmeans_iter_init,
            k_kmeans_iter_init=args.k_kmeans_iter_init if args.kmeans_iter_init is None else args.kmeans_iter_init,
            q_kmeans_iter_step=args.q_kmeans_iter_step if args.kmeans_iter_step is None else args.kmeans_iter_step,
            k_kmeans_iter_step=args.k_kmeans_iter_step if args.kmeans_iter_step is None else args.kmeans_iter_step,
            q_kmeans_init_method=args.q_kmeans_init_method,
            k_kmeans_init_method=args.k_kmeans_init_method,
            zero_step_kmeans_init=getattr(args, "zero_step_kmeans_init", False),
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "PURE_PISA":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            logging_file=args.logging_file,
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "MIX_PISA":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            logging_file=args.logging_file,
            stability_analysis_dir=args.stability_analysis_dir,
        )
    elif args.pattern == "GROUP_PISA":
        replace_wan_attention(
            pipe,
            args.height,
            args.width,
            args.num_frames,
            density=args.density,
            first_layers_fp=args.first_layers_fp,
            first_times_fp=args.first_times_fp,
            pattern=args.pattern,
            logging_file=args.logging_file,
            stability_analysis_dir=args.stability_analysis_dir,
        )

    # Print time logger
    for block in pipe.transformer.blocks:
        block.register_forward_hook(print_operator_log_data)

    #########################################################
    # Generate the video
    #########################################################
    output = pipe(
        prompt=args.prompt, negative_prompt=args.negative_prompt, height=args.height, width=args.width, num_frames=args.num_frames, guidance_scale=5.0, num_inference_steps=args.num_inference_steps
    ).frames[0]

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=16)
    
    # 注意：稳定性分析的可视化会在每个去噪 step 聚类完成后自动生成
    # 图片保存在 --stability_analysis_dir 指定的目录中
    if args.stability_analysis_dir is not None:
        logger.info(f"Stability analysis step-by-step visualizations saved to: {args.stability_analysis_dir}")
