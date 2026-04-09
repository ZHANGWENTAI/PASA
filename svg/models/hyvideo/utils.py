"""Mask Mod for Image2Video"""

import bisect
import math
from functools import lru_cache
from math import floor

import torch
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from torch.nn.attention.flex_attention import (
    create_block_mask,
)


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda", _compile=False):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=_compile)
    return block_mask


def generate_temporal_head_mask_mod(
    context_length: int = 226, prompt_length: int = 226, num_frames: int = 13, token_per_frame: int = 1350, mul: int = 2
):

    def round_to_multiple(idx):
        return floor(idx / 128) * 128

    real_length = num_frames * token_per_frame + prompt_length

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        real_mask = (kv_idx < real_length) & (q_idx < real_length)
        fake_mask = (kv_idx >= real_length) & (q_idx >= real_length)

        two_frame = round_to_multiple(mul * token_per_frame)
        temporal_head_mask = torch.abs(q_idx - kv_idx) < two_frame

        text_column_mask = (num_frames * token_per_frame <= kv_idx) & (kv_idx < real_length)
        text_row_mask = (num_frames * token_per_frame <= q_idx) & (q_idx < real_length)

        video_mask = temporal_head_mask | text_column_mask | text_row_mask
        real_mask = real_mask & video_mask

        return real_mask | fake_mask

    return temporal_mask_mod


def get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size, device="cuda"):

    attention_mask = torch.zeros(
        (context_length + num_frame * frame_size, context_length + num_frame * frame_size), device="cpu"
    )

    # TODO: fix hard coded mask
    if mask_name == "spatial":
        pixel_attn_mask = torch.zeros_like(
            attention_mask[:-context_length, :-context_length], dtype=torch.bool, device="cpu"
        )
        block_size, block_thres = 128, frame_size * 1.5
        num_block = math.ceil(num_frame * frame_size / block_size)
        for i in range(num_block):
            for j in range(num_block):
                if abs(i - j) < block_thres // block_size:
                    pixel_attn_mask[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size] = 1
        attention_mask[:-context_length, :-context_length] = pixel_attn_mask

        attention_mask[-context_length:, :] = 1
        attention_mask[:, -context_length:] = 1
        # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_spatial.pt", map_location="cpu")

    else:
        pixel_attn_mask = torch.zeros_like(
            attention_mask[:-context_length, :-context_length], dtype=torch.bool, device=device
        )

        block_size, block_thres = 128, frame_size * 1.5
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
        attention_mask[:-context_length, :-context_length] = pixel_attn_mask

        attention_mask[-context_length:, :] = 1
        attention_mask[:, -context_length:] = 1
        # attention_mask = torch.load(f"/data/home/xihaocheng/andy_develop/tmp_data/hunyuanvideo/I2VSparse/sparseattn/v5/mask_tensor/mask_temporal.pt", map_location="cpu")
    attention_mask = attention_mask[:sample_mse_max_row].cuda()
    return attention_mask


def get_prompt_length(pipe, prompt, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_sequence_length=256, device="cuda"):
    """
    Compute the prompt length for the prompt. In HunyuanVideo, we have prompt_length + unprompt_length = context_length, where context_length is a fixed value.
    We need to compute the prompt_length for the prompt in advance if using SVG to pre-compile the attention mask.
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    prompt = [prompt_template["template"].format(p) for p in prompt]

    crop_start = prompt_template.get("crop_start", None)
    if crop_start is None:
        prompt_template_input = pipe.tokenizer(
            prompt_template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = prompt_template_input["input_ids"].shape[-1]
        # Remove <|eot_id|> token and placeholder {}
        crop_start -= 2

    max_sequence_length += crop_start
    text_inputs = pipe.tokenizer(
        prompt,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    text_input_ids = text_inputs.input_ids.to(device=device)
    prompt_attention_mask = text_inputs.attention_mask.to(device=device)

    if crop_start is not None and crop_start > 0:
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

    prompt_length = prompt_attention_mask.sum()
    return prompt_length


def sparsity_to_width(sparsity, context_length, num_frame, frame_size):
    seq_len = context_length + num_frame * frame_size
    total_elements = seq_len**2

    sparsity = (sparsity * total_elements - 2 * seq_len * context_length) / total_elements

    width = seq_len * (1 - math.sqrt(1 - sparsity))
    width_frame = width / frame_size

    return width_frame


rescaled_factor = {
    980: 1.6152366403841978,
    977: 1.4244029491374073,
    973: 1.2456681678710728,
    969: 1.1577642551524232,
    965: 1.087926426180865,
    961: 1.0259881680507759,
    956: 0.9882446640033729,
    952: 0.998799764326042,
    947: 0.976715734175757,
    942: 0.9477546592234024,
    937: 1.0769424030861474,
    931: 0.9294812964749741,
    925: 1.0657480013838343,
    919: 0.9829626617416454,
    913: 0.9267012194852744,
    906: 0.9763504224744779,
    899: 0.881764156653075,
    891: 0.8393725940255019,
    883: 0.8388535229660975,
    875: 0.8281609252565499,
    865: 0.9164300181371487,
    856: 0.8481337559091641,
    846: 0.8545773921107588,
    835: 0.8255081833973007,
    823: 0.8232195053020991,
    810: 0.8222604824037629,
    797: 0.8989242548341655,
    782: 0.8668879556184683,
    767: 0.8278520354957465,
    750: 0.8240282394569347,
    731: 0.8416698289227471,
    710: 0.83350478810412,
    688: 0.8543081007474298,
    663: 0.8488151060279007,
    636: 0.859377515215382,
    605: 0.88276764659049,
    571: 0.8928392386920468,
    532: 0.9087985042129907,
    488: 0.9239564591192239,
    437: 0.9691458885876514,
    378: 1.038657955349757,
    308: 1.1553916971248515,
    225: 1.4314307717394603,
    125: 2.2366760448475045,
}

def lookup_rescaled_density(rescaled_density: dict, ts: int):
    """Exact timestep key if present; else nearest key (scheduler timesteps may not match calibration keys)."""
    if ts in rescaled_density:
        return rescaled_density[ts]
    keys = sorted(rescaled_density.keys())
    if not keys:
        return None
    if ts <= keys[0]:
        return rescaled_density[keys[0]]
    if ts >= keys[-1]:
        return rescaled_density[keys[-1]]
    i = bisect.bisect_left(keys, ts)
    lo, hi = keys[i - 1], keys[i]
    return rescaled_density[lo] if (ts - lo) <= (hi - ts) else rescaled_density[hi]