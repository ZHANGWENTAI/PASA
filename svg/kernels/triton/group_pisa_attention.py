"""
PISA with Per-Group First-Order Correction (PISA-PG)

改进 vs 原始 PISA:
  - Phase 2: 零阶 + per-group 一阶修正 (替代全局 H 的 Phase 3)
  - Phase 3: 删除
  - Block selection: covariance-aware top-k
  - GROUP_SIZE 固定 32

结构:
  Prepare:  fused_chunk_reduce → qc, kc, vc, hc, h_norm
            compute_h_group    → h_group [BH, N_GROUPS, D, D]
            covariance-aware topk → indices [BH, NT, NS]
  Kernel:
    Phase 1: 精确 attention (S_i)
    Phase 2: 零阶 + per-group 一阶 (U_i = 所有非 S_i 的 block)
"""

import math
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl

def _ensure_triton_allocator():
    """Set Triton global scratch allocator to PyTorch CUDA if unset.
    Kernels using runtime scratch (e.g. fused_chunk_reduce) require this."""
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


_ensure_triton_allocator()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# =====================================================================
# Kernel 1: Fused block reduce
#   输出: qc [BH, NT, D], kc [BH, NT, D], vc [BH, NT, D],
#         hc [BH, NT, D, D], h_norm [BH, NT]
# =====================================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T"],
)
@triton.jit
def fused_chunk_reduce_kernel(
    q, k, v,
    qc, kc, vc, hc, h_norm,
    T,
    NT: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_kv, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_k = i_kv // tl.cdiv(V, BV)
    i_v = i_kv % tl.cdiv(V, BV)

    BLOCK_SIZE = tl.minimum(BT, T - i_t * BT)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))

    b_q = p_q.load([i_t * BT, i_k * BK])
    b_k = p_k.load([i_t * BT, i_k * BK])
    b_v = p_v.load([i_t * BT, i_v * BV])

    b_qc = tl.sum(b_q, axis=0) / BLOCK_SIZE
    b_kc = tl.sum(b_k, axis=0) / BLOCK_SIZE
    b_vc = tl.sum(b_v, axis=0)

    # hc = Σ k^T v - kc ⊗ vc (代数简化)
    b_hc = tl.dot(tl.trans(b_k).to(b_v.dtype), b_v)
    b_hc -= b_kc[:, None] * b_vc[None, :]

    # ||H_j||_F^2 的 partial (当 BK=K, BV=V 时就是完整的)
    b_norm_sq = tl.sum(b_hc * b_hc)

    # Store
    p_qc = tl.make_block_ptr(qc + i_bh * NT * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_kc = tl.make_block_ptr(kc + i_bh * NT * K + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * NT * V + i_t * V, (V,), (1,), (i_v * BV,), (BV,), (0,))

    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))

    p_hc = tl.make_block_ptr(
        hc + i_bh * NT * K * V + i_t * K * V,
        (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0),
    )
    tl.store(p_hc, b_hc.to(p_hc.dtype.element_ty), boundary_check=(0, 1))

    # h_norm: 只在第一个 kv tile 写入 (避免竞争)
    if i_k == 0 and i_v == 0:
        tl.store(h_norm + i_bh * NT + i_t, b_norm_sq.to(tl.float32))


# =====================================================================
# Kernel 2: Main attention — Phase 1 (exact) + Phase 2 (0th + 1st per-group)
# =====================================================================
@triton.jit
def pisa_pg_fwd_kernel(
    q, k, v,
    kc, vc,
    h_group,        # [BH, N_GROUPS, K, V]
    o,
    indices,        # [BH, NT, NS]
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_q = p_q.load([i_t * BT, 0])

    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float('inf')
    sm_scale = scale * 1.44269504

    # =================================================================
    # Phase 1: Exact Attention
    # =================================================================
    p_k_desc = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v_desc = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT

        b_k = p_k_desc.load([bos, 0])
        b_v = p_v_desc.load([bos, i_v * BV])

        b_s = tl.dot(b_q, tl.trans(b_k)) * sm_scale
        b_s += tl.where((bos + tl.arange(0, BT))[None, :] < T, 0, float("-inf"))

        new_m_i = tl.maximum(m_i, tl.max(b_s, -1))
        alpha = tl.math.exp2(m_i - new_m_i)
        score = tl.math.exp2(b_s - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(score, -1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m_i

    # =================================================================
    # Phase 2: Zero-order + Per-group first-order
    # =================================================================
    last_chunk_len = T - (NT - 1) * BT

    NS_POW2: tl.constexpr = triton.next_power_of_2(NS)
    offs_ns = tl.arange(0, NS_POW2)
    loaded_indices = tl.load(
        indices + i_bh * NT * NS + i_t * NS + offs_ns,
        mask=offs_ns < NS,
        other=-1,
    )

    g_l = tl.zeros([BT], dtype=tl.float32)

    p_kc_desc = tl.make_tensor_descriptor(
        kc + i_bh * NT * K, (NT, K), (K, 1), (GROUP_SIZE, BK),
    )
    p_vc_desc = tl.make_tensor_descriptor(
        vc + i_bh * NT * V, (NT, V), (V, 1), (GROUP_SIZE, BV),
    )

    indices_base = indices + i_bh * NT * NS + i_t * NS

    for start_n in range(0, NT, GROUP_SIZE):
        group_idx = start_n // GROUP_SIZE

        # --- Zero-order (all non-S_i blocks) ---
        b_kc = p_kc_desc.load([start_n, 0])
        b_s_mean = tl.dot(b_q, tl.trans(b_kc)) * sm_scale

        chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
        current_lens = tl.where(chunk_indices == NT - 1, last_chunk_len, BT).to(tl.float32)

        # 排除 S_i
        mask_is_selected = tl.zeros([GROUP_SIZE], dtype=tl.int32)
        for s_idx in range(NS):
            sel_idx = tl.load(indices_base + s_idx).to(tl.int32)
            mask_is_selected |= (chunk_indices == sel_idx).to(tl.int32)

        valid_mask = (chunk_indices < NT) & (mask_is_selected == 0)
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))

        new_m_i = tl.maximum(m_i, tl.max(b_s_mean, 1))
        alpha = tl.math.exp2(m_i - new_m_i)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        m_i = new_m_i

        prob_chunk = tl.math.exp2(b_s_mean - m_i[:, None])  # [BT, GROUP_SIZE]

        b_vc = p_vc_desc.load([start_n, i_v * BV])
        acc += tl.dot(prob_chunk.to(b_vc.dtype), b_vc)

        weighted_prob = prob_chunk * current_lens[None, :]
        g_l += tl.sum(weighted_prob, axis=1)

        # --- First-order per-group correction ---
        p_hg = tl.make_tensor_descriptor(
            h_group + i_bh * N_GROUPS * K * V + group_idx * K * V,
            (K, V), (V, 1), (BK, BV),
        )
        b_hg = p_hg.load([0, i_v * BV])              # [BK, BV]
        b_r = tl.dot(b_q, b_hg.to(b_q.dtype))        # [BT, BV]
        group_prob_sum = tl.sum(prob_chunk, axis=1)    # [BT]
        acc += b_r * (group_prob_sum * scale)[:, None]

    # =================================================================
    # Final
    # =================================================================
    l_i += g_l
    acc /= l_i[:, None]

    p_o = tl.make_tensor_descriptor(o + i_bh * T * V, (T, V), (V, 1), (BT, BV))
    p_o.store([i_t * BT, i_v * BV], acc.to(b_q.dtype))


# =====================================================================
# Python helpers
# =====================================================================
def fused_chunk_reduce(q, k, v, block_size):
    B, H, T, K = q.shape
    V = v.shape[-1]
    NT = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    qc = torch.empty(B, H, NT, K, device=q.device, dtype=q.dtype)
    kc = torch.empty(B, H, NT, K, device=q.device, dtype=q.dtype)
    vc = torch.empty(B, H, NT, V, device=v.device, dtype=v.dtype)
    hc = torch.empty(B, H, NT, K, V, device=v.device, dtype=v.dtype)
    h_norm = torch.zeros(B, H, NT, device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(K, BK) * triton.cdiv(V, BV), NT, B * H)
    fused_chunk_reduce_kernel[grid](
        q, k, v, qc, kc, vc, hc, h_norm,
        T, NT, K, V, block_size, BK, BV,
    )

    h_norm = h_norm.sqrt()
    return qc, kc, vc, hc, h_norm


def compute_h_group(hc, group_size):
    """hc [B, H, NT, D, D] → h_group [B*H, N_GROUPS, D, D]"""
    B, H, NT, K, V = hc.shape
    pad = (group_size - NT % group_size) % group_size
    if pad > 0:
        hc_padded = torch.nn.functional.pad(hc, (0, 0, 0, 0, 0, pad))
    else:
        hc_padded = hc
    N_GROUPS = hc_padded.shape[2] // group_size
    h_group = hc_padded.reshape(B, H, N_GROUPS, group_size, K, V).mean(dim=3)
    return h_group.reshape(B * H, N_GROUPS, K, V).contiguous(), N_GROUPS


# =====================================================================
# Main entry
# =====================================================================
def pisa_pg_attention(
    q, k, v,
    density=0.15,
    timestep=0,
    block_size=64,
    group_size=32,
    scale=None,
    use_cov_bias=True,
    use_random=False,
    verbose=False,
    prompt_token_length: Optional[int] = None,
):
    """
    PISA with Per-Group first-order correction.

    Args:
        q, k, v: [B, H, T, D]
        density:  fraction of blocks for exact attention
        block_size: token block size
        group_size: fixed GROUP_SIZE for per-group H
        scale: attention scale (None = 1/sqrt(D))
        use_cov_bias: use covariance-aware selection
        verbose: print debug info
        prompt_token_length: 若为 None/0，整段 Q 走原稀疏 kernel（全 score top-k）。
            若为正整数（如 256），仅在本函数内构造 indices（score 去掉前 strip_blocks 行/列后对子块 top-k，
            并合并 K block 0..strip_blocks-1）；仍对整段 q,k,v 调用 pisa_pg_fwd_kernel，前若干 Q block 的输出
            由调用方用 flash_attention 覆盖。须为 block_size 的整数倍。
    """
    if not supports_host_descriptor():
        warnings.warn(
            "Best performance on Hopper (H100). Current platform may be sub-optimal.",
            UserWarning,
        )

    B, H, T, K = q.shape
    V = v.shape[-1]
    scale = K ** -0.5 if scale is None else scale

    NT = triton.cdiv(T, block_size)
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))

    # =================================================================
    # Prepare Phase
    # =================================================================
    qc, kc, vc, hc, h_norm = fused_chunk_reduce(q, k, v, block_size)

    # Per-group H
    h_group, N_GROUPS = compute_h_group(hc, group_size)

    if verbose:
        print(f"NT={NT}, N_GROUPS={N_GROUPS}, GROUP_SIZE={group_size}")
        print(f"h_norm: mean={h_norm.mean():.2f}, max={h_norm.max():.2f}")
        print(f"h_group: {h_group.shape}")

    # =================================================================
    # Covariance-aware top-k selection
    # =================================================================
    score = torch.einsum('bhid, bhjd -> bhij', qc, kc * scale)

    if use_cov_bias:
        log_bias = torch.log(h_norm + 1e-6)
        if use_random:
            log_bias = log_bias + torch.randn_like(log_bias)
        score = score + log_bias.unsqueeze(2)  # broadcast over query dim
    
    top_k = min(max(1, int(density * NT)), NT)
    indices = torch.topk(score, k=top_k, dim=-1).indices  # [B, H, NT, top_k]
    indices = indices.reshape(B * H, NT, top_k).contiguous().to(torch.int32)

    if verbose:
        print(f"top_k={top_k}, density={density}")

    o = torch.empty_like(v)
    
    grid = (triton.cdiv(V, BV), NT, B * H)
    pisa_pg_fwd_kernel[grid](
        q=q, k=k, v=v,
        kc=kc, vc=vc,
        h_group=h_group,
        o=o,
        indices=indices,
        scale=scale,
        T=T,
        K=K, V=V,
        BT=block_size, BK=BK, BV=BV,
        NT=NT, NS=top_k,
        N_GROUPS=N_GROUPS,
        GROUP_SIZE=group_size,
    )

    return o

