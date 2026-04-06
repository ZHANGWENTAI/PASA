import torch

from svg.kmeans_utils import dynamic_block_sparse_fwd_flashinfer


def print_gapr_mask_sparsity(gapr_mask, prefix=""):
    """
    Print sparsity statistics of gapr_mask.
    
    Args:
        gapr_mask: (B, H, NQ, NK) - boolean mask indicating which blocks should be rectified
        prefix: optional prefix string for the print statement
    """
    B, H, NQ, NK = gapr_mask.shape
    total_elements = gapr_mask.numel()
    true_count = gapr_mask.sum().item()
    false_count = total_elements - true_count
    
    density = true_count / total_elements if total_elements > 0 else 0.0
    sparsity = false_count / total_elements if total_elements > 0 else 0.0
    
    print(f"{prefix}GAPR Mask Sparsity Statistics:")
    print(f"  Shape: (B={B}, H={H}, NQ={NQ}, NK={NK})")
    print(f"  Total elements: {total_elements}")
    print(f"  True (rectify): {true_count} ({density*100:.2f}%)")
    print(f"  False (skip): {false_count} ({sparsity*100:.2f}%)")
    
    # Print per-head statistics if H > 1
    if H > 1:
        print(f"  Per-head statistics:")
        for h in range(H):
            head_mask = gapr_mask[:, h, :, :]
            head_total = head_mask.numel()
            head_true = head_mask.sum().item()
            head_density = head_true / head_total if head_total > 0 else 0.0
            print(f"    Head {h}: {head_true}/{head_total} ({head_density*100:.2f}%)")


def estimate_pr_gain(query, key, qlabels, klabels, qcentroids, kcentroids, 
                     qcluster_sizes, kcluster_sizes, attention_scores):
    """
    Estimates the attention gains and pooling errors of the Pooling Rectification.
    
    Args:
        query: (B, H, S, D) - query tensor
        key: (B, H, S, D) - key tensor
        qlabels: (B, H, S) - query cluster labels
        klabels: (B, H, S) - key cluster labels
        qcentroids: (B, H, NQ, D) - query centroids
        kcentroids: (B, H, NK, D) - key centroids
        qcluster_sizes: (B, H, NQ) - query cluster sizes
        kcluster_sizes: (B, H, NK) - key cluster sizes
        attention_scores: (B, H, NQ, NK) - attention scores between clusters
    
    Returns:
        gapr_mask: (B, H, NQ, NK) A mask where the pooling correction gain exceeds the pooling error.
    """
    B, H, S, D = query.shape
    NQ = qcentroids.shape[2]
    NK = kcentroids.shape[2]
    device = query.device
    dtype = query.dtype
    
    # Compute delta_q and delta_k for each token
    # delta_q[b, h, s] = query[b, h, s] - qcentroids[b, h, qlabels[b, h, s]]
    qlabels_expanded = qlabels.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, S, D)
    q_assigned_centroids = torch.gather(qcentroids, dim=2, index=qlabels_expanded)  # (B, H, S, D)
    delta_q = query - q_assigned_centroids  # (B, H, S, D)
    
    klabels_expanded = klabels.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, H, S, D)
    k_assigned_centroids = torch.gather(kcentroids, dim=2, index=klabels_expanded)  # (B, H, S, D)
    delta_k = key - k_assigned_centroids  # (B, H, S, D)
    
    # Compute mean absolute delta per cluster
    # For each q_cluster: mean(|delta_q|) for tokens in that cluster
    delta_q_abs = delta_q.abs()  # (B, H, S, D)
    delta_q_sum = torch.zeros(B, H, NQ, D, dtype=dtype, device=device)
    # scatter_add requires index to have same shape as src, with indices along dim=2
    delta_q_sum.scatter_add_(dim=2, index=qlabels.unsqueeze(-1).expand(-1, -1, -1, D), src=delta_q_abs)
    delta_q_mean = torch.where(
        qcluster_sizes.unsqueeze(-1) > 0,
        delta_q_sum / qcluster_sizes.unsqueeze(-1),
        torch.zeros_like(delta_q_sum)
    )  # (B, H, NQ, D)
    
    delta_k_abs = delta_k.abs()  # (B, H, S, D)
    delta_k_sum = torch.zeros(B, H, NK, D, dtype=dtype, device=device)
    delta_k_sum.scatter_add_(dim=2, index=klabels.unsqueeze(-1).expand(-1, -1, -1, D), src=delta_k_abs)
    delta_k_mean = torch.where(
        kcluster_sizes.unsqueeze(-1) > 0,
        delta_k_sum / kcluster_sizes.unsqueeze(-1),
        torch.zeros_like(delta_k_sum)
    )  # (B, H, NK, D)
    
    # Reshape for batch matrix multiplication
    BH = B * H
    delta_q_mean_flat = delta_q_mean.reshape(BH, NQ, D)  # (BH, NQ, D)
    delta_k_mean_flat = delta_k_mean.reshape(BH, NK, D)  # (BH, NK, D)
    k_pools_flat = kcentroids.reshape(BH, NK, D)  # (BH, NK, D)
    q_pools_flat = qcentroids.reshape(BH, NQ, D)  # (BH, NQ, D)
    
    # Compute error scores: err_q = |delta_q_mean @ k_centroid|
    # For each (q_cluster, k_cluster) pair (average error per pair)
    dot_q = torch.bmm(delta_q_mean_flat, k_pools_flat.transpose(-1, -2))  # (BH, NQ, NK)
    err_q_sum = dot_q.abs().view(B, H, NQ, NK)
    
    # err_k = |q_centroid @ delta_k_mean|
    dot_k = torch.bmm(q_pools_flat, delta_k_mean_flat.transpose(-1, -2))  # (BH, NQ, NK)
    err_k_sum = dot_k.abs().view(B, H, NQ, NK)
    
    err_score = err_q_sum + err_k_sum  # (B, H, NQ, NK)

    # Compute gain score: multiply by IQ to get block-level total gain
    IQ_expanded = qcluster_sizes.unsqueeze(-1).expand(-1, -1, -1, NK)  # (B, H, NQ, NK)
    Gain_score = attention_scores.abs() * IQ_expanded
    
    # Print statistics for debugging
    print(f"[estimate_pr_gain] Gain_score and err_score statistics:")
    print(f"  Shape: (B={B}, H={H}, NQ={NQ}, NK={NK})")
    print(f"  Gain_score: min={Gain_score.min().item():.6f}, max={Gain_score.max().item():.6f}, mean={Gain_score.mean().item():.6f}, median={Gain_score.median().item():.6f}")
    print(f"  err_score: min={err_score.min().item():.6f}, max={err_score.max().item():.6f}, mean={err_score.mean().item():.6f}, median={err_score.median().item():.6f}")
    
    # Print some sample values (first batch, first head, first few Q-K pairs)
    if B > 0 and H > 0:
        sample_size = min(10, NQ * NK)
        gain_flat = Gain_score[0, 0].flatten()
        err_flat = err_score[0, 0].flatten()
        print(f"  Sample values (B=0, H=0, first {sample_size} elements):")
        for i in range(sample_size):
            print(f"    [{i}] Gain_score={gain_flat[i].item():.6f}, err_score={err_flat[i].item():.6f}, ratio={gain_flat[i].item()/err_flat[i].item() if err_flat[i].item() > 0 else float('inf'):.6f}")
        
        # Print statistics about the ratio
        ratio = Gain_score / (err_score + 1e-10)  # Add small epsilon to avoid division by zero
        print(f"  Gain/err ratio: min={ratio.min().item():.6f}, max={ratio.max().item():.6f}, mean={ratio.mean().item():.6f}, median={ratio.median().item():.6f}")
        print(f"  Number of elements where Gain > err: {(Gain_score > err_score).sum().item()} / {Gain_score.numel()}")
    
    # gapr_mask: only rectify when gain > error
    gapr_mask = Gain_score > err_score
    
    return gapr_mask


def rectified_sparse_attention(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,    # [B, H, S, D]
    value: torch.Tensor,  # [B, H, S, D]
    qlabels: torch.Tensor,  # [B, H, S]
    qcentroids: torch.Tensor,  # [B, H, num_q_centroids, D]
    qcluster_sizes: torch.Tensor,  # [B, H, num_q_centroids]
    klabels: torch.Tensor,  # [B, H, S]
    kcentroids: torch.Tensor,  # [B, H, num_k_centroids, D]
    kcluster_sizes: torch.Tensor,  # [B, H, num_k_centroids]
    triple_dynamic_map: torch.Tensor,  # [B, H, num_q_centroids, num_k_centroids]
    weighted_attn_probs: torch.Tensor,  # [B, H, num_q_centroids, num_k_centroids]
    num_q_centroids: int,
    num_k_centroids: int,
    top_p_kmeans: float,
    min_kc_ratio: float,
    sage: bool = False,
):
    """
    Combined attention processing for visual blocks:
    Return:
        output_normal: (B, H, S, D) - output of rectified sparse attention
    """
    batch_size, num_heads, context_size, head_dim = query.shape

    # Create masks for different computation types
    # 2 = fine-grained (逐个token计算), 1 = coarse-grained (rectified方式), 0 = 不计算
    fine_grained_mask = (triple_dynamic_map == 2)  # (B, H, qc_num, kc_num)
    coarse_grained_mask = (triple_dynamic_map == 1)  # (B, H, qc_num, kc_num)

    fine_grained_dynamic_map = fine_grained_mask.to(torch.bool)
    if sage:
        assert False, "SAGE is not supported yet"
    else:
        output_fine = dynamic_block_sparse_fwd_flashinfer(
            query, key, value, fine_grained_dynamic_map, 
            qcluster_sizes, kcluster_sizes, is_cpu=False
        )  # (B, H, S, D)
    
    # Rectifying the Attention Bias of Critical Tokens (for fine-grained blocks)
    attn_pool = weighted_attn_probs.masked_fill(~fine_grained_mask, 0.0)
    attn_pool_sum = torch.sum(attn_pool, dim=-1)  # (B, H, qc_num)
    rectified_factor_R = torch.gather(attn_pool_sum, dim=-1, index=qlabels)  # (B, H, context_size)

    attn_pool_novalid = weighted_attn_probs.masked_fill(~coarse_grained_mask, 0.0)  # (B, H, qc_num, kc_num)
    klabels_expanded = klabels.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # (B, H, context_size, head_dim)
    value_pool = torch.zeros(batch_size, num_heads, num_k_centroids, head_dim, 
                             dtype=value.dtype, device=value.device)
    value_pool.scatter_add_(dim=2, index=klabels_expanded, src=value)
    k_cluster_sizes_expanded = kcluster_sizes.unsqueeze(-1).to(value.dtype)  # (B, H, kc_num, 1)
    # Handle zero cluster sizes: if k_cluster_sizes is 0, value_pool is already 0, so keep it as 0
    # This means the cluster has no tokens, so it contributes nothing to the attention
    value_pool = torch.where(k_cluster_sizes_expanded > 0, 
                            value_pool / k_cluster_sizes_expanded, 
                            torch.zeros_like(value_pool))
    rectified_noncriattention_cluster = torch.matmul(attn_pool_novalid, value_pool)  # (B, H, qc_num, head_dim)
    # Expand cluster-level attention to token-level using qlabels
    # rectified_noncriattention_cluster[b, h, qc_id, :] -> rectified_noncriattention[b, h, token, :]
    output_coarse = torch.gather(
        rectified_noncriattention_cluster, 
        dim=2, 
        index=qlabels.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    )  # (B, H, context_size, head_dim)
    
    # Combine fine-grained and coarse-grained results
    # Reshape to (B*H, S, D) for element-wise operations
    output_fine = output_fine.reshape(batch_size * num_heads, context_size, head_dim)
    rectified_factor_R = rectified_factor_R.reshape(batch_size * num_heads, context_size)  # (B*H, S)
    output_coarse = output_coarse.reshape(batch_size * num_heads, context_size, head_dim)  # (B*H, S, D)

    # Fine-grained output needs rectification factor, coarse-grained output is already rectified
    output = output_fine * rectified_factor_R.to(query.dtype).unsqueeze(-1) + output_coarse

    return output.view(batch_size, num_heads, context_size, head_dim)
