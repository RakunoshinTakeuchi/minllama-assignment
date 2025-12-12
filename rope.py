from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.
    """
    _, seqlen, _, _ = query.shape
    device = query.device
    dtype = query.dtype

    # query/key を複素数表現に分解
    # shape: (..., head_dim/2) になる
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # -----------------------------
    # 1. 角度 theta_pos_k を計算
    # -----------------------------
    # 位置インデックス: [0, 1, ..., seqlen-1]
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)  # (seqlen,)

    # 周波数: inv_freq の次元は head_dim/2
    # 典型的な定義: 1 / (theta^(2i / head_dim))
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
    # freqs: (seqlen, half_dim)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)

    # cos, sin: (seqlen, half_dim)
    cos = freqs.cos()
    sin = freqs.sin()

    # query_real: (B, seqlen, H, half_dim)
    # reshape_for_broadcast で (1, seqlen, 1, half_dim) にしてブロードキャスト
    cos = reshape_for_broadcast(cos, query_real)
    sin = reshape_for_broadcast(sin, query_real)

    # -----------------------------------------
    # 2. 複素数の回転: (real + i imag) * (cos + i sin)
    # -----------------------------------------
    # out_real = real * cos - imag * sin
    # out_imag = real * sin + imag * cos
    q_out_real = query_real * cos - query_imag * sin
    q_out_imag = query_real * sin + query_imag * cos

    k_out_real = key_real * cos - key_imag * sin
    k_out_imag = key_real * sin + key_imag * cos

    # -----------------------------
    # 3. 実部・虚部を元の形に戻す
    # -----------------------------
    # stack(..., dim=-1) で (..., half_dim, 2) → reshape で (..., head_dim)
    query_out = torch.stack((q_out_real, q_out_imag), dim=-1).reshape_as(query).type_as(query)
    key_out = torch.stack((k_out_real, k_out_imag), dim=-1).reshape_as(key).type_as(key)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
