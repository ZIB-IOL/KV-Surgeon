"""
Highest-attention keys: keep the t keys with the largest attention over the
calibration queries (max, mean or RMS), fit beta by NNLS and values by least squares.
"""
import torch
from typing import Tuple
from .base import CompactionAlgorithm


class HighestAttentionKeysCompaction(CompactionAlgorithm):
    """Keep the keys with the highest attention scores."""

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None,
                 score_method: str = 'max', c2_method: str = 'lsq', beta_method: str = 'nnls',
                 c2_ridge_lambda: float = 0, c2_solver: str = 'lstsq', c2_ridge_scale: str = 'spectral'):
        """
        Parameters
        ----------
        nnls_iters : int
            Projected-gradient iterations for NNLS (0 = clamped least squares).
        nnls_lower_bound, nnls_upper_bound : float, optional
            Box constraints on exp(beta).
        score_method : {'max', 'mean', 'rms'}
            Aggregation of each key's attention over queries.
        c2_method : {'lsq', 'direct'}
            Value fit: least squares or the original values.
        beta_method : {'nnls', 'zero'}
        c2_solver, c2_ridge_scale : str
            See CompactionAlgorithm._compute_C2.
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        if score_method not in ['mean', 'rms', 'max']:
            raise ValueError(f"score_method must be 'mean', 'rms', or 'max', got '{score_method}'")
        self.score_method = score_method
        self.c2_method = c2_method
        if beta_method not in ['nnls', 'zero']:
            raise ValueError(f"beta_method must be 'nnls' or 'zero', got '{beta_method}'")
        self.beta_method = beta_method
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale

    def name(self) -> str:
        return "HighestAttentionKeys"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Select keys by attention score, then fit beta and values."""
        C1, beta, indices = self._select_keys_highest_attention(K, queries, t, attention_bias)

        C2 = self._compute_C2_with_method(
            C1, beta, K, V, queries,
            method=self.c2_method,
            indices=indices,
            attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale
        )

        return C1, beta, C2, indices

    def _select_keys_highest_attention(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ):
        """Top-t keys by score_method; returns (C1, beta, indices)."""
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype

        inv_sqrt_d = (1.0 / d) ** 0.5

        scores_raw = queries @ K.T                                 # (n, T) original dtype
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d       # (n, T) fp32
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores32.shape
                )
                scores32 = scores32 + bias32                       # (n, T)
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores32.max(dim=1, keepdim=True)[0]          # (n, 1) fp32
        exp_scores = torch.exp(scores32 - max_scores)              # (n, T) fp32

        sum_exp = exp_scores.sum(dim=1, keepdim=True)              # (n, 1)
        attention_weights = exp_scores / sum_exp                   # (n, T)

        if self.score_method == 'rms':
            key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,) fp32
        elif self.score_method == 'max':
            key_scores = attention_weights.max(dim=0)[0]  # (T,) fp32
        else:  # 'mean'
            key_scores = attention_weights.mean(dim=0)              # (T,) fp32

        _, top_indices = torch.topk(key_scores, t, largest=True)
        selected_indices_tensor = top_indices

        C1 = K[selected_indices_tensor]  # (t, d)

        if self.beta_method == 'zero':
            beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        else:  # 'nnls'
            target = exp_scores.sum(dim=1)  # (n,) fp32

            M = exp_scores[:, selected_indices_tensor]  # (n, t) fp32

            B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)  # (t,) fp32 (>=0)
            beta32 = torch.log(B)  # (t,) fp32

        beta = beta32.to(dtype_param)

        selected_indices = selected_indices_tensor.cpu().tolist()

        return C1, beta, selected_indices
