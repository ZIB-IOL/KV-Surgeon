"""
Magnitude-Evict: calibration-free eviction without refit.

score='norm':    s_j = ||v_j||^b / ||k_j||^a (small key norm predicts high
                 attention, Devoto et al. 2024, arXiv:2406.11430).
score='recency': s_j = j; with n_sink > 0 this is StreamingLLM.
"""
import torch
from typing import Tuple

from .base import CompactionAlgorithm


class MagnitudeEvictCompaction(CompactionAlgorithm):
    """Calibration-free eviction scored from key/value magnitudes."""

    def __init__(
        self,
        score: str = 'norm',
        key_exponent: float = 2.0,
        value_exponent: float = 2.0,
        norm_eps: float = 1e-12,
        n_sink: int = 0,
        n_recent: int = 0,
    ):
        """
        Parameters
        ----------
        score : {'norm', 'recency'}
        key_exponent : float
            Exponent a on the inverse key norm.
        value_exponent : float
            Exponent b on the value norm.
        n_sink, n_recent : int
            Always keep the first n_sink and last n_recent positions.
        """
        if score not in ('norm', 'recency'):
            raise ValueError(f"score must be 'norm' or 'recency', got '{score}'")
        if n_sink < 0 or n_recent < 0:
            raise ValueError(
                f"n_sink and n_recent must be >= 0, got {n_sink}, {n_recent}"
            )
        self.score = score
        self.key_exponent = key_exponent
        self.value_exponent = value_exponent
        self.norm_eps = norm_eps
        self.n_sink = int(n_sink)
        self.n_recent = int(n_recent)

    def name(self) -> str:
        return "MagnitudeEvict"

    def requires_queries(self) -> bool:
        """Scoring is calibration-free and there is no refit."""
        return False

    def _norm_scores(self, K32: torch.Tensor, V32: torch.Tensor) -> torch.Tensor:
        """s_j = ||v_j||^b / ||k_j||^a."""
        k_norm = K32.norm(dim=1)                                    # (T,)
        inv_k = 1.0 / (k_norm.clamp_min(self.norm_eps) ** self.key_exponent)
        payload = V32.norm(dim=1)                                   # (T,)
        return inv_k * payload ** self.value_exponent               # (T,)

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Keep the top-t positions by score; queries and attention_bias are ignored."""
        T = K.shape[0]
        t = min(int(t), T)

        if self.score == 'recency':
            key_scores = torch.arange(
                T, dtype=torch.float32, device=K.device)                # (T,)
        else:
            key_scores = self._norm_scores(
                K.to(torch.float32), V.to(torch.float32))               # (T,)

        # Protected positions always win the top-k.
        if self.n_sink > 0:
            key_scores[:min(self.n_sink, T)] = float('inf')
        if self.n_recent > 0:
            key_scores[max(0, T - self.n_recent):] = float('inf')

        _, top_idx = torch.topk(key_scores, t, largest=True)        # (t,)
        C1 = K[top_idx]                                             # (t, d)
        indices = top_idx.cpu().tolist()

        beta = torch.zeros(t, dtype=K.dtype, device=K.device)
        C2 = self._direct_C2(C1, K, V, indices)

        return C1, beta, C2, indices
