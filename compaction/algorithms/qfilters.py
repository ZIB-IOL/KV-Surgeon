"""
Q-Filters (Godey et al. 2025, arXiv:2503.02812): calibration-free eviction.

Keeps the top-t keys by s_j = <k_j, u>, where u is the principal direction of
the head's queries, estimated offline by scripts/compute_qfilters.py. No refit.
"""
import torch
from typing import Tuple

from .base import CompactionAlgorithm

# One load per (path, device); the wrapper builds a new instance per head.
_FILTER_CACHE = {}


def _load_filters(path: str, device) -> torch.Tensor:
    key = (path, str(device))
    if key not in _FILTER_CACHE:
        payload = torch.load(path, map_location=device)
        filters = payload['filters'] if isinstance(payload, dict) else payload
        _FILTER_CACHE[key] = filters.to(device=device, dtype=torch.float32)
    return _FILTER_CACHE[key]


class QFiltersCompaction(CompactionAlgorithm):
    """Query-free eviction scored by key projection onto the Q-filter."""

    def __init__(
        self,
        filter_path: str,
        n_sink: int = 0,
        n_recent: int = 0,
    ):
        """
        Parameters
        ----------
        filter_path : str
            .pt file from scripts/compute_qfilters.py (e.g.
            qfilters/pile/<Model>.pt) holding 'filters' of shape
            (num_layers, num_kv_heads, head_dim).
        n_sink, n_recent : int
            Always keep the first n_sink and last n_recent positions.
        """
        if not filter_path:
            raise ValueError("qfilters requires filter_path (see "
                             "scripts/compute_qfilters.py)")
        if n_sink < 0 or n_recent < 0:
            raise ValueError(
                f"n_sink and n_recent must be >= 0, got {n_sink}, {n_recent}")
        self.filter_path = filter_path
        self.n_sink = int(n_sink)
        self.n_recent = int(n_recent)
        self._layer_idx = None
        self._head_idx = None

    def name(self) -> str:
        return "QFilters"

    def requires_queries(self) -> bool:
        """Filters are precomputed offline."""
        return False

    def set_layer_head(self, layer_idx: int, head_idx: int):
        """Set the (layer, KV head) whose filter to use; called by the wrapper."""
        self._layer_idx = int(layer_idx)
        self._head_idx = int(head_idx)

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Keep the top-t keys by <k_j, u> with original values and zero beta."""
        if self._layer_idx is None or self._head_idx is None:
            raise RuntimeError(
                "QFiltersCompaction needs set_layer_head(layer, head) before "
                "compute_compacted_cache (per-layer-head wrapper required)")
        T = K.shape[0]
        t = min(int(t), T)

        filters = _load_filters(self.filter_path, K.device)
        L, H, d = filters.shape
        if self._layer_idx >= L or self._head_idx >= H:
            raise ValueError(
                f"filter file {self.filter_path} has shape {(L, H, d)}, "
                f"no entry for layer {self._layer_idx} head {self._head_idx}")
        u = filters[self._layer_idx, self._head_idx]                # (d,)

        key_scores = K.to(torch.float32) @ u                        # (T,)

        if self.n_sink > 0:
            key_scores[:min(self.n_sink, T)] = float('inf')
        if self.n_recent > 0:
            key_scores[max(0, T - self.n_recent):] = float('inf')

        _, top_idx = torch.topk(key_scores, t, largest=True)        # (t,)
        C1 = K[top_idx]
        indices = top_idx.cpu().tolist()

        beta = torch.zeros(t, dtype=K.dtype, device=K.device)
        C2 = self._direct_C2(C1, K, V, indices)

        return C1, beta, C2, indices
