"""Base class for methods that compact the whole KV cache, plus head-budget helpers."""
import json
import torch
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod

from ..query_generation import QueryConfig

class FullCacheCompactionAlgorithm(ABC):
    """Compacts the KV cache of all layers and heads into a cache usable for generation."""

    @abstractmethod
    def name(self) -> str:
        pass

    def returns_cache(self) -> bool:
        """True if compact_kv_cache returns a (C1, beta, C2) cache, False if it returns context text."""
        return True

    def requires_preextracted_cache(self) -> bool:
        """False if the method prefills the context itself, so past_key_values may be None."""
        return True

    @abstractmethod
    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: object,
        tokenizer: object,
        formatted_context: str,
        vllm_model: Optional[object] = None,
        sliding_layer_indices: Optional[set] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact the KV cache of every layer and head.

        Parameters
        ----------
        past_key_values : tuple of (keys, values)
            Per-layer tensors of shape (batch, num_heads, seq_len, head_dim).
        target_size : int
            Total sequence length after compaction, including positions outside `indices`.
        indices : range, optional
            Positions to compact; the rest are kept unchanged. None compacts everything.
        vllm_model : object, optional
            vLLM model used for self-study query generation.
        sliding_layer_indices : set, optional
            Sliding-window layers; these keep their original cache.

        Returns
        -------
        compacted_cache : tuple of (C1, beta, C2)
            Per layer: keys (batch, heads, t, d), biases (batch, heads, t), values (batch, heads, t, d).
        stats : dict
        """
        pass

    def apply_compacted_cache(
        self,
        compacted_cache: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        query_tensor: torch.Tensor,
        layer_idx: int,
        head_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Attention output of `query_tensor` against the compacted cache (optional for subclasses)."""
        raise NotImplementedError(
            "This method should be implemented by subclasses that support "
            "inference-time application of compacted cache."
        )


def load_budgets_from_json(
    path: str,
    num_layers: int,
    num_heads: int,
) -> Dict[Tuple[int, int], Optional[float]]:
    """
    Load per-head budget proportions from JSON of the form {"L0H0": 0.014, ...}.

    Proportions sum to about 1 over all heads. Returns {(layer, head): proportion or None}.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    budgets = {}
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key_str = f'L{layer_idx}H{head_idx}'
            if key_str in data:
                budgets[(layer_idx, head_idx)] = data[key_str]
            else:
                budgets[(layer_idx, head_idx)] = None

    return budgets


def apply_max_ratio_cap(
    proportions: Dict[Tuple[int, int], Optional[float]],
    max_ratio_per_head: float,
    target_ratio: float,
    total_heads: int,
    verbose: bool = True,
) -> Dict[Tuple[int, int], Optional[float]]:
    """
    Cap each head's kept ratio at max_ratio_per_head by water-filling.

    Excess budget of capped heads is redistributed to the other heads in proportion
    to their share.

    Parameters
    ----------
    proportions : dict
        {(layer, head): proportion or None}.
    max_ratio_per_head : float
        Largest fraction of its tokens any head may keep.
    target_ratio : float
        Overall target size divided by context length.
    total_heads : int
        Number of compacted heads (global layers times heads).
    """
    if max_ratio_per_head >= 1.0 and target_ratio * total_heads <= 1.0:
        return proportions

    # Largest proportion a head can have before exceeding max_ratio_per_head.
    if target_ratio * total_heads > 0:
        max_proportion = max_ratio_per_head / (target_ratio * total_heads)
    else:
        return proportions

    valid_items = [(k, v) for k, v in proportions.items() if v is not None]
    if not valid_items:
        return proportions

    max_opt_proportion = max(v for _, v in valid_items)
    if max_opt_proportion <= max_proportion:
        return proportions

    result = {k: v for k, v in proportions.items()}

    for iteration in range(total_heads + 1):
        excess_total = 0.0
        uncapped_total = 0.0
        capped_keys = set()

        for key, prop in result.items():
            if prop is None:
                continue
            if prop > max_proportion:
                excess_total += prop - max_proportion
                capped_keys.add(key)
            else:
                uncapped_total += prop

        if excess_total == 0:
            break

        if uncapped_total == 0:
            for key in result:
                if result[key] is not None:
                    result[key] = min(result[key], max_proportion)
            break

        for key in result:
            if result[key] is None:
                continue
            if key in capped_keys:
                result[key] = max_proportion
            else:
                result[key] = result[key] + (result[key] / uncapped_total) * excess_total

    if verbose:
        new_max = max((p for p in result.values() if p is not None), default=0)
        original_max = max((p for p in proportions.values() if p is not None), default=0)
        print(f"Applied max ratio cap (max_ratio_per_head={max_ratio_per_head})")
        print(f"  Target ratio: {target_ratio:.4f}")
        print(f"  Max allowed proportion: {max_proportion:.6f}")
        print(f"  Original max proportion: {original_max:.6f}")
        print(f"  New max proportion: {new_max:.6f}")

    return result
