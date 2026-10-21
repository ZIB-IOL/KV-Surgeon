"""Applies a single-head compaction algorithm to every layer and KV head independently."""
import torch
from typing import Tuple, Dict, Optional, Type, Any

from .base import FullCacheCompactionAlgorithm, load_budgets_from_json, apply_max_ratio_cap
from ..algorithms.base import CompactionAlgorithm
from ..query_generation import QueryConfig
from models.cache import CompactedPrefixCache, CompactedPrefixLayer, DynamicSlidingWindowLayer


class PerLayerHeadCompaction(FullCacheCompactionAlgorithm):
    """Runs a CompactionAlgorithm on each (layer, head) of the full cache."""

    def __init__(
        self,
        algorithm_class: Type[CompactionAlgorithm],
        algorithm_kwargs: Optional[Dict] = None,
        config_name: Optional[str] = None,
        precomputed_budget_path: Optional[str] = None,
        max_ratio_per_head: float = 1.0,
    ):
        """
        Parameters
        ----------
        algorithm_class : type
            CompactionAlgorithm subclass, instantiated once per head.
        precomputed_budget_path : str, optional
            JSON of per-head budget proportions; None gives uniform budgets.
        max_ratio_per_head : float
            Cap on the fraction of its tokens a head may keep under precomputed budgets.
        """
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs or {}
        self._name_instance = algorithm_class(**self.algorithm_kwargs)
        self.config_name = config_name
        self.precomputed_budget_path = precomputed_budget_path
        self.max_ratio_per_head = max_ratio_per_head

    def name(self) -> str:
        if self.config_name:
            return self.config_name
        return f"per_layer_head_{self._name_instance.name()}"

    @staticmethod
    def _split_layer_cache(layer_tuple):
        """Normalize (K, V) or (K, bias, V) to (K, bias or None, V)."""
        if not isinstance(layer_tuple, (list, tuple)):
            raise TypeError(f"Expected layer cache to be tuple/list, got {type(layer_tuple)}")
        if len(layer_tuple) == 2:
            keys, values = layer_tuple
            bias = None
        elif len(layer_tuple) == 3:
            keys, bias, values = layer_tuple
        else:
            raise ValueError(f"Layer cache must have 2 or 3 elements (K,[bias],V); got {len(layer_tuple)}")
        return keys, bias, values

    @staticmethod
    def _compacted_prefix_to_tuple(cache: CompactedPrefixCache):
        """Convert a CompactedPrefixCache to per-layer (K, beta, V); sliding layers get beta = 0."""
        layers = []
        for layer in cache.layers:
            if isinstance(layer, CompactedPrefixLayer):
                layers.append((layer.keys, layer.beta, layer.values))
            elif isinstance(layer, DynamicSlidingWindowLayer):
                keys = layer.keys
                values = layer.values
                beta = torch.zeros(
                    keys.shape[0],
                    keys.shape[1],
                    keys.shape[2],
                    device=keys.device,
                    dtype=keys.dtype,
                )
                layers.append((keys, beta, values))
            else:
                raise TypeError(f"Unsupported layer type in CompactedPrefixCache: {type(layer)}")
        return tuple(layers)

    def compact_kv_cache(
        self,
        past_key_values: Tuple,
        target_size: int,
        indices: Optional[range],
        query_config: QueryConfig,
        model: Any,
        tokenizer: Any,
        formatted_context: str,
        verbose_logging: bool = False,
        vllm_model: Optional[Any] = None,
        sliding_layer_indices: Optional[set] = None,
        past_key_values_for_queries: Optional[Any] = None,
        full_query_extraction: bool = False,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact each (layer, head) independently. See FullCacheCompactionAlgorithm.

        Additional parameters
        ---------------------
        past_key_values : tuple or CompactedPrefixCache
            Per layer (K, V) or (K, bias, V).
        past_key_values_for_queries : optional
            Cache used only for query generation (chunked compaction passes the
            full prefix so queries see correct context and RoPE positions).
        full_query_extraction : bool
            Generate queries over the whole context instead of only `indices`.
        """
        kv_for_queries = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        if isinstance(past_key_values, CompactedPrefixCache):
            if sliding_layer_indices is None:
                sliding_layer_indices = getattr(past_key_values, "_sliding_layer_indices", None)
            past_key_values = self._compacted_prefix_to_tuple(past_key_values)

        num_layers = len(past_key_values)

        # Sliding-window layers may hold fewer tokens, so read a full-attention layer.
        sliding_layer_indices = sliding_layer_indices or set()
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        if batch_size != 1:
            raise NotImplementedError(
                "PerLayerHeadCompaction currently only supports batch_size=1"
            )

        num_global_layers = num_layers - len(sliding_layer_indices)
        if sliding_layer_indices:
            print(f"Using {len(sliding_layer_indices)} sliding window layers (indices: {sorted(sliding_layer_indices)})")
            print(f"Only compacting {num_global_layers} global attention layers")

        if not self._name_instance.requires_queries():
            print(f"{self._name_instance.name()} is calibration-free: skipping query generation")
            queries = None
            query_stats = {'skipped': True, 'final_n_queries_per_kv_head': 0}
        else:
            from ..query_generation import QueryGenerator

            device = past_key_values[0][0].device
            dtype = past_key_values[0][0].dtype
            generator = QueryGenerator(
                model=model,
                tokenizer=tokenizer,
                config=query_config,
                device=device,
                dtype=dtype,
                vllm_model=vllm_model,
            )

            query_indices = indices
            if full_query_extraction:
                full_len = tokenizer(
                    formatted_context,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.shape[1]
                query_indices = range(0, full_len)
            base_queries, query_stats, _ = generator.generate_queries(
                formatted_context=formatted_context,
                past_key_values=kv_for_queries,
                indices=query_indices,
            )

            # (1, num_layers, num_kv_heads, n_queries, head_dim)
            queries = base_queries.unsqueeze(0)

            print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")
            for method, method_stats in query_stats.get('methods_used', {}).items():
                print(f"  {method}: {method_stats['n_queries_actual_per_kv_head']} queries ({method_stats['fraction']:.1%})")
            print("Train queries shape:", queries.shape)

        if indices is not None:
            indices_list = list(indices)
            num_to_compact = len(indices_list)
            num_to_keep = seq_len - num_to_compact

            sub_target_size = target_size - num_to_keep

            if sub_target_size <= 0:
                raise ValueError(
                    f"target_size ({target_size}) must be greater than the number of "
                    f"positions to keep ({num_to_keep}). Got sub_target_size = {sub_target_size}"
                )

            all_indices = torch.arange(seq_len)
            compact_mask = torch.zeros(seq_len, dtype=torch.bool)
            compact_mask[indices_list] = True
            keep_mask = ~compact_mask
            keep_indices = all_indices[keep_mask].tolist()

            actual_target_size = sub_target_size
            is_partial_compaction = True
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size
            is_partial_compaction = False

        # Budgets apply to the compacted span only, not the kept tokens.
        per_head_budgets = None
        if self.precomputed_budget_path is not None:
            print(f"Loading per-head budgets from {self.precomputed_budget_path}...")
            per_head_proportions = load_budgets_from_json(
                self.precomputed_budget_path, num_layers, num_heads
            )

            # Head budget = proportion * actual_target_size * total_heads, so the total matches
            # a uniform budget over all global (non-sliding) heads.
            article_len = len(indices_list) if is_partial_compaction else seq_len
            target_ratio = actual_target_size / article_len if article_len > 0 else 1.0
            total_heads = num_global_layers * num_heads

            per_head_proportions = apply_max_ratio_cap(
                per_head_proportions,
                self.max_ratio_per_head,
                target_ratio,
                total_heads,
            )

            per_head_budgets = {}
            for (layer_idx, head_idx), proportion in per_head_proportions.items():
                if proportion is not None:
                    per_head_budgets[(layer_idx, head_idx)] = int(proportion * actual_target_size * total_heads)
                else:
                    per_head_budgets[(layer_idx, head_idx)] = actual_target_size

            total_budget = sum(per_head_budgets.values())
            total_proportions = sum(p for p in per_head_proportions.values() if p is not None)
            max_head_budget = max(per_head_budgets.values()) if per_head_budgets else 0
            print(f"Per-head proportions sum to: {total_proportions:.4f}")
            print(f"Per-head budget distribution: total={total_budget}, "
                  f"min={min(per_head_budgets.values()) if per_head_budgets else 0}, "
                  f"max={max_head_budget}, "
                  f"mean={total_budget / (num_global_layers * num_heads):.1f}")

        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_global_layers,
        }

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }
            all_stats['keep_indices'] = {
                'num_positions': len(keep_indices),
            }

        compacted_layers, all_stats = self._compact_sequential(
            past_key_values, queries, actual_target_size,
            is_partial_compaction, indices_list, keep_indices,
            all_stats, query_stats,
            model, verbose_logging, per_head_budgets,
            sliding_layer_indices
        )

        return tuple(compacted_layers), all_stats

    def _compact_sequential(
        self,
        past_key_values,
        queries,
        actual_target_size,
        is_partial_compaction,
        indices_list,
        keep_indices,
        all_stats,
        query_stats,
        model=None,
        verbose_logging=False,
        per_head_budgets=None,
        sliding_layer_indices=None,
    ):
        """Compact every (layer, head) pair in turn."""
        num_layers = len(past_key_values)
        sliding_layer_indices = sliding_layer_indices or set()

        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        num_heads = past_key_values[ref_layer_idx][0].shape[1]
        seq_len = past_key_values[ref_layer_idx][0].shape[2]
        head_dim = past_key_values[ref_layer_idx][0].shape[3]
        compacted_layers = []

        total_effective_article_tokens = 0

        for layer_idx in range(num_layers):
            keys_layer, bias_layer, values_layer = self._split_layer_cache(past_key_values[layer_idx])
            if bias_layer is not None and bias_layer.dim() == 2:
                bias_layer = bias_layer.unsqueeze(0)

            if layer_idx in sliding_layer_indices:
                print(f"Layer {layer_idx+1}/{num_layers}: sliding window (keeping original KV)")
                # Empty placeholder; CompactedPrefixCache uses a DynamicSlidingWindowLayer here.
                placeholder_C1 = keys_layer.new_zeros(1, num_heads, 0, head_dim)
                placeholder_beta = keys_layer.new_zeros(1, num_heads, 0)
                placeholder_C2 = values_layer.new_zeros(1, num_heads, 0, head_dim)
                compacted_layers.append((placeholder_C1, placeholder_beta, placeholder_C2))
                continue

            print(f"Compacting layer {layer_idx+1}/{num_layers}")

            C1_heads = []
            beta_heads = []
            C2_heads = []

            for head_idx in range(num_heads):
                K_full = keys_layer[0, head_idx, :, :]  # (seq_len, head_dim)
                V_full = values_layer[0, head_idx, :, :]  # (seq_len, head_dim)

                if is_partial_compaction:
                    K = K_full[indices_list, :]  # (num_to_compact, head_dim)
                    V = V_full[indices_list, :]  # (num_to_compact, head_dim)
                    attn_bias_head = bias_layer[0, head_idx, indices_list] if bias_layer is not None else None

                    K_keep = K_full[keep_indices, :]  # (num_to_keep, head_dim)
                    V_keep = V_full[keep_indices, :]  # (num_to_keep, head_dim)
                    bias_keep = bias_layer[0, head_idx, keep_indices] if bias_layer is not None else None
                else:
                    K = K_full
                    V = V_full
                    attn_bias_head = bias_layer[0, head_idx, :] if bias_layer is not None else None
                    bias_keep = None

                # None for calibration-free algorithms.
                queries_head = queries[0, layer_idx, head_idx, :, :] if queries is not None else None

                if per_head_budgets is not None:
                    head_target_size = per_head_budgets.get((layer_idx, head_idx), actual_target_size)
                else:
                    head_target_size = actual_target_size

                if head_target_size == 0:
                    C1_compact = K.new_zeros(0, head_dim)
                    beta_compact = K.new_zeros(0)
                    C2_compact = V.new_zeros(0, head_dim)
                    selected_indices = []
                else:
                    algorithm = self.algorithm_class(**self.algorithm_kwargs)
                    # For algorithms with per-(layer, head) state, e.g. Q-Filters.
                    if hasattr(algorithm, 'set_layer_head'):
                        algorithm.set_layer_head(layer_idx, head_idx)

                    C1_compact, beta_compact, C2_compact, selected_indices = algorithm.compute_compacted_cache(
                        K, V, queries_head, head_target_size, attention_bias=attn_bias_head
                    )

                # An algorithm may return fewer keys than its budget.
                actual_returned_size = C1_compact.shape[0]
                total_effective_article_tokens += actual_returned_size
                beta_for_stats = beta_compact
                if actual_returned_size < head_target_size:
                    num_padding = head_target_size - actual_returned_size
                    C1_padding = K.new_zeros(num_padding, head_dim)
                    C2_padding = V.new_zeros(num_padding, head_dim)
                    # beta = -inf masks the padding out of attention.
                    beta_padding = K.new_full((num_padding,), float('-inf'))

                    C1_compact = torch.cat([C1_compact, C1_padding], dim=0)
                    beta_compact = torch.cat([beta_compact, beta_padding], dim=0)
                    C2_compact = torch.cat([C2_compact, C2_padding], dim=0)

                if is_partial_compaction:
                    # Layout: [kept before span, compacted span, kept after span].
                    compact_start = indices_list[0]
                    compact_end = indices_list[-1] + 1

                    keep_before_mask = [idx < compact_start for idx in keep_indices]
                    keep_after_mask = [idx >= compact_end for idx in keep_indices]

                    K_keep_before = K_keep[keep_before_mask, :] if any(keep_before_mask) else K.new_zeros(0, head_dim)
                    V_keep_before = V_keep[keep_before_mask, :] if any(keep_before_mask) else V.new_zeros(0, head_dim)
                    K_keep_after = K_keep[keep_after_mask, :] if any(keep_after_mask) else K.new_zeros(0, head_dim)
                    V_keep_after = V_keep[keep_after_mask, :] if any(keep_after_mask) else V.new_zeros(0, head_dim)

                    if bias_keep is not None:
                        bias_keep_before = bias_keep[keep_before_mask]
                        bias_keep_after = bias_keep[keep_after_mask]
                    else:
                        bias_keep_before = K.new_zeros(K_keep_before.shape[0])
                        bias_keep_after = K.new_zeros(K_keep_after.shape[0])

                    beta_keep_before = bias_keep_before
                    beta_keep_after = bias_keep_after

                    C1 = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=0)
                    beta = torch.cat([beta_keep_before, beta_compact, beta_keep_after], dim=0)
                    C2 = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=0)
                else:
                    C1 = C1_compact
                    beta = beta_compact
                    C2 = C2_compact

                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))  # (1, 1, t, head_dim)
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))  # (1, 1, t)
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))  # (1, 1, t, head_dim)

                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_for_stats.min().item()) if len(beta_for_stats) > 0 else None,
                        'max': float(beta_for_stats.max().item()) if len(beta_for_stats) > 0 else None,
                        'mean': float(beta_for_stats.mean().item()) if len(beta_for_stats) > 0 else None,
                        'std': float(beta_for_stats.std().item()) if len(beta_for_stats) > 1 else None,
                        'num_less_than_minus_7': int((beta_for_stats < -7).sum().item()) if len(beta_for_stats) > 0 else 0,
                    }} if verbose_logging else {})
                }

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

            # Pad heads to the layer's longest head; layers may differ in length.
            target_seq_len = max(h.shape[2] for h in C1_heads)
            for i in range(len(C1_heads)):
                curr_len = C1_heads[i].shape[2]
                if curr_len < target_seq_len:
                    pad_len = target_seq_len - curr_len
                    C1_heads[i] = torch.cat([
                        C1_heads[i],
                        C1_heads[i].new_zeros(1, 1, pad_len, head_dim)
                    ], dim=2)
                    C2_heads[i] = torch.cat([
                        C2_heads[i],
                        C2_heads[i].new_zeros(1, 1, pad_len, head_dim)
                    ], dim=2)
                    beta_heads[i] = torch.cat([
                        beta_heads[i],
                        beta_heads[i].new_full((1, 1, pad_len), float('-inf'))
                    ], dim=2)

            C1_layer = torch.cat(C1_heads, dim=1)  # (1, num_heads, t, head_dim)
            beta_layer = torch.cat(beta_heads, dim=1)  # (1, num_heads, t)
            C2_layer = torch.cat(C2_heads, dim=1)  # (1, num_heads, t, head_dim)

            compacted_layers.append((C1_layer, beta_layer, C2_layer))

        # Average stored length over global layers.
        total_tensor_len = 0
        num_global_layers = num_layers - len(sliding_layer_indices)
        for layer_idx, layer_data in enumerate(compacted_layers):
            if layer_idx not in sliding_layer_indices:
                total_tensor_len += layer_data[0].shape[2]

        if num_global_layers > 0:
            avg_tensor_compacted_len = total_tensor_len / num_global_layers
        else:
            num_kept = len(keep_indices) if keep_indices is not None else 0
            avg_tensor_compacted_len = actual_target_size + num_kept
        all_stats['tensor_compacted_seq_len'] = avg_tensor_compacted_len

        # Effective lengths exclude -inf padding.
        total_global_heads = num_global_layers * num_heads
        if total_global_heads > 0:
            effective_article_tokens = total_effective_article_tokens / total_global_heads
        else:
            effective_article_tokens = 0
        num_kept = len(keep_indices) if keep_indices is not None else 0
        effective_compacted_seq_len = effective_article_tokens + num_kept

        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats['effective_article_tokens'] = effective_article_tokens
        all_stats['tensor_article_tokens'] = tensor_article_tokens
        all_stats['effective_compacted_seq_len'] = effective_compacted_seq_len
        all_stats['query_generation'] = query_stats

        return compacted_layers, all_stats
