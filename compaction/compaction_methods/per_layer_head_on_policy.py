"""
Per-layer-head compaction with on-policy queries.

Layers are compacted in order. Layer 0 uses the original queries; each later layer
uses queries re-extracted with the already compacted earlier layers in place.
"""
import torch
import time
from typing import Tuple, Dict, Optional, Type, Any

from .base import FullCacheCompactionAlgorithm, load_budgets_from_json, apply_max_ratio_cap
from ..algorithms.base import CompactionAlgorithm
from ..query_generation import QueryConfig


class PerLayerHeadOnPolicyCompaction(FullCacheCompactionAlgorithm):
    """Per-layer-head compaction where layer l > 0 is calibrated with queries from the partly compacted cache."""

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
        return f"per_layer_head_on_policy_{self._name_instance.name()}"

    def _get_sliding_window(self, model) -> Optional[int]:
        return getattr(model.config, 'sliding_window', None)

    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
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
        query_cache_article_boundaries: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact layer by layer with on-policy queries. See FullCacheCompactionAlgorithm.

        Additional parameters
        ---------------------
        verbose_logging : bool
            Store selected indices and beta statistics in the stats.
        past_key_values_for_queries : optional
            Cache used only for query generation (chunked compaction).
        query_cache_article_boundaries : (int, int), optional
            (start, end) of the article span inside past_key_values_for_queries,
            used to split it into prefix, article and suffix.
        """
        num_layers = len(past_key_values)
        # Sliding-window layers may hold fewer tokens, so read a full-attention layer.
        sliding_layer_indices = sliding_layer_indices or set()
        ref_layer_idx = 0
        for i in range(num_layers):
            if i not in sliding_layer_indices:
                ref_layer_idx = i
                break
        batch_size, num_heads, seq_len, head_dim = past_key_values[ref_layer_idx][0].shape

        num_global_layers = num_layers - len(sliding_layer_indices)
        if sliding_layer_indices:
            print(f"Using {len(sliding_layer_indices)} sliding window layers (indices: {sorted(sliding_layer_indices)})")
            print(f"Only compacting {num_global_layers} global attention layers")

        if batch_size != 1:
            raise NotImplementedError(
                "PerLayerHeadOnPolicyCompaction currently only supports batch_size=1"
            )

        from ..query_generation import QueryGenerator

        device = past_key_values[ref_layer_idx][0].device
        dtype = past_key_values[ref_layer_idx][0].dtype

        print("Generating original queries for all layers...")
        generator = QueryGenerator(
            model=model,
            tokenizer=tokenizer,
            config=query_config,
            device=device,
            dtype=dtype,
            vllm_model=vllm_model,
        )

        # The generated sequences are kept for on-policy re-extraction.
        kv_for_queries = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        original_queries, query_stats, sequences = self._generate_queries_with_sequences(
            generator=generator,
            formatted_context=formatted_context,
            past_key_values=kv_for_queries,
            indices=indices,
        )

        has_self_study = 'self_study' in query_stats.get("methods_used", {})
        self_study_query_range = None
        if has_self_study:
            # Pop the subsample indices so they do not bloat the stats.
            ss = query_stats.get("methods_used", {}).get("self_study", {})
            ss_stats = ss.get("stats", {})
            token_idx_list = ss_stats.pop("subsample_indices", None)
            kv_idx_list = ss.pop("subsample_indices_kv", None)
            token_idx = (torch.tensor(token_idx_list, device=device, dtype=torch.long)
                        if token_idx_list is not None else None)
            kv_idx = (torch.tensor(kv_idx_list, device=device, dtype=torch.long)
                    if kv_idx_list is not None else None)

            # Position of the self-study queries in the concatenated query tensor.
            self_study_query_range = ss.get("query_range", None)
            if self_study_query_range is not None:
                print(f"Self-study queries occupy range {self_study_query_range[0]}-{self_study_query_range[1]} "
                      f"in final queries (total: {query_stats['final_n_queries_per_kv_head']})")
        else:
            token_idx = None
            kv_idx = None

        print(f"Generated {query_stats['final_n_queries_per_kv_head']} original queries per KV head")
        print("Original queries shape:", original_queries.shape)

        is_partial_compaction = indices is not None
        if is_partial_compaction:
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
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size

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
            print(f"Per-head proportions sum to: {total_proportions:.4f}")
            print(f"Per-head budget distribution: total={total_budget}, "
                  f"min={min(per_head_budgets.values()) if per_head_budgets else 0}, "
                  f"max={max(per_head_budgets.values()) if per_head_budgets else 0}, "
                  f"mean={total_budget / (num_global_layers * num_heads):.1f}")

        compacted_layers = []
        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'on_policy_query_extraction_time': 0.0,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_global_layers,
        }
        total_effective_article_tokens = 0

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }
            all_stats['keep_indices'] = {
                'num_positions': len(keep_indices),
            }

        for layer_idx in range(num_layers):
            if layer_idx in sliding_layer_indices:
                print(f"Layer {layer_idx+1}/{num_layers}: sliding window (keeping original KV)")
                # Empty placeholder; sliding layers keep their own cache.
                placeholder_C1 = past_key_values[ref_layer_idx][0].new_zeros(1, num_heads, 0, head_dim)
                placeholder_beta = past_key_values[ref_layer_idx][0].new_zeros(1, num_heads, 0)
                placeholder_C2 = past_key_values[ref_layer_idx][1].new_zeros(1, num_heads, 0, head_dim)
                compacted_layers.append((placeholder_C1, placeholder_beta, placeholder_C2))
                continue

            print(f"Compacting layer {layer_idx+1}/{num_layers}")
            keys_layer = past_key_values[layer_idx][0]  # (1, num_heads, seq_len, head_dim)
            values_layer = past_key_values[layer_idx][1]

            C1_heads = []
            beta_heads = []
            C2_heads = []

            num_compacted_global_layers = len([l for l in range(layer_idx) if l not in sliding_layer_indices])
            if num_compacted_global_layers > 0:
                compacted_cache_prefix = tuple(compacted_layers[:layer_idx])

                # Each extractor re-extracts the queries of one method over its range of the
                # query tensor; queries of the other methods stay original.
                on_policy_extractors = []

                if has_self_study and sequences and self_study_query_range is not None:
                    ss_start, ss_end = self_study_query_range
                    # Bind current values, not the loop variables.
                    def make_self_study_extractor(cache_prefix, lid, kv_for_q, article_bounds):
                        return lambda: self._extract_on_policy_queries_for_layer(
                            model=model,
                            tokenizer=tokenizer,
                            sequences=sequences,
                            formatted_context=formatted_context,
                            compacted_cache_prefix=cache_prefix,
                            original_seq_len=seq_len,
                            layer_idx=lid,
                            device=device,
                            dtype=dtype,
                            token_subsample_idx=token_idx,
                            kv_subsample_idx=kv_idx,
                            sliding_layer_indices=sliding_layer_indices,
                            sliding_window=self._get_sliding_window(model),
                            past_key_values=past_key_values,
                            past_key_values_for_queries=kv_for_q,
                            query_cache_article_boundaries=article_bounds,
                        )
                    on_policy_extractors.append({
                        'method': 'self_study',
                        'range': (ss_start, ss_end),
                        'extract_fn': make_self_study_extractor(compacted_cache_prefix, layer_idx, kv_for_queries, query_cache_article_boundaries)
                    })

                # context_prefill queries are not re-extracted.

                if layer_idx == 1:
                    if not on_policy_extractors:
                        print("  No on-policy methods - using original queries")
                    else:
                        for extractor in on_policy_extractors:
                            method = extractor['method']
                            start, end = extractor['range']
                            print(f"  Extracting on-policy queries from {method} (range {start}:{end})")

                        total_queries = original_queries.shape[2]
                        all_ranges = sorted([ext['range'] for ext in on_policy_extractors])

                        if all_ranges[0][0] > 0:
                            print(f"  Keeping original queries for non-on-policy methods (range 0:{all_ranges[0][0]})")
                        for i in range(len(all_ranges) - 1):
                            gap_start = all_ranges[i][1]
                            gap_end = all_ranges[i + 1][0]
                            if gap_end > gap_start:
                                print(f"  Keeping original queries for non-on-policy methods (range {gap_start}:{gap_end})")
                        if all_ranges[-1][1] < total_queries:
                            print(f"  Keeping original queries for non-on-policy methods (range {all_ranges[-1][1]}:{total_queries})")

                if not on_policy_extractors:
                    queries_on_policy_layer = original_queries[layer_idx]
                else:
                    on_policy_queries = {}
                    for extractor in on_policy_extractors:
                        method = extractor['method']
                        extraction_start = time.time()
                        on_policy_queries[method] = extractor['extract_fn']()
                        all_stats['on_policy_query_extraction_time'] += time.time() - extraction_start

                    sorted_extractors = sorted(on_policy_extractors, key=lambda x: x['range'][0])

                    parts = []
                    current_pos = 0
                    total_queries = original_queries.shape[2]

                    for extractor in sorted_extractors:
                        method = extractor['method']
                        start, end = extractor['range']

                        if start > current_pos:
                            parts.append(original_queries[layer_idx, :, current_pos:start, :])

                        parts.append(on_policy_queries[method])
                        current_pos = end

                    if current_pos < total_queries:
                        parts.append(original_queries[layer_idx, :, current_pos:, :])

                    queries_on_policy_layer = torch.cat(parts, dim=1)  # (num_heads, total_queries, head_dim)

            for head_idx in range(num_heads):
                K_full = keys_layer[0, head_idx, :, :]  # (seq_len, head_dim)
                V_full = values_layer[0, head_idx, :, :]

                if is_partial_compaction:
                    K = K_full[indices_list, :]  # (num_to_compact, head_dim)
                    V = V_full[indices_list, :]  # (num_to_compact, head_dim)

                    K_keep = K_full[keep_indices, :]  # (num_to_keep, head_dim)
                    V_keep = V_full[keep_indices, :]  # (num_to_keep, head_dim)
                else:
                    K = K_full
                    V = V_full

                queries_original = original_queries[layer_idx, head_idx, :, :]  # (n_queries, head_dim)

                # The first global layer has no compacted predecessor, so its queries are original.
                if num_compacted_global_layers == 0:
                    queries_for_compaction = queries_original
                else:
                    queries_for_compaction = queries_on_policy_layer[head_idx]  # (total_q, head_dim)
                    assert queries_for_compaction.shape == queries_original.shape, (
                        f"shape mismatch L{layer_idx} H{head_idx}: "
                        f"on_policy={tuple(queries_for_compaction.shape)} original={tuple(queries_original.shape)}"
                    )

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
                    if hasattr(algorithm, 'set_layer_head'):
                        algorithm.set_layer_head(layer_idx, head_idx)
                    C1_compact, beta_compact, C2_compact, selected_indices = algorithm.compute_compacted_cache(
                        K, V, queries_for_compaction, head_target_size
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

                    beta_keep_before = K.new_zeros(K_keep_before.shape[0])
                    beta_keep_after = K.new_zeros(K_keep_after.shape[0])

                    C1 = torch.cat([K_keep_before, C1_compact, K_keep_after], dim=0)
                    beta = torch.cat([beta_keep_before, beta_compact, beta_keep_after], dim=0)
                    C2 = torch.cat([V_keep_before, C2_compact, V_keep_after], dim=0)
                else:
                    C1 = C1_compact
                    beta = beta_compact
                    C2 = C2_compact

                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))

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

            C1_layer = torch.cat(C1_heads, dim=1)  # (1, num_heads, target_size, head_dim)
            beta_layer = torch.cat(beta_heads, dim=1)
            C2_layer = torch.cat(C2_heads, dim=1)

            compacted_layers.append((C1_layer, beta_layer, C2_layer))

        # Average stored length over global layers.
        total_tensor_len = 0
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
        effective_article_tokens = total_effective_article_tokens / total_global_heads if total_global_heads > 0 else 0
        num_kept = len(keep_indices) if keep_indices is not None else 0
        effective_compacted_seq_len = effective_article_tokens + num_kept

        tensor_article_tokens = avg_tensor_compacted_len - num_kept

        all_stats['effective_article_tokens'] = effective_article_tokens
        all_stats['tensor_article_tokens'] = tensor_article_tokens
        all_stats['effective_compacted_seq_len'] = effective_compacted_seq_len

        # Counted as query-generation time so it is excluded from compaction time.
        query_stats['on_policy_query_extraction_time'] = all_stats['on_policy_query_extraction_time']
        if 'query_generation_time' in query_stats:
            query_stats['query_generation_time'] += all_stats['on_policy_query_extraction_time']
        else:
            query_stats['query_generation_time'] = all_stats['on_policy_query_extraction_time']
        all_stats['query_generation'] = query_stats

        return tuple(compacted_layers), all_stats

    def _generate_queries_with_sequences(
        self,
        generator,
        formatted_context: str,
        past_key_values: Tuple,
        indices: Optional[range] = None,
    ) -> Tuple[torch.Tensor, Dict, list]:
        """Generate queries (num_layers, num_kv_heads, n_queries, head_dim) plus the self-study sequences."""
        queries, stats, sequences = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=past_key_values,
            indices=indices,
            return_sequences=True,
        )

        if sequences is None:
            sequences = []

        return queries, stats, sequences

    def _extract_on_policy_queries_for_layer(
        self,
        model,
        tokenizer,
        sequences: list,
        formatted_context: str,
        compacted_cache_prefix: Tuple,
        original_seq_len: int,
        layer_idx: int,
        device,
        dtype,
        token_subsample_idx: Optional[torch.Tensor] = None,
        kv_subsample_idx: Optional[torch.Tensor] = None,
        sliding_layer_indices: Optional[set] = None,
        sliding_window: Optional[int] = None,
        past_key_values: Optional[Tuple] = None,
        past_key_values_for_queries: Optional[Tuple] = None,
        query_cache_article_boundaries: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Re-extract the self-study queries of `layer_idx` with layers < layer_idx compacted.

        The same token and KV subsampling as in the original extraction is applied, so the
        result (num_kv_heads, total_q, head_dim) lines up with the original queries.
        """
        from models.cache import CompactedPrefixCache
        from ..query_generation.self_study import SelfStudyQueryGenerator

        head_dim = getattr(model.config, "head_dim",
                        model.config.hidden_size // model.config.num_attention_heads)

        num_layers = model.config.num_hidden_layers
        num_attn_heads = model.config.num_attention_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_attn_heads)
        num_kv_groups = num_attn_heads // num_kv_heads

        sliding_indices = sliding_layer_indices or set()

        base_kv = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        is_compacted_prefix_cache = base_kv is not None and isinstance(base_kv, CompactedPrefixCache)

        full_cache_tuple = []

        if is_compacted_prefix_cache:
            # Chunked compaction: base_kv holds [prefix, chunk, suffix]; swap in the compacted chunk.
            for layer_i in range(num_layers):
                if layer_i < len(compacted_cache_prefix) and layer_i not in sliding_indices:
                    layer = base_kv.layers[layer_i]
                    full_keys = layer.keys
                    full_values = layer.values
                    full_beta = layer.beta
                    full_seq_len = full_keys.shape[2]

                    if query_cache_article_boundaries is not None:
                        article_start, article_end = query_cache_article_boundaries
                        prefix_len = article_start
                        suffix_len = full_seq_len - article_end
                    else:
                        prefix_len = 0
                        suffix_len = 0

                    prefix_keys = full_keys[:, :, :prefix_len, :]
                    prefix_values = full_values[:, :, :prefix_len, :]
                    prefix_beta = full_beta[:, :, :prefix_len]

                    if suffix_len > 0:
                        suffix_keys = full_keys[:, :, -suffix_len:, :]
                        suffix_values = full_values[:, :, -suffix_len:, :]
                        suffix_beta = full_beta[:, :, -suffix_len:]
                    else:
                        suffix_keys = full_keys[:, :, 0:0, :]
                        suffix_values = full_values[:, :, 0:0, :]
                        suffix_beta = full_beta[:, :, 0:0]

                    compacted_keys = compacted_cache_prefix[layer_i][0]
                    compacted_beta = compacted_cache_prefix[layer_i][1]
                    compacted_values = compacted_cache_prefix[layer_i][2]

                    combined_keys = torch.cat([prefix_keys, compacted_keys, suffix_keys], dim=2)
                    combined_beta = torch.cat([prefix_beta, compacted_beta, suffix_beta], dim=2)
                    combined_values = torch.cat([prefix_values, compacted_values, suffix_values], dim=2)
                    full_cache_tuple.append((combined_keys, combined_beta, combined_values))

                elif layer_i < len(base_kv.layers):
                    # Sliding layer, or a global layer not compacted yet.
                    layer = base_kv.layers[layer_i]
                    if layer_i in sliding_indices:
                        keys = layer.keys
                        values = layer.values
                        beta = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2],
                                          device=device, dtype=dtype)
                        full_cache_tuple.append((keys, beta, values))
                    else:
                        full_cache_tuple.append((layer.keys, layer.beta, layer.values))
                else:
                    empty_keys = torch.zeros((1, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
                    empty_beta = torch.full((1, num_kv_heads, 0), float('-inf'), device=device, dtype=dtype)
                    empty_values = torch.zeros((1, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
                    full_cache_tuple.append((empty_keys, empty_beta, empty_values))
        else:
            # Compacted layers from compacted_cache_prefix, sliding layers from past_key_values,
            # and empty placeholders for global layers not compacted yet.
            for layer_i in range(num_layers):
                if layer_i < len(compacted_cache_prefix):
                    if layer_i in sliding_indices and past_key_values is not None:
                        orig_keys = past_key_values[layer_i][0]
                        orig_values = past_key_values[layer_i][1]
                        orig_beta = torch.zeros(
                            (orig_keys.shape[0], orig_keys.shape[1], orig_keys.shape[2]),
                            device=device, dtype=dtype
                        )
                        full_cache_tuple.append((orig_keys, orig_beta, orig_values))
                    else:
                        full_cache_tuple.append(compacted_cache_prefix[layer_i])
                else:
                    if layer_i in sliding_indices and past_key_values is not None:
                        orig_keys = past_key_values[layer_i][0]
                        orig_values = past_key_values[layer_i][1]
                        orig_beta = torch.zeros(
                            (orig_keys.shape[0], orig_keys.shape[1], orig_keys.shape[2]),
                            device=device, dtype=dtype
                        )
                        full_cache_tuple.append((orig_keys, orig_beta, orig_values))
                    else:
                        empty_keys = torch.zeros((1, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
                        empty_beta = torch.full((1, num_kv_heads, 0), float('-inf'), device=device, dtype=dtype)
                        empty_values = torch.zeros((1, num_kv_heads, 0, head_dim), device=device, dtype=dtype)
                        full_cache_tuple.append((empty_keys, empty_beta, empty_values))
        
        if is_compacted_prefix_cache:
            full_original_seq_len = base_kv.rope_base() + base_kv.get_seq_length()
        else:
            full_original_seq_len = original_seq_len

        cache = CompactedPrefixCache(
            compacted_cache=tuple(full_cache_tuple),
            original_seq_len=full_original_seq_len,
            sliding_layer_indices=sliding_indices,
            sliding_window=sliding_window,
        )

        temp_generator = SelfStudyQueryGenerator(
            model=model,
            tokenizer=tokenizer,
            config=None,
            device=device,
            dtype=dtype,
            verbose=False,
            vllm_model=None,
        )

        per_seq_q_layer = []

        for seq_info in sequences:
            # Stored token ids (answer prompt + answer) guarantee the same tokenization as before.
            input_ids = seq_info["suffix_token_ids"].to(device)  # (1, suffix_len)

            q_all_layers = temp_generator._extract_query_vectors_from_prefill(
                input_ids=input_ids,
                past_key_values=cache,
                head_dim=head_dim,
                start_token_idx=0,
                max_layer=layer_idx,
            )
            if q_all_layers is None:
                continue

            q_layer = q_all_layers[layer_idx]  # (num_attn_heads, n_tokens, head_dim)

            per_seq_q_layer.append(q_layer)

        if not per_seq_q_layer:
            return torch.zeros((num_kv_heads, 0, head_dim), device=device, dtype=dtype)

        # Same token order as the original extraction. (num_attn_heads, total_tokens, head_dim)
        q_layer_cat = torch.cat(per_seq_q_layer, dim=1)

        if token_subsample_idx is not None:
            q_layer_cat = q_layer_cat[:, token_subsample_idx, :]

        # KV grouping as in QueryGenerator._attention_to_kv.
        q_grouped = q_layer_cat.view(num_kv_heads, num_kv_groups, q_layer_cat.shape[1], head_dim)

        # (kv, group, token, dim) -> (kv, group * token, dim)
        q_kv_cat = q_grouped.reshape(num_kv_heads, -1, head_dim)

        if kv_subsample_idx is not None:
            q_kv_cat = q_kv_cat[:, kv_subsample_idx, :]

        return q_kv_cat
