"""
Global highest-attention-keys compaction.

Scores keys with generated queries and keeps the global top keys across all layers
and heads, so heads receive non-uniform budgets.
"""
import torch
import json
import datetime
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

from .base import FullCacheCompactionAlgorithm
from ..algorithms.highest_attention_keys import HighestAttentionKeysCompaction
from ..query_generation import QueryConfig


class GlobalHighestAttentionKeysCompaction(FullCacheCompactionAlgorithm):
    """Keeps the highest-scoring keys pooled over all global layers and heads, then refits beta and C2."""

    def __init__(
        self,
        score_method: str = 'max',
        beta_method: str = 'nnls',
        c2_method: str = 'lsq',
        nnls_iters: int = 0,
        nnls_lower_bound: Optional[float] = None,
        nnls_upper_bound: Optional[float] = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        c2_ridge_scale: str = 'spectral',
        config_name: Optional[str] = None,
        save_head_proportions: bool = False,
    ):
        """
        Parameters
        ----------
        score_method : str
            Key score over queries: 'max', 'mean' or 'rms' attention.
        beta_method : str
            'nnls' fits beta by NNLS, 'zero' sets beta = 0.
        c2_method : str
            'lsq' fits values by least squares, 'direct' keeps the selected values.
        nnls_iters : int
            Projected-gradient iterations for NNLS; 0 uses lstsq with clamping.
        nnls_lower_bound, nnls_upper_bound : float, optional
            Bounds on the NNLS solution (lower defaults to 1e-12).
        c2_ridge_lambda : float
            Ridge strength for the C2 fit.
        c2_solver : str
            'pinv', 'cholesky' or 'lstsq'.
        c2_ridge_scale : str
            Scaling of c2_ridge_lambda: 'spectral', 'frobenius' or 'fixed'.
        save_head_proportions : bool
            Write the resulting per-head key proportions to JSON.
        """
        self.score_method = score_method
        self.beta_method = beta_method
        self.c2_method = c2_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.config_name = config_name
        self.save_head_proportions = save_head_proportions

        # Used for key scoring and the per-head beta/C2 fits.
        self.algorithm = HighestAttentionKeysCompaction(
            nnls_iters=nnls_iters,
            nnls_lower_bound=nnls_lower_bound,
            nnls_upper_bound=nnls_upper_bound,
            score_method=score_method,
            c2_method=c2_method,
            beta_method=beta_method,
            c2_ridge_lambda=c2_ridge_lambda,
            c2_solver=c2_solver,
            c2_ridge_scale=c2_ridge_scale,
        )

    def name(self) -> str:
        if self.config_name:
            return self.config_name
        return f"nonuniform_score={self.score_method}_beta={self.beta_method}_c2={self.c2_method}"

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
    ) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...], Dict]:
        """
        Compact with global top-k key selection. See FullCacheCompactionAlgorithm.

        Only positions in `indices` compete for the global budget; the rest are kept.
        Heads are padded (beta = -inf) to the longest head of their layer.
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

        if batch_size != 1:
            raise NotImplementedError(
                "GlobalHighestAttentionKeysCompaction currently only supports batch_size=1"
            )

        from ..query_generation import QueryGenerator

        device = past_key_values[0][0].device
        dtype = past_key_values[0][0].dtype

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

            actual_target_size = sub_target_size
            print(f"Partial compaction: compacting positions {indices_list[0]}-{indices_list[-1]} "
                  f"({num_to_compact} tokens) to {actual_target_size} tokens")
            print(f"Keeping {num_to_keep} tokens unchanged")

            all_indices = torch.arange(seq_len)
            compact_mask = torch.zeros(seq_len, dtype=torch.bool)
            compact_mask[indices_list] = True
            keep_mask = ~compact_mask
            keep_indices = all_indices[keep_mask].tolist()
        else:
            indices_list = None
            keep_indices = None
            actual_target_size = target_size
            is_partial_compaction = False
            print(f"Full compaction: compacting all {seq_len} tokens to {actual_target_size} tokens")

        generator = QueryGenerator(
            model=model,
            tokenizer=tokenizer,
            config=query_config,
            device=device,
            dtype=dtype,
            vllm_model=vllm_model,
        )

        kv_for_queries = past_key_values_for_queries if past_key_values_for_queries is not None else past_key_values

        # (num_layers, num_kv_heads, n_queries, head_dim)
        base_queries, query_stats, _ = generator.generate_queries(
            formatted_context=formatted_context,
            past_key_values=kv_for_queries,
        )

        print(f"Generated {query_stats['final_n_queries_per_kv_head']} queries per KV head")
        for method, method_stats in query_stats.get('methods_used', {}).items():
            print(f"  {method}: {method_stats['n_queries_actual_per_kv_head']} queries ({method_stats['fraction']:.1%})")
        print("Train queries shape:", base_queries.shape)

        num_global_layers = num_layers - len(sliding_layer_indices)
        if sliding_layer_indices:
            print(f"Using {len(sliding_layer_indices)} sliding window layers (indices: {sorted(sliding_layer_indices)})")
            print(f"Only compacting {num_global_layers} global attention layers")

        # Split each global layer into the span to compact and the kept parts around it.
        K_all_layers = []
        V_all_layers = []
        K_keep_before_all = []
        V_keep_before_all = []
        K_keep_after_all = []
        V_keep_after_all = []
        global_layer_indices = []

        for layer_idx in range(num_layers):
            if layer_idx in sliding_layer_indices:
                continue

            global_layer_indices.append(layer_idx)
            K_layer = []
            V_layer = []
            K_keep_before_layer = []
            V_keep_before_layer = []
            K_keep_after_layer = []
            V_keep_after_layer = []

            for head_idx in range(num_heads):
                K_full = past_key_values[layer_idx][0][0, head_idx, :, :]  # (seq_len, head_dim)
                V_full = past_key_values[layer_idx][1][0, head_idx, :, :]  # (seq_len, head_dim)

                if is_partial_compaction:
                    K = K_full[indices_list, :]  # (num_to_compact, head_dim)
                    V = V_full[indices_list, :]  # (num_to_compact, head_dim)

                    compact_start = indices_list[0]
                    compact_end = indices_list[-1] + 1

                    keep_before_mask = [idx < compact_start for idx in keep_indices]
                    keep_after_mask = [idx >= compact_end for idx in keep_indices]

                    K_keep_before = K_full[[keep_indices[i] for i, m in enumerate(keep_before_mask) if m], :] if any(keep_before_mask) else K.new_zeros(0, head_dim)
                    V_keep_before = V_full[[keep_indices[i] for i, m in enumerate(keep_before_mask) if m], :] if any(keep_before_mask) else V.new_zeros(0, head_dim)
                    K_keep_after = K_full[[keep_indices[i] for i, m in enumerate(keep_after_mask) if m], :] if any(keep_after_mask) else K.new_zeros(0, head_dim)
                    V_keep_after = V_full[[keep_indices[i] for i, m in enumerate(keep_after_mask) if m], :] if any(keep_after_mask) else V.new_zeros(0, head_dim)
                else:
                    K = K_full
                    V = V_full
                    K_keep_before = K.new_zeros(0, head_dim)
                    V_keep_before = V.new_zeros(0, head_dim)
                    K_keep_after = K.new_zeros(0, head_dim)
                    V_keep_after = V.new_zeros(0, head_dim)

                K_layer.append(K)
                V_layer.append(V)
                K_keep_before_layer.append(K_keep_before)
                V_keep_before_layer.append(V_keep_before)
                K_keep_after_layer.append(K_keep_after)
                V_keep_after_layer.append(V_keep_after)

            K_all_layers.append(K_layer)
            V_all_layers.append(V_layer)
            K_keep_before_all.append(K_keep_before_layer)
            V_keep_before_all.append(V_keep_before_layer)
            K_keep_after_all.append(K_keep_after_layer)
            V_keep_after_all.append(V_keep_after_layer)

        print("Computing global attention scores...")
        queries_global_layers = base_queries[global_layer_indices, :, :, :]
        global_scores, layer_head_info, key_scores_dict = self._compute_global_attention_scores(
            K_all_layers, queries_global_layers, global_layer_indices
        )

        total_positions = len(global_scores)

        # Top-k over all (layer, head, position) entries of the global layers.
        num_to_select = actual_target_size * num_global_layers * num_heads

        selected_by_layer_head = self._select_global_top_k(
            global_scores, layer_head_info, num_to_select
        )

        if self.save_head_proportions:
            source_size = num_to_compact if is_partial_compaction else seq_len
            compaction_ratio = actual_target_size / source_size
            self._save_head_proportions_to_json(
                selected_by_layer_head, num_global_layers, num_heads, compaction_ratio
            )

        print(f"Selecting top {num_to_select} positions out of {total_positions} total "
              f"({100 * num_to_select / total_positions:.2f}%)")

        print(f"Selected positions across {len(selected_by_layer_head)} layer/head combinations")

        compacted_layers = []
        all_stats = {
            'per_layer_head_metrics': {},
            'is_partial_compaction': is_partial_compaction,
            'num_sliding_layers': len(sliding_layer_indices),
            'num_global_layers': num_global_layers,
            'global_selection_info': {
                'total_positions': total_positions,
                'num_selected': num_to_select,
                'selection_ratio': num_to_select / total_positions,
                'num_layer_head_combinations': len(selected_by_layer_head),
            }
        }

        if is_partial_compaction:
            all_stats['compaction_indices'] = {
                'start': indices_list[0],
                'end': indices_list[-1] + 1,
                'num_positions': len(indices_list),
            }

        layer_to_global_idx = {layer_idx: global_idx for global_idx, layer_idx in enumerate(global_layer_indices)}

        total_effective_article_tokens = 0

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

            global_idx = layer_to_global_idx[layer_idx]
            C1_heads = []
            beta_heads = []
            C2_heads = []

            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)

                K = K_all_layers[global_idx][head_idx]
                V = V_all_layers[global_idx][head_idx]
                K_keep_before = K_keep_before_all[global_idx][head_idx]
                V_keep_before = V_keep_before_all[global_idx][head_idx]
                K_keep_after = K_keep_after_all[global_idx][head_idx]
                V_keep_after = V_keep_after_all[global_idx][head_idx]

                queries_head = base_queries[layer_idx, head_idx, :, :]  # (n_queries, head_dim)

                if key in selected_by_layer_head:
                    selected_indices = selected_by_layer_head[key]
                    num_selected = len(selected_indices)

                    K_selected = K[selected_indices, :]  # (num_selected, head_dim)
                    V_selected = V[selected_indices, :]  # (num_selected, head_dim)

                    if self.beta_method == 'zero':
                        beta_selected32 = torch.zeros(num_selected, dtype=torch.float32, device=device)
                    else:  # 'nnls'
                        beta_selected32 = self._compute_beta_for_selected(
                            K, queries_head, selected_indices
                        )

                    beta_selected = beta_selected32.to(K.dtype)

                    if self.c2_method == 'direct':
                        C2_selected = V_selected.clone()
                    else:  # 'lsq'
                        C2_selected = self.algorithm._compute_C2(
                            K_selected, beta_selected, K, V, queries_head,
                            ridge_lambda=self.c2_ridge_lambda,
                            solver=self.c2_solver,
                            ridge_scale=self.c2_ridge_scale
                        )
                else:
                    num_selected = 0
                    K_selected = K.new_zeros(0, head_dim)
                    beta_selected = torch.zeros(0, dtype=K.dtype, device=device)
                    C2_selected = V.new_zeros(0, head_dim)
                    selected_indices = []

                # Layout: [kept before span, compacted span, kept after span].
                beta_keep_before = K.new_zeros(K_keep_before.shape[0])
                beta_keep_after = K.new_zeros(K_keep_after.shape[0])

                C1 = torch.cat([K_keep_before, K_selected, K_keep_after], dim=0)
                beta = torch.cat([beta_keep_before, beta_selected, beta_keep_after], dim=0)
                C2 = torch.cat([V_keep_before, C2_selected, V_keep_after], dim=0)

                total_effective_article_tokens += num_selected

                C1_heads.append(C1.unsqueeze(0).unsqueeze(0))  # (1, 1, seq_len, head_dim)
                beta_heads.append(beta.unsqueeze(0).unsqueeze(0))  # (1, 1, seq_len)
                C2_heads.append(C2.unsqueeze(0).unsqueeze(0))  # (1, 1, seq_len, head_dim)

                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'num_selected': num_selected,
                    'selection_ratio': num_selected / K.shape[0] if K.shape[0] > 0 else 0.0,
                    **({'selected_indices': [int(idx) for idx in selected_indices]} if verbose_logging else {}),
                    'selected_indices_stats': {
                        'count': len(selected_indices),
                        'min': int(min(selected_indices)) if len(selected_indices) > 0 else None,
                        'max': int(max(selected_indices)) if len(selected_indices) > 0 else None,
                    },
                    **({'beta_stats': {
                        'min': float(beta_selected.min().item()) if len(beta_selected) > 0 else None,
                        'max': float(beta_selected.max().item()) if len(beta_selected) > 0 else None,
                        'mean': float(beta_selected.mean().item()) if len(beta_selected) > 0 else None,
                        'std': float(beta_selected.std().item()) if len(beta_selected) > 1 else None,
                        'num_less_than_minus_7': int((beta_selected < -7).sum().item()) if len(beta_selected) > 0 else 0,
                    }} if verbose_logging else {})
                }

                all_stats['per_layer_head_metrics'][f'L{layer_idx}H{head_idx}'] = head_stats

            # Pad heads to the layer's longest head; layers may differ in length.
            layer_max_seq_len = max(h.shape[2] for h in C1_heads)
            for i in range(len(C1_heads)):
                curr_len = C1_heads[i].shape[2]
                if curr_len < layer_max_seq_len:
                    pad_len = layer_max_seq_len - curr_len
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

            C1_layer = torch.cat(C1_heads, dim=1)  # (1, num_heads, layer_seq_len, head_dim)
            beta_layer = torch.cat(beta_heads, dim=1)  # (1, num_heads, layer_seq_len)
            C2_layer = torch.cat(C2_heads, dim=1)  # (1, num_heads, layer_seq_len, head_dim)

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
        all_stats['query_generation'] = query_stats

        return tuple(compacted_layers), all_stats

    def _compute_global_attention_scores(
        self,
        K_all_layers: list,
        queries: torch.Tensor,
        global_layer_indices: list,
    ) -> Tuple[torch.Tensor, list, Dict[Tuple[int, int], torch.Tensor]]:
        """
        Score every key of every global layer and head.

        K_all_layers[g][h] is (T, d) and queries is (num_global_layers, num_heads, n, d).
        Returns the concatenated scores (total_positions,), the (layer, head, pos) of each
        entry, and a per-(layer, head) score dict. Layer indices are model layer indices.
        """
        num_global_layers = len(K_all_layers)
        num_heads = len(K_all_layers[0])

        all_scores = []
        layer_head_info = []
        key_scores_dict = {}

        for global_idx in range(num_global_layers):
            layer_idx = global_layer_indices[global_idx]
            for head_idx in range(num_heads):
                K = K_all_layers[global_idx][head_idx]  # (T, d)
                queries_head = queries[global_idx, head_idx, :, :]  # (n_queries, head_dim)

                n, d = queries_head.shape
                T = K.shape[0]
                inv_sqrt_d = (1.0 / d) ** 0.5

                # Matmul in model dtype, softmax in fp32.
                scores_raw = queries_head @ K.T  # (n, T)
                scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
                max_scores = scores32.max(dim=1, keepdim=True)[0]
                exp_scores = torch.exp(scores32 - max_scores)

                sum_exp = exp_scores.sum(dim=1, keepdim=True)
                attention_weights = exp_scores / sum_exp  # (n, T)

                if self.score_method == 'rms':
                    key_scores = torch.sqrt((attention_weights ** 2).mean(dim=0))  # (T,)
                elif self.score_method == 'max':
                    key_scores = attention_weights.max(dim=0)[0]  # (T,)
                else:  # 'mean'
                    key_scores = attention_weights.mean(dim=0)  # (T,)

                all_scores.append(key_scores)
                key_scores_dict[(layer_idx, head_idx)] = key_scores.cpu()

                for pos_idx in range(T):
                    layer_head_info.append((layer_idx, head_idx, pos_idx))

        global_scores = torch.cat(all_scores, dim=0)  # (total_positions,)

        return global_scores, layer_head_info, key_scores_dict

    def _save_head_proportions_to_json(
        self,
        selected_by_layer_head: Dict[Tuple[int, int], list],
        num_layers: int,
        num_heads: int,
        compaction_ratio: float
    ) -> None:
        """Write each head's share of the selected keys to logs/head_proportions/*.json."""
        total_selected = sum(len(idxs) for idxs in selected_by_layer_head.values())
        if total_selected == 0:
            print("Warning: total_selected is 0, cannot compute allocation proportions")
            return

        proportions = {}
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = (layer_idx, head_idx)
                n = len(selected_by_layer_head.get(key, []))
                proportions[f"L{layer_idx}H{head_idx}"] = n / total_selected

        output_dir = Path("logs/head_proportions")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_suffix = (
            self.config_name.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            if self.config_name else "nonuniform"
        )
        filename = f"head_proportions_t{compaction_ratio:.2f}_{config_suffix}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(proportions, f, indent=2)

        print(f"Head proportions saved to: {filepath}")

    def _compute_beta_for_selected(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        selected_indices: list,
    ) -> torch.Tensor:
        """
        Fit beta = log B for the selected keys.

        With E = exp(Q K^T / sqrt(d)), solves min ||E[:, S] B - E 1||^2 over B >= 0, so the
        selected keys reproduce each query's softmax normalizer over all of K.
        """
        device = K.device
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        scores_raw = queries @ K.T  # (n, T)
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)  # (n, T)

        exp_scores_selected = exp_scores[:, selected_indices]  # (n, num_selected)

        target = exp_scores.sum(dim=1)  # (n,)

        B = self.algorithm._nnls_pg(
            exp_scores_selected, target,
            self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound
        )
        beta = torch.log(B)

        return beta

    def _select_global_top_k(
        self,
        global_scores: torch.Tensor,
        layer_head_info: list,
        num_to_select: int,
    ) -> Dict[Tuple[int, int], list]:
        """Top-k over all (layer, head, position) entries, returned as sorted positions per (layer, head)."""
        k = min(num_to_select, len(global_scores))
        _, top_global_indices = torch.topk(global_scores, k, largest=True)

        selected_by_layer_head = {}
        for global_idx in top_global_indices.cpu().tolist():
            layer_idx, head_idx, pos_idx = layer_head_info[global_idx]
            key = (layer_idx, head_idx)
            if key not in selected_by_layer_head:
                selected_by_layer_head[key] = []
            selected_by_layer_head[key].append(pos_idx)

        for key in selected_by_layer_head:
            selected_by_layer_head[key] = sorted(selected_by_layer_head[key])

        return selected_by_layer_head
