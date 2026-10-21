"""Dispatcher that mixes query generation methods and maps queries to KV heads."""

import torch
import time
from typing import Optional, Tuple, Dict, Any
from transformers import AutoTokenizer
from models.cache import CompactedPrefixCache
from transformers.cache_utils import DynamicCache

from .config import QueryConfig


class QueryGenerator:
    """Generate calibration queries with the methods and fractions in a QueryConfig."""

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        config: QueryConfig,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        vllm_model: Optional[Any] = None,
    ):
        """vllm_model is an optional pre-initialized vLLM engine for self-study."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype
        self.vllm_model = vllm_model

        self.method_generators = {}
        self._initialize_generators()

    def _validate_past_key_values_for_queries(self, past_key_values: Optional[Tuple]) -> Optional[Tuple]:
        """Accept CompactedPrefixCache, DynamicCache or per-layer (keys, values) tuples."""
        if past_key_values is None:
            return None

        if isinstance(past_key_values, (CompactedPrefixCache, DynamicCache)):
            return past_key_values

        if isinstance(past_key_values, (tuple, list)):
            for layer_idx, layer in enumerate(past_key_values):
                if not isinstance(layer, (tuple, list)):
                    raise TypeError(f"past_key_values layer {layer_idx} must be a tuple/list, got {type(layer)}")
                if len(layer) == 2:
                    continue
                if len(layer) == 3:
                    raise ValueError(
                        "past_key_values with attention bias (keys, bias, values) is not supported for query generation; "
                        f"found 3-tuple at layer {layer_idx}. Pass a CompactedPrefixCache or (keys, values) tuples."
                    )
                raise ValueError(
                    f"past_key_values layer {layer_idx} must have length 2 (keys, values); got length {len(layer)}"
                )
            return past_key_values

        raise TypeError(
            f"Unsupported past_key_values type {type(past_key_values)}; expected CompactedPrefixCache, DynamicCache, or tuple-of-(keys, values)."
        )

    def _initialize_generators(self):
        """Initialize method-specific generators based on config."""
        for method_config in self.config.method_configs:
            method = method_config.method

            if method == 'self_study':
                from .self_study import SelfStudyQueryGenerator
                self.method_generators[method] = SelfStudyQueryGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    config=method_config.config,
                    device=self.device,
                    dtype=self.dtype,
                    verbose=self.config.verbose,
                    vllm_model=self.vllm_model,
                )

            elif method == 'context_prefill':
                from .context_prefill import ContextPrefillQueryGenerator
                self.method_generators[method] = ContextPrefillQueryGenerator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    config=method_config.config,
                    device=self.device,
                    dtype=self.dtype,
                )

            else:
                raise ValueError(f"Unknown query generation method: {method}")
            
    @staticmethod
    def _attention_to_kv(
        queries_attention_heads: torch.Tensor,
        num_kv_heads: int,
        heads_per_kv_head: int,
    ) -> torch.Tensor:
        """Pool the queries of each GQA group: (L, H, n, D) -> (L, H_kv, group * n, D)."""
        num_layers, num_attention_heads, n_queries_per_head, head_dim = queries_attention_heads.shape

        if num_kv_heads == num_attention_heads:
            return queries_attention_heads

        queries_reshaped = queries_attention_heads.view(
            num_layers,
            num_kv_heads,
            heads_per_kv_head,
            n_queries_per_head,
            head_dim,
        )
        queries_kv = queries_reshaped.reshape(
            num_layers,
            num_kv_heads,
            heads_per_kv_head * n_queries_per_head,
            head_dim,
        )
        return queries_kv

    def generate_queries(
        self,
        formatted_context: Optional[str] = None,
        past_key_values: Optional[Tuple] = None,
        head_dim: Optional[int] = None,
        indices: Optional[range] = None,
        return_sequences: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Optional[list]]:
        """
        Generate queries with all configured methods.

        Parameters
        ----------
        past_key_values : optional
            Precomputed cache of the context (see _validate_past_key_values_for_queries).
        indices : range, optional
            Positions to compact (the article span).
        return_sequences : bool
            Also return the self-study sequences, for on-policy re-extraction.

        Returns
        -------
        queries : Tensor, shape (num_layers, num_kv_heads, n, head_dim)
        stats : dict
        sequences : list or None
        """
        query_gen_start_time = time.time()

        if head_dim is None:
            head_dim = getattr(
                self.model.config,
                "head_dim",
                self.model.config.hidden_size // self.model.config.num_attention_heads,
            )

        num_layers = self.model.config.num_hidden_layers
        num_attention_heads = self.model.config.num_attention_heads
        num_kv_heads = getattr(
            self.model.config,
            "num_key_value_heads",
            num_attention_heads,
        )
        heads_per_kv_head = max(1, num_attention_heads // num_kv_heads)

        n_train_per_kv_head = self.config.max_query_vectors_per_kv_head

        stats: Dict[str, Any] = {
            "n_total_requested_per_kv_head": n_train_per_kv_head,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "heads_per_kv_head": heads_per_kv_head,
            "methods_used": {},
        }
        normalized_past_key_values = self._validate_past_key_values_for_queries(past_key_values)

        all_queries_by_method_kv = []
        sequences = None

        for method_config in self.config.method_configs:
            method = method_config.method
            fraction = method_config.fraction

            if fraction <= 0.0:
                continue

            n_queries_per_kv_head_for_method = int(n_train_per_kv_head * fraction)
            if n_queries_per_kv_head_for_method <= 0:
                continue

            n_queries_per_attention_head = max(
                1, (n_queries_per_kv_head_for_method + heads_per_kv_head - 1) // heads_per_kv_head
            )

            generator = self.method_generators[method]

            if self.config.verbose:
                print(
                    f"Generating ~{n_queries_per_kv_head_for_method} queries per KV head "
                    f"using '{method}' method..."
                )

            if method == "self_study":
                method_queries_attention, method_stats, method_sequences = generator.generate_queries(
                    n_queries_per_attention_head=n_queries_per_attention_head,
                    formatted_context=formatted_context,
                    past_key_values=normalized_past_key_values,
                    return_sequences=return_sequences,
                    indices=indices,
                )
                method_queries_kv = self._attention_to_kv(
                    method_queries_attention,
                    num_kv_heads=num_kv_heads,
                    heads_per_kv_head=heads_per_kv_head,
                )
                if return_sequences and method_sequences is not None:
                    sequences = method_sequences

            elif method == "context_prefill":
                method_queries_attention, method_stats = generator.generate_queries(
                    n_queries_per_attention_head=n_queries_per_attention_head,
                    formatted_context=formatted_context,
                    past_key_values=normalized_past_key_values,
                    indices=indices,
                )
                method_queries_kv = self._attention_to_kv(
                    method_queries_attention,
                    num_kv_heads=num_kv_heads,
                    heads_per_kv_head=heads_per_kv_head,
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            if method_queries_kv.shape[0] != num_layers or method_queries_kv.shape[1] != num_kv_heads:
                raise ValueError(
                    f"Method '{method}' produced KV queries of shape "
                    f"{tuple(method_queries_kv.shape)}, expected "
                    f"(num_layers={num_layers}, num_kv_heads={num_kv_heads}, *, head_dim={head_dim})"
                )

            all_queries_by_method_kv.append(method_queries_kv)

            stats["methods_used"][method] = {
                "fraction": fraction,
                "n_queries_requested_per_kv_head": n_queries_per_kv_head_for_method,
                "n_queries_actual_per_kv_head": method_queries_kv.shape[2],
                "stats": method_stats,
            }

        if not all_queries_by_method_kv:
            raise ValueError("No queries generated from any method")

        # If a method returned fewer queries than requested, shrink all methods
        # by the same factor so the fractions still hold.
        method_data = []
        queries_idx = 0
        for method_config in self.config.method_configs:
            if method_config.fraction <= 0.0:
                continue
            method_data.append((method_config, all_queries_by_method_kv[queries_idx]))
            queries_idx += 1

        scaling_factors = []
        for method_config, method_queries_kv in method_data:
            n_requested = int(n_train_per_kv_head * method_config.fraction)
            n_actual = method_queries_kv.shape[2]
            if n_requested > 0:
                scaling_factors.append(n_actual / n_requested)

        min_scaling = min(scaling_factors) if scaling_factors else 1.0

        final_queries_list = []
        query_offset = 0
        for method_config, method_queries_kv in method_data:
            method = method_config.method
            fraction = method_config.fraction
            n_actual = method_queries_kv.shape[2]

            n_target = int(n_train_per_kv_head * fraction * min_scaling)

            if n_target > n_actual:
                n_target = n_actual

            if n_target < n_actual:
                indices = torch.randperm(n_actual, device=method_queries_kv.device)[:n_target]
                indices = indices.sort()[0]
                method_queries_kv = method_queries_kv[:, :, indices, :]

                if method == "self_study" and return_sequences:
                    stats["methods_used"][method]["subsample_indices_kv"] = indices.detach().cpu().tolist()

                if self.config.verbose:
                    print(
                        f"Subsampled '{method}' from {n_actual} to {n_target} queries per KV head "
                        f"to maintain fraction {fraction:.2f}"
                    )
            else:
                if method == "self_study" and return_sequences:
                    stats["methods_used"][method]["subsample_indices_kv"] = None

            final_queries_list.append(method_queries_kv)

            n_queries_for_method = method_queries_kv.shape[2]
            stats["methods_used"][method]["n_queries_final_per_kv_head"] = n_queries_for_method
            stats["methods_used"][method]["query_range"] = (query_offset, query_offset + n_queries_for_method)
            query_offset += n_queries_for_method

        final_queries = torch.cat(final_queries_list, dim=2)

        stats["final_n_queries_per_kv_head"] = final_queries.shape[2]
        stats["scaling_factor_applied"] = min_scaling

        query_gen_end_time = time.time()
        stats["query_generation_time"] = query_gen_end_time - query_gen_start_time

        return final_queries, stats, sequences
