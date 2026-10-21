"""Factory that builds a full-cache compaction method from its config name and kwargs."""
from typing import Dict, Any, Optional

from .base import FullCacheCompactionAlgorithm
from .per_layer_head import PerLayerHeadCompaction
from .per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from .global_highest_attention_keys import GlobalHighestAttentionKeysCompaction
from .summarize import SummarizeCompaction
from ..algorithms import ALGORITHM_REGISTRY


def get_compaction_method(
    method_name: str,
    method_kwargs: Optional[Dict[str, Any]] = None
) -> FullCacheCompactionAlgorithm:
    """
    Build a full-cache compaction method by name.

    Parameters
    ----------
    method_name : str
        Config name: 'original', 'global_highest_attention_keys', 'summarize', or any key
        of algorithms.ALGORITHM_REGISTRY (run per layer and head).
    method_kwargs : dict, optional
        Constructor kwargs. 'algorithm' overrides the base algorithm, 'chunking' wraps the
        method in ChunkedCompaction ('raw_chunks' lists chunks it leaves uncompacted),
        'on_policy' selects the on-policy per-layer-head wrapper.
    """
    method_kwargs = method_kwargs or {}

    if method_name == 'original':
        return OriginalCacheMethod()

    # Checked first so that any method can be wrapped with chunking.
    chunking = method_kwargs.get('chunking', None)

    if chunking is not None and chunking.lower() != 'none':
        # Lazy imports avoid a circular dependency.
        from .chunked import ChunkedCompaction
        from ..chunking import get_chunking_strategy

        chunking_strategy = get_chunking_strategy(
            chunking,
            chunk_size=method_kwargs.get('chunk_size', 4096)
        )

        # use_kv_based is a harness flag; chunked compaction is always KV-based.
        excluded_keys = ('chunking', 'chunk_size', 'use_kv_based', 'raw_chunks')
        inner_kwargs = {k: v for k, v in method_kwargs.items() if k not in excluded_keys}

        def inner_method_factory():
            return get_compaction_method(method_name, inner_kwargs)

        return ChunkedCompaction(
            inner_compaction_method=inner_method_factory,
            chunking_strategy=chunking_strategy,
            config_name=method_name,
            raw_chunks=method_kwargs.get('raw_chunks'),
        )

    base_algorithm = method_kwargs.get('algorithm', method_name)

    if base_algorithm == 'global_highest_attention_keys':
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k != 'algorithm'}
        return GlobalHighestAttentionKeysCompaction(config_name=method_name, **algorithm_kwargs)

    # The evaluation harness injects head-budget kwargs into every method;
    # methods without a per-head cache must drop them.
    budget_keys = ('precomputed_budget_path', 'max_ratio_per_head')

    if base_algorithm == 'summarize':
        excluded_keys = ('algorithm',) + budget_keys
        algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k not in excluded_keys}
        return SummarizeCompaction(config_name=method_name, **algorithm_kwargs)

    # Wrapper-level fields are consumed by PerLayerHead(OnPolicy)Compaction, not the algorithm.
    wrapper_fields = ('algorithm', 'on_policy', 'precomputed_budget_path', 'max_ratio_per_head')
    algorithm_kwargs = {k: v for k, v in method_kwargs.items() if k not in wrapper_fields}

    use_on_policy = method_kwargs.get('on_policy', False)

    precomputed_budget_path = method_kwargs.get('precomputed_budget_path', None)

    max_ratio_per_head = method_kwargs.get('max_ratio_per_head', 1.0)

    if base_algorithm in ALGORITHM_REGISTRY:
        algorithm_class = ALGORITHM_REGISTRY[base_algorithm]
        if use_on_policy:
            return PerLayerHeadOnPolicyCompaction(
                config_name=method_name,
                algorithm_class=algorithm_class,
                algorithm_kwargs=algorithm_kwargs,
                precomputed_budget_path=precomputed_budget_path,
                max_ratio_per_head=max_ratio_per_head,
            )
        else:
            return PerLayerHeadCompaction(
                config_name=method_name,
                algorithm_class=algorithm_class,
                algorithm_kwargs=algorithm_kwargs,
                precomputed_budget_path=precomputed_budget_path,
                max_ratio_per_head=max_ratio_per_head,
            )
    else:
        raise ValueError(
            f"Unknown base algorithm: {base_algorithm}. "
            f"Available algorithms: 'original', 'global_highest_attention_keys', "
            f"'summarize', {', '.join(ALGORITHM_REGISTRY.keys())}"
        )


class OriginalCacheMethod(FullCacheCompactionAlgorithm):
    """No-op baseline: returns the full context text, which is then generated from with vLLM."""

    def name(self) -> str:
        return "original"

    def returns_cache(self) -> bool:
        return False

    def requires_preextracted_cache(self) -> bool:
        return False

    def compact_kv_cache(
        self,
        past_key_values,
        target_size,
        indices,
        query_config,
        model=None,
        tokenizer=None,
        formatted_context=None,
        vllm_model=None,
        verbose_logging=False,
        sliding_layer_indices=None,
    ):
        """Return the unchanged formatted context and stats with ratio 1.0."""
        if past_key_values is not None:
            # Sliding-window layers may hold fewer tokens, so read a full-attention layer.
            sliding_layer_indices = sliding_layer_indices or set()
            ref_layer_idx = 0
            for i in range(len(past_key_values)):
                if i not in sliding_layer_indices:
                    ref_layer_idx = i
                    break
            seq_len = past_key_values[ref_layer_idx][0].shape[2]
        else:
            seq_len = len(tokenizer.encode(formatted_context, add_special_tokens=False)) if tokenizer and formatted_context else 0

        stats = {
            'method': 'original',
            'compaction_ratio': 1.0,
            'tensor_compacted_seq_len': seq_len,
            'effective_compacted_seq_len': seq_len,
        }

        return formatted_context, stats
