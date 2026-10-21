"""Full-cache compaction methods, per-layer-head wrappers and the method factory."""
from .base import FullCacheCompactionAlgorithm
from .per_layer_head import PerLayerHeadCompaction
from .per_layer_head_on_policy import PerLayerHeadOnPolicyCompaction
from .global_highest_attention_keys import GlobalHighestAttentionKeysCompaction
from .summarize import SummarizeCompaction
from .registry import get_compaction_method, OriginalCacheMethod


# Lazy import avoids a circular import.
def __getattr__(name):
    if name == 'ChunkedCompaction':
        from .chunked import ChunkedCompaction
        return ChunkedCompaction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'FullCacheCompactionAlgorithm',
    'PerLayerHeadCompaction',
    'PerLayerHeadOnPolicyCompaction',
    'GlobalHighestAttentionKeysCompaction',
    'SummarizeCompaction',
    'ChunkedCompaction',
    'get_compaction_method',
    'OriginalCacheMethod',
]
