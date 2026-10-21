"""Per-head KV cache compaction algorithms (interface: base.CompactionAlgorithm)."""
from .base import CompactionAlgorithm
from .omp import OMPCompaction
from .highest_attention_keys import HighestAttentionKeysCompaction
from .loo_evict import LOOEvictCompaction
from .kv_surgeon import KVSurgeonCompaction
from .greedy_evict import GreedyEvictCompaction
from .magnitude_evict import MagnitudeEvictCompaction
from .qfilters import QFiltersCompaction
from .kvmerger import KVMergerCompaction

__all__ = [
    'CompactionAlgorithm',
    'OMPCompaction',
    'HighestAttentionKeysCompaction',
    'LOOEvictCompaction',
    'KVSurgeonCompaction',
    'GreedyEvictCompaction',
    'MagnitudeEvictCompaction',
    'QFiltersCompaction',
    'KVMergerCompaction',
]

# Config names of the per-head algorithms, used by compaction_methods/registry.py.
ALGORITHM_REGISTRY = {
    'omp': OMPCompaction,
    'highest_attention_keys': HighestAttentionKeysCompaction,
    'loo_evict': LOOEvictCompaction,
    'kv_surgeon': KVSurgeonCompaction,
    'greedy_evict': GreedyEvictCompaction,
    'magnitude_evict': MagnitudeEvictCompaction,
    'qfilters': QFiltersCompaction,
    'kvmerger': KVMergerCompaction,
}
