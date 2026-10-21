# Query vectors of the last 64 article positions (SnapKV-style observation window).
from compaction.query_generation.config import QueryConfig, QueryMethodConfig, ContextPrefillConfig

config = QueryConfig(
    method_configs=[
        QueryMethodConfig(
            method='context_prefill',
            fraction=1.0,
            config=ContextPrefillConfig(last_n=64),
        ),
    ],
    max_query_vectors_per_kv_head=100000,
    verbose=True
)
