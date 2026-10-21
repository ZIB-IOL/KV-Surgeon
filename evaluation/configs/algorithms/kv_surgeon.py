import math

exp = math.exp

# Use kv_surgeon with --query-config ss-plus-repeat and kv_surgeon_fast with
# --query-config repeat.
_shared = {
    'algorithm': 'kv_surgeon',
    'on_policy': True,
    'pool_factor': 4.0,
    'block_size': 32,
    'gamma0': 1.0,
    'ridge': 0.01,
    'refresh_interval': 0,
    'refit': 'vm',
    'nnls_iters': 2,
    'nnls_lower_bound': exp(-3),
    'nnls_upper_bound': exp(3),
}

config = {
    'kv_surgeon': {
        **_shared,
        'pool_score': 'loo',
        'c2_solver': 'lstsq',
    },
    'kv_surgeon_fast': {
        **_shared,
        'pool_score': 'truncated',
        'truncation_order': 2,
        'c2_solver': 'cholesky',
        'score_max_queries': 8192,
        'build_max_queries': 4096,
        'build_queries_per_key': 16,
        'fallback_threshold': 0.3,
    },
}
