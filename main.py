"""wandb entry point for KV cache compaction QA evaluation.

Reads the run config from ``wandb.config``, runs :func:`run_evaluation_from_config`
and logs the aggregate metrics to wandb.

Usage::

    python main.py --debug                       # local run with the defaults below, wandb disabled
    wandb sweep -p <project> sweeps/frontier/qwen3-4b/quality/optimized/kv_surgeon.yaml
    python controller.py --sweep <sweep_id>=25   # SLURM jobs running the agent
"""
import math
import os
import socket
import sys
import traceback

import wandb

from evaluation.run_qa_evaluation import run_evaluation_from_config

debug = "--debug" in sys.argv

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "compaction")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)

# Every option a run accepts. The values are used only with --debug. In sweep
# runs every key starts as None, so unset keys fall back to
# evaluation.run_qa_evaluation.DEFAULTS.
defaults = dict(
    # Algorithm-config keys to evaluate; 'original' is the uncompacted baseline,
    # 'all' runs every method in algorithm_config.
    methods=['original', 'all'],

    dataset_name='quality',          # quality, longhealth, longhealthX (e.g. longhealth5), lca
    n_articles=1,                    # -1 for all
    start_article=0,
    n_questions_per_article=None,    # None -> all

    target_size=0.1,                 # compacted article length; a fraction if in (0, 1)

    chunking=None,                   # None|none, longhealth, lqa_pack
    chunk_size=4096,                 # max chunk tokens for 'lqa_pack'
    use_kv_based=1,                  # slice chunks from one full-context cache

    max_new_tokens=2048,
    batch_size=20,                   # 0/None -> sequential
    verbose_logging=0,
    compute_perplexity=1,            # perplexity of the original answers under the compacted cache
    seed=None,

    model_name='Qwen/Qwen3-4B',
    device=None,                     # None -> cuda if available
    max_model_len=None,              # None -> model default

    # Used only when compaction_algorithm is None.
    algorithm_config='fast',         # evaluation/configs/algorithms/<name>.py
    query_config='repeat',           # evaluation/configs/query_generation/<name>.py
    precomputed_budget_path=None,    # nonuniform head-budget JSON
    max_ratio_per_head=0.75,         # cap on per-head ratio with precomputed budgets

    # Inline single method: every non-reserved config key is forwarded to the
    # algorithm constructor (see build_method_overrides).
    compaction_algorithm="kv_surgeon",
    method_name="kv_surgeon",         # None -> compaction_algorithm
    eval_original=True,              # also evaluate the uncompacted baseline
    method_kwargs=dict(
        pool_factor=4.0, block_size=32, gamma0=1.0, ridge=0.01,
        refresh_interval=0, pool_score='loo', refit='vm', c2_solver='lstsq',
        nnls_iters=2, nnls_log_lower_bound=-3, nnls_log_upper_bound=3,
    ),

    log_dir='logs/qa_evaluation',
    name=None,                       # None -> derived from target_size
)

# Harness keys; every other config key is an algorithm hyperparameter.
RESERVED_CONFIG_KEYS = set(defaults) | {'computer', 'method_overrides'}


def build_method_overrides(config: dict):
    """Build ``{method_name: kwargs}`` from the flat config, or None if no ``compaction_algorithm``."""
    algorithm = config.get('compaction_algorithm')
    if not algorithm:
        return None

    method_name = config.get('method_name') or algorithm
    kwargs = {'algorithm': algorithm}

    for key, value in config.items():
        if key in RESERVED_CONFIG_KEYS or key.startswith('_'):
            continue
        if value is None:
            continue
        kwargs[key] = value

    # Nested dict for kwargs that must be swept jointly.
    extra = config.get('method_kwargs')
    if isinstance(extra, dict):
        kwargs.update(extra)

    # NNLS bounds may be given in log space.
    log_low = kwargs.pop('nnls_log_lower_bound', None)
    if log_low is not None:
        kwargs['nnls_lower_bound'] = math.exp(log_low)
    log_high = kwargs.pop('nnls_log_upper_bound', None)
    if log_high is not None:
        kwargs['nnls_upper_bound'] = math.exp(log_high)

    return {method_name: kwargs}


# Top-level wandb metric name -> key path into overall_stats[primary_method].
HEADLINE_METRICS = {
    "accuracy": ("overall_accuracy",),  # exact-match line score on lca
    "parse_rate": ("overall_parse_rate",),
    "perplexity": ("overall_avg_perplexity",),
    "log_perplexity": ("overall_avg_log_perplexity",),
    "num_articles": ("num_articles",),
    "num_questions": ("total_questions",),
    "num_correct": ("total_correct",),
    "total_compaction_time": ("total_compaction_time",),
    "avg_compaction_time_per_article": ("avg_compaction_time_per_article",),
    "total_extraction_time": ("total_extraction_time",),
    "avg_extraction_time_per_article": ("avg_extraction_time_per_article",),
    "total_query_generation_time": ("total_query_generation_time",),
    "avg_query_generation_time_per_article": ("avg_query_generation_time_per_article",),
    "total_generation_time": ("total_generation_time",),
    "avg_generation_time_per_question": ("avg_generation_time_per_question",),
    "total_generated_tokens": ("total_generated_tokens",),
    "tokens_per_second": ("avg_tokens_per_second",),
    "avg_time_per_token": ("avg_time_per_token",),
    "compaction_ratio": ("article_compaction_stats", "avg_article_compaction_ratio"),
    "avg_original_article_tokens": ("article_compaction_stats", "avg_original_article_tokens"),
    "avg_effective_article_tokens": ("article_compaction_stats", "avg_effective_article_tokens"),
    "memory_reduction_pct": ("memory_stats", "avg_memory_reduction_pct"),
    "compacted_cache_size": ("memory_stats", "avg_compacted_cache_size"),
}


def _dig(stats: dict, path: tuple):
    cur = stats
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def log_overall_stats(overall_stats: dict, primary_methods) -> None:
    """Log every scalar as ``{method}/{metric}``, plus top-level headline metrics for the primary method."""
    flat = {}

    def add_scalars(prefix: str, d: dict):
        for key, value in d.items():
            if isinstance(value, dict):
                add_scalars(f"{prefix}/{key}", value)
            elif isinstance(value, (int, float, bool)):
                flat[f"{prefix}/{key}"] = value

    for method, stats in overall_stats.items():
        if not isinstance(stats, dict):
            continue
        for key, value in stats.items():
            if key == "per_article":
                continue
            if isinstance(value, dict):
                add_scalars(f"{method}/{key}", value)
            elif isinstance(value, (int, float, bool)):
                flat[f"{method}/{key}"] = value

    # Primary method: the first non-original method that ran.
    primary = None
    for m in primary_methods:
        if m != "original" and m in overall_stats:
            primary = m
            break
    if primary is None and overall_stats:
        primary = next(iter(overall_stats))

    if primary is not None:
        primary_stats = overall_stats[primary]
        flat["primary_method"] = primary
        for name, path in HEADLINE_METRICS.items():
            value = _dig(primary_stats, path)
            if isinstance(value, (int, float, bool)):
                flat[name] = value

        # Compaction cost per article: query generation plus compaction,
        # excluding the full-cache prefill and answer generation.
        pipeline_parts = [
            flat.get("avg_query_generation_time_per_article"),
            flat.get("avg_compaction_time_per_article"),
        ]
        if all(isinstance(p, (int, float)) for p in pipeline_parts):
            flat["avg_pipeline_time_per_article"] = sum(pipeline_parts)

        if primary != "original" and isinstance(overall_stats.get("original"), dict):
            base = overall_stats["original"]
            base_acc = base.get("overall_accuracy")
            if isinstance(base_acc, (int, float)):
                flat["baseline_accuracy"] = base_acc
                if "accuracy" in flat:
                    flat["accuracy_drop"] = base_acc - flat["accuracy"]

    if flat:
        wandb.log(flat)
        for key, value in flat.items():
            wandb.run.summary[key] = value


def main():
    # Outside debug mode only wandb-supplied values take effect.
    if debug:
        config_defaults = dict(defaults)
    else:
        config_defaults = {key: None for key in defaults}
    config_defaults["computer"] = socket.gethostname()

    wandb.init(
        config=config_defaults,
        project=WANDB_PROJECT,   # overridden inside a sweep
        entity=WANDB_ENTITY,     # overridden inside a sweep
        mode="disabled" if debug else "online",
    )

    exit_code = 0
    try:
        config = dict(wandb.config)

        method_overrides = build_method_overrides(config)
        if method_overrides:
            config['method_overrides'] = method_overrides
            method_name = next(iter(method_overrides))
            eval_original = config.get('eval_original')
            if eval_original is None:
                eval_original = True
            config['methods'] = (
                ['original', method_name] if eval_original else [method_name]
            )
            print(f"Inline method '{method_name}': {method_overrides[method_name]}")

        output = run_evaluation_from_config(config)

        overall_stats = output.get("overall_stats", {}) if isinstance(output, dict) else {}
        methods_run = output.get("config", {}).get("methods", []) if isinstance(output, dict) else []
        log_overall_stats(overall_stats, methods_run)
    except BaseException:
        # Catch everything so wandb.finish() still marks the run as crashed.
        exit_code = 1
        traceback.print_exc()
    finally:
        try:
            wandb.finish(exit_code=exit_code)
        except Exception:
            pass
        # vLLM's engine process and non-daemon threads can block interpreter
        # shutdown, so hard-exit.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
