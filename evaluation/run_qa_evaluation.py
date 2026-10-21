"""Command-line QA evaluation of KV cache compaction methods.

Example usage:
    # Evaluate the full-context baseline and every method in evaluation/configs/algorithms/fast.py
    python -m evaluation.run_qa_evaluation --n-articles 3

    # Evaluate a single method from a config file with a custom target size
    python -m evaluation.run_qa_evaluation --algorithm-config kv_surgeon \
        --methods kv_surgeon --query-config ss-plus-repeat --target-size 0.05

    # Evaluate with chunking on a long context dataset with nonuniform budgets
    python -m evaluation.run_qa_evaluation --dataset-name longhealth5 --chunking longhealth \
        --model-name google/gemma-3-12b-it --query-config repeat \
        --precomputed-budget-path head_budget_optimization/head_budgets/gemma-3-12b-it/optimized_agnostic.json

Single-method runs with arbitrary hyperparameters go through main.py.
"""
import argparse
import torch

from .qa_evaluator import QAEvaluator
from .configs.utils import load_algorithm_config, load_query_config


# All run options, shared by the CLI below and main.py.
DEFAULTS = dict(
    methods=['original', 'all'],
    dataset_name='quality',
    n_articles=1,
    start_article=0,
    n_questions_per_article=None,
    target_size=0.1,
    chunking=None,
    chunk_size=4096,
    # Slice chunks from one full-context cache so they keep their original RoPE positions.
    use_kv_based=1,
    max_new_tokens=2048,
    batch_size=20,
    verbose_logging=0,
    compute_perplexity=1,
    model_name='Qwen/Qwen3-4B',
    device=None,
    max_model_len=None,
    algorithm_config='fast',
    query_config='repeat',
    precomputed_budget_path=None,
    max_ratio_per_head=0.75,
    # Seeds python/numpy/torch and the vLLM engine. None = unseeded.
    seed=None,
    log_dir='logs/qa_evaluation',
    name=None,
    # {method_name: kwargs}; replaces loading `algorithm_config` from disk.
    method_overrides=None,
)


def run_evaluation_from_config(config) -> dict:
    """Run a QA evaluation from a flat config mapping; missing or None keys fall back to ``DEFAULTS``.

    Returns the output of ``QAEvaluator.run_evaluation``.
    """
    def get(key):
        value = config.get(key, None) if hasattr(config, 'get') else getattr(config, key, None)
        if value is None and DEFAULTS.get(key) is not None:
            return DEFAULTS[key]
        return value

    methods = get('methods')
    dataset_name = get('dataset_name')
    n_articles = get('n_articles')
    start_article = get('start_article')
    n_questions_per_article = get('n_questions_per_article')
    target_size = get('target_size')
    chunking = get('chunking')
    chunk_size = get('chunk_size')
    use_kv_based = get('use_kv_based')
    max_new_tokens = get('max_new_tokens')
    batch_size = get('batch_size')
    verbose_logging = get('verbose_logging')
    compute_perplexity = get('compute_perplexity')
    model_name = get('model_name')
    device = get('device')
    max_model_len = get('max_model_len')
    algorithm_config = get('algorithm_config')
    query_config_name = get('query_config')
    precomputed_budget_path = get('precomputed_budget_path')
    max_ratio_per_head = get('max_ratio_per_head')
    log_dir = get('log_dir')
    name = get('name')

    method_overrides = get('method_overrides')

    if isinstance(methods, str):
        methods = [m for m in methods.replace(',', ' ').split() if m]

    if method_overrides:
        method_config = dict(method_overrides)
        algorithm_config = None
        print(f"Using inline method_overrides: {list(method_config.keys())}")
    else:
        method_config = load_algorithm_config(algorithm_config, target_size=target_size)
        print(f"Loaded algorithm config from: {algorithm_config}")
    print(f"Available configs: {list(method_config.keys())}")

    methods_to_run = []
    if 'all' in methods:
        if 'original' in methods:
            methods_to_run.append('original')
        methods_to_run.extend(list(method_config.keys()))
    else:
        methods_to_run = list(methods)
        invalid_configs = [m for m in methods_to_run if m != 'original' and m not in method_config]
        if invalid_configs:
            raise ValueError(
                f"Unknown configurations: {invalid_configs}. "
                f"Available in {algorithm_config}: {list(method_config.keys())}"
            )

    method_kwargs = {}
    for method_name in methods_to_run:
        if method_name == 'original':
            continue

        if method_name in method_config:
            algo_config = method_config[method_name]
            method_kwargs[method_name] = {k: v for k, v in algo_config.items()}

            if chunking is not None and chunking != 'none':
                method_kwargs[method_name]['chunking'] = chunking
                method_kwargs[method_name]['chunk_size'] = chunk_size
                method_kwargs[method_name]['use_kv_based'] = bool(use_kv_based)

            if precomputed_budget_path is not None:
                method_kwargs[method_name]['precomputed_budget_path'] = precomputed_budget_path
                method_kwargs[method_name]['max_ratio_per_head'] = max_ratio_per_head

    query_config = load_query_config(query_config_name)
    if verbose_logging:
        query_config.verbose = True
    print(f"Loaded query config from: {query_config_name}")
    print(f"  - methods: {[mc.method for mc in query_config.method_configs]}")
    print(f"  - fractions: {[mc.fraction for mc in query_config.method_configs]}")
    print(f"  - max_query_vectors_per_kv_head: {query_config.max_query_vectors_per_kv_head}")

    # Seed before model load and data selection.
    seed = get('seed')
    if seed is not None:
        seed = int(seed)
        import random as _random
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except ImportError:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seeded run with seed={seed} (python/numpy/torch + vLLM engine)")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device != 'cuda':
        raise RuntimeError(
            "CUDA is currently required for evaluation. "
            "Please run on a machine with a GPU."
        )

    if name is None:
        if chunking and chunking != 'none':
            name = f"target{target_size}_chunked_{chunking}"
        else:
            name = f"target{target_size}"

    if chunking and chunking != 'none':
        print(f"\nChunked compaction enabled:")
        print(f"  - strategy: {chunking}")
        if chunking == 'lqa_pack':
            print(f"  - chunk_size: {chunk_size}")

    evaluator = QAEvaluator(
        model_name=model_name,
        device=device,
        max_model_len=max_model_len,
        seed=seed,
    )

    # Always tear down vLLM so its engine process is not left orphaned on the GPU.
    try:
        output = evaluator.run_evaluation(
            dataset_name=dataset_name,
            compaction_methods=methods_to_run,
            target_size=target_size,
            n_articles=n_articles,
            start_article=start_article,
            n_questions_per_article=n_questions_per_article,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            query_config=query_config,
            verbose_logging=bool(verbose_logging),
            compute_perplexity=bool(compute_perplexity),
            method_kwargs=method_kwargs,
            log_dir=log_dir,
            experiment_name=name,
            algorithm_config_file=algorithm_config,
            query_config_file=query_config_name,
        )
    finally:
        _shutdown_evaluator(evaluator)
    return output


def _shutdown_evaluator(evaluator) -> None:
    """Best-effort release of the evaluator's vLLM engine and CUDA memory."""
    try:
        vllm_model = getattr(evaluator, 'vllm_model', None)
        if vllm_model is not None:
            del vllm_model
            evaluator.vllm_model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate KV cache compaction on QA tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Method selection
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['original', 'all'],
        help='Compaction methods to evaluate: "original" (full context) and/or method names '
             'defined in the --algorithm-config file, or "all" for every method in it '
             '(default: original all)'
    )

    # Data arguments
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='quality',
        help='Dataset name. Options: quality, longhealth, longhealthX (X patients per article, '
             'e.g. longhealth5), lca (default: quality)'
    )
    parser.add_argument(
        '--n-articles',
        type=int,
        default=1,
        help='Number of articles to evaluate. Use -1 for all articles (default: 1)'
    )
    parser.add_argument(
        '--start-article',
        type=int,
        default=0,
        help='Starting article index. Evaluates articles from start_article to start_article + n_articles (default: 0)'
    )
    parser.add_argument(
        '--n-questions-per-article',
        type=int,
        default=None,
        help='Number of questions per article. If specified, uses a random shuffled set of n questions per article (or all questions if less than n) (default: None, uses all questions)'
    )

    # Compaction arguments
    parser.add_argument(
        '--target-size',
        type=float,
        default=0.1,
        help='Target compacted article sequence length. If between 0 and 1, treated as fraction of original size (default: 0.1 = 10%%). Note that this specifies only the target compaction size of the article portion, keeping the system prompt and chat template intact.'
    )

    # Chunked compaction arguments
    parser.add_argument(
        '--chunking',
        type=str,
        default=None,
        choices=['none', 'longhealth', 'lqa_pack'],
        help='Chunking strategy for compaction. Options: none (default), '
             'longhealth (split on <text_0>, grouping note chains), '
             'lqa_pack (repository files split on [start of ...] markers and packed up to '
             '--chunk-size tokens; the lca benchmark uses --chunk-size 10000). '
             'The full cache is extracted once and chunks are sliced from it.'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=4096,
        help='Maximum chunk size in tokens for lqa_pack chunking (default: 4096)'
    )

    # Evaluation arguments
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens to generate for each answer (default: 2048)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='Batch size for generation. If specified and nonzero, processes questions in batches for faster generation (default: 20)'
    )
    parser.add_argument(
        '--verbose-logging',
        type=int,
        default=0,
        help='Enable verbose logging for compaction and query generation (saves detailed stats like beta_stats, selected_indices). 1 to enable, 0 to disable (default: 0)'
    )
    parser.add_argument(
        '--compute-perplexity',
        type=int,
        default=1,
        help='Compute perplexity of original generation under compacted cache. 1 to enable, 0 to disable (default: 1)'
    )

    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen3-4B',
        help='HuggingFace model name (default: Qwen/Qwen3-4B)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cpu or cuda). If not specified, uses cuda if available, else cpu'
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=None,
        help='Maximum model context length. Sets max_model_len for vLLM and enables '
             'YaRN rope scaling for extended context support. Example: 131072 for 128K context.'
    )

    # Configuration files
    parser.add_argument(
        '--algorithm-config',
        type=str,
        default='fast',
        help='Name of a config in evaluation/configs/algorithms/ (e.g. "fast", "kv_surgeon")'
    )
    parser.add_argument(
        '--query-config',
        type=str,
        default='repeat',
        help='Name of query generation config (repeat, ss-plus-repeat, context-prefill, '
             'context-prefill-last64) - automatically uses evaluation/configs/query_generation/ directory'
    )
    parser.add_argument(
        '--precomputed-budget-path',
        type=str,
        default=None,
        help='Path to precomputed head budget proportions JSON file (for nonuniform head budgets)'
    )
    parser.add_argument(
        '--max-ratio-per-head',
        type=float,
        default=0.75,
        help='Maximum ratio per head when using precomputed budgets (default: 0.75). '
             'If budgets would assign a higher ratio, proportions are blended towards uniform.'
    )

    # Logging arguments
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/qa_evaluation',
        help='Directory for logs (default: logs/qa_evaluation)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name for logging (default: None)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Replicate seed (python/numpy/torch + vLLM); None = unseeded (default: None)'
    )

    args = parser.parse_args()

    run_evaluation_from_config(vars(args))


if __name__ == "__main__":
    main()
