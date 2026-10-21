"""
Compute nonuniform per-head budgets.

Measures per-head influence curves (log-perplexity of reference answers as one head's
budget varies, the others at a uniform baseline), averages them over articles and solves
for head proportions. Outputs go to logs/budget_optimization/<model>/optimized_<timestamp>/.
Copy optimized_agnostic.json to head_budgets/<model>/ to use it via --precomputed-budget-path.

Recipe for the released head_budgets/<Model>/optimized_agnostic.json:

Step 1, curves only, one job per held-out QuALITY article 90-99:

    python -m head_budget_optimization.run \
        --model-name Qwen/Qwen3-4B \
        --dataset-name quality --n-articles 1 --start-article 90 \
        --target-ratio 0.05 --n-eval-points 51 --max-ratio 0.5 \
        --algorithm-config best --method highest_attn_keys_rms_nnls2_-3_3_lsq \
        --query-config ss-plus-repeat \
        --skip-solve

Step 2, collect the ten per_article_curves/article_*.json into one directory and solve
for one set of proportions across all --solve-ratios:

    python -m head_budget_optimization.run \
        --model-name Qwen/Qwen3-4B \
        --per-article-curves-dir <dir with article_*.json> \
        --solve-ratios 0.01,0.02,0.05,0.1
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import json
import torch
import numpy as np
from datetime import datetime

from evaluation.utils import load_model_and_tokenizer, extract_full_kv_cache, initialize_vllm
from evaluation.datasets import load_dataset
from evaluation.configs.utils import load_algorithm_config, load_query_config
from models.generate import get_sliding_layer_info

from .influence import (
    HeadInfluenceComputer,
    aggregate_head_curves,
    save_head_curves,
    load_and_aggregate_article_curves,
)
from .solver import HeadBudgetSolver, analyze_head_curves


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute optimal nonuniform head budget allocations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Baseline configuration
    parser.add_argument(
        '--baseline-schedule',
        type=str,
        default=None,
        help='Path to baseline schedule JSON (default: auto-generate uniform schedule for global layers)'
    )
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=0.05,
        help='Baseline target compaction ratio for computing influence curves (default: 0.05)'
    )

    # Evaluation points
    parser.add_argument(
        '--max-ratio',
        type=float,
        default=1.0,
        help='Maximum ratio to probe for each head (default: 1.0)'
    )
    parser.add_argument(
        '--n-eval-points',
        type=int,
        default=5,
        help='Number of evaluation points per head (linspaced from 0 to max-ratio) (default: 5)'
    )
    parser.add_argument(
        '--max-above-baseline',
        type=float,
        default=None,
        help='If set, each head\'s max ratio is baseline_ratio + this value (clamped to [0, max-ratio]). '
             'E.g., 0.2 means sweep from 0 to baseline_ratio + 0.2 for each head. '
             'This reduces OMP computation time for heads with small baseline allocations.'
    )

    # Data arguments
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='quality',
        help='Dataset name (default: quality)'
    )
    parser.add_argument(
        '--n-articles',
        type=int,
        default=1,
        help='Number of articles to use for computing curves (default: 1)'
    )
    parser.add_argument(
        '--start-article',
        type=int,
        default=0,
        help='Starting article index (default: 0)'
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
        help='Device to use (default: cuda if available)'
    )

    # Configuration files
    parser.add_argument(
        '--algorithm-config',
        type=str,
        default='best',
        help='Name of algorithm config file (default: best)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='highest_attn_keys_rms_nnls2_-3_3_lsq',
        help='Method name within the algorithm config (default: highest_attn_keys_rms_nnls2_-3_3_lsq)'
    )
    parser.add_argument(
        '--query-config',
        type=str,
        default='repeat',
        help='Name of query generation config (default: repeat)'
    )

    # Load precomputed curves
    parser.add_argument(
        '--per-article-curves-dir',
        type=str,
        default=None,
        help='Path to directory with per-article curves to aggregate (skip curve computation if provided)'
    )

    # Solver arguments
    parser.add_argument(
        '--solve-ratios',
        type=str,
        default='0.01,0.02,0.05,0.1',
        help='Comma-separated list of target ratios to solve for (default: 0.01,0.02,0.05,0.1)'
    )
    parser.add_argument(
        '--step-size',
        type=float,
        default=0.001,
        help='Step size for solver (default: 0.001)'
    )
    parser.add_argument(
        '--solver-method',
        type=str,
        default='ratio-agnostic',
        choices=['greedy', 'swap', 'annealing', 'ratio-agnostic'],
        help='Solver method: "greedy" (start from zero, build up), '
             '"swap" (start from uniform, swap between heads), '
             '"annealing" (simulated annealing), or '
             '"ratio-agnostic" (find single proportions good across all target ratios). Default: ratio-agnostic'
    )
    parser.add_argument(
        '--ratio-weights',
        type=str,
        default=None,
        help='Comma-separated weights for each target ratio when using ratio-agnostic solver. '
             'E.g., "1,1,2,2" to weight larger ratios more heavily. Default: uniform weights.'
    )
    parser.add_argument(
        '--smoothing-window',
        type=int,
        default=0,
        help='Sliding window size for smoothing curves before solving (default: 0 = no smoothing). '
             'Smoothed curves are used for decision-making, but final loss is computed on original curves.'
    )

    # Other
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Max tokens for reference answer generation (default: 2048)'
    )
    parser.add_argument(
        '--skip-solve',
        action='store_true',
        help='Only compute curves, skip solving for optimal allocations'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    solve_ratios = [float(r.strip()) for r in args.solve_ratios.split(',')]

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device != 'cuda':
        raise RuntimeError(
            "CUDA is currently required for evaluation. "
            "Please run on a machine with a GPU."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    output_dir = Path(f'logs/budget_optimization/{model_short_name}/optimized_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_output = str(output_dir / 'head_curves.json')

    print(f"\n{'='*60}")
    print("HEAD BUDGET OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Baseline schedule: {args.baseline_schedule or '(auto-uniform for global layers)'}")
    print(f"Target ratio for curves: {args.target_ratio}")
    print(f"Eval points: {args.n_eval_points} (0 to {args.max_ratio})")
    print(f"Solve ratios: {solve_ratios}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    def infer_model_dims(head_curves):
        head_keys = list(head_curves.keys())
        layers = set()
        heads = set()
        for key in head_keys:
            layer_idx = int(key[1:].split('H')[0])
            head_idx = int(key.split('H')[1])
            layers.add(layer_idx)
            heads.add(head_idx)
        return max(layers) + 1, max(heads) + 1

    if args.per_article_curves_dir:
        print(f"Loading and aggregating per-article curves from {args.per_article_curves_dir}")
        head_curves, article_metadata = load_and_aggregate_article_curves(args.per_article_curves_dir)
        num_layers, num_heads = infer_model_dims(head_curves)
        print(f"Aggregated curves for {len(head_curves)} heads ({num_layers} layers x {num_heads} heads)")
        print(f"From {len(article_metadata)} articles")

    else:
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(
            args.model_name,
            args.device,
        )

        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_key_value_heads

        sliding_layer_indices, sliding_window = get_sliding_layer_info(model)
        global_layer_indices = [i for i in range(num_layers) if i not in sliding_layer_indices]
        num_global_layers = len(global_layer_indices)

        if sliding_layer_indices:
            print(f"Detected {len(sliding_layer_indices)} sliding window layers (window={sliding_window})")
            print(f"Will optimize {num_global_layers} global attention layers only")

        if args.baseline_schedule is not None:
            print(f"Loading baseline schedule from {args.baseline_schedule}")
            with open(args.baseline_schedule, 'r') as f:
                baseline_proportions = json.load(f)
        else:
            print("Generating uniform baseline schedule for global layers")
            total_global_heads = num_global_layers * num_heads
            uniform_proportion = 1.0 / total_global_heads
            baseline_proportions = {}
            for layer_idx in global_layer_indices:
                for head_idx in range(num_heads):
                    head_key = f"L{layer_idx}H{head_idx}"
                    baseline_proportions[head_key] = uniform_proportion
            print(f"Created uniform schedule with {len(baseline_proportions)} heads, each with proportion {uniform_proportion:.6f}")

        query_config = load_query_config(args.query_config)
        print(f"Query config: {args.query_config}")

        algorithm_config = load_algorithm_config(args.algorithm_config, target_size=args.target_ratio)
        if args.method not in algorithm_config:
            raise ValueError(
                f"Method '{args.method}' not found in config '{args.algorithm_config}'. "
                f"Available methods: {list(algorithm_config.keys())}"
            )
        method_config = algorithm_config[args.method]
        algorithm_name = method_config.get('algorithm', 'highest_attention_keys')
        algorithm_kwargs = {k: v for k, v in method_config.items() if k != 'algorithm'}
        print(f"Algorithm config: {args.algorithm_config}, method: {args.method}")
        print(f"Algorithm: {algorithm_name}")
        print(f"Algorithm kwargs: {algorithm_kwargs}")

        print(f"Loading dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name)

        if args.n_articles == -1:
            article_indices = list(range(args.start_article, len(dataset)))
        else:
            end_article = min(args.start_article + args.n_articles, len(dataset))
            article_indices = list(range(args.start_article, end_article))

        print(f"Using {len(article_indices)} articles: {article_indices}")

        if args.max_above_baseline is not None:
            # Per-head sweep from 0 to baseline_ratio + max_above_baseline.
            total_global_heads = num_global_layers * num_heads
            per_head_eval_ratios = {}
            for layer_idx in global_layer_indices:
                for head_idx in range(num_heads):
                    head_key = f"L{layer_idx}H{head_idx}"
                    proportion = baseline_proportions.get(head_key, 1.0 / total_global_heads)
                    baseline_ratio = proportion * args.target_ratio * total_global_heads

                    head_max = min(args.max_ratio, baseline_ratio + args.max_above_baseline)
                    head_eval_ratios = np.linspace(0, head_max, args.n_eval_points).tolist()
                    per_head_eval_ratios[head_key] = head_eval_ratios

            eval_ratios = per_head_eval_ratios
            print(f"Per-head evaluation ratios (0 to baseline + {args.max_above_baseline}):")
            example_heads = list(per_head_eval_ratios.keys())[:3]
            for head_key in example_heads:
                print(f"  {head_key}: {[f'{r:.3f}' for r in per_head_eval_ratios[head_key]]}")
            print(f"  ...")
        else:
            eval_ratios = np.linspace(0, args.max_ratio, args.n_eval_points).tolist()
            print(f"Evaluation ratios: {eval_ratios}")

        vllm_model = None
        self_study_config = query_config.get_method_config('self_study')
        if self_study_config is not None:
            print("Initializing vLLM for self-study query generation...")
            vllm_model = initialize_vllm(args.model_name)

        computer = HeadInfluenceComputer(
            model=model,
            tokenizer=tokenizer,
            query_config=query_config,
            algorithm_kwargs=algorithm_kwargs,
            algorithm_name=algorithm_name,
            device=args.device,
            vllm_model=vllm_model,
        )

        all_article_curves = []

        for i, article_idx in enumerate(article_indices):
            article_data = dataset[article_idx]

            print(f"\n{'='*60}")
            print(f"Article {i+1}/{len(article_indices)}: {article_data['title']}")
            print(f"{'='*60}")

            print("Extracting KV cache...")
            seq_len, past_key_values, article_indices_range, formatted_context, _ = extract_full_kv_cache(
                model=model,
                tokenizer=tokenizer,
                article_text=article_data['article'],
                device=args.device,
            )

            print(f"Sequence length: {seq_len}")
            print(f"Article indices: {article_indices_range.start}-{article_indices_range.stop} ({len(article_indices_range)} tokens)")

            article_curves = computer.compute_all_head_curves_for_article(
                article_data=article_data,
                past_key_values=past_key_values,
                article_indices=article_indices_range,
                formatted_context=formatted_context,
                baseline_proportions=baseline_proportions,
                target_ratio=args.target_ratio,
                eval_ratios=eval_ratios,
                model_name=args.model_name,
                max_new_tokens=args.max_new_tokens,
            )

            all_article_curves.append(article_curves)

            article_curves_dir = output_dir / 'per_article_curves'
            article_curves_dir.mkdir(parents=True, exist_ok=True)
            article_curve_path = article_curves_dir / f'article_{article_idx}.json'
            suffix = 1
            while article_curve_path.exists():
                article_curve_path = article_curves_dir / f'article_{article_idx}_{suffix}.json'
                suffix += 1
            article_metadata = {
                'article_idx': article_idx,
                'article_title': article_data.get('title', ''),
                'article_id': article_data.get('article_id', ''),
            }
            save_head_curves(article_curves, str(article_curve_path), article_metadata)

            del past_key_values
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("Aggregating curves across articles...")
        print(f"{'='*60}")

        head_curves = aggregate_head_curves(all_article_curves)

        metadata = {
            'model_name': args.model_name,
            'baseline_schedule': args.baseline_schedule,
            'target_ratio': args.target_ratio,
            'n_articles': len(article_indices),
            'article_indices': article_indices,
            'eval_ratios': eval_ratios,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'algorithm_config': args.algorithm_config,
            'algorithm_kwargs': algorithm_kwargs,
            'method': args.method,
            'query_config': args.query_config,
            'timestamp': datetime.now().isoformat(),
        }

        curves_output_path = Path(curves_output)
        base_stem = curves_output_path.stem
        suffix = 1
        while curves_output_path.exists():
            curves_output_path = curves_output_path.parent / f'{base_stem}_{suffix}.json'
            suffix += 1
        save_head_curves(head_curves, str(curves_output_path), metadata)

    print(f"\n{'='*60}")
    print("Analyzing head curves...")
    print(f"{'='*60}")

    analysis = analyze_head_curves(head_curves)

    print("\nTop 10 most important heads (highest delta at ratio=0):")
    for head_key, metrics in analysis['top_10_most_important']:
        print(f"  {head_key}: delta_at_zero={metrics['delta_at_zero']:.4f}, "
              f"auc={metrics['auc']:.4f}")

    print("\nTop 10 least important heads:")
    for head_key, metrics in analysis['top_10_least_important']:
        print(f"  {head_key}: delta_at_zero={metrics['delta_at_zero']:.4f}, "
              f"auc={metrics['auc']:.4f}")

    if args.skip_solve:
        print("\nSkipping solve step (--skip-solve)")
        return

    print(f"\n{'='*60}")
    print("Solving for optimal allocations...")
    print(f"{'='*60}")

    solver = HeadBudgetSolver(
        head_curves=head_curves,
        num_layers=num_layers,
        num_heads=num_heads,
        smoothing_window=args.smoothing_window,
    )

    if args.smoothing_window > 0:
        print(f"Using smoothing window of {args.smoothing_window} for decision-making")

    if args.solver_method == 'ratio-agnostic':
        ratio_weights = None
        if args.ratio_weights:
            ratio_weights = [float(w.strip()) for w in args.ratio_weights.split(',')]
            if len(ratio_weights) != len(solve_ratios):
                raise ValueError(
                    f"Number of weights ({len(ratio_weights)}) must match "
                    f"number of target ratios ({len(solve_ratios)})"
                )

        proportions, solve_stats = solver.solve_ratio_agnostic(
            target_ratios=solve_ratios,
            step_size=args.step_size,
            weights=ratio_weights,
        )

        all_proportions = {'agnostic': proportions}
    else:
        all_proportions, solve_stats = solver.solve_for_ratios(
            target_ratios=solve_ratios,
            step_size=args.step_size,
            method=args.solver_method,
        )

    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    if args.solver_method == 'ratio-agnostic':
        agnostic_path = output_dir / 'optimized_agnostic.json'
        solver.save_proportions(proportions, str(agnostic_path))

        stats_path = output_dir / 'solve_stats.json'
        with open(stats_path, 'w') as f:
            stats_for_json = {
                'agnostic_stats': solve_stats,
                'per_ratio_stats': {
                    str(k): v for k, v in solve_stats.get('per_ratio_stats', {}).items()
                }
            }
            json.dump(stats_for_json, f, indent=2)
        print(f"Saved solve stats to {stats_path}")
    else:
        solver.save_all_proportions(
            all_proportions=all_proportions,
            output_dir=str(output_dir),
            prefix='optimized',
        )

        stats_path = output_dir / 'solve_stats.json'
        solver.save_solve_stats(solve_stats, str(stats_path))

    analysis_path = output_dir / 'analysis.json'
    suffix = 1
    while analysis_path.exists():
        analysis_path = output_dir / f'analysis_{suffix}.json'
        suffix += 1
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Head curves saved to: {curves_output}")
    print(f"Optimized proportions saved to: {output_dir}")
    print(f"Solve stats saved to: {stats_path}")
    print(f"Analysis saved to: {analysis_path}")


if __name__ == '__main__':
    main()
