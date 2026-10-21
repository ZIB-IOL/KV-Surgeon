# KV-Surgeon: Compensation-Aware Offline KV-Cache Compaction

Code for [KV-Surgeon](https://arxiv.org/abs/2610.25933).

Offline compaction builds a smaller KV cache for a fixed context once and
reuses it for every later query. KV-Surgeon decides which cache entries to
delete with a quadratic surrogate of the attention-output error. It deletes
entries greedily, and after each deletion it updates the remaining entries in
closed form to compensate. A final refit sets the retained keys' logit biases
and values.

This repository is built on the
[Attention Matching code base](https://github.com/adamzweiger/compaction)
(Zweiger et al., 2026). It also contains the baselines and ablations from the
paper.

## Installation

Python 3.12 and a CUDA GPU are required.

```bash
pip install -r requirements.txt
```

## Quick start

Compact the KV cache of a short article with KV-Surgeon and compare answers
before and after:

```bash
python -m examples.qa_demo --model Qwen/Qwen3-4B --target-size 0.02
```

Evaluate on QuALITY without wandb:

```bash
python -m evaluation.run_qa_evaluation \
  --model-name Qwen/Qwen3-4B \
  --algorithm-config kv_surgeon --methods original kv_surgeon \
  --query-config ss-plus-repeat \
  --precomputed-budget-path head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json \
  --dataset-name quality --n-articles 1 --target-size 0.02
```

For KV-Surgeon-fast use `--methods kv_surgeon_fast --query-config repeat`.
`--target-size 0.02` keeps 2% of the context (50x compaction).

## Reproducing the paper

Every experiment is a wandb grid sweep in `sweeps/`. See
[`sweeps/README.md`](sweeps/README.md) for the layout and how to launch them.

| Experiment | Sweeps |
|---|---|
| Main results, all models and ratios | `sweeps/frontier/<model>/{quality,longhealth}/` |
| Long Code Arena | `sweeps/frontier/qwen3-4b/lca/` |
| Component ablations | `sweeps/ablation_*.yaml` |
| Head budgets and final refit | `sweeps/frontier/qwen3-4b/quality/uniform/` |
| Ported baselines, published vs. our pipeline | `sweeps/frontier/qwen3-4b/quality/{kvzip.yaml,uniform/,optimized/}` |

Each run writes a result JSON to the `log_dir` of its sweep. The ablation
tables are aggregated with `scripts/aggregate_refit_budget_ablation.py` and
`scripts/aggregate_selection_ablations.py`. The loss-level analyses of
compensation use `scripts/surrogate_check.py`, `scripts/kv_surgeon_oracle.py`
and `scripts/refresh_diag.py`.

**Data.**
- QuALITY is included.
- LongHealth is downloaded from its repository on first use.
- Long Code Arena must be built once:
  `python scripts/build_lca_repo_completion.py`.

**Precomputed files.**
- Per-head budgets: `head_budget_optimization/head_budgets/<Model>/optimized_agnostic.json`.
  Recompute them with `scripts/head-budget-optimization.sh`, then run the
  solve step described in `head_budget_optimization/run.py`.
- Q-Filters vectors: `qfilters/pile/<Model>.pt`. Recompute them with
  `scripts/compute_qfilters.py`.

The reference answers used for perplexity are generated and cached on first
use. They are sampled, so perplexities can differ slightly from ours.

## Methods

| Name in the paper | Algorithm | Reference |
|---|---|---|
| KV-Surgeon, KV-Surgeon-fast | `kv_surgeon` | this work |
| LOO | `loo_evict` | this work; generalizes [CAOTE](https://arxiv.org/abs/2504.14051) and [ReST-KV](https://arxiv.org/abs/2605.08840) |
| AM-HAK, AM-OMP, AM-OMP-CS | `highest_attention_keys`, `omp` | [Attention Matching](https://arxiv.org/abs/2602.16284) |
| OBCache | `loo_evict` (`score: obcache_k`) | [Gu et al., 2025](https://arxiv.org/abs/2510.07651) |
| H2O | `highest_attention_keys` (`score_method: mean`) | [Zhang et al., 2023](https://arxiv.org/abs/2306.14048) |
| KVzip | `highest_attention_keys` (`score_method: max`), `global_highest_attention_keys` | [Kim et al., 2025](https://arxiv.org/abs/2505.23416) |
| KVMerger | `kvmerger` | [Wang et al., 2024](https://arxiv.org/abs/2407.08454) |
| Q-Filters | `qfilters` | [Godey et al., 2025](https://arxiv.org/abs/2503.02812) |
| StreamingLLM | `magnitude_evict` (`score: recency`) | [Xiao et al., 2024](https://arxiv.org/abs/2309.17453) |
| L2 key-norm | `magnitude_evict` (`score: norm`) | [Devoto et al., 2024](https://arxiv.org/abs/2406.11430) |
| Summarization | `summarize` | |

Calibration queries come from self-study
([Cartridges](https://arxiv.org/abs/2506.06266)) or a repeat pass
([KVzip](https://arxiv.org/abs/2505.23416)).

Benchmarks: [QuALITY](https://arxiv.org/abs/2112.08608),
[LongHealth](https://arxiv.org/abs/2401.14490),
[Long Code Arena](https://arxiv.org/abs/2406.11612).
Models: [Qwen3](https://arxiv.org/abs/2505.09388),
[Llama 3.1](https://arxiv.org/abs/2407.21783),
[Gemma 3](https://arxiv.org/abs/2503.19786).
The Q-Filters vectors are calibrated on
[the Pile](https://arxiv.org/abs/2101.00027) (`NeelNanda/pile-10k`).
Generation uses [vLLM](https://arxiv.org/abs/2309.06180) and
[Transformers](https://github.com/huggingface/transformers).

## Repository layout

| Path | Contents |
|---|---|
| `compaction/algorithms/` | Per-head compaction algorithms |
| `compaction/compaction_methods/` | Whole-cache wrappers: per-head budgets, calibration queries, chunking |
| `compaction/query_generation/` | Self-study, repeat-pass and prefill calibration queries |
| `evaluation/` | Datasets, QA evaluation, algorithm and query configs |
| `head_budget_optimization/` | Per-head budget computation and the budget files |
| `models/` | Qwen3, Qwen3-MoE, Llama and Gemma 3 with compacted-cache support |
| `sweeps/` | Experiment definitions |
| `scripts/` | Data building, Q-Filters, ablation aggregation and analyses |
| `main.py`, `controller.py`, `runner.sh` | wandb sweep entry point and SLURM launcher |

## Citation

If you use this code, please cite KV-Surgeon and Attention Matching:

```bibtex
@misc{wagner2026kvsurgeon,
      title={The KV-Surgeon: Compensation-Aware Offline KV-Cache Compaction},
      author={Moritz Wagner and Christophe Roux and Max Zimmer and Sebastian Pokutta},
      year={2026},
      eprint={2610.25933},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.25933},
}
```

```bibtex
@misc{zweiger2026fastkvcompaction,
      title={Fast {KV} Compaction via {Attention Matching}},
      author={Adam Zweiger and Xinghong Fu and Han Guo and Yoon Kim},
      year={2026},
      eprint={2602.16284},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.16284},
}
```

## License

MIT, see [LICENSE](LICENSE). Some files are under other licenses; see
[NOTICE](NOTICE).
