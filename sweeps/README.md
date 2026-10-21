# Sweeps

Each yaml is one wandb grid sweep: one method on one model and benchmark,
over the compaction ratios and five seeds.

```
sweeps/frontier/<model>/<dataset>/[<budget>/]<method>.yaml

  model    qwen3-4b | llama-3.1-8b | gemma-3-12b | qwen3-30b-a3b
  dataset  quality | longhealth | lca
  budget   optimized | uniform
```

Methods without per-head budgets (`original`, `summarize`, KVzip as
published) sit directly under `<dataset>/`. The component ablations are
`sweeps/ablation_*.yaml` (Qwen3-4B, QuALITY, 50x): candidate-pool width and
reweighted refresh (`pool_refresh`), consistent refresh (`consistent_refresh`),
what the final refit re-solves (`final_refit`, `loo_refit`), the mass term
(`mass_term`), greedy elimination without compensation (`no_compensation`) and
the mild-compaction fallback of KV-Surgeon-fast (`fallback_fast`).
`sweeps/ablation_mass_term_longhealth/` ablates the mass term and the served
multipliers of KV-Surgeon-fast on LongHealth (Qwen3-4B-Instruct-2507, 100x and
50x): `gamma0` 1/0 times `refit` vm/v, with every patient compacted (`multi.yaml`)
or with a raw mask of patients served uncompacted next to the compacted ones
(`mixed.yaml`, `raw_chunks`). Under `refit: v` the multipliers left by the
elimination are served, clamped to the NNLS box; with `gamma0: 0` there are
none and unit multipliers are served. `scripts/aggregate_mass_term_longhealth.py`
splits accuracy by whether the question's patient was compacted or raw.

| File | Method |
|---|---|
| `kv_surgeon.yaml` | KV-Surgeon |
| `kv_surgeon_fast.yaml` | KV-Surgeon-fast |
| `loo.yaml` | LOO |
| `am_hak.yaml` | AM-HAK and AM-HAK-fast (split by `query_config`) |
| `am_omp.yaml` | AM-OMP |
| `am_omp_fast.yaml` | AM-OMP-CS |
| `h2o_calib.yaml`, `kvzip_calib.yaml`, `kvmerger_calib.yaml`, `obcache_calib_fast.yaml` | H2O, KVzip, KVMerger, OBCache in our pipeline |
| `qfilters_pile.yaml` | Q-Filters |
| `streaming.yaml` | StreamingLLM |
| `l2norm.yaml` | L2 key-norm |
| `summarize.yaml` | Summarization |
| `original.yaml` | Full context |
| `kvzip.yaml`, `uniform/{h2o,kvmerger,obcache}.yaml` | Baselines as published (Qwen3-4B, QuALITY) |
| `optimized/{h2o,kvmerger}.yaml` | Baselines with their own queries and our budgets (Qwen3-4B, QuALITY) |
| `optimized/kv_surgeon_fast_offpolicy.yaml` | KV-Surgeon-fast with off-policy queries, control for `ablation_fallback_fast.yaml` |
| `uniform/<method>.yaml`, `uniform/<method>_raw.yaml` | Uniform head budgets, with and without the final refit (Qwen3-4B, QuALITY) |

`target_size` 0.01, 0.02, 0.05, 0.1, 0.2 corresponds to 100x, 50x, 20x,
10x and 5x compaction. Reported numbers are means over seeds 0 to 4.

## Launching

```bash
export WANDB_ENTITY=<entity> WANDB_PROJECT=<project>
wandb sweep sweeps/frontier/<model>/<dataset>/[<budget>/]<method>.yaml
python controller.py --sweep <sweep_id>=<runs>
```

`<runs>` is the grid size. `controller.py` submits one SLURM job per run,
each running `runner.sh`; adapt the configuration at the top of both files
to your cluster. Without SLURM, run `wandb agent <entity>/<project>/<sweep_id>`
on a GPU machine. Results go to the `log_dir` set in each yaml.

## Smoke tests

`sweeps/smoke/` runs every method once on a small input (one context, 10
questions, 50x, seed 0) to check that the pipeline works end to end. Each run
takes a single `method_kwargs` entry; all runs of a sweep share the method
name `smoke`, so tell them apart by `method_kwargs` in the wandb config.

| File | Runs | Covers |
|---|---|---|
| `self_study.yaml` | 15 | KV-Surgeon, LOO, AM-HAK, AM-OMP, AM-OMP-CS, ablation arms |
| `repeat.yaml` | 12 | KV-Surgeon-fast, AM-HAK-fast, ported baselines, query-free baselines, summarization |
| `uniform.yaml` | 6 | Uniform budgets, refit off, KVzip as published |
| `prefill.yaml`, `prefill_last64.yaml` | 2, 1 | H2O, KVMerger and OBCache as published |
| `algorithm_config.yaml` | 1 | Full context and KV-Surgeon from `evaluation/configs/algorithms/kv_surgeon.py` |
| `longhealth_repeat.yaml`, `longhealth_self_study.yaml` | 3, 1 | Chunked LongHealth, including the data download |
| `lca.yaml` | 2 | Long Code Arena (build the data file first) |
| `model_{llama,gemma,qwen30b}.yaml` | 1 each | Full context and KV-Surgeon-fast on the other models |
