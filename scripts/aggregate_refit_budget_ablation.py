#!/usr/bin/env python
"""Refit x head-budget ablation table (Qwen3-4B / QuALITY, 50x) from result jsons.

Runs are classified from their stored hyperparameters:
  refit  on  = refit 'vm'; HAK c2 lsq + beta nnls; KVMerger beta nnls
  refit  off = refit 'none'; HAK c2 direct + beta zero; KVMerger beta zero
  budget on  = precomputed_budget_path set; off = uniform per-head budgets
Rows are (method, query tier); columns are the four refit/budget combinations. Only
full-evaluation-set runs count; the newest json wins per (cell, seed).

    python scripts/aggregate_refit_budget_ablation.py [--latex]
"""
import argparse, collections, glob, json, os, statistics as st
RATIOS = (0.02,)
STD = {'quality': (50, 894)}
LOG_DIRS = ('logs/qa_evaluation/frontier/qwen3-4b/quality', 'logs/qa_evaluation/ablation_loo_refit')
ORDER = ['h2o', 'kvzip', 'kvmerger', 'obcache', 'hak', 'loo', 'kv_surgeon_fast', 'kv_surgeon', 'am_omp_fast', 'am_omp']
TIERS = {'loo': ('ss',), 'kv_surgeon': ('ss',), 'am_omp': ('ss',), 'am_omp_fast': ('ss',),
         'kv_surgeon_fast': ('repeat',), 'hak': ('ss', 'repeat'),
         'h2o': ('repeat', 'context-prefill'), 'kvzip': ('repeat',), 'kvmerger': ('repeat', 'context-prefill'),
         'obcache': ('repeat', 'context-prefill-last64')}   # rows of the ablation
LABEL = {'h2o': 'H2O', 'kvzip': 'KVzip', 'kvmerger': 'KVMerger', 'obcache': 'OBCache',
         'hak': 'AM-HAK', 'loo': 'LOO', 'kv_surgeon_fast': 'KV-Surgeon-fast', 'kv_surgeon': 'KV-Surgeon',
         'am_omp_fast': 'AM-OMP-CS', 'am_omp': 'AM-OMP'}


def classify(h, tier):
    """-> (method, refit-state) or (None, None) for runs outside the ablation."""
    if 'k_choice' in h:
        return ('am_omp_fast' if h.get('k_choice') == 4 else 'am_omp'), 'omp'
    if 'threshold_search_steps' in h:
        return 'kvmerger', {'nnls': 'on', 'zero': 'off'}.get(h.get('beta_method'), '?')
    if 'gamma0' in h and 'refit' in h:
        if h.get('pool_factor') not in (None, 4.0) or h.get('gamma0') not in (None, 1.0):
            return None, None
        if tier != 'ss' and not h.get('on_policy'):
            return None, None   # off-policy fast tier is not part of this table
        m = 'kv_surgeon' if tier == 'ss' else 'kv_surgeon_fast'
        return m, {'vm': 'on', 'none': 'off'}.get(h['refit'], h['refit'])
    if 'refit' in h and 'objective' not in h:
        m = {'loo': 'loo', 'obcache_k': 'obcache'}.get(h.get('score', 'loo'))
        if m is None:
            return None, None
        return m, {'vm': 'on', 'none': 'off'}.get(h['refit'], h['refit'])
    if h.get('algorithm') == 'highest_attention_keys' and 'c2_method' in h:   # not global_* (KVZip as published)
        c2, b = h.get('c2_method'), h.get('beta_method')
        m = {'rms': 'hak', 'mean': 'h2o', 'max': 'kvzip'}.get(h.get('score_method'), 'hak')
        return m, ('on' if (c2, b) == ('lsq', 'nnls') else 'off' if (c2, b) == ('direct', 'zero') else f'{c2}/{b}')
    return None, None


def collect():
    runs = collections.defaultdict(dict)   # (ds, method, tier, refit, budget, rho) -> {seed: acc}
    files = [f for d in LOG_DIRS for f in glob.glob(f'{d}/**/*.json', recursive=True)]
    for f in sorted(files, key=os.path.getmtime):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if 'overall_stats' not in j or 'hyperparameters' not in j:
            continue
        c = j['config']
        if 'Qwen3-4B' not in str(c.get('model_name')) or c.get('dataset_name') not in STD:
            continue
        try:
            rho = float(c['target_size'])
        except Exception:
            continue
        if rho not in RATIOS or c.get('seed') in (None, 'None'):
            continue
        name = next(iter(j['hyperparameters'])); h = j['hyperparameters'][name]; ms = j['overall_stats'][name]
        if (ms.get('num_articles'), ms.get('total_questions')) != STD[c['dataset_name']]:
            continue
        qc = os.path.basename(str(j.get('query_config_file') or 'none'))
        tier = 'ss' if 'ss-plus-repeat' in qc else ('repeat' if qc.startswith('repeat') else qc)
        method, refit = classify(h, tier)
        if method is None:
            continue
        budget = 'on' if h.get('precomputed_budget_path') else 'off'
        acc = ms['overall_accuracy']; acc = acc * 100 if acc <= 1 else acc
        runs[(c['dataset_name'], method, tier, refit, budget, rho)][int(c['seed'])] = acc
    return runs


def cell(runs, ds, m, tier, refit, budget, rho):
    v = list(runs.get((ds, m, tier, refit, budget, rho), {}).values())
    if not v:
        return None
    return (v[0], 0.0, 1) if len(v) == 1 else (st.mean(v), st.stdev(v) / len(v) ** 0.5, len(v))


COLS = [('A refit on / budget off', 'on', 'off'), ('B refit off / budget off', 'off', 'off'),
        ('C both on', 'on', 'on'), ('D refit off / budget on', 'off', 'on')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latex', action='store_true')
    a = ap.parse_args()
    runs = collect()
    fmt = lambda r: '  --  ' if r is None else (f'{r[0]:.1f}+-{r[1]:.1f}' + ('' if r[2] == 5 else f'({r[2]})'))
    for ds in STD:
        rows = sorted({(k[1], k[2]) for k in runs if k[0] == ds and k[2] in TIERS.get(k[1], ())},
                      key=lambda x: (ORDER.index(x[0]) if x[0] in ORDER else 99, x[1]))
        for rho in RATIOS:
            print(f'\n== {ds}  rho={rho:g}  ({int(round(1 / rho))}x)')
            if a.latex:
                print('\\begin{tabular}{ll' + 'r' * len(COLS) + '}\n\\toprule\nmethod & tier & ' + ' & '.join(c[0] for c in COLS) + ' \\\\\n\\midrule')
            else:
                print(f"{'method':<14}{'tier':<16}" + ''.join(f'{c[0]:>26}' for c in COLS))
            for m, tier in rows:
                vals = []
                for _, rf, bd in COLS:
                    r = cell(runs, ds, m, tier, rf, bd, rho)
                    if r is None and m in ('am_omp', 'am_omp_fast') and rf == 'off':
                        vals.append('  n/a ')
                    elif r is None and m in ('am_omp', 'am_omp_fast'):
                        vals.append(fmt(cell(runs, ds, m, tier, 'omp', bd, rho)))
                    else:
                        vals.append(fmt(r))
                if a.latex:
                    print(f"{LABEL.get(m, m)} & {tier} & " + ' & '.join(v.replace('+-', '$\\pm$') for v in vals) + ' \\\\')
                else:
                    print(f'{LABEL.get(m, m):<14}{tier:<16}' + ''.join(f'{v:>26}' for v in vals))
            if a.latex:
                print('\\bottomrule\n\\end{tabular}')


if __name__ == '__main__':
    main()
