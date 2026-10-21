"""Aggregate the selection-objective ablations (Qwen3-4B, QuALITY, rho=0.02, articles 0-24).

Arms: greedy elimination on the surrogate or exact objective without compensation,
KV-Surgeon with consistent refreshes, and references (KV-Surgeon at pool factor 4, LOO
and KV-Surgeon frontier runs). Prints mean +- std over seeds and seed-paired differences
to the frozen KV-Surgeon arm.

    python scripts/aggregate_selection_ablations.py
"""
import glob, json
from collections import defaultdict
import numpy as np

N_ART = 25

def acc_first_articles(j, n=N_ART):
    c = t = 0; ct = 0.0
    for r in j['results']:
        if r['article_idx'] >= n: continue
        qa = r['qa_results']
        c += qa['correct_answers']
        t += qa['total_questions']
        ct += r['compaction_time']
    return 100.0 * c / t, ct / n

def hp(j):
    return list(j['hyperparameters'].values())[0]

def label(d, h):
    a = h['algorithm']
    tag = {'ablation_pool_refresh': '[pool] ', 'ablation_consistent_refresh': '[cons] '}.get(d, '')
    if d.startswith('frontier'):
        return f"frontier {a}"
    if a == 'greedy_evict':
        return f"greedy_evict objective={h['objective']}"
    if a == 'loo_evict':
        return f"loo_evict score={h['score']}"
    if a == 'kv_surgeon':
        mode = h.get('refresh_mode', 'reweighted')
        return tag + f"kv_surgeon pf={h['pool_factor']:g} refresh={h.get('refresh_interval', 0)} mode={mode}"
    return a

DIRS = ['ablation_no_compensation', 'ablation_consistent_refresh', 'ablation_pool_refresh',
        'frontier/qwen3-4b/quality/optimized/loo', 'frontier/qwen3-4b/quality/optimized/kv_surgeon']
arms = defaultdict(dict)   # label -> seed -> (acc, ctime)
for d in DIRS:
    for f in glob.glob(f'logs/qa_evaluation/{d}/target0.02_*.json'):
        j = json.load(open(f)); h = hp(j)
        if d == 'ablation_pool_refresh' and h['pool_factor'] != 4: continue
        if j['config']['n_articles'] < N_ART: continue
        try:
            a, ct = acc_first_articles(j)
        except Exception as e:
            print('skip', f, e); continue
        arms[label(d, h)][j['config']['seed']] = (a, ct)

REF = '[pool] kv_surgeon pf=4 refresh=0 mode=reweighted'
ref = arms[REF]
PAIRS = [('greedy_evict objective=exact', 'greedy_evict objective=surrogate'),
         ('greedy_evict objective=surrogate', REF),
         ('greedy_evict objective=exact', REF),
         ('frontier loo_evict', 'greedy_evict objective=surrogate'),
         ('[cons] kv_surgeon pf=4 refresh=64 mode=consistent', '[pool] kv_surgeon pf=4 refresh=64 mode=reweighted'),
         ('[cons] kv_surgeon pf=4 refresh=256 mode=consistent', '[pool] kv_surgeon pf=4 refresh=256 mode=reweighted')]
print(f"{'arm':<75} {'n':>2} {'acc':>13} {'d vs frozen(paired)':>22} {'ctime/art':>9}")
for k in sorted(arms):
    s = arms[k]; acc = np.array([v[0] for v in s.values()]); ct = np.mean([v[1] for v in s.values()])
    common = sorted(set(s) & set(ref))
    diff = np.array([s[i][0] - ref[i][0] for i in common])
    dtxt = f"{diff.mean():+.2f} ± {diff.std(ddof=1)/np.sqrt(len(diff)):.2f} (n={len(diff)})" if len(diff) > 1 else 'n/a'
    print(f"{k:<75} {len(acc):>2} {acc.mean():6.2f} ± {acc.std(ddof=1):4.2f} {dtxt:>22} {ct:8.1f}s")

print()
for a, b in PAIRS:
    if a not in arms or b not in arms: print('missing', a, b); continue
    common = sorted(set(arms[a]) & set(arms[b]))
    d = np.array([arms[a][i][0] - arms[b][i][0] for i in common])
    print(f"{a}\n  - {b}\n  = {d.mean():+.2f} ± {d.std(ddof=1)/np.sqrt(len(d)):.2f} (paired, n={len(d)}; per-seed {np.round(d,2).tolist()})")
