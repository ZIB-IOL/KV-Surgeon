#!/usr/bin/env python
"""Mass-term ablation on LongHealth (sweeps/ablation_mass_term_longhealth/*.yaml).

Four arms, gamma0 x refit, in two serving regimes on longhealth5 contexts:
  multi  every patient compacted (the paper's protocol)
  mixed  two complementary raw masks, patients {1,3} raw and {0,2,4} raw, so
         every patient is served once compacted and once raw
Accuracy is split per question by the role of the patient it asks about:
'compacted' or 'raw'. Patients are matched to chunks through the order in
which their questions appear (the LongHealth chunker emits one chunk per
patient in group order, and questions are appended per patient in the same
order); the raw mask comes from the method's stored `raw_chunks`.

Per (arm, regime, role) the mean accuracy over seeds and its standard error
over seeds are reported. The newest json wins per (arm, regime, seed, ratio).

    python scripts/aggregate_mass_term_longhealth.py [--log-dir DIR ...] [--latex]
"""
import argparse
import collections
import glob
import json
import os
import statistics as st

LOG_DIRS = ('logs/qa_evaluation/ablation_mass_term_longhealth',)
ARMS = [
    ((1.0, 'vm'), 'gamma0=1, refit vm (default)'),
    ((0.0, 'vm'), 'gamma0=0, refit vm'),
    ((1.0, 'v'), 'gamma0=1, refit v (elimination multipliers)'),
    ((0.0, 'v'), 'gamma0=0, refit v (unit multipliers)'),
]
REGIMES = ['multi', 'mixed']
ROLES = ['all', 'compacted', 'raw']


def arm_of(h):
    if 'gamma0' not in h or 'refit' not in h:
        return None
    return (float(h['gamma0']), h['refit'])


def load(log_dirs):
    """-> {(arm, ratio, regime, seed): {role: [is_correct, ...]}}, newest json per key."""
    files = []
    for d in log_dirs:
        files += glob.glob(os.path.join(d, '**', '*.json'), recursive=True)
    files.sort(key=os.path.getmtime)
    runs = {}
    for f in files:
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if 'results' not in d:
            continue
        cfg = d.get('config', {})
        seed = cfg.get('seed')
        ratio = cfg.get('target_size')
        for m, h in d.get('hyperparameters', {}).items():
            arm = arm_of(h)
            if arm is None:
                continue
            raw = sorted(int(i) for i in (h.get('raw_chunks') or []))
            regime = 'mixed' if raw else 'multi'
            tallies = collections.defaultdict(list)
            for r in d['results']:
                if r['method'] != m:
                    continue
                qs = r['qa_results']['results_per_question']
                order = []
                for q in qs:
                    pid = q['question_id'].rsplit('_q', 1)[0]
                    if pid not in order:
                        order.append(pid)
                chunk_of = {pid: i for i, pid in enumerate(order)}
                for q in qs:
                    pid = q['question_id'].rsplit('_q', 1)[0]
                    c = bool(q.get('is_correct'))
                    tallies['all'].append(c)
                    tallies['raw' if chunk_of[pid] in raw else 'compacted'].append(c)
            if tallies['all']:
                # later files overwrite earlier ones (sorted by mtime)
                runs[(arm, ratio, regime, seed, tuple(raw))] = dict(tallies)
    return runs


def table(runs, latex=False):
    ratios = sorted({k[1] for k in runs})
    for ratio in ratios:
        print(f"\n=== target_size {ratio} ({1/ratio:.0f}x) ===")
        hdr = f"{'arm':40}"
        for reg in REGIMES:
            for role in ROLES:
                if reg == 'multi' and role != 'all':
                    continue
                hdr += f" {reg+'/'+role:>18}"
        print(hdr)
        rows = []
        for arm, label in ARMS:
            cells = []
            for reg in REGIMES:
                for role in ROLES:
                    if reg == 'multi' and role != 'all':
                        continue
                    # per seed, pool the two raw masks (each question counted once per mask)
                    per_seed = collections.defaultdict(list)
                    for (a, r, g, seed, raw), t in runs.items():
                        if a == arm and r == ratio and g == reg:
                            per_seed[seed] += t.get(role, [])
                    accs = [100 * sum(v) / len(v) for v in per_seed.values() if v]
                    if not accs:
                        cells.append(('-', '', 0, 0))
                        continue
                    mean = st.mean(accs)
                    se = st.stdev(accs) / len(accs) ** 0.5 if len(accs) > 1 else 0.0
                    n = sum(len(v) for v in per_seed.values()) // len(per_seed)
                    cells.append((mean, se, len(accs), n))
            rows.append((label, cells))
            line = f"{label:40}"
            for c in cells:
                if c[0] == '-':
                    line += f" {'-':>18}"
                else:
                    line += f" {c[0]:5.1f}±{c[1]:3.1f} (s={c[2]},n={c[3]:3d})"
            print(line)
        if latex:
            print()
            for label, cells in rows:
                tex = label
                for c in cells:
                    tex += ' & ' + ('--' if c[0] == '-' else f"{c[0]:.1f}\\se{{{c[1]:.1f}}}")
                print(tex + ' \\\\')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', nargs='+', default=list(LOG_DIRS))
    ap.add_argument('--latex', action='store_true')
    args = ap.parse_args()
    runs = load(args.log_dir)
    if not runs:
        print('no runs found under', args.log_dir)
        return
    table(runs, latex=args.latex)


if __name__ == '__main__':
    main()
