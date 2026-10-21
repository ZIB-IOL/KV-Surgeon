#!/usr/bin/env python
"""Summary tables for the CSV written by scripts/surrogate_check.py.

For each (calib, query_set) combination prints: surrogate/exact ratios vs. fraction
deleted, the small-set anchor and retained-mass predictors, within-head Spearman on
random pool subsets, cross-head Spearman at fixed fraction, and the compensation
losses of kv_surgeon_actual sets. Also writes <input>_perhead.csv.

Usage:
  python scripts/surrogate_check_analysis.py results/surrogate_check.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

Q = [.25, .5, .75]


def band(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return f"{'n/a':>24}"
    q = np.nanquantile(x, Q)
    return f"{q[1]:6.3f} [{q[0]:5.3f}, {q[2]:5.3f}]"


def sp(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or len(np.unique(a)) < 3 or len(np.unique(b)) < 3:
        return np.nan
    return spearmanr(a, b).statistic


def main(path):
    d = pd.read_csv(path)
    if 'calib' not in d:
        d['calib'] = 'slow'; d['query_set'] = 'calib'
    d = d[d.exact > 0].copy()
    for c in ('quad', 'diag', 'loo'):
        d[c + '_r'] = d[c] / d.exact
    d['rel'] = d.exact / d.O_norm2
    d['fr'] = d.frac_deleted.round(2)
    combos = sorted(set(zip(d.calib, d.query_set)))
    print(f"{path}: {len(d)} rows, combos {combos}, "
          f"models {sorted(d.model.unique())}")

    for calib, qs in combos:
        g = d[(d.calib == calib) & (d.query_set == qs)]
        print(f"\n\n######## calib={calib}  query_set={qs}  "
              f"(scores from {calib} queries, exact on {qs} queries)")

        print("\n=== 1. loo_prefix calibration curve, median [IQR] over heads")
        print(f"{'frac':>5} {'n':>5}  {'quad/exact':>24} {'diag/exact':>24} "
              f"{'loo/exact':>24}  {'exact/|O|^2':>11}  {'mu2_mean':>8}")
        for f, gg in g[g.sampler == 'loo_prefix'].groupby('fr'):
            print(f"{f:5.2f} {len(gg):5d}  {band(gg.quad_r):>24} "
                  f"{band(gg.diag_r):>24} {band(gg.loo_r):>24}  "
                  f"{gg.rel.median():11.4f}  "
                  f"{gg.mu2_mean.median() if 'mu2_mean' in gg else float('nan'):8.3f}")
        for smp, gg in g[g.sampler.str.startswith('kv_surgeon_actual')].groupby('sampler'):
            print(f"{smp:>18} n={len(gg)} frac={gg.frac_deleted.median():.3f}  "
                  f"quad/exact {band(gg.quad_r)}  loo/exact {band(gg.loo_r)}  "
                  f"exact/|O|^2 {gg.rel.median():.4f}")

        print("\n=== 2. small-set anchor (loo == exact at size 1 by construction)")
        for k, gg in g[g.sampler == 'pool_small'].groupby('set_size'):
            line = (f"  size={k} n={len(gg):6d}  quad/exact {band(gg.quad_r)}  "
                    f"loo/exact {band(gg.loo_r)}")
            if 'mu2_mean' in gg:
                line += (f"  Spearman(quad/exact, mu2_mean)="
                         f"{sp(gg.quad_r, gg.mu2_mean):.3f}")
            line += (f"  Spearman(quad/exact, (1-a_max)^2)="
                     f"{sp(gg.quad_r, (1 - gg.mass_deleted_max) ** 2):.3f}")
            print(line)
        print("  quad/exact vs retained-mass predictors at the operating point:")
        for smp in sorted(g.sampler.unique()):
            if not (smp.startswith('kv_surgeon_actual') or smp == 'loo_prefix'):
                continue
            gg = g[g.sampler == smp]
            if smp == 'loo_prefix':
                gg = gg[gg.fr >= 0.9]
            line = f"    {smp:>18} n={len(gg)}"
            if 'mu2_mean' in gg:
                line += f"  Sp(qr, mu2_mean)={sp(gg.quad_r, gg.mu2_mean):.3f}"
            line += (f"  Sp(qr, (1-a_mean)^2)={sp(gg.quad_r, (1 - gg.mass_deleted_mean) ** 2):.3f}"
                     f"  Sp(qr, (1-a_max)^2)={sp(gg.quad_r, (1 - gg.mass_deleted_max) ** 2):.3f}")
            print(line)

        print("\n=== 3. ranking on random pool subsets: Spearman vs exact")
        pr = g[g.sampler.str.startswith('pool_random')]
        keys = ['article', 'layer', 'head', 'sampler']
        pooled = pr.groupby(keys).apply(
            lambda x: pd.Series({c: sp(x[c], x.exact) for c in ('quad', 'diag', 'loo')}),
            include_groups=False)
        within = pr.groupby(keys + ['set_size']).apply(
            lambda x: pd.Series({c: sp(x[c], x.exact) for c in ('quad', 'diag', 'loo')})
            if len(x) >= 5 else pd.Series({c: np.nan for c in ('quad', 'diag', 'loo')}),
            include_groups=False).dropna()
        print(f"  {'':6s} {'within-(head,size)  median [IQR]':>34}  frac<0.5   "
              f"{'pooled over sizes (INFLATED)':>28}")
        for c in ('quad', 'loo', 'diag'):
            print(f"  {c:6s} {band(within[c]):>34}  {(within[c] < 0.5).mean():8.3f}   "
                  f"{band(pooled[c]):>28}")

        print("\n=== 4. cross-head Spearman at fixed frac (allocation comparability)")
        rows = []
        for (a, f), gg in g[g.sampler == 'loo_prefix'].groupby(['article', 'fr']):
            rows.append(dict(fr=f, quad=sp(gg.quad, gg.exact), loo=sp(gg.loo, gg.exact),
                             quad_rel=sp(gg.quad / gg.O_norm2, gg.rel),
                             loo_rel=sp(gg.loo / gg.O_norm2, gg.rel)))
        R = pd.DataFrame(rows).groupby('fr').median()
        print("  loo_prefix (per-article median over articles):")
        print(R.to_string(float_format=lambda x: f'{x:.3f}'))
        for smp, gg in g[g.sampler.str.startswith('kv_surgeon_actual')].groupby('sampler'):
            v = [sp(x.quad, x.exact) for _, x in gg.groupby('article')]
            v2 = [sp(x.loo, x.exact) for _, x in gg.groupby('article')]
            print(f"  {smp}: quad {np.nanmedian(v):.3f}  loo {np.nanmedian(v2):.3f}")

        if 'exact_comp' in g and g.exact_comp.notna().any():
            print("\n=== 5. compensation link (kv_surgeon_actual)")
            cc = g[g.exact_comp.notna()].copy()
            for smp, gg in cc.groupby('sampler'):
                raw, comp = gg.exact_raw, gg.exact_comp
                line = (f"  {smp} n={len(gg)}\n"
                        f"    stored/raw exact loss     {band(comp / raw)}   "
                        f"(fraction of the raw loss the refit leaves)\n"
                        f"    decoupled_comp/exact_comp {band(gg.decoupled_comp / comp)}   "
                        f"Sp across heads {sp(gg.decoupled_comp, comp):.3f}\n")
                if gg.exact_elim.notna().any():
                    sw = gg[gg.exact_elim.notna()]
                    line += (f"    elim/raw exact loss       {band(sw.exact_elim / sw.exact_raw)}\n"
                             f"    stored/elim exact loss    {band(sw.exact_comp / sw.exact_elim)}\n"
                             f"    decoupled_elim/exact_elim {band(sw.decoupled_elim / sw.exact_elim)}   "
                             f"Sp {sp(sw.decoupled_elim, sw.exact_elim):.3f}\n")
                line += (f"    stored loss / |O|^2       {band(comp / gg.O_norm2)}   "
                         f"(raw: {band(raw / gg.O_norm2)})")
                print(line)

    # Per-head file.
    ph = d[d.sampler.isin(['loo_prefix']) | d.sampler.str.startswith('kv_surgeon_actual')]
    cols = ['calib', 'query_set', 'model', 'article', 'layer', 'head', 'sampler',
            'fr', 'set_size', 'T', 'exact', 'quad', 'diag', 'loo', 'O_norm2']
    cols += [c for c in ('mu2_mean', 'exact_raw', 'exact_elim', 'exact_comp',
                         'decoupled_elim', 'decoupled_comp') if c in ph]
    out = path.rsplit('.', 1)[0] + '_perhead.csv'
    ph[cols].rename(columns={'fr': 'frac'}).to_csv(out, index=False)
    print(f"\nwrote {out}: {len(ph)} rows")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/surrogate_check.csv')
