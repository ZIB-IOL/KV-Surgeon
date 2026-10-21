"""Diagnose the effect of reweighted refreshes in the KV-Surgeon elimination on real KV.

Runs instrumented KV-Surgeon eliminations on one article (rho = 0.02, 'repeat' queries) with
variants that separate what a refresh changes: the ridge rescale, the
rho_i = 1 / max(mu_i, floor)^2 reweighting, and the consistent re-linearisation
(design P / mu). Writes, per head and variant, the exact loss before and after
the final refit (calibration and held-out queries) relative to the frozen elimination, the
support overlap with the frozen elimination and refresh statistics.

    python scripts/refresh_diag.py --layers 0,6,12,18,24,30,35
"""
import argparse, math, os, sys
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluation.utils import load_model_and_tokenizer, extract_full_kv_cache
from evaluation.datasets import load_dataset
from evaluation.configs.utils import load_query_config
from compaction.query_generation import QueryGenerator
from compaction.algorithms.loo_evict import LOOEvictCompaction
from compaction.algorithms.kv_surgeon import KVSurgeonCompaction
from compaction.compaction_methods.base import load_budgets_from_json, apply_max_ratio_cap
from surrogate_check import attention, subsample, refit_losses, head_budget, heldout_queries, KV_SURGEON_KW, BUDGET_DIRS
from compaction.query_generation.self_study import SelfStudyQueryGenerator


class Instrumented(KVSurgeonCompaction):
    def __init__(self, *a, uniform_rho=False, fixed_ridge=False, **kw):
        super().__init__(*a, **kw)
        self.uniform_rho, self.fixed_ridge = uniform_rho, fixed_ridge
        self.log, self._lam_abs = [], None

    def _ridge_fit(self, X, Y, row_weights=None):
        rec = {}
        if row_weights is not None:
            rec['rho_ratio'] = float(row_weights.max() / row_weights.min())
            rec['frac_nonmax'] = float((row_weights < row_weights.max() * (1 - 1e-6)).float().mean())
            if self.uniform_rho:
                row_weights = None
        dm = float((X * X).sum(dim=0).mean())
        if self._lam_abs is None:
            self._lam_abs = self.ridge * dm
            rec['kind'] = 'init'
        else:
            rec['kind'] = 'refresh'
            rec['ridge_ratio'] = self.ridge * dm / self._lam_abs
        self.log.append(rec)
        if self.fixed_ridge and rec['kind'] == 'refresh':
            if row_weights is not None:
                sw = row_weights.sqrt().unsqueeze(1); X = X * sw; Y = Y * sw
            A = X.T @ X; A = 0.5 * (A + A.T); A.diagonal().add_(self._lam_abs)
            L = torch.linalg.cholesky(A); Ainv = torch.cholesky_inverse(L)
            return Ainv, Ainv @ (X.T @ Y)
        return super()._ridge_fit(X, Y, row_weights)


VARIANTS = {
    'frozen':            dict(),
    'rw64_f0.05':        dict(refresh_interval=64, refresh_mu_floor=0.05),
    'rw64_f1':           dict(refresh_interval=64, refresh_mu_floor=1.0),
    'refit64_newridge':  dict(refresh_interval=64, refresh_mu_floor=1.0, uniform_rho=True),
    'refit64_fixridge':  dict(refresh_interval=64, refresh_mu_floor=1.0, uniform_rho=True, fixed_ridge=True),
    'rw64_f0.05_fixrdg': dict(refresh_interval=64, refresh_mu_floor=0.05, fixed_ridge=True),
    'cons64':            dict(refresh_interval=64, refresh_mode='consistent'),
    'cons256':           dict(refresh_interval=256, refresh_mode='consistent'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--article', type=int, default=0)
    ap.add_argument('--layers', default='0,6,12,18,24,30,35')
    ap.add_argument('--heads', default='')
    ap.add_argument('--rho', type=float, default=0.02)
    ap.add_argument('--max-queries', type=int, default=4096)
    ap.add_argument('--out', default='results/refresh_diag.csv')
    args = ap.parse_args()
    device = 'cuda'
    torch.manual_seed(0)
    model, tok = load_model_and_tokenizer(args.model, device); model.eval()
    art = load_dataset('quality')[args.article]
    seq_len, pkv, art_idx, fmt_ctx, _ = extract_full_kv_cache(model, tok, art['article'], device, model_name=args.model)
    gen = QueryGenerator(model=model, tokenizer=tok, config=load_query_config('repeat'),
                         device=device, dtype=pkv[0][0].dtype)
    Qall, _, _ = gen.generate_queries(formatted_context=fmt_ctx, past_key_values=pkv, indices=art_idx)
    cfg = model.config
    head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hpk = max(1, cfg.num_attention_heads // n_kv)
    ss_gen = SelfStudyQueryGenerator(model, tok, config=None, device=device)
    Hq = heldout_queries(ss_gen, tok, art, fmt_ctx, pkv, args.model, head_dim, n_kv, hpk, device)
    print('heldout queries', None if Hq is None else Hq.shape, flush=True)
    idx = list(art_idx); T = len(idx)
    n_layers, n_heads = len(pkv), pkv[0][0].shape[1]
    t_uni = max(1, int(args.rho * T))
    props = load_budgets_from_json(f'head_budget_optimization/head_budgets/{BUDGET_DIRS[args.model]}/optimized_agnostic.json', n_layers, n_heads)
    props = apply_max_ratio_cap(props, 0.75, t_uni / T, n_layers * n_heads)
    bmap = {k: int(v * t_uni * n_layers * n_heads) for k, v in props.items() if v is not None}
    scorer = LOOEvictCompaction(score='loo')
    layers = [int(x) for x in args.layers.split(',')]
    heads = [int(x) for x in args.heads.split(',')] if args.heads else range(n_heads)
    print(f"T={T} Nq={Qall.shape[2]} t_uni={t_uni}", flush=True)
    rows = []
    for l in layers:
        for h in heads:
            K32 = pkv[l][0][0, h, idx].float(); V32 = pkv[l][-1][0, h, idx].float()
            Q = subsample(Qall[l, h].to(device, torch.float32), args.max_queries)
            E, s, o = attention(K32, V32, Q, None)
            Eh, sh, oh = attention(K32, V32, Hq[l, h].to(device, torch.float32), None)
            loo_s = scorer._loo_scores_streamed(E, s, o, V32)
            t = min(T, max(1, head_budget(bmap, l, h, T, t_uni)))
            pool = min(T, max(t, int(math.ceil(4.0 * t))))
            if pool <= t:
                continue
            pool_idx = torch.topk(loo_s, pool, largest=True).indices
            Ppool = E[:, pool_idx] / s.unsqueeze(1)
            gamma = float(o.square().sum(dim=1).mean().sqrt())
            res = {}
            for name, over in VARIANTS.items():
                kw = dict(KV_SURGEON_KW); kw.update({k: v for k, v in over.items() if k not in ('uniform_rho', 'fixed_ridge')})
                alg = Instrumented(uniform_rho=over.get('uniform_rho', False), fixed_ridge=over.get('fixed_ridge', False), **kw)
                keep, B = alg._eliminate(Ppool, o, gamma, t)
                top = pool_idx[keep]; U = B[keep, :-1]; w = B[keep, -1] / gamma
                beta, C2 = alg._final_refit('vm', E, s, o, top, K32, V32, top.cpu().tolist())
                cl = refit_losses(E, s, o, top, beta, C2, U, w, gamma)
                ch = refit_losses(Eh, sh, oh, top, beta, C2, U, w, gamma)
                ref = alg.log[1:]
                rank = torch.empty(pool, device=device); rank[torch.argsort(loo_s[pool_idx], descending=True)] = torch.arange(pool, device=device, dtype=torch.float32)
                res[name] = dict(exact_elim=cl['exact_elim'], exact_comp=cl['exact_comp'], top=set(top.tolist()),
                                 heldout_comp=ch['exact_comp'], heldout_elim=ch['exact_elim'], mean_rank=float(rank[keep].mean() / pool),
                                 n_refresh=len(ref),
                                 ridge_ratio=(sum(r.get('ridge_ratio', 1) for r in ref) / len(ref)) if ref else 1.0,
                                 frac_nonmax=(sum(r.get('frac_nonmax', 0) for r in ref) / len(ref)) if ref else 0.0,
                                 rho_ratio=(sum(r.get('rho_ratio', 1) for r in ref) / len(ref)) if ref else 1.0)
            f = res['frozen']
            for name, r in res.items():
                jac = len(r['top'] & f['top']) / len(r['top'] | f['top'])
                rows.append((l, h, T, t, pool, name, r['exact_elim'], r['exact_comp'], f['exact_elim'], f['exact_comp'],
                             jac, r['n_refresh'], r['ridge_ratio'], r['frac_nonmax'], r['rho_ratio'],
                             r['heldout_elim'], r['heldout_comp'], f['heldout_elim'], f['heldout_comp'], r['mean_rank']))
            print(f"L{l}H{h} t={t} " + " ".join(f"{n}:{r['exact_comp']/f['exact_comp']:.3f}/{r['heldout_comp']/f['heldout_comp']:.3f}" for n, r in res.items()), flush=True)
    import csv
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['layer', 'head', 'T', 't', 'pool', 'variant', 'exact_elim', 'exact_comp', 'frozen_elim', 'frozen_comp',
                    'jaccard_vs_frozen', 'n_refresh', 'ridge_ratio', 'frac_nonmax_weight', 'rho_max_over_min',
                    'heldout_elim', 'heldout_comp', 'frozen_heldout_elim', 'frozen_heldout_comp', 'mean_pool_rank'])
        w.writerows(rows)
    import numpy as np
    print(f"\n{'variant':<20}{'comp/frozen med':>16}{'<1 frac':>9}{'elim/frozen med':>18}{'jaccard':>9}{'ridge x':>9}{'nonmax w':>10}{'rho ratio':>10}{'heldout comp/frozen':>20}{'<1':>6}{'pool rank':>10}")
    for name in VARIANTS:
        R = [r for r in rows if r[5] == name]
        c = np.array([r[7] / r[9] for r in R]); sw = np.array([r[6] / r[8] for r in R])
        print(f"{name:<20}{np.median(c):>16.3f}{(c < 1).mean():>9.2f}{np.median(sw):>18.3f}{np.median([r[10] for r in R]):>9.3f}"
              f"{np.median([r[12] for r in R]):>9.2f}{np.median([r[13] for r in R]):>10.2f}{np.median([r[14] for r in R]):>10.1f}"
              f"{np.median([r[16]/r[18] for r in R]):>20.3f}{np.mean([r[16]/r[18] < 1 for r in R]):>6.2f}{np.median([r[19] for r in R]):>10.3f}")


if __name__ == '__main__':
    main()
