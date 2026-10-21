#!/usr/bin/env python
"""Compare KV-Surgeon selection against an exact-refit oracle on the same candidate pool.

Same pool, greedy search and budget t; only the scoring rule changes:
  oracle  greedy backward elimination on the exact loss, refitting the values by
          least squares after every candidate eviction
  kv_surgeon  KVSurgeonCompaction
  loo     top-t of the pool by leave-one-out score
  random  uniform t of the pool

Writes per-head losses and gap_remaining = (L - L_oracle) / (L_loo - L_oracle), and
per-step Spearman correlations of each score with the exact marginal loss along the
oracle's trajectory. Cost grows like t^5, so --max-t skips heads with larger budgets.

Usage:
  python scripts/kv_surgeon_oracle.py --n-articles 2 --max-t 64 --out results/kv_surgeon_oracle.csv
"""
import argparse, csv, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluation.utils  # noqa: F401  (import first: breaks a circular import)
from evaluation.utils import (load_model_and_tokenizer, extract_full_kv_cache,
                              initialize_vllm)
from evaluation.datasets import load_dataset
from evaluation.configs.utils import load_query_config
from compaction.query_generation import QueryGenerator
from compaction.algorithms.loo_evict import LOOEvictCompaction
from compaction.algorithms.kv_surgeon import KVSurgeonCompaction
from compaction.compaction_methods.base import (
    load_budgets_from_json, apply_max_ratio_cap)

KV_SURGEON_KW = dict(pool_factor=4.0, block_size=32, gamma0=1.0, ridge=0.01,
              refresh_interval=0, refresh_mu_floor=0.05, pool_score='loo',
              refit='vm', nnls_iters=2,
              nnls_lower_bound=0.049787068367863944,
              nnls_upper_bound=20.085536923187668,
              c2_solver='lstsq', fallback_threshold=1.0)

SET_FIELDS = ["model", "article", "layer", "head", "T", "Nq", "t", "pool",
              "method", "loss", "loss_rel_O", "gap_remaining",
              "overlap_with_oracle"]
STEP_FIELDS = ["model", "article", "layer", "head", "t", "pool", "step",
               "support", "retained_mass_mean",
               "spearman_kv_surgeon", "spearman_loo", "spearman_quad"]


def exact_refit_losses(Es, m, O, ridge):
    """Exact loss after refit, for every single-eviction candidate.

    Es : (n, s) unnormalised attention exponentials on the current support.
    m  : (n,)   their row sums.  O : (n, dv) full-cache outputs.
    Returns (s,) losses ||O - Ptilde_j C*_j||_F^2, where Ptilde_j renormalises
    over the support minus column j and C*_j is the least-squares refit.
    """
    n, s = Es.shape
    O_n2 = (O * O).sum()
    out = torch.empty(s, dtype=torch.float64, device=Es.device)
    eye = torch.eye(s, device=Es.device)
    # Chunk candidates so the (b, n, s) design stays in memory.
    b = max(1, min(s, int(2e8 // max(n * s, 1))))
    for c0 in range(0, s, b):
        J = torch.arange(c0, min(c0 + b, s), device=Es.device)
        W = Es.unsqueeze(0) / (m.view(1, -1, 1) - Es[:, J].T.unsqueeze(-1))
        bi = torch.arange(len(J), device=Es.device)
        W[bi, :, J] = 0.0                                    # drop column j
        # Row/col j of A and row j of h are zero; a unit pivot keeps the Cholesky
        # defined and the dropped column contributes nothing to `expl`.
        A = W.transpose(1, 2) @ W                            # (b, s, s)
        h = W.transpose(1, 2) @ O                            # (b, s, dv)
        A = A + eye.unsqueeze(0) * ridge
        A[bi, J, J] = 1.0
        L = torch.linalg.cholesky(A)
        expl = (h * torch.cholesky_solve(h, L)).sum(dim=(1, 2))
        out[c0:c0 + len(J)] = (O_n2 - expl).clamp_min(0).double()
        del W, A, h, L
    return out


def support_loss(Es, m, O, ridge):
    """Exact refit loss of the current support itself (no eviction)."""
    P = Es / m.unsqueeze(1)
    A = P.T @ P + torch.eye(P.shape[1], device=P.device) * ridge
    h = P.T @ O
    L = torch.linalg.cholesky(A)
    return float(((O * O).sum() - (h * torch.cholesky_solve(h, L)).sum()).clamp_min(0))


def saliency(Es, s_full, O, gamma0, ridge):
    """KV-Surgeon saliency s_j = ||B_j||^2 / [A^-1]_jj on the current support.

    Uses the full-cache row sums (as kv_surgeon.py does) and targets [O, gamma*1].
    """
    P = Es / s_full.unsqueeze(1)
    gamma = gamma0 * float((O * O).sum(1).mean().sqrt())
    tgt = torch.cat([O, torch.full((O.shape[0], 1), gamma, device=O.device)], 1)
    A = P.T @ P
    A = A + torch.eye(A.shape[0], device=A.device) * (ridge * float(A.diagonal().mean()))
    Ainv = torch.linalg.inv(A)
    B = Ainv @ (P.T @ tgt)
    return (B * B).sum(1) / Ainv.diagonal().clamp_min(1e-30)


def frozen_quad(Es, s_full, O, V):
    """Frozen-quadratic single-deletion score on the current support."""
    P = Es / s_full.unsqueeze(1)
    r = P.unsqueeze(-1) * (V.unsqueeze(0) - O.unsqueeze(1))   # (n, s, dv)
    return (r * r).sum(-1).sum(0)


def spearman(x, y):
    def rk(z):
        o = torch.argsort(z)
        r = torch.empty_like(o)
        r[o] = torch.arange(len(z), device=z.device)
        return r.double()
    a, b = rk(x), rk(y)
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).clamp_min(1e-30)
    return float((a * b).sum() / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--dataset', default='quality')
    ap.add_argument('--n-articles', type=int, default=2)
    ap.add_argument('--query-config', default='ss-plus-repeat')
    ap.add_argument('--budget-path',
                    default='head_budget_optimization/head_budgets/Qwen3-4B/'
                            'optimized_agnostic.json')
    ap.add_argument('--max-ratio-per-head', type=float, default=0.75)
    ap.add_argument('--target-size', type=float, default=0.02)
    ap.add_argument('--max-t', type=int, default=64,
                    help='skip heads whose budget exceeds this (cost ~ t^5)')
    ap.add_argument('--queries-per-pool-col', type=int, default=8,
                    help='N_q = this x pool, so the refit stays overdetermined')
    ap.add_argument('--ridge', type=float, default=1e-6)
    ap.add_argument('--layers', default='')
    ap.add_argument('--heads', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-vllm', action='store_true')
    ap.add_argument('--out', default='results/kv_surgeon_oracle.csv')
    ap.add_argument('--steps-out', default='results/kv_surgeon_oracle_steps.csv')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    model.eval()
    qcfg = load_query_config(args.query_config)
    needs_vllm = any(not sp.is_prefill() or not sp.is_direct()
                     for mc in qcfg.method_configs
                     for sp in (getattr(mc.config, 'conversation_specs', None) or []))
    vllm_model = None
    if needs_vllm and not args.no_vllm:
        vllm_model = initialize_vllm(args.model, seed=args.seed)
    elif needs_vllm:
        raise SystemExit("--no-vllm requires --query-config repeat")
    data = load_dataset(args.dataset)[:args.n_articles]
    scorer = LOOEvictCompaction(score='loo')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fh = open(args.out, 'w', newline=''); wr = csv.DictWriter(fh, SET_FIELDS); wr.writeheader()
    sh = open(args.steps_out, 'w', newline=''); sw = csv.DictWriter(sh, STEP_FIELDS); sw.writeheader()
    n_cov = n_skip = 0

    for a_i, art in enumerate(data):
        t0 = time.time()
        seq_len, pkv, art_idx, fmt, _ = extract_full_kv_cache(
            model, tokenizer, art['article'], device, model_name=args.model)
        gen = QueryGenerator(model=model, tokenizer=tokenizer, config=qcfg,
                             device=device, dtype=pkv[0][0].dtype,
                             vllm_model=vllm_model)
        queries, _, _ = gen.generate_queries(formatted_context=fmt,
                                             past_key_values=pkv, indices=art_idx)
        n_layers, n_heads = len(pkv), pkv[0][0].shape[1]
        idx = list(art_idx); T = len(idx)
        t_uni = max(1, int(args.target_size * T))
        props = load_budgets_from_json(args.budget_path, n_layers, n_heads)
        props = apply_max_ratio_cap(props, args.max_ratio_per_head,
                                    t_uni / T, n_layers * n_heads, verbose=False)
        bmap = {k: int(v * t_uni * n_layers * n_heads)
                for k, v in props.items() if v is not None}
        print(f"[article {a_i}] T={T} Nq_avail={queries.shape[2]} "
              f"({time.time()-t0:.0f}s)", flush=True)

        layers = ([int(x) for x in args.layers.split(',')] if args.layers
                  else range(n_layers))
        heads = ([int(x) for x in args.heads.split(',')] if args.heads
                 else range(n_heads))
        for l in layers:
            for h in heads:
                t = max(1, bmap.get((l, h), t_uni))
                if t > args.max_t:
                    n_skip += 1
                    continue
                K = pkv[l][0][0, h, idx, :].to(torch.float32)
                V = pkv[l][-1][0, h, idx, :].to(torch.float32)
                pool_n = min(T, max(t, int(math.ceil(KV_SURGEON_KW['pool_factor'] * t))))
                nq = min(queries.shape[2], args.queries_per_pool_col * pool_n)
                Q = queries[l, h].to(torch.float32)
                rows = torch.linspace(0, Q.shape[0] - 1, nq, device=Q.device).round().long()
                Q = Q[rows]
                E = (Q @ K.T).mul_(K.shape[1] ** -0.5)
                E.sub_(E.max(1, keepdim=True)[0]).exp_()
                sfull = E.sum(1)
                O = (E @ V) / sfull.unsqueeze(1)

                loo_all = scorer._loo_scores_streamed(E, sfull, O, V)
                pool = torch.topk(loo_all, pool_n, largest=True).indices
                Ep, Vp = E[:, pool], V[pool]

                # Oracle: greedy backward elimination on the exact refit loss.
                alive = torch.ones(pool_n, dtype=torch.bool, device=E.device)
                step = 0
                while int(alive.sum()) > t:
                    S = alive.nonzero(as_tuple=True)[0]
                    Es = Ep[:, S]; m = Es.sum(1)
                    cand = exact_refit_losses(Es, m, O, args.ridge)
                    if step % max(1, (pool_n - t) // 8) == 0 and len(S) > 4:
                        sal = saliency(Es, sfull, O, KV_SURGEON_KW['gamma0'],
                                           KV_SURGEON_KW['ridge'])
                        sw.writerow(dict(
                            model=args.model, article=a_i, layer=l, head=h,
                            t=t, pool=pool_n, step=step, support=int(len(S)),
                            retained_mass_mean=float((m / sfull).mean()),
                            # cand is the loss after evicting j: large = important.
                            spearman_kv_surgeon=spearman(sal, cand.float()),
                            spearman_loo=spearman(loo_all[pool[S]], cand.float()),
                            spearman_quad=spearman(frozen_quad(Es, sfull, O, Vp[S]),
                                                   cand.float())))
                    alive[S[int(torch.argmin(cand))]] = False
                    step += 1
                keep_oracle = pool[alive.nonzero(as_tuple=True)[0]]

                # Comparators on the same pool and budget.
                alg = KVSurgeonCompaction(**KV_SURGEON_KW)
                _, _, _, keep_kvs = alg.compute_compacted_cache(K, V, Q, t)
                keep_kvs = torch.tensor(sorted(keep_kvs), device=E.device)
                keep_loo = pool[torch.topk(loo_all[pool], t, largest=True).indices]
                g = torch.Generator(device='cpu').manual_seed(
                    args.seed * 7919 + l * 131 + h)
                keep_rnd = pool[torch.randperm(pool_n, generator=g)[:t].to(E.device)]

                def L(keep):
                    Es = E[:, keep]
                    return support_loss(Es, Es.sum(1), O, args.ridge)
                losses = {k: L(v) for k, v in
                          (('oracle', keep_oracle), ('kv_surgeon', keep_kvs),
                           ('loo', keep_loo), ('random', keep_rnd))}
                On2 = float((O * O).sum())
                denom = losses['loo'] - losses['oracle']
                so = set(keep_oracle.tolist())
                for name, keep in (('oracle', keep_oracle), ('kv_surgeon', keep_kvs),
                                   ('loo', keep_loo), ('random', keep_rnd)):
                    wr.writerow(dict(
                        model=args.model, article=a_i, layer=l, head=h, T=T,
                        Nq=nq, t=t, pool=pool_n, method=name,
                        loss=losses[name], loss_rel_O=losses[name] / On2,
                        gap_remaining=((losses[name] - losses['oracle']) / denom
                                     if abs(denom) > 1e-12 else ''),
                        overlap_with_oracle=len(so & set(keep.tolist())) / t))
                n_cov += 1
                del E, O, Ep
            fh.flush(); sh.flush()
            print(f"  layer {l}: {n_cov} heads covered, {n_skip} skipped "
                  f"(t>{args.max_t}), {time.time()-t0:.0f}s", flush=True)
        del pkv, queries
        torch.cuda.empty_cache()
    fh.close(); sh.close()
    print(f"wrote {args.out} and {args.steps_out}: {n_cov} heads covered, "
          f"{n_skip} skipped ({n_cov/max(n_cov+n_skip,1):.1%} coverage)")


if __name__ == '__main__':
    main()
