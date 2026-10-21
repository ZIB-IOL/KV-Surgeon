#!/usr/bin/env python
"""Compare the deletion-loss surrogates against the exact loss on real deletion sets.

Per head and deletion set D (values fixed), with a_i = sum_{j in D} p_ij and
r_i = sum_{j in D} p_ij v_j - a_i o_i:

    quad  = sum_i ||r_i||^2                 exact = sum_i ||r_i||^2 / (1 - a_i)^2
    diag  = sum_{j in D} sum_i p_ij^2 ||v_j - o_i||^2
    loo   = the same with each term divided by (1 - p_ij)^2 + eps

Scores are computed from slow (--query-config) or fast (--fast-config) calibration
queries (column `calib`); `exact` from the slow queries or from teacher-forced
benchmark questions (column `query_set` = calib / heldout). Only compare values
within one row.

Samplers: loo_prefix (the k lowest-LOO positions), pool_random and pool_small (random
subsets of the KV-Surgeon candidate pool), kv_surgeon_actual (the set KV-Surgeon deletes). kv_surgeon_actual rows
also report the exact and decoupled losses of the elimination coefficients, the stored
refit, one reweighted refresh and one consistent refresh.

Usage:
  python scripts/surrogate_check.py --n-articles 5 --out results/surrogate_check.csv
"""
import argparse, csv, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils import (load_model_and_tokenizer, extract_full_kv_cache,
                             initialize_vllm, format_question)
from evaluation.datasets import load_dataset
from evaluation.configs.utils import load_query_config
from compaction.query_generation import QueryGenerator
from compaction.query_generation.self_study import SelfStudyQueryGenerator
from compaction.algorithms.loo_evict import LOOEvictCompaction
from compaction.algorithms.kv_surgeon import KVSurgeonCompaction
from compaction.compaction_methods.base import (
    load_budgets_from_json, apply_max_ratio_cap)

# KV-Surgeon configuration used in the experiments (same for all models).
KV_SURGEON_KW = dict(pool_factor=4.0, block_size=32, gamma0=1.0, ridge=0.01,
              refresh_interval=0, refresh_mu_floor=0.05, pool_score='loo',
              refit='vm', nnls_iters=2,
              nnls_lower_bound=0.049787068367863944,
              nnls_upper_bound=20.085536923187668,
              c2_solver='lstsq', fallback_threshold=1.0)

BUDGET_DIRS = {
    'Qwen/Qwen3-4B': 'Qwen3-4B',
    'Qwen/Qwen3-4B-Instruct-2507': 'Qwen3-4B-Instruct-2507',
    'Qwen/Qwen3-30B-A3B': 'Qwen3-30B-A3B',
    'meta-llama/Llama-3.1-8B-Instruct': 'Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it': 'gemma-3-12b-it',
}

FRACS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
POOL_FRACS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
RHOS = [0.01, 0.02, 0.05]
FIELDS = ["model", "article", "layer", "head", "sampler", "set_size",
          "frac_deleted", "T", "Nq", "Nq_exact", "calib", "query_set",
          "exact", "quad", "diag", "loo",
          "O_norm2", "mass_deleted_max", "mass_deleted_mean", "mu2_mean",
          "exact_raw", "exact_elim", "exact_comp",
          "decoupled_elim", "decoupled_comp", "gamma",
          "exact_refresh", "decoupled_refresh", "wobj_elim", "wobj_refresh",
          "mu_floor_frac", "exact_cons", "decoupled_cons"]
COMBOS = [('slow', 'calib'), ('slow', 'heldout'),
          ('fast', 'calib'), ('fast', 'heldout')]


def attention(K32, V32, Q32, bias):
    """E (max-shifted exponentials), s (row sums), o (full outputs)."""
    E = (Q32 @ K32.T).mul_(K32.shape[1] ** -0.5)
    if bias is not None:
        E.add_(torch.broadcast_to(bias.to(torch.float32), E.shape))
    E.sub_(E.max(dim=1, keepdim=True)[0]).exp_()
    s = E.sum(dim=1)
    o = (E @ V32) / s.unsqueeze(1)
    return E, s, o


def eval_sets(E, s, o, V32, masks, q_chunk, set_chunk):
    """quad, exact, a_max, a_mean, mu2_mean per deletion set, streamed.

    masks : (T, S) float 0/1 indicator columns.
    """
    T, S = masks.shape
    dv = V32.shape[1]
    n = E.shape[0]
    quad = torch.zeros(S, dtype=torch.float64, device=E.device)
    exact = torch.zeros(S, dtype=torch.float64, device=E.device)
    a_max = torch.zeros(S, dtype=torch.float32, device=E.device)
    a_sum = torch.zeros(S, dtype=torch.float64, device=E.device)
    mu2_sum = torch.zeros(S, dtype=torch.float64, device=E.device)
    for c0 in range(0, S, set_chunk):
        cs = slice(c0, min(c0 + set_chunk, S))
        M = masks[:, cs]                                        # (T, b)
        b = M.shape[1]
        W = (M.unsqueeze(-1) * V32.unsqueeze(1)).reshape(T, b * dv)
        for r0 in range(0, n, q_chunk):
            rows = slice(r0, min(r0 + q_chunk, n))
            P = E[rows] / s[rows].unsqueeze(1)                  # (c, T)
            A = P @ M                                           # (c, b)
            B = (P @ W).view(-1, b, dv)                         # (c, b, dv)
            r = B - A.unsqueeze(-1) * o[rows].unsqueeze(1)      # (c, b, dv)
            rn = (r * r).sum(-1)                                # (c, b)
            mu = (1.0 - A).clamp_min(1e-6)
            quad[cs] += rn.sum(0).double()
            exact[cs] += (rn / (mu * mu)).sum(0).double()
            a_max[cs] = torch.maximum(a_max[cs], A.max(dim=0).values)
            a_sum[cs] += A.sum(0).double()
            mu2_sum[cs] += (mu * mu).sum(0).double()
        del W
    return quad, exact, a_max, a_sum / n, mu2_sum / n


def refresh_fit(alg, E, s, o, S, w_sw, gamma):
    """One reweighted refresh on support S, as in kv_surgeon._eliminate.

    Returns (U, w, mu_floor_frac) of the 1/mu^2-weighted fit and (U, w) of the
    consistent refresh.
    """
    P = E[:, S] / s.unsqueeze(1)
    mu = P @ w_sw
    floor_frac = float((mu < alg.refresh_mu_floor).float().mean())
    rho = 1.0 / mu.clamp_min(alg.refresh_mu_floor).square()
    rho = rho / rho.mean().clamp_min(1e-30)
    Y = torch.cat([o, o.new_full((o.shape[0], 1), gamma)], dim=1)
    _, B = alg._ridge_fit(P, Y, row_weights=rho)
    # Consistent refresh: design P / mu, targets [O, gamma].
    muc = mu.clamp_min(alg.refresh_mu_floor)
    Xc = P / muc.unsqueeze(1)
    _, Bc = alg._ridge_fit(Xc, Y)
    return (B[:, :-1], B[:, -1] / gamma, floor_frac,
            Bc[:, :-1], Bc[:, -1] / gamma)


def weighted_objective(E, s, o, S, U, w, gamma, w_ref, mu_floor, ridge):
    """The 1/mu^2-weighted decoupled objective (weights from w_ref), including the
    ridge term _ridge_fit adds, so the refreshed fit is its exact minimiser."""
    P = E[:, S] / s.unsqueeze(1)
    rho = 1.0 / (P @ w_ref).clamp_min(mu_floor).square()
    rho = rho / rho.mean().clamp_min(1e-30)
    Aw = (P * rho.unsqueeze(1)).T @ P
    lam = ridge * float(Aw.diagonal().mean())
    B = torch.cat([U, (gamma * w).unsqueeze(1)], dim=1)
    return float((rho * ((o - P @ U) ** 2).sum(dim=1)).sum()
                 + gamma ** 2 * (rho * (1.0 - P @ w) ** 2).sum()
                 + lam * (B * B).sum())


def refit_losses(E, s, o, S, beta, C2, U_sw, w_sw, gamma):
    """Exact and decoupled losses of stored / elimination coefficients on (E, s, o).

    S : (t,) kept positions. beta, C2 : the stored refit. U_sw, w_sw : the elimination
    coefficients split into value rows and multipliers (None if there was no elimination).
    """
    P = E[:, S] / s.unsqueeze(1)                                 # (n, t)
    w = beta.to(torch.float32).exp()
    Xw = E[:, S] * w.unsqueeze(0)
    X = Xw / Xw.sum(dim=1, keepdim=True).clamp_min(1e-38)
    C2f = C2.to(torch.float32)
    exact_comp = ((X @ C2f - o) ** 2).sum().double()
    U = w.unsqueeze(1) * C2f
    dec_comp = (((o - P @ U) ** 2).sum()
                + gamma ** 2 * ((1.0 - P @ w) ** 2).sum()).double()
    out = dict(exact_comp=float(exact_comp), decoupled_comp=float(dec_comp),
               exact_elim='', decoupled_elim='',
               exact_refresh='', decoupled_refresh='', wobj_elim='', wobj_refresh='')
    if U_sw is not None:
        Pw = (P @ w_sw).clamp_min(1e-6)
        Ohat = (P @ U_sw) / Pw.unsqueeze(1)
        out['exact_elim'] = float(((Ohat - o) ** 2).sum().double())
        out['decoupled_elim'] = float((((o - P @ U_sw) ** 2).sum()
            + gamma ** 2 * ((1.0 - P @ w_sw) ** 2).sum()).double())
    return out


def subsample(Q, max_q):
    if 0 < max_q < Q.shape[0]:
        rows = torch.linspace(0, Q.shape[0] - 1, max_q,
                              device=Q.device).round().long()
        return Q[rows]
    return Q


def heldout_queries(ss_gen, tokenizer, art, fmt_ctx, pkv, model_name,
                    head_dim, n_kv, hpk, device):
    """Teacher-forced benchmark-question queries, in KV-head space."""
    n_ctx = tokenizer(fmt_ctx, return_tensors='pt',
                      add_special_tokens=False).input_ids.shape[1]
    outs = []
    for q in art['questions']:
        prompt = format_question(tokenizer, q['question'], q.get('options'),
                                 model_name)
        if prompt.startswith('<bos>'):
            prompt = prompt[len('<bos>'):]
        gold = q.get('gold_label')
        answer = chr(64 + int(gold)) if gold else ''
        ids = tokenizer(fmt_ctx + prompt + answer, return_tensors='pt',
                        add_special_tokens=False).input_ids.to(device)
        qv = ss_gen._extract_query_vectors_from_prefill(
            input_ids=ids[:, n_ctx:], past_key_values=pkv,
            head_dim=head_dim, start_token_idx=0)
        if qv is not None:
            outs.append(qv.cpu())
    if not outs:
        return None
    Q = torch.cat(outs, dim=2)
    return QueryGenerator._attention_to_kv(Q, n_kv, hpk)


def head_budget(bmap, layer, head, T, uniform_t):
    return bmap.get((layer, head), uniform_t) if bmap else uniform_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-4B')
    ap.add_argument('--dataset', default='quality')
    ap.add_argument('--n-articles', type=int, default=5)
    ap.add_argument('--start-article', type=int, default=0)
    ap.add_argument('--query-config', default='ss-plus-repeat',
                    help='slow calibration queries (sets are chosen on these)')
    ap.add_argument('--fast-config', default='repeat',
                    help="cheap calibration queries; '' disables the fast rows")
    ap.add_argument('--no-heldout', action='store_true',
                    help='skip the teacher-forced benchmark-question queries')
    ap.add_argument('--budget-path', default=None,
                    help='per-head budgets; default derived from --model, '
                         'uniform if the file is missing')
    ap.add_argument('--max-ratio-per-head', type=float, default=0.75)
    ap.add_argument('--max-queries', type=int, default=8192,
                    help='even-stride query subsample per query set (0 = all)')
    ap.add_argument('--layers', default='', help='comma list, default all')
    ap.add_argument('--heads', default='', help='comma list, default all')
    ap.add_argument('--draws', type=int, default=8, help='random sets per size')
    ap.add_argument('--q-chunk', type=int, default=512)
    ap.add_argument('--set-chunk', type=int, default=8)
    ap.add_argument('--no-kv-surgeon', action='store_true', help='skip kv_surgeon_actual')
    ap.add_argument('--only-kv-surgeon', action='store_true',
                    help='emit kv_surgeon_actual rows only (fast rerun for the '
                         'compensation / refresh columns)')
    ap.add_argument('--no-vllm', action='store_true',
                    help='refuse to start vLLM (only with --query-config repeat)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/surrogate_check.csv')
    args = ap.parse_args()

    if args.budget_path is None:
        sub = BUDGET_DIRS.get(args.model)
        args.budget_path = (f'head_budget_optimization/head_budgets/{sub}/'
                            'optimized_agnostic.json') if sub else ''
    if not (args.budget_path and os.path.exists(args.budget_path)):
        print(f"WARNING: no budget file at '{args.budget_path}', "
              "using uniform per-head budgets", flush=True)
        args.budget_path = ''

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
        print(f"query config '{args.query_config}' needs generation; "
              f"starting vLLM", flush=True)
        vllm_model = initialize_vllm(args.model, seed=args.seed)
    elif needs_vllm:
        raise SystemExit(
            f"--no-vllm given but query config '{args.query_config}' has "
            "generated specs; use --query-config repeat instead")
    fast_cfg = load_query_config(args.fast_config) if args.fast_config else None
    data = load_dataset(args.dataset)[args.start_article:
                                      args.start_article + args.n_articles]
    scorer = LOOEvictCompaction(score='loo')
    diag_scorer = LOOEvictCompaction(score='truncated', truncation_order=0)
    # Extraction-only self-study generator for the held-out prefill hook.
    ss_gen = SelfStudyQueryGenerator(model, tokenizer, config=None,
                                     device=device)
    cfg = model.config
    head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    n_kv = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
    hpk = max(1, cfg.num_attention_heads // n_kv)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fh = open(args.out, 'w', newline='')
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    wr.writeheader()
    n_rows = 0

    for a_i, art in enumerate(data):
        a_id = args.start_article + a_i
        t0 = time.time()
        seq_len, pkv, art_idx, fmt_ctx, _ = extract_full_kv_cache(
            model, tokenizer, art['article'], device, model_name=args.model)
        dt = pkv[0][0].dtype
        gen = QueryGenerator(model=model, tokenizer=tokenizer, config=qcfg,
                             device=device, dtype=dt, vllm_model=vllm_model)
        qsets = {}
        qsets['slow'], _, _ = gen.generate_queries(
            formatted_context=fmt_ctx, past_key_values=pkv, indices=art_idx)
        if fast_cfg is not None:
            fgen = QueryGenerator(model=model, tokenizer=tokenizer,
                                  config=fast_cfg, device=device, dtype=dt)
            qsets['fast'], _, _ = fgen.generate_queries(
                formatted_context=fmt_ctx, past_key_values=pkv, indices=art_idx)
        if not args.no_heldout:
            hq = heldout_queries(ss_gen, tokenizer, art, fmt_ctx, pkv,
                                 args.model, head_dim, n_kv, hpk, device)
            if hq is not None:
                qsets['heldout'] = hq
        combos = [(c, q) for c, q in COMBOS
                  if c in qsets and (q == 'calib' or q in qsets)]
        n_layers = len(pkv)
        n_heads = pkv[0][0].shape[1]
        idx = list(art_idx)
        T = len(idx)
        print(f"[article {a_id}] T={T} seq_len={seq_len} "
              + " ".join(f"Nq_{k}={v.shape[2]}" for k, v in qsets.items())
              + f" ({time.time()-t0:.0f}s)", flush=True)

        layers = ([int(x) for x in args.layers.split(',')] if args.layers
                  else range(n_layers))
        heads = ([int(x) for x in args.heads.split(',')] if args.heads
                 else range(n_heads))

        bmaps = {}
        for rho in RHOS:
            t_uni = max(1, int(rho * T))
            if not args.budget_path:
                bmaps[rho] = ({}, t_uni)
                continue
            props = load_budgets_from_json(args.budget_path, n_layers, n_heads)
            props = apply_max_ratio_cap(props, args.max_ratio_per_head,
                                        t_uni / T, n_layers * n_heads)
            bmaps[rho] = ({k: int(v * t_uni * n_layers * n_heads)
                           for k, v in props.items() if v is not None}, t_uni)

        for l in layers:
            lay = pkv[l]
            bias_l = lay[1] if (isinstance(lay, (tuple, list))
                                and len(lay) == 3) else None
            if bias_l is not None and bias_l.dim() == 2:
                bias_l = bias_l.unsqueeze(0)
            for h in heads:
                K32 = pkv[l][0][0, h, idx, :].to(torch.float32)
                V32 = pkv[l][-1][0, h, idx, :].to(torch.float32)
                bias_h = bias_l[0, h, idx] if bias_l is not None else None
                att = {}
                for name, Qall in qsets.items():
                    Q = subsample(Qall[l, h].to(K32.device, torch.float32),
                                  args.max_queries)
                    att[name] = attention(K32, V32, Q, bias_h)
                E, s, o = att['slow']
                Nq = E.shape[0]
                O_norm2 = {name: float((oo * oo).sum()) for name, (_, _, oo) in att.items()}
                scores = {}
                for name in ('slow', 'fast'):
                    if name in att:
                        Ee, ss, oo = att[name]
                        scores[name] = (scorer._loo_scores_streamed(Ee, ss, oo, V32),
                                        diag_scorer._truncated_scores_streamed(Ee, ss, oo, V32))
                loo_s = scores['slow'][0]
                order = torch.argsort(loo_s)                    # ascending

                sets, meta = [], []
                for f in ([] if args.only_kv_surgeon else FRACS):       # loo_prefix
                    k = max(1, int(round(f * T)))
                    if k >= T:
                        continue
                    sets.append(order[:k]); meta.append(('loo_prefix', k))
                gcpu = torch.Generator(device='cpu').manual_seed(
                    args.seed * 1000003 + l * 1009 + h)
                pools = {}
                for rho in RHOS:
                    bmap, t_uni = bmaps[rho]
                    t = max(1, head_budget(bmap, l, h, T, t_uni))
                    pool = min(T, max(t, int(math.ceil(4.0 * t))))
                    pools[rho] = torch.topk(loo_s, pool, largest=True).indices
                    for f in ([] if args.only_kv_surgeon else POOL_FRACS):  # pool_random
                        k = max(1, int(round(f * pool)))
                        for _ in range(args.draws):
                            perm = torch.randperm(pool, generator=gcpu)[:k]
                            sets.append(pools[rho][perm.to(loo_s.device)])
                            meta.append((f'pool_random_rho{rho}', k))
                for k in (() if args.only_kv_surgeon else (1, 2, 4, 8)):  # pool_small
                    pool = pools[0.02]
                    if k > pool.numel():
                        continue
                    for _ in range(args.draws + 1):
                        perm = torch.randperm(pool.numel(), generator=gcpu)[:k]
                        sets.append(pool[perm.to(loo_s.device)])
                        meta.append(('pool_small', k))
                comp = {}                                        # set index -> refits
                if not args.no_kv_surgeon:                              # kv_surgeon_actual
                    for rho in RHOS:
                        bmap, t_uni = bmaps[rho]
                        t = max(1, head_budget(bmap, l, h, T, t_uni))
                        t = min(t, T)
                        pool_idx = pools[rho]
                        alg = KVSurgeonCompaction(**KV_SURGEON_KW)
                        U_sw = w_sw = None
                        gamma = 0.0
                        if pool_idx.numel() > t:
                            Ppool = E[:, pool_idx] / s.unsqueeze(1)
                            gamma = float(KV_SURGEON_KW['gamma0']
                                          * o.square().sum(dim=1).mean().sqrt())
                            keep_local, B_sw = alg._eliminate(Ppool, o, gamma, t)
                            del Ppool
                            top_idx = pool_idx[keep_local]
                            U_sw = B_sw[keep_local, :-1]
                            w_sw = B_sw[keep_local, -1] / gamma
                            (U_ir, w_ir, ffrac,
                             U_co, w_co) = refresh_fit(alg, E, s, o, top_idx, w_sw, gamma)
                        else:
                            top_idx = pool_idx
                            U_ir = w_ir = U_co = w_co = None
                            ffrac = ''
                        beta, C2 = alg._final_refit(
                            'vm', E, s, o, top_idx, K32, V32, top_idx.cpu().tolist())
                        mask = torch.ones(T, dtype=torch.bool, device=E.device)
                        mask[top_idx] = False
                        d = mask.nonzero(as_tuple=True)[0]
                        if d.numel():
                            comp[len(sets)] = (top_idx, beta, C2, U_sw, w_sw, gamma,
                                               U_ir, w_ir, ffrac, U_co, w_co)
                            sets.append(d)
                            meta.append((f'kv_surgeon_actual_rho{rho}', int(d.numel())))

                masks = torch.zeros(T, len(sets), device=E.device)
                for si, D in enumerate(sets):
                    masks[D, si] = 1.0
                # exact side: per query set. surrogate side: per calib set.
                ex = {}
                for name in ('slow', 'heldout'):
                    if name in att:
                        Ee, ss, oo = att[name]
                        ex[name] = eval_sets(Ee, ss, oo, V32, masks,
                                             args.q_chunk, args.set_chunk)
                quad_of = {'slow': ex['slow'][0]}
                if 'fast' in att:
                    Ee, ss, oo = att['fast']
                    quad_of['fast'] = eval_sets(Ee, ss, oo, V32, masks,
                                                args.q_chunk, args.set_chunk)[0]
                sums = {name: ((sc[1].unsqueeze(1) * masks).sum(0),
                               (sc[0].unsqueeze(1) * masks).sum(0))
                        for name, sc in scores.items()}          # (diag, loo)
                refits = {}
                for si, (top_idx, beta, C2, U_sw, w_sw, gamma,
                         U_ir, w_ir, ffrac, U_co, w_co) in comp.items():
                    for name in ex:
                        Ee, ss, oo = att[name]
                        cl = refit_losses(
                            Ee, ss, oo, top_idx, beta, C2, U_sw, w_sw, gamma)
                        cl['gamma'] = gamma
                        cl['mu_floor_frac'] = ffrac
                        if U_ir is not None:
                            ir = refit_losses(Ee, ss, oo, top_idx, beta, C2,
                                                U_ir, w_ir, gamma)
                            cl['exact_refresh'] = ir['exact_elim']
                            cl['decoupled_refresh'] = ir['decoupled_elim']
                            co = refit_losses(Ee, ss, oo, top_idx, beta, C2,
                                                U_co, w_co, gamma)
                            cl['exact_cons'] = co['exact_elim']
                            cl['decoupled_cons'] = co['decoupled_elim']
                            cl['wobj_elim'] = weighted_objective(
                                Ee, ss, oo, top_idx, U_sw, w_sw, gamma, w_sw,
                                alg.refresh_mu_floor, alg.ridge)
                            cl['wobj_refresh'] = weighted_objective(
                                Ee, ss, oo, top_idx, U_ir, w_ir, gamma, w_sw,
                                alg.refresh_mu_floor, alg.ridge)
                        refits[(si, name)] = cl

                for si, (sampler, k) in enumerate(meta):
                    for calib, qs in combos:
                        exname = 'slow' if qs == 'calib' else 'heldout'
                        quad, exact, amax, amean, mu2 = ex[exname]
                        row = dict(
                            model=args.model, article=a_id, layer=l, head=h,
                            sampler=sampler, set_size=k, frac_deleted=k / T,
                            T=T, Nq=att[calib][0].shape[0],
                            Nq_exact=att[exname][0].shape[0],
                            calib=calib, query_set=qs,
                            exact=float(exact[si]),
                            quad=float(quad_of[calib][si]),
                            diag=float(sums[calib][0][si]),
                            loo=float(sums[calib][1][si]),
                            O_norm2=O_norm2[exname],
                            mass_deleted_max=float(amax[si]),
                            mass_deleted_mean=float(amean[si]),
                            mu2_mean=float(mu2[si]),
                            exact_raw='', exact_elim='', exact_comp='',
                            decoupled_elim='', decoupled_comp='', gamma='',
                            exact_refresh='', decoupled_refresh='', wobj_elim='',
                            wobj_refresh='', mu_floor_frac='', exact_cons='',
                            decoupled_cons='')
                        if calib == 'slow' and (si, exname) in refits:
                            row['exact_raw'] = float(exact[si])
                            row.update(refits[(si, exname)])
                        wr.writerow(row)
                        n_rows += 1
                del att, E, s, o, masks, K32, V32, ex, quad_of, refits
            fh.flush()
            print(f"  layer {l} done, {n_rows} rows, "
                  f"{time.time()-t0:.0f}s", flush=True)
        del pkv, qsets
        torch.cuda.empty_cache()
    fh.close()
    print(f"wrote {args.out}: {n_rows} rows")


if __name__ == '__main__':
    main()
