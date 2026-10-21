"""
KV-Surgeon: greedy elimination with closed-form compensation.

On the ridge regression [O, gamma*1] ~ P_S B with G = (P_S^T P_S + lambda I)^-1,
repeatedly delete the entries with the smallest saliency
s_j = ||B_j||^2 / G_jj and update B and G in closed form. The elimination runs
on the top ceil(pool_factor * t) entries by LOO score; the final refit of
multipliers and values is shared with LOOEvictCompaction.
"""
import math
import torch
from typing import Optional, Tuple

from .loo_evict import LOOEvictCompaction


class KVSurgeonCompaction(LOOEvictCompaction):
    """KV-Surgeon: compensation-aware elimination on a LOO-selected candidate pool."""

    def __init__(
        self,
        # elimination
        pool_factor: float = 4.0,
        block_size: int = 32,
        gamma0: float = 1.0,
        ridge: float = 0.01,
        refresh_interval: int = 0,
        refresh_mu_floor: float = 0.05,
        refresh_mode: str = 'reweighted',
        # candidate pool
        pool_score: str = 'loo',
        loo_eps: float = 1e-3,
        truncation_order: int = 2,
        # final refit and cost knobs, see LOOEvictCompaction
        refit: str = 'vm',
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        c2_ridge_lambda: float = 0,
        c2_solver: str = 'lstsq',
        score_max_queries: int = 0,
        build_max_queries: int = 0,
        build_queries_per_key: int = 8,
        fallback_threshold: float = 1.0,
        score_chunk_size: int = 4096,
    ):
        """
        Parameters (the rest are as in LOOEvictCompaction)
        ----------
        pool_factor : float
            Candidate pool size as a multiple of t (>= 1; 1 is plain LOO
            selection with the final refit).
        block_size : int
            Entries deleted per elimination step, compensated jointly.
        gamma0 : float
            Mass-term weight, gamma = gamma0 * RMS_i ||o_i||. 0 disables it.
        ridge : float
            Ridge lambda relative to the mean diagonal of P_S^T P_S.
        refresh_interval : int
            If > 0, re-solve the surrogate at the current normalization ratios
            mu every this many deletions.
        refresh_mu_floor : float
            Lower clamp on mu_i in the refresh.
        refresh_mode : {'reweighted', 'consistent'}
            'reweighted': residual weights 1/mu_i^2 on the fixed design.
            'consistent': design diag(1/mu) P_S with unit weights, so that the
            refreshed objective agrees with the served output.
        pool_score : {'loo', 'truncated'}
            Score that ranks entries for the candidate pool.
        """
        if pool_factor < 1.0:
            raise ValueError(f"pool_factor must be >= 1.0, got {pool_factor}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if gamma0 < 0:
            raise ValueError(f"gamma0 must be >= 0, got {gamma0}")
        if ridge <= 0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        if refresh_interval < 0:
            raise ValueError(f"refresh_interval must be >= 0, got {refresh_interval}")
        if not 0 < refresh_mu_floor <= 1:
            raise ValueError(
                f"refresh_mu_floor must be in (0, 1], got {refresh_mu_floor}")
        if refresh_mode not in ('reweighted', 'consistent'):
            raise ValueError(
                f"refresh_mode must be 'reweighted' or 'consistent', got {refresh_mode!r}")
        if pool_score not in ('loo', 'truncated'):
            raise ValueError(f"pool_score must be 'loo' or 'truncated', got '{pool_score}'")
        super().__init__(
            score=pool_score,
            refit=refit,
            loo_eps=loo_eps,
            truncation_order=truncation_order,
            nnls_iters=nnls_iters,
            nnls_lower_bound=nnls_lower_bound,
            nnls_upper_bound=nnls_upper_bound,
            c2_ridge_lambda=c2_ridge_lambda,
            c2_solver=c2_solver,
            score_max_queries=score_max_queries,
            build_max_queries=build_max_queries,
            build_queries_per_key=build_queries_per_key,
            fallback_threshold=fallback_threshold,
            score_chunk_size=score_chunk_size,
        )
        self.pool_factor = float(pool_factor)
        self.block_size = int(block_size)
        self.gamma0 = float(gamma0)
        self.ridge = float(ridge)
        self.refresh_interval = int(refresh_interval)
        self.refresh_mu_floor = float(refresh_mu_floor)
        self.refresh_mode = refresh_mode

    def name(self) -> str:
        return "KV-Surgeon"

    def _ridge_fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        row_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weighted ridge fit of Y (n, c) on X (n, k); returns (G, B).

        The ridge grows 10x on Cholesky failure, since near-duplicate attention
        columns can make the Gram matrix numerically singular.
        """
        if row_weights is not None:
            sw = row_weights.sqrt().unsqueeze(1)
            X = X * sw
            Y = Y * sw
        A = X.T @ X                                                 # (k, k)
        A = 0.5 * (A + A.T)
        diag_mean = A.diagonal().mean().clamp_min(1e-30)
        lam = self.ridge
        for attempt in range(4):
            A_damped = A.clone()
            A_damped.diagonal().add_(lam * diag_mean)
            try:
                L = torch.linalg.cholesky(A_damped)
                break
            except Exception:
                lam *= 10.0
        else:
            raise RuntimeError("Gram matrix not positive definite after ridge escalation")
        G = torch.cholesky_inverse(L)                               # (k, k)
        B = G @ (X.T @ Y)                                           # (k, c)
        return G, B

    def _eliminate(
        self,
        Ppool: torch.Tensor,
        o: torch.Tensor,
        gamma: float,
        t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy elimination of the pool columns of Ppool (n, p) down to t survivors.

        Returns (kept local indices, B). B is (p, c) with deleted rows zero;
        with the mass term on, its last column is gamma * w.
        """
        n, p = Ppool.shape
        n_delete = p - t
        device = Ppool.device

        has_mass = gamma > 0
        if has_mass:
            Y = torch.cat(
                [o, o.new_full((n, 1), gamma)], dim=1)              # (n, d_v+1)
        else:
            Y = o
        c = Y.shape[1]

        G, B = self._ridge_fit(Ppool, Y)
        if n_delete <= 0:
            return torch.arange(p, device=device), B
        alive = torch.ones(p, dtype=torch.bool, device=device)

        deleted = 0
        since_refresh = 0
        while deleted < n_delete:
            b = min(self.block_size, n_delete - deleted)
            diag = G.diagonal().clamp_min(1e-30)
            sal = (B * B).sum(dim=1) / diag                         # (p,)
            sal = sal.masked_fill(~alive, float('inf'))
            J = torch.topk(sal, b, largest=False).indices           # (b,)

            # One b x b solve gives both the compensation and the G update.
            GJJ = G[J][:, J]                                        # (b, b)
            GJJ = 0.5 * (GJJ + GJJ.T)
            GJJ.diagonal().add_(1e-10 * GJJ.diagonal().mean().clamp_min(1e-30))
            rhs = torch.cat([B[J], G[J]], dim=1)                    # (b, c + p)
            sol = torch.linalg.solve(GJJ, rhs)
            GJ = G[:, J]                                            # (p, b)
            B = B - GJ @ sol[:, :c]
            G = G - GJ @ sol[:, c:]
            # Zero deleted rows and columns so round-off does not accumulate.
            alive[J] = False
            B[J] = 0
            G[J, :] = 0
            G[:, J] = 0

            deleted += b
            since_refresh += b

            if (self.refresh_interval > 0 and since_refresh >= self.refresh_interval
                    and deleted < n_delete):
                since_refresh = 0
                idx = alive.nonzero(as_tuple=True)[0]               # (k,)
                if has_mass:
                    w_cur = (B[:, -1] / gamma).clamp_min(0.0)       # (p,)
                    mu = Ppool @ w_cur                              # deleted rows are 0
                else:
                    mu = Ppool[:, idx].sum(dim=1)
                mu = mu.clamp_min(self.refresh_mu_floor)
                if self.refresh_mode == 'consistent':
                    Xc = Ppool[:, idx] / mu.unsqueeze(1)
                    G_sub, B_sub = self._ridge_fit(Xc, Y)
                    del Xc
                else:
                    rho = 1.0 / mu.square()
                    rho = rho / rho.mean().clamp_min(1e-30)         # keep the ridge scale
                    G_sub, B_sub = self._ridge_fit(Ppool[:, idx], Y, row_weights=rho)
                G = torch.zeros(p, p, dtype=G_sub.dtype, device=device)
                G[idx.unsqueeze(1), idx.unsqueeze(0)] = G_sub
                B = torch.zeros(p, c, dtype=B_sub.dtype, device=device)
                B[idx] = B_sub

        return alive.nonzero(as_tuple=True)[0], B

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Same contract as LOOEvictCompaction.compute_compacted_cache."""
        device = K.device
        d = queries.shape[1]
        T = K.shape[0]
        t = min(int(t), T)
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Mild-compaction fallback: plain LOO eviction without refit.
        if t > self.fallback_threshold * T:
            return super().compute_compacted_cache(
                K, V, queries, t, attention_bias=attention_bias)

        pool = min(T, max(t, int(math.ceil(self.pool_factor * t))))

        # The query subsample must cover the pool, the number of unknowns.
        pre_build = self.build_max_queries
        if pre_build > 0:
            pre_build = max(pre_build, self.build_queries_per_key * pool)
        if 0 < pre_build < queries.shape[0]:
            n_orig = queries.shape[0]
            rows = torch.linspace(
                0, n_orig - 1, pre_build, device=device
            ).round().long()
            queries = queries[rows]
            if (attention_bias is not None and attention_bias.dim() >= 2
                    and attention_bias.shape[-2] == n_orig):
                attention_bias = attention_bias[..., rows, :]

        # One (n, T) attention pass shared by pool score, elimination and refit.
        scores_raw = queries @ K.T                                  # (n, T)
        E = scores_raw.to(torch.float32)
        del scores_raw
        E.mul_(inv_sqrt_d)
        if attention_bias is not None:
            try:
                E.add_(torch.broadcast_to(attention_bias.to(torch.float32), E.shape))
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {tuple(E.shape)}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        E.sub_(E.max(dim=1, keepdim=True)[0])
        E.exp_()
        s = E.sum(dim=1)                                            # (n,)
        V32 = V.to(torch.float32)
        o = (E @ V32) / s.unsqueeze(1)                              # (n, d_v)

        n_all = E.shape[0]
        if 0 < self.score_max_queries < n_all:
            score_rows = torch.linspace(
                0, n_all - 1, self.score_max_queries, device=device
            ).round().long()
            E_sc, s_sc, o_sc = E[score_rows], s[score_rows], o[score_rows]
        else:
            E_sc, s_sc, o_sc = E, s, o
        if self.score == 'truncated':
            key_scores = self._truncated_scores_streamed(E_sc, s_sc, o_sc, V32)
        else:
            key_scores = self._loo_scores_streamed(E_sc, s_sc, o_sc, V32)
        del E_sc, s_sc, o_sc

        _, pool_idx = torch.topk(key_scores, pool, largest=True)   # (pool,)

        keep_local, B_elim, gamma = None, None, 0.0
        if pool > t:
            Ppool = E[:, pool_idx] / s.unsqueeze(1)                 # (n, pool)
            if self.gamma0 > 0:
                gamma = float(
                    self.gamma0 * o.square().sum(dim=1).mean().sqrt())
            keep_local, B_elim = self._eliminate(Ppool, o, gamma, t)
            del Ppool
            top_idx = pool_idx[keep_local]                          # (t,)
        else:
            top_idx = pool_idx

        C1 = K[top_idx]                                             # (t, d)
        indices = top_idx.cpu().tolist()

        # Partial refits keep the rest from the elimination coefficients, which
        # requires an elimination with the mass term; otherwise refit as LOO does.
        if (self.refit in ('none', 'v', 'm')
                and B_elim is not None and gamma > 0):
            beta, C2 = self._refit_from_elimination(
                E, s, o, top_idx, K, V, keep_local, B_elim, gamma)
        else:
            beta, C2 = self._final_refit(
                self.refit, E, s, o, top_idx, K, V, indices)

        return C1, beta, C2, indices

    def _refit_from_elimination(
        self,
        E: torch.Tensor,
        s: torch.Tensor,
        o: torch.Tensor,
        top_idx: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        keep_local: torch.Tensor,
        B: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Final refit that keeps whatever it does not re-solve from the elimination coefficients B.

        'none': multipliers and values from B. 'v': multipliers from B, refit
        values. 'm': box-constrained multipliers, values from B. The multipliers
        in B are w = B[:, -1] / gamma; entries with w <= w_floor keep their
        original value. Under 'v' the served multipliers are clamped to the box
        [nnls_lower_bound, nnls_upper_bound] (e^-3, e^3 by default) and the
        values are refit against the clamped multipliers.
        """
        dtype_param = K.dtype
        w_floor = 1e-8

        w = (B[keep_local, -1] / gamma).to(torch.float32)           # (t,)
        healthy = w > w_floor
        w_pos = w.clamp_min(w_floor)

        if self.refit == 'v':
            lo = self.nnls_lower_bound if self.nnls_lower_bound is not None else math.exp(-3.0)
            hi = self.nnls_upper_bound if self.nnls_upper_bound is not None else math.exp(3.0)
            w_pos = w.clamp(min=float(lo), max=float(hi))
            healthy = torch.ones_like(healthy)

        if self.refit == 'm':
            M = E[:, top_idx]
            Bnnls = self._nnls_pg(M, s, self.nnls_iters,
                                  self.nnls_lower_bound, self.nnls_upper_bound)
            beta32 = torch.log(Bnnls)
        else:
            beta32 = torch.log(w_pos)

        if self.refit == 'v':
            Xw = E[:, top_idx].clone()                              # (n, t) fp32
            Xw.mul_(w_pos.unsqueeze(0))
            X = Xw.div_(Xw.sum(dim=1, keepdim=True).clamp_min(1e-38))
            C2 = self._solve_C2_fused(X, o).to(dtype_param)
        else:
            U = B[keep_local, :-1]                                  # (t, d_v)
            C2_32 = U / w_pos.unsqueeze(1)
            if not bool(healthy.all()):
                C2_32 = torch.where(
                    healthy.unsqueeze(1), C2_32, V[top_idx].to(torch.float32))
            C2 = C2_32.to(dtype_param)

        return beta32.to(dtype_param), C2
