"""
LOO-Evict: exact leave-one-out eviction for softmax attention.

Keeps the top-t keys by the exact single-deletion cost
s_j = sum_i P_ij^2 / (1 - P_ij)^2 ||v_j - o_i||^2, then applies a refit
refit: 'none', 'm' (box-constrained multipliers), 'v' (least-squares values) or 'vm' (both).
Also provides the truncated LOO score used by KV-Surgeon-fast and the OBCache key score.
"""
import torch
from typing import Tuple

from .base import CompactionAlgorithm


class LOOEvictCompaction(CompactionAlgorithm):
    """Exact leave-one-out eviction with an optional final refit."""

    def __init__(
        self,
        score: str = 'loo',
        refit: str = 'vm',
        loo_eps: float = 1e-3,
        truncation_order: int = 2,
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
        Parameters
        ----------
        score : {'loo', 'truncated', 'obcache_k'}
            'loo': exact single-deletion cost. 'truncated': the LOO score with
            (1 - P)^-2 replaced by its series sum_{k<=truncation_order} (k+1) P^k.
            'obcache_k': OBCache key score.
        refit : {'none', 'm', 'v', 'vm'}
            Final refit: 'm' fits multipliers by box-constrained least squares, 'v' refits values
            by least squares, 'vm' does both.
        loo_eps : float
            Floor on (1 - P_ij)^2 in the LOO score.
        truncation_order : int
            Series order for score='truncated'; 0 gives the second-order diagonal H_jj.
        nnls_iters : int
            Projected-gradient iterations for the multiplier fit (0 = clamped least
            squares).
        nnls_lower_bound, nnls_upper_bound : float, optional
            Box constraints on exp(beta).
        c2_ridge_lambda : float
            Ridge for the value refit, relative to the mean Gram diagonal.
        c2_solver : {'lstsq', 'cholesky'}
            Value-refit solver.
        score_max_queries : int
            If > 0, score on an even-stride subsample of this many queries.
        build_max_queries : int
            If > 0, subsample queries to this many rows before building the
            attention matrix, so the whole pipeline scales with it.
        build_queries_per_key : int
            With build_max_queries, keep at least this many rows per kept key.
        fallback_threshold : float
            Mild-compaction fallback: heads with t / T above this skip the refit.
        score_chunk_size : int
            Query-row chunk size for streamed scoring.
        """
        if score not in ('loo', 'truncated', 'obcache_k'):
            raise ValueError(
                f"score must be one of 'loo', 'truncated', 'obcache_k', got '{score}'")
        if truncation_order < 0:
            raise ValueError(f"truncation_order must be >= 0, got {truncation_order}")
        refit = str(refit).lower()
        if refit not in ('none', 'm', 'v', 'vm'):
            raise ValueError(
                f"refit must be one of 'none', 'm', 'v', 'vm', got '{refit}'"
            )
        if score_max_queries < 0:
            raise ValueError(f"score_max_queries must be >= 0, got {score_max_queries}")
        if build_max_queries < 0:
            raise ValueError(f"build_max_queries must be >= 0, got {build_max_queries}")
        if build_queries_per_key < 0:
            raise ValueError(
                f"build_queries_per_key must be >= 0, got {build_queries_per_key}")
        if not 0.0 < fallback_threshold <= 1.0:
            raise ValueError(
                f"fallback_threshold must be in (0, 1], got {fallback_threshold}")
        if c2_solver not in ('lstsq', 'cholesky'):
            raise ValueError(f"c2_solver must be 'lstsq' or 'cholesky', got '{c2_solver}'")
        if score_chunk_size < 1:
            raise ValueError(f"score_chunk_size must be >= 1, got {score_chunk_size}")
        self.score = score
        self.refit = refit
        self.loo_eps = loo_eps
        self.truncation_order = int(truncation_order)
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.score_max_queries = int(score_max_queries)
        self.build_max_queries = int(build_max_queries)
        self.build_queries_per_key = int(build_queries_per_key)
        self.fallback_threshold = float(fallback_threshold)
        self.score_chunk_size = int(score_chunk_size)

    def name(self) -> str:
        return "LOOEvict"

    def _loo_scores_streamed(
        self, E: torch.Tensor, s: torch.Tensor, o: torch.Tensor, V32: torch.Tensor
    ) -> torch.Tensor:
        """LOO score from exponentials E (n, T), row sums s and outputs o."""
        n, T = E.shape
        v_sq = (V32 * V32).sum(dim=1)                               # (T,)
        o_sq = (o * o).sum(dim=1)                                   # (n,)
        scores = torch.zeros(T, dtype=torch.float32, device=E.device)
        for start in range(0, n, self.score_chunk_size):
            rows = slice(start, min(start + self.score_chunk_size, n))
            P = E[rows] / s[rows].unsqueeze(1)                      # (c, T)
            cross = o[rows] @ V32.T                                 # (c, T)
            novelty_sq = (v_sq.unsqueeze(0) - 2.0 * cross
                          + o_sq[rows].unsqueeze(1)).clamp_min_(0.0)
            one_minus_P = 1.0 - P
            w = (P * P) / (one_minus_P * one_minus_P + self.loo_eps)
            scores += (w * novelty_sq).sum(dim=0)
        return scores

    def _truncated_scores_streamed(
        self, E: torch.Tensor, s: torch.Tensor, o: torch.Tensor, V32: torch.Tensor
    ) -> torch.Tensor:
        """Score sum_i g(P_ij) ||v_j - o_i||^2, g(P) = P^2 sum_k (k+1) P^k.

        The square is expanded so no (c, T) novelty tensor is formed.
        """
        n, T = E.shape
        v_sq = (V32 * V32).sum(dim=1)                               # (T,)
        o_sq = (o * o).sum(dim=1)                                   # (n,)
        a = torch.zeros(T, dtype=torch.float32, device=E.device)
        Bmat = torch.zeros(T, V32.shape[1], dtype=torch.float32, device=E.device)
        c = torch.zeros(T, dtype=torch.float32, device=E.device)
        for start in range(0, n, self.score_chunk_size):
            rows = slice(start, min(start + self.score_chunk_size, n))
            P = E[rows] / s[rows].unsqueeze(1)                      # (c, T)
            # Horner evaluation of sum_k (k+1) P^k, then * P^2.
            poly = torch.full_like(P, float(self.truncation_order + 1))
            for k in range(self.truncation_order - 1, -1, -1):
                poly.mul_(P).add_(float(k + 1))
            g = poly.mul_(P).mul_(P)                                # (c, T)
            a += g.sum(dim=0)
            Bmat += g.T @ o[rows]                                   # (T, d_v)
            c += g.T @ o_sq[rows]                                   # (T,)
        return (v_sq * a - 2.0 * (Bmat * V32).sum(dim=1) + c).clamp_min_(0.0)

    def _obcache_scores_streamed(
        self, E: torch.Tensor, s: torch.Tensor, o: torch.Tensor,
        V32: torch.Tensor, rowmax: torch.Tensor,
    ) -> torch.Tensor:
        """OBCache key score (Gu et al. 2025, arXiv:2510.07651).

        s_j = sum_i (P_ij z_ij)^2 ||v_j - o_i||^2 with raw logits
        z_ij = log E_ij + rowmax_i. Unlike the official code, query rows of a
        GQA group are scored separately instead of averaged.
        """
        n, T = E.shape
        v_sq = (V32 * V32).sum(dim=1)                               # (T,)
        o_sq = (o * o).sum(dim=1)                                   # (n,)
        scores = torch.zeros(T, dtype=torch.float32, device=E.device)
        for start in range(0, n, self.score_chunk_size):
            rows = slice(start, min(start + self.score_chunk_size, n))
            Ec = E[rows]
            P = Ec / s[rows].unsqueeze(1)                           # (c, T)
            Z = torch.where(Ec > 0, Ec.clamp_min(1e-38).log() + rowmax[rows],
                            torch.zeros((), device=E.device))       # logits (c, T)
            cross = o[rows] @ V32.T                                 # (c, T)
            novelty_sq = (v_sq.unsqueeze(0) - 2.0 * cross
                          + o_sq[rows].unsqueeze(1)).clamp_min_(0.0)
            PZ = P * Z
            scores += (PZ * PZ * novelty_sq).sum(dim=0)
        return scores.clamp_min_(0.0)

    def _solve_C2_fused(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Solve min_C ||X C - Y||_F^2 (+ ridge) for X (n, t), Y (n, d_v)."""
        if self.c2_solver == 'cholesky':
            XtX = X.T @ X                                           # (t, t)
            lam = self.c2_ridge_lambda if self.c2_ridge_lambda > 0 else 1e-6
            XtX.diagonal().add_(lam * XtX.diagonal().mean())
            R = torch.linalg.cholesky(0.5 * (XtX + XtX.T))
            return torch.cholesky_solve(X.T @ Y, R)                 # (t, d_v)
        sol = torch.linalg.lstsq(X, Y).solution                     # (t, d_v)
        if self.c2_ridge_lambda > 0 or torch.isnan(sol).any():
            # Ridge normal equations, also the fallback for lstsq NaNs.
            XtX = X.T @ X
            lam = max(self.c2_ridge_lambda, 1e-6)
            XtX.diagonal().add_(lam * XtX.diagonal().mean())
            R = torch.linalg.cholesky(0.5 * (XtX + XtX.T))
            sol = torch.cholesky_solve(X.T @ Y, R)
        return sol

    def _final_refit(
        self,
        refit: str,
        E: torch.Tensor,
        s: torch.Tensor,
        o: torch.Tensor,
        top_idx: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        indices: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multiplier (box-constrained) and value refit on a selected support; returns (beta, C2).

        Reuses the exponentials E (n, T), row sums s and outputs o.
        """
        device = E.device
        dtype_param = K.dtype
        t = top_idx.numel()

        B = None
        if refit in ('m', 'vm'):
            M = E[:, top_idx]                                       # (n, t)
            B = self._nnls_pg(M, s, self.nnls_iters,
                              self.nnls_lower_bound, self.nnls_upper_bound)  # (t,) > 0
            beta32 = torch.log(B)                                   # (t,)
        else:
            beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        beta = beta32.to(dtype_param)

        # Value refit design: X = normalize(E[:, S] * exp(beta)), targets o.
        if refit in ('none', 'm'):
            C2 = self._direct_C2(K[top_idx], K, V, indices)
        else:
            Xw = E[:, top_idx].clone()                              # (n, t) fp32
            if B is not None:
                Xw.mul_(B.unsqueeze(0))
            X = Xw.div_(Xw.sum(dim=1, keepdim=True).clamp_min(1e-38))
            C2 = self._solve_C2_fused(X, o).to(dtype_param)

        return beta, C2

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Select t keys by score and apply the final refit (see CompactionAlgorithm)."""
        device = K.device
        d = queries.shape[1]
        T = K.shape[0]
        t = min(int(t), T)
        inv_sqrt_d = (1.0 / d) ** 0.5

        # Heads that keep most keys skip the refit: eviction is nearly
        # lossless there and the value solve is costly and ill-determined.
        refit = self.refit
        if refit != 'none' and t > self.fallback_threshold * T:
            refit = 'none'

        # Subsample queries before the attention build. Without a refit the
        # queries only serve scoring, so score_max_queries applies here too.
        pre_build = self.build_max_queries
        if pre_build > 0 and refit != 'none':
            # The value solve has t unknowns, so keep enough rows per key.
            pre_build = max(pre_build, self.build_queries_per_key * t)
        if refit == 'none' and self.score_max_queries > 0:
            pre_build = (min(pre_build, self.score_max_queries)
                         if pre_build > 0 else self.score_max_queries)
        if 0 < pre_build < queries.shape[0]:
            n_orig = queries.shape[0]
            rows = torch.linspace(
                0, n_orig - 1, pre_build, device=device
            ).round().long()
            queries = queries[rows]
            # Per-query biases (n, T) are subsampled too; broadcast shapes are not.
            if (attention_bias is not None and attention_bias.dim() >= 2
                    and attention_bias.shape[-2] == n_orig):
                attention_bias = attention_bias[..., rows, :]

        # One (n, T) attention pass shared by scorer and refits: E = max-shifted
        # exponentials, s = row sums, o = full-cache outputs.
        scores_raw = queries @ K.T                                  # (n, T) param dtype
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
        rowmax = E.max(dim=1, keepdim=True)[0]                      # (n, 1) fp32
        E.sub_(rowmax)
        E.exp_()
        s = E.sum(dim=1)                                            # (n,) fp32
        if self.score != 'obcache_k':
            rowmax = None

        V32 = V.to(torch.float32)
        o = (E @ V32) / s.unsqueeze(1)                              # (n, d_v)

        # Even-stride row subsample for scoring only.
        n_all = E.shape[0]
        if 0 < self.score_max_queries < n_all:
            score_rows = torch.linspace(
                0, n_all - 1, self.score_max_queries, device=device
            ).round().long()
            E_sc, s_sc, o_sc = E[score_rows], s[score_rows], o[score_rows]
            rm_sc = rowmax[score_rows] if rowmax is not None else None
        else:
            E_sc, s_sc, o_sc, rm_sc = E, s, o, rowmax

        if self.score == 'truncated':
            key_scores = self._truncated_scores_streamed(E_sc, s_sc, o_sc, V32)  # (T,)
        elif self.score == 'obcache_k':
            key_scores = self._obcache_scores_streamed(
                E_sc, s_sc, o_sc, V32, rm_sc)                       # (T,)
        else:
            key_scores = self._loo_scores_streamed(E_sc, s_sc, o_sc, V32)  # (T,)
        del E_sc, s_sc, o_sc, rm_sc, rowmax

        _, top_idx = torch.topk(key_scores, t, largest=True)        # (t,)
        C1 = K[top_idx]                                             # (t, d) param dtype
        indices = top_idx.cpu().tolist()

        beta, C2 = self._final_refit(refit, E, s, o, top_idx, K, V, indices)

        return C1, beta, C2, indices
