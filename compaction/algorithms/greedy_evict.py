"""
Greedy elimination on the KV-Surgeon candidate pool without compensation (ablation).

With deletion set D, a_i = sum_{j in D} p_ij and r_i = sum_{j in D} p_ij (v_j - o_i),
each step evicts the keys j with the smallest cost
'surrogate': sum_i ||r_i + p_ij (v_j - o_i)||^2, or
'exact':     the same divided by (1 - a_i - p_ij)^2 (renormalized output).
"""
import math
from typing import Tuple

import torch

from .loo_evict import LOOEvictCompaction


class GreedyEvictCompaction(LOOEvictCompaction):
    """Greedy elimination without compensation (ablation of KV-Surgeon)."""

    def __init__(
        self,
        objective: str = 'surrogate',
        pool_factor: float = 4.0,
        block_size: int = 32,
        **kwargs,
    ):
        """
        Parameters
        ----------
        objective : {'surrogate', 'exact'}
            Elimination cost: the fixed quadratic or the exact renormalized loss.
        pool_factor : float
            Pool size as a multiple of t, chosen by the LOO score.
        block_size : int
            Keys evicted per step.
        **kwargs
            Passed to LOOEvictCompaction; `score` is forced to 'loo'.
        """
        if objective not in ('surrogate', 'exact'):
            raise ValueError(f"objective must be 'surrogate' or 'exact', got {objective!r}")
        if pool_factor < 1.0:
            raise ValueError(f"pool_factor must be >= 1, got {pool_factor}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        kwargs['score'] = 'loo'
        super().__init__(**kwargs)
        self.objective = objective
        self.pool_factor = float(pool_factor)
        self.block_size = int(block_size)

    def name(self) -> str:
        return f"GreedyEvict-{self.objective}"

    def _greedy(
        self,
        P: torch.Tensor,       # (n, p) normalized attention on the pool
        o: torch.Tensor,       # (n, d_v)
        Vp: torch.Tensor,      # (p, d_v)
        t: int,
    ) -> torch.Tensor:
        """Indices (into the pool) of the t survivors."""
        n, p = P.shape
        device = P.device
        exact = self.objective == 'exact'
        chunk = max(1, self.score_chunk_size)

        vn2 = (Vp * Vp).sum(dim=1)                                  # (p,)
        on2 = (o * o).sum(dim=1)                                    # (n,)
        # ||g_ij||^2 = ||v_j||^2 - 2 <o_i, v_j> + ||o_i||^2
        G2 = vn2.unsqueeze(0) - 2.0 * (o @ Vp.T) + on2.unsqueeze(1)  # (n, p)
        P2G = P * P * G2                                            # (n, p)
        del G2

        R = torch.zeros(n, o.shape[1], dtype=torch.float32, device=device)
        A = torch.zeros(n, dtype=torch.float32, device=device)
        rn2 = torch.zeros(n, dtype=torch.float32, device=device)
        alive = torch.ones(p, dtype=torch.bool, device=device)
        n_evict = p - t
        while n_evict > 0:
            b = min(self.block_size, n_evict)
            cost = torch.zeros(p, dtype=torch.float64, device=device)
            for r0 in range(0, n, chunk):
                rows = slice(r0, min(r0 + chunk, n))
                Pc = P[rows]                                        # (c, p)
                RV = R[rows] @ Vp.T                                 # (c, p)
                ro = (R[rows] * o[rows]).sum(dim=1, keepdim=True)   # (c, 1)
                num = (rn2[rows].unsqueeze(1)
                       + 2.0 * Pc * (RV - ro) + P2G[rows])         # (c, p)
                if exact:
                    mu = (1.0 - A[rows].unsqueeze(1) - Pc).clamp_min(1e-6)
                    num = num / (mu * mu)
                cost += num.sum(dim=0).double()
            cost[~alive] = float('inf')
            ev = torch.topk(cost, b, largest=False).indices          # (b,)
            alive[ev] = False
            Pe = P[:, ev]                                           # (n, b)
            pe = Pe.sum(dim=1)                                      # (n,)
            R += Pe @ Vp[ev] - pe.unsqueeze(1) * o
            A += pe
            rn2 = (R * R).sum(dim=1)
            n_evict -= b
        return alive.nonzero(as_tuple=True)[0]

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

        # Heads that keep most keys use plain LOO eviction.
        if t > self.fallback_threshold * T:
            return super().compute_compacted_cache(
                K, V, queries, t, attention_bias=attention_bias)

        pool = min(T, max(t, int(math.ceil(self.pool_factor * t))))

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
        E = (queries @ K.T).to(torch.float32)
        E.mul_(inv_sqrt_d)
        if attention_bias is not None:
            E.add_(torch.broadcast_to(attention_bias.to(torch.float32), E.shape))
        E.sub_(E.max(dim=1, keepdim=True)[0])
        E.exp_()
        s = E.sum(dim=1)
        V32 = V.to(torch.float32)
        o = (E @ V32) / s.unsqueeze(1)

        n_all = E.shape[0]
        if 0 < self.score_max_queries < n_all:
            score_rows = torch.linspace(
                0, n_all - 1, self.score_max_queries, device=device
            ).round().long()
            E_sc, s_sc, o_sc = E[score_rows], s[score_rows], o[score_rows]
        else:
            E_sc, s_sc, o_sc = E, s, o
        key_scores = self._loo_scores_streamed(E_sc, s_sc, o_sc, V32)
        del E_sc, s_sc, o_sc

        _, pool_idx = torch.topk(key_scores, pool, largest=True)

        if pool > t:
            P = E[:, pool_idx] / s.unsqueeze(1)
            keep_local = self._greedy(P, o, V32[pool_idx], t)
            del P
            top_idx = pool_idx[keep_local]
        else:
            top_idx = pool_idx

        C1 = K[top_idx]
        indices = top_idx.cpu().tolist()
        beta, C2 = self._final_refit(self.refit, E, s, o, top_idx, K, V, indices)
        return C1, beta, C2, indices
