"""
KVMerger (Wang et al. 2024, arXiv:2407.08454): cluster runs of adjacent keys
with high cosine similarity and merge each cluster by Gaussian-kernel weighted
averaging around its most-attended key. Optionally fits beta by NNLS.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, List
from .base import CompactionAlgorithm


class KVMergerCompaction(CompactionAlgorithm):
    """Merge consecutive similar keys via Gaussian kernel weighted averaging."""

    def __init__(
        self,
        top_k_ratio: float = 0.0,
        c2_method: str = 'merge',
        beta_method: str = 'zero',
        nnls_iters: int = 0,
        nnls_lower_bound: float = None,
        nnls_upper_bound: float = None,
        threshold_search_steps: int = 20,
    ):
        """
        Parameters
        ----------
        top_k_ratio : float
            Fraction of most-attended keys kept as singleton clusters.
        c2_method : {'merge'}
            Only the kernel-weighted value merge is supported.
        beta_method : {'zero', 'nnls'}
            'zero' is plain KVMerger; 'nnls' fits beta to the partition function.
        nnls_iters : int
            Projected-gradient iterations for NNLS (0 = clamped least squares).
        nnls_lower_bound, nnls_upper_bound : float, optional
            Box constraints on exp(beta).
        threshold_search_steps : int
            Binary-search steps for the similarity threshold that yields t clusters.
        """
        self.top_k_ratio = top_k_ratio
        if c2_method != 'merge':
            raise ValueError(f"c2_method must be 'merge', got '{c2_method}'")
        self.c2_method = c2_method
        if beta_method not in ['zero', 'nnls']:
            raise ValueError(f"beta_method must be 'zero' or 'nnls', got '{beta_method}'")
        self.beta_method = beta_method
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.threshold_search_steps = threshold_search_steps

    def name(self) -> str:
        return "KVMerger"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Cluster and merge to t entries; indices are the cluster pivots."""
        T, d = K.shape
        device = K.device
        dtype_param = K.dtype

        if t >= T:
            beta = torch.zeros(T, dtype=dtype_param, device=device)
            return K.clone(), beta, V.clone(), list(range(T))

        agg_attn = self._compute_aggregated_attention(K, queries, attention_bias)  # (T,)

        n_protected = max(1, int(T * self.top_k_ratio))
        n_protected = min(n_protected, t)
        protected_indices = agg_attn.topk(n_protected).indices  # (n_protected,)
        protected_set = set(protected_indices.cpu().tolist())

        K_norm = F.normalize(K.float(), dim=-1)  # (T, d) fp32
        adj_sim = (K_norm[:-1] * K_norm[1:]).sum(dim=-1)  # (T-1,)

        merging_sets = self._find_merging_sets_with_target(
            adj_sim, protected_set, T, t
        )

        C1_list = []
        C2_list = []
        pivotal_indices = []

        for s in merging_sets:
            if len(s) == 1:
                idx = s[0]
                C1_list.append(K[idx])
                C2_list.append(V[idx])
                pivotal_indices.append(idx)
            else:
                # The pivot is the most-attended member.
                set_attn = agg_attn[s]  # (|s|,)
                pivot_local = set_attn.argmax().item()
                pivot_idx = s[pivot_local]

                keys_in_set = K[s].float()  # (|s|, d)
                vals_in_set = V[s].float()  # (|s|, d)
                pivot_key = K[pivot_idx].float()  # (d,)

                diffs = keys_in_set - pivot_key.unsqueeze(0)  # (|s|, d)
                dists_sq = (diffs ** 2).sum(dim=-1)  # (|s|,)

                # Paper: sigma = sum(g) / (sqrt(2) |S|), with g from a preliminary
                # kernel whose width is the mean squared distance.
                g_raw = torch.exp(-dists_sq / (2 * dists_sq.mean().clamp(min=1e-12)))
                sigma = g_raw.sum() / (len(s) * (2 ** 0.5))
                sigma = sigma.clamp(min=1e-8)

                g = torch.exp(-dists_sq / (2 * sigma ** 2))  # (|s|,)

                g_sum = g.sum().clamp(min=1e-12)
                w_keys = g / g_sum  # (|s|,)

                merged_key = (w_keys.unsqueeze(-1) * keys_in_set).sum(dim=0)  # (d,)

                w_vals = w_keys * len(s)
                merged_val = (w_vals.unsqueeze(-1) * vals_in_set).sum(dim=0) / len(s)  # (d,)

                C1_list.append(merged_key.to(dtype_param))
                C2_list.append(merged_val.to(dtype_param))
                pivotal_indices.append(pivot_idx)

        C1 = torch.stack(C1_list)  # (t', d)
        C2 = torch.stack(C2_list)  # (t', d)
        actual_t = C1.shape[0]

        if self.beta_method == 'zero':
            beta = torch.zeros(actual_t, dtype=dtype_param, device=device)
        else:  # 'nnls'
            beta = self._compute_beta_nnls(C1, K, queries, attention_bias)

        return C1, beta, C2, pivotal_indices

    def _compute_aggregated_attention(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Attention weight of each key summed over queries, shape (T,)."""
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        scores_raw = queries @ K.T  # (n, T)
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        if attention_bias is not None:
            bias32 = torch.broadcast_to(
                attention_bias.to(torch.float32), scores32.shape
            )
            scores32 = scores32 + bias32
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)
        attn = exp_scores / exp_scores.sum(dim=1, keepdim=True)  # (n, T)
        return attn.sum(dim=0)  # (T,)

    def _find_merging_sets_with_target(
        self,
        adj_sim: torch.Tensor,
        protected_set: set,
        T: int,
        t: int,
    ) -> List[List[int]]:
        """Binary-search the similarity threshold, then trim or split to exactly t clusters."""
        adj_sim_cpu = adj_sim.cpu()

        lo, hi = 0.0, 1.0
        best_sets = None
        best_diff = float('inf')

        for _ in range(self.threshold_search_steps):
            mid = (lo + hi) / 2
            sets = self._greedy_ahc(adj_sim_cpu, protected_set, T, mid)
            n_sets = len(sets)
            diff = abs(n_sets - t)

            if diff < best_diff:
                best_diff = diff
                best_sets = sets

            if n_sets == t:
                break
            elif n_sets < t:
                lo = mid
            else:
                hi = mid

        if len(best_sets) > t:
            best_sets = self._trim_sets(best_sets, t)
        elif len(best_sets) < t:
            best_sets = self._split_sets(best_sets, t)

        return best_sets

    @staticmethod
    def _greedy_ahc(
        adj_sim: torch.Tensor,
        protected_set: set,
        T: int,
        threshold: float,
    ) -> List[List[int]]:
        """Greedy clustering of adjacent keys (Algorithm 1 of the paper).

        Protected keys are always singletons.
        """
        sets: List[List[int]] = []
        current_set = [T - 1]

        for i in range(T - 2, -1, -1):
            if (i in protected_set
                    or (i + 1) in protected_set
                    or adj_sim[i].item() < threshold):
                sets.append(current_set)
                current_set = [i]
            else:
                current_set.append(i)

        sets.append(current_set)

        return [sorted(s) for s in reversed(sets)]

    @staticmethod
    def _trim_sets(sets: List[List[int]], t: int) -> List[List[int]]:
        """Merge the smallest adjacent pairs until t sets remain."""
        while len(sets) > t:
            min_combined = float('inf')
            merge_idx = 0
            for i in range(len(sets) - 1):
                combined = len(sets[i]) + len(sets[i + 1])
                if combined < min_combined:
                    min_combined = combined
                    merge_idx = i
            merged = sets[merge_idx] + sets[merge_idx + 1]
            sets = sets[:merge_idx] + [merged] + sets[merge_idx + 2:]
        return sets

    @staticmethod
    def _split_sets(sets: List[List[int]], t: int) -> List[List[int]]:
        """Split the largest sets in half until there are t sets."""
        while len(sets) < t:
            max_size = 0
            split_idx = 0
            for i, s in enumerate(sets):
                if len(s) > max_size:
                    max_size = len(s)
                    split_idx = i
            if max_size <= 1:
                break
            s = sets[split_idx]
            mid = len(s) // 2
            sets = sets[:split_idx] + [s[:mid], s[mid:]] + sets[split_idx + 1:]
        return sets

    def _compute_beta_nnls(
        self,
        C1: torch.Tensor,
        K: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fit beta via NNLS to match the original partition function."""
        d = K.shape[1]
        dtype_param = K.dtype
        device = K.device
        inv_sqrt_d = (1.0 / d) ** 0.5

        scores_raw = queries @ K.T
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        if attention_bias is not None:
            bias32 = torch.broadcast_to(
                attention_bias.to(torch.float32), scores32.shape
            )
            scores32 = scores32 + bias32
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores_orig = torch.exp(scores32 - max_scores)  # (n, T)

        sC_raw = queries @ C1.T
        sC32 = sC_raw.to(torch.float32) * inv_sqrt_d
        max_C = sC32.max(dim=1, keepdim=True)[0]
        # Shift by the original row max so M and target share a scale.
        M = torch.exp(sC32 - max_scores)  # (n, t)

        target = exp_scores_orig.sum(dim=1)  # (n,)

        B = self._nnls_pg(M, target, self.nnls_iters,
                          self.nnls_lower_bound, self.nnls_upper_bound)
        beta32 = torch.log(B)  # (t,)
        return beta32.to(dtype_param)
