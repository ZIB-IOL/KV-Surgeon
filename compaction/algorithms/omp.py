"""
OMP compaction: greedily select keys whose exp-scores best explain each query's
partition function sum_j exp(q k_j), with NNLS weights B (beta = log B), then
fit values by least squares.
"""
import torch
from typing import Tuple, List, Optional
from .base import CompactionAlgorithm


class OMPCompaction(CompactionAlgorithm):
    """Orthogonal matching pursuit on the partition function."""

    def __init__(self, nnls_iters: int = 0, nnls_lower_bound: float = None, nnls_upper_bound: float = None,
                 c2_method: str = 'lsq', k_choice: int = 1,
                 c2_ridge_lambda: float = 0, c2_solver: str = 'lstsq', c2_ridge_scale: str = 'spectral',
                 nnls_interval: int = 1, drop_key_beta_cutoff: float = None):
        """
        Parameters
        ----------
        nnls_iters : int
            Projected-gradient iterations for NNLS (0 = clamped least squares).
        nnls_lower_bound, nnls_upper_bound : float, optional
            Box constraints on the NNLS weights.
        c2_method : {'lsq', 'direct'}
            Value fit: least squares or the original values.
        k_choice : int
            Keys added per iteration.
        c2_solver, c2_ridge_scale : str
            See CompactionAlgorithm._compute_C2.
        nnls_interval : int
            Solve NNLS only every this many iterations.
        drop_key_beta_cutoff : float, optional
            If set, after selecting t keys, drop keys with beta below this and
            select replacements (up to 3 rounds).
        """
        self.nnls_iters = nnls_iters
        self.nnls_lower_bound = nnls_lower_bound
        self.nnls_upper_bound = nnls_upper_bound
        self.c2_method = c2_method
        self.k_choice = k_choice
        self.c2_ridge_lambda = c2_ridge_lambda
        self.c2_solver = c2_solver
        self.c2_ridge_scale = c2_ridge_scale
        self.drop_key_beta_cutoff = drop_key_beta_cutoff
        self.nnls_interval = nnls_interval

    def name(self) -> str:
        return "OMP"

    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        cached_selection_order: Optional[List[int]] = None,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """OMP key selection plus value fit.

        cached_selection_order, if given, is a selection order from a run with
        larger t; its first t keys are used instead of running OMP.
        """
        if cached_selection_order is not None:
            C1, beta, indices = self._select_keys_from_cached_order(
                K, queries, t, cached_selection_order, attention_bias
            )
        else:
            C1, beta, indices = self._select_keys_omp(K, queries, t, self.k_choice, attention_bias)

        C2 = self._compute_C2_with_method(
            C1, beta, K, V, queries,
            method=self.c2_method,
            indices=indices,
            attention_bias=attention_bias,
            ridge_lambda=self.c2_ridge_lambda,
            solver=self.c2_solver,
            ridge_scale=self.c2_ridge_scale
        )

        return C1, beta, C2, indices

    def get_full_selection_order(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        max_keys: Optional[int] = None,
        attention_bias: torch.Tensor = None,
    ) -> List[int]:
        """Greedy OMP selection order of up to max_keys keys, for reuse across t."""
        T = K.shape[0]
        if max_keys is None:
            max_keys = T

        _, _, indices = self._select_keys_omp(K, queries, max_keys, self.k_choice, attention_bias)
        return indices

    def _select_keys_from_cached_order(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        cached_order: List[int],
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Take the first t keys of a cached order and fit their beta by NNLS."""
        n, d = queries.shape
        device = K.device
        dtype_param = K.dtype

        indices = cached_order[:t]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)

        inv_sqrt_d = (1.0 / d) ** 0.5
        scores_raw = queries @ K.T
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores32.shape
                )
                scores32 = scores32 + bias32
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores32.max(dim=1, keepdim=True)[0]
        exp_scores = torch.exp(scores32 - max_scores)
        target = exp_scores.sum(dim=1)

        M = exp_scores[:, indices_tensor]
        B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
        beta32 = torch.log(B)

        C1 = K[indices_tensor]
        beta = beta32.to(dtype_param)

        return C1, beta, indices

    def _solve_nnls(
        self,
        M: torch.Tensor,
        target: torch.Tensor,
        prev_B: torch.Tensor,
        iteration: int,
    ) -> Tuple[torch.Tensor, bool]:
        """NNLS on the selected columns, or reuse prev_B between nnls_interval solves.

        Returns (B, solved).
        """
        interval = self.nnls_interval

        should_solve = False

        if prev_B is None:
            should_solve = True
        elif interval is not None:
            should_solve = (iteration % interval == 0)

        if should_solve:
            B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
            return B, True
        else:
            # Reuse the previous solution; new keys start at the lower bound.
            i = M.shape[1]
            prev_i = prev_B.shape[0]
            B = torch.zeros(i, dtype=torch.float32, device=M.device)
            B[:prev_i] = prev_B
            min_val = 1e-12 if self.nnls_lower_bound is None else self.nnls_lower_bound
            B[prev_i:] = min_val
            return B, False

    def _select_keys_omp(
        self,
        K: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        k_choice: int = 1,
        attention_bias: torch.Tensor = None,
    ):
        """Greedy OMP selection of up to t keys; returns (C1, beta, indices)."""
        n, d = queries.shape
        T = K.shape[0]
        device = K.device
        dtype_param = K.dtype
        inv_sqrt_d = (1.0 / d) ** 0.5
        scores_raw = queries @ K.T                               # (n, T) original dtype
        scores32 = scores_raw.to(torch.float32) * inv_sqrt_d     # (n, T) fp32
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    scores32.shape
                )
                scores32 = scores32 + bias32                     # (n, T)
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {scores32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        max_scores = scores32.max(dim=1, keepdim=True)[0]        # (n, 1) fp32
        exp_scores = torch.exp(scores32 - max_scores)            # (n, T) fp32
        target = exp_scores.sum(dim=1)                           # (n,)  fp32

        selected_indices_tensor = torch.zeros(t, dtype=torch.long, device=device)
        beta32 = torch.zeros(t, dtype=torch.float32, device=device)
        current = torch.zeros_like(target)                      # (n,) fp32
        mask_selected = torch.zeros(T, dtype=torch.bool, device=device)
        mask_excluded = torch.zeros(T, dtype=torch.bool, device=device)  # dropped by refinement

        i = 0
        prev_B = None
        iteration = 0
        refinement_count = 0

        while True:
            # i < t: selection. i == t: optional refinement that drops
            # low-weight keys and selects replacements.
            if i < t:
                residual = target - current                         # (n,)

                corr = (exp_scores * residual.unsqueeze(1)).sum(dim=0)  # (T,)

                num_remaining = T - (mask_selected | mask_excluded).sum().item()
                k_select = min(k_choice, num_remaining, t - i)

                if k_select == 0:
                    break

                corr[mask_selected] = -float('inf')
                corr[mask_excluded] = -float('inf')

                top_k_values, top_k_indices = torch.topk(corr, k_select, largest=True)  # (k_select,)
                new_indices_list = []
                prev_i = i
                for idx in top_k_indices:
                    if i >= t:
                        break
                    if mask_selected[idx]:
                        continue
                    selected_indices_tensor[i] = idx
                    mask_selected[idx] = True
                    new_indices_list.append(idx.item())
                    i += 1

                M = exp_scores[:, selected_indices_tensor[:i]]    # (n, i) fp32

                B, solved = self._solve_nnls(M, target, prev_B, iteration)  # (i,) fp32 (>=0)
                prev_B = B

                beta32[:i] = torch.log(B)
                current = M @ B
                iteration += 1

            elif i == t and self.drop_key_beta_cutoff is not None:
                refinement_count += 1

                if refinement_count > 3:
                    break

                # Exact NNLS here regardless of nnls_interval: the drop decision needs it.
                M = exp_scores[:, selected_indices_tensor[:i]]
                B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
                prev_B = B

                beta32[:i] = torch.log(B)

                keys_to_drop_mask = beta32[:i] < self.drop_key_beta_cutoff
                num_keys_to_drop = keys_to_drop_mask.sum().item()

                if num_keys_to_drop > 0:
                    drop_positions = torch.where(keys_to_drop_mask)[0]
                    keys_to_exclude = selected_indices_tensor[drop_positions]

                    mask_excluded[keys_to_exclude] = True

                    keep_mask = ~keys_to_drop_mask
                    keep_positions = torch.where(keep_mask)[0]
                    num_kept = keep_positions.shape[0]

                    selected_indices_tensor[:num_kept] = selected_indices_tensor[keep_positions]
                    beta32[:num_kept] = beta32[keep_positions]

                    mask_selected[keys_to_exclude] = False

                    i = num_kept

                    M = exp_scores[:, selected_indices_tensor[:i]]
                    B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
                    current = M @ B

                    prev_B = None

                    iteration += 1
                    continue
                else:
                    break
            else:
                # Final NNLS solve in case nnls_interval skipped the last one.
                if self.nnls_interval > 1:
                    M = exp_scores[:, selected_indices_tensor[:i]]
                    B = self._nnls_pg(M, target, self.nnls_iters, self.nnls_lower_bound, self.nnls_upper_bound)
                    beta32[:i] = torch.log(B)
                break

        # Refinement may leave fewer than t keys.
        selected_indices = selected_indices_tensor[:i].cpu().tolist()
        C1 = K[selected_indices_tensor[:i]]
        beta = beta32[:i].to(dtype_param)
        return C1, beta, selected_indices
