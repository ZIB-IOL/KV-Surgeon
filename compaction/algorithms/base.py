"""Base class and shared solvers for per-head KV cache compaction algorithms."""
import torch
from typing import Tuple
from abc import ABC, abstractmethod


class CompactionAlgorithm(ABC):
    """Base class for KV cache compaction algorithms."""

    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name for logging."""
        pass

    @abstractmethod
    def compute_compacted_cache(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        t: int,
        attention_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Compact one head's cache to t entries.

        Parameters
        ----------
        K : Tensor, shape (T, d)
            Original keys.
        V : Tensor, shape (T, d_v)
            Original values.
        queries : Tensor, shape (n, d)
            Calibration queries.
        t : int
            Number of entries to keep.
        attention_bias : Tensor, optional
            Additive bias on the original attention logits, broadcastable to (n, T).

        Returns
        -------
        C1 : Tensor, shape (t, d)
            Compacted keys.
        beta : Tensor, shape (t,)
            Per-key logit biases.
        C2 : Tensor, shape (t, d_v)
            Compacted values.
        indices : list of int
            Indices of selected keys, if the method selects a subset.
        """
        pass

    def requires_queries(self) -> bool:
        """Whether compute_compacted_cache needs calibration queries.

        If False, the caller may skip query generation and pass queries=None.
        """
        return True

    def _compute_C2(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        attention_bias: torch.Tensor = None,
        ridge_lambda: float = 0,
        solver: str = 'lstsq',
        ridge_scale: str = 'spectral'
    ) -> torch.Tensor:
        """
        Solve softmax(q K^T) V ~ softmax(q C1^T + beta) C2 for C2 by (ridge) least squares.

        Parameters
        ----------
        C1 : Tensor, shape (t, d)
            Reduced set of keys.
        beta : Tensor, shape (t,)
            Bias terms for the reduced keys.
        K, V : Tensors, shapes (T, d) and (T, d_v)
            Original keys and values.
        queries : Tensor, shape (n, d)
        attention_bias : Tensor, optional
            Additive bias on the original attention logits, broadcastable to (n, T).
        solver : {'lstsq', 'pinv', 'cholesky'}
        ridge_scale : {'spectral', 'frobenius', 'fixed'}
            Scale of ridge_lambda: ||X||_2^2, ||X||_F^2 / t, or none.
        """
        # Matmuls in the original dtype, softmax in fp32.
        dtype_param = K.dtype
        d = K.shape[1]
        inv_sqrt_d = (1.0 / d) ** 0.5

        sK_raw = queries @ K.T                                   # (n, T) original dtype
        sK32 = sK_raw.to(torch.float32) * inv_sqrt_d             # (n, T)
        if attention_bias is not None:
            try:
                bias32 = torch.broadcast_to(
                    attention_bias.to(torch.float32),
                    sK32.shape
                )
                sK32 = sK32 + bias32                             # (n, T)
            except Exception as e:
                raise ValueError(
                    f"attention_bias must be broadcastable to {sK32.shape}, "
                    f"got {tuple(attention_bias.shape)}"
                ) from e
        m_K = sK32.max(dim=1, keepdim=True)[0]                   # (n, 1) for stability
        exp_sK = torch.exp(sK32 - m_K)                           # (n, T)
        sum_exp_K = exp_sK.sum(dim=1, keepdim=True)              # (n, 1)
        attn_K = exp_sK / sum_exp_K                              # (n, T) normalized
        Y = attn_K @ V.to(torch.float32)                         # (n, d) fp32

        sC_raw = queries @ C1.T                                  # (n, t) original dtype
        sC32 = sC_raw.to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)  # (n, t)
        m_C = sC32.max(dim=1, keepdim=True)[0]                   # (n, 1) for stability
        exp_sC = torch.exp(sC32 - m_C)                           # (n, t)
        sum_exp_C = exp_sC.sum(dim=1, keepdim=True)              # (n, 1)
        X = exp_sC / sum_exp_C                                   # (n, t) normalized

        n, t = X.shape
        # ||X||_F^2 / t is the mean eigenvalue of X^T X.
        if ridge_lambda == 0:
            lam = 0
        else:
            if ridge_scale == 'spectral':
                try:
                    lam = ridge_lambda * (torch.linalg.matrix_norm(X, ord=2)**2)
                except Exception as e:
                    print(f"Warning: Spectral norm computation failed ({e}), falling back to Frobenius norm")
                    lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro')**2) / t)
            elif ridge_scale == 'frobenius':
                lam = ridge_lambda * ((torch.linalg.matrix_norm(X, ord='fro')**2) / t)
            elif ridge_scale == 'fixed':
                lam = ridge_lambda
            else:
                raise ValueError(f"Unknown ridge_scale: {ridge_scale}. Must be 'spectral', 'frobenius', or 'fixed'.")

        if solver == 'lstsq':
            try:
                C2_32 = torch.linalg.lstsq(X, Y, driver='gels').solution    # (t,d)
                if torch.isnan(C2_32).any():
                    raise RuntimeError("NaNs in lstsq solution")
            except Exception as e:
                print(f"lstsq failed ({e}), increasing lambda to 1e-6 and retrying")
                print(f"  C1 has NaN: {torch.isnan(C1).any().item()}, Inf: {torch.isinf(C1).any().item()}")
                print(f"  beta has NaN: {torch.isnan(beta).any().item()}, Inf: {torch.isinf(beta).any().item()}")
                print(f"  K has NaN: {torch.isnan(K).any().item()}, Inf: {torch.isinf(K).any().item()}")
                print(f"  V has NaN: {torch.isnan(V).any().item()}, Inf: {torch.isinf(V).any().item()}")
                print(f"  queries has NaN: {torch.isnan(queries).any().item()}, Inf: {torch.isinf(queries).any().item()}")
                print(f"  X has NaN: {torch.isnan(X).any().item()}, Inf: {torch.isinf(X).any().item()}")
                print(f"  Y has NaN: {torch.isnan(Y).any().item()}, Inf: {torch.isinf(Y).any().item()}")

                lam = 1e-6
                if n < t:
                    XXt = X @ X.T
                    XXt = 0.5 * (XXt + XXt.T)
                    XXt.diagonal().add_(lam)
                    L = torch.linalg.cholesky(XXt)
                    Z = torch.cholesky_solve(Y, L)
                    C2_32 = X.T @ Z
                else:
                    XtX = X.T @ X
                    XtX = 0.5 * (XtX + XtX.T)
                    XtX.diagonal().add_(lam)
                    L = torch.linalg.cholesky(XtX)
                    XtY = X.T @ Y
                    C2_32 = torch.cholesky_solve(XtY, L)
                if torch.isnan(C2_32).any():
                    raise RuntimeError("NaNs in cholesky solution")
        elif solver == 'pinv':
            if lam == 0:
                X_pinv = torch.linalg.pinv(X)
                C2_32 = X_pinv @ Y
            elif n >= t:
                # C2 = (XᵀX + λI)^{-1} Xᵀ Y
                XtX = X.T @ X
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                XtY = X.T @ Y
                C2_32 = torch.linalg.pinv(XtX) @ XtY         # (t,d)
            else:
                # Underdetermined: C2 = X^T (X X^T + lam I)^-1 Y
                XXt = X @ X.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                XXt_pinv = torch.linalg.pinv(XXt)
                Z = XXt_pinv @ Y
                C2_32 = X.T @ Z
        elif solver == 'cholesky':
            if n < t:
                # C2 = Xᵀ (XXᵀ + λI)^{-1} Y
                XXt = X @ X.T
                XXt = 0.5 * (XXt + XXt.T)
                XXt.diagonal().add_(lam)
                L = torch.linalg.cholesky(XXt)               # (n,n)
                Z = torch.cholesky_solve(Y, L)               # solves (XXᵀ+λI)Z = Y
                C2_32 = X.T @ Z                              # (t,d)
            else:
                # C2 = (XᵀX + λI)^{-1} Xᵀ Y
                XtX = X.T @ X
                XtX = 0.5 * (XtX + XtX.T)
                XtX.diagonal().add_(lam)
                L = torch.linalg.cholesky(XtX)               # (t,t)
                XtY = X.T @ Y
                C2_32 = torch.cholesky_solve(XtY, L)         # (t,d)
        else:
            raise ValueError(f"Unknown solver: {solver}. Must be 'pinv', 'cholesky', or 'lstsq'.")

        return C2_32.to(dtype_param)

    def _direct_C2(
        self,
        C1: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        indices: list = None
    ) -> torch.Tensor:
        """Take C2 from V at `indices`, or at each C1 row's nearest key in K."""
        if indices is not None:
            if isinstance(indices, list):
                indices = torch.tensor(indices, device=V.device, dtype=torch.long)
            C2 = V[indices]  # (t, d)
            return C2

        t, d = C1.shape
        T = K.shape[0]

        C1_norms_sq = (C1 ** 2).sum(dim=1)  # (t,)
        K_norms_sq = (K ** 2).sum(dim=1)    # (T,)
        pairwise_dots = C1 @ K.T            # (t, T)

        squared_distances = C1_norms_sq.unsqueeze(1) + K_norms_sq.unsqueeze(0) - 2 * pairwise_dots

        nearest_indices = squared_distances.argmin(dim=1)  # (t,)

        C2 = V[nearest_indices]  # (t, d)

        return C2

    def _compute_C2_with_method(
        self,
        C1: torch.Tensor,
        beta: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        queries: torch.Tensor,
        method: str = 'lsq',
        indices: list = None,
        attention_bias: torch.Tensor = None,
        ridge_lambda: float = 0,
        solver: str = 'lstsq',
        ridge_scale: str = 'spectral',
        **kwargs
    ) -> torch.Tensor:
        """C2 by least squares ('lsq', see _compute_C2) or by selection ('direct', see _direct_C2)."""
        if method == 'direct':
            return self._direct_C2(C1, K, V, indices)
        elif method == 'lsq':
            return self._compute_C2(
                C1, beta, K, V, queries,
                attention_bias=attention_bias,
                ridge_lambda=ridge_lambda,
                solver=solver,
                ridge_scale=ridge_scale
            )
        else:
            raise ValueError(f"Unknown C2 computation method: {method}. Must be 'lsq' or 'direct'.")

    @staticmethod
    def _nnls_pg(M: torch.Tensor, y: torch.Tensor, iters: int = 0,
                   lower_bound: float = 1e-12, upper_bound: float = None) -> torch.Tensor:
        """
        Solve min_B ||M B - y||^2 s.t. lower_bound <= B <= upper_bound (fp32).

        Starts from clamped least squares, then runs `iters` projected-gradient
        steps with step size 1 / ||M||_2^2.
        """
        n, t = M.shape
        min_val = 1e-12 if lower_bound is None else lower_bound

        lam = 0

        if lam == 0:
            try:
                B = torch.linalg.lstsq(M, y.unsqueeze(1), driver='gels').solution.squeeze(1)
                if torch.isnan(B).any():
                    raise RuntimeError("NaNs in NNLS lstsq solution")
            except Exception as e:
                print(f"lstsq failed ({e}), increasing lambda to 1e-5*(||M||_2^2) and retrying")
                print(f"  M has NaN: {torch.isnan(M).any().item()}, Inf: {torch.isinf(M).any().item()}")
                print(f"  y has NaN: {torch.isnan(y).any().item()}, Inf: {torch.isinf(y).any().item()}")
                lam = 1e-6
                if n < t:
                    MMt = M @ M.T
                    MMt = 0.5 * (MMt + MMt.T)
                    MMt.diagonal().add_(lam)
                    R = torch.linalg.cholesky(MMt)
                    alpha = torch.cholesky_solve(y.unsqueeze(1), R).squeeze(1)
                    B = M.T @ alpha
                else:
                    MtM = M.T @ M
                    MtM = 0.5 * (MtM + MtM.T)
                    MtM.diagonal().add_(lam)
                    R = torch.linalg.cholesky(MtM)
                    Mty = M.T @ y
                    B = torch.cholesky_solve(Mty.unsqueeze(1), R).squeeze(1)
                if torch.isnan(B).any():
                    raise RuntimeError("NaNs in NNLS cholesky solution")
        elif n < t:  # underdetermined
            MMt = M @ M.T
            MMt.diagonal().add_(lam)
            R = torch.linalg.cholesky(MMt)
            alpha = torch.cholesky_solve(y.unsqueeze(1), R).squeeze(1)
            B = M.T @ alpha
        else:  # overdetermined
            MtM = M.T @ M
            MtM.diagonal().add_(lam)
            R = torch.linalg.cholesky(MtM)
            Mty = M.T @ y
            B = torch.cholesky_solve(Mty.unsqueeze(1), R).squeeze(1)

        B = B.clamp_min_(min_val)
        if upper_bound is not None:
            B = B.clamp_max_(upper_bound)

        if iters == 0:
            return B

        # Power iteration for ||M||_2.
        u = torch.randn(t, device=M.device, dtype=M.dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(3):
            v = M @ u
            if v.norm() == 0:
                break
            v = v / v.norm()
            u = M.T @ v
            if u.norm() == 0:
                break
            u = u / u.norm()
        sigma = (u @ (M.T @ (M @ u))).sqrt().clamp_min(1e-6)  # ~||M||_2
        L = (sigma ** 2).clamp_min(1e-6)
        eta = 1.0 / L

        for _ in range(iters):
            grad = M.T @ (M @ B - y)
            B = B - eta * grad
            B = B.clamp_min_(min_val)
            if upper_bound is not None:
                B = B.clamp_max_(upper_bound)

        return B
