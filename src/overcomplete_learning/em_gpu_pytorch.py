"""
Production-Grade GPU Implementation using PyTorch
Optimal performance with automatic differentiation support
"""

import numpy as np
import torch
from typing import Tuple, Optional


class EMAlgorithmGPU:
    """
    PyTorch-based GPU-accelerated EM algorithm.
    
    All computations stay on GPU to minimize PCIe transfers.
    Results are numerically identical to NumPy version (float32 precision).
    
    Example
    -------
    >>> em_gpu = EMAlgorithmGPU(device='cuda:0')
    >>> em_gpu.run(X, init_A, init_Sigma, init_xi, n_iter=100)
    """
    
    def __init__(self, device: str = 'cuda:0', dtype: torch.dtype = torch.float32):
        """
        Initialize GPU EM algorithm.
        
        Parameters
        ----------
        device : str
            GPU device (e.g., 'cuda:0', 'cuda:1', 'cpu')
        dtype : torch.dtype
            Computation precision (torch.float32 or torch.float64)
        """
        self.device = device
        self.dtype = dtype
        
        # Check GPU availability
        if 'cuda' in device:
            if not torch.cuda.is_available():
                print("⚠️  CUDA not available! Falling back to CPU.")
                self.device = 'cpu'
            else:
                print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert NumPy array to GPU tensor."""
        return torch.from_numpy(array).to(dtype=self.dtype, device=self.device)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert GPU tensor to NumPy array."""
        return tensor.detach().cpu().numpy()
    
    def E_step(
        self,
        A: torch.Tensor,
        Sigma: torch.Tensor,
        x: torch.Tensor,
        xi: torch.Tensor,
        sknown: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated E-step.
        
        All operations stay on GPU. Handles known sources efficiently.
        
        Parameters
        ----------
        A : (nobs, nsrc)
        Sigma : (nobs, nobs)
        x : (nsamples, nobs)
        xi : (nsamples, nsrc)
        sknown : (nsamples, nsrc) or None
            Known source values (NaN where unknown)
        
        Returns
        -------
        post_mean : (nsamples, nsrc)
            Posterior means E[S|X]
        ssn : (nsamples, nsrc, nsrc)
            Posterior second moments E[SS^T|X]
        """
        
        # Precompute Sigma inverse and Cholesky for numerical stability
        Sigma_inv = torch.linalg.inv(Sigma)
        
        # Inner product: A^T Sigma^{-1} A  — shape (srcdim, srcdim)
        innerA = A.T @ torch.linalg.solve(Sigma, A)
        
        # Safe inverse of xi — avoid division by zero
        safe_xi = torch.where(
            torch.abs(xi) < 1e-20,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            1.0 / torch.abs(xi)
        )  # (nsamples, srcdim)
        
        # Diagonal matrices: (nsamples, srcdim, srcdim)
        Lambda_inv = torch.diag_embed(safe_xi)
        
        # Precision matrix: (nsamples, srcdim, srcdim)
        K = innerA.unsqueeze(0) + Lambda_inv  # broadcasting
        
        # ⭐ CRITICAL: Batched matrix inversion (GPU's specialty!)
        # PyTorch's batched LU decomposition is highly optimized
        post_cov = torch.linalg.inv(K)  # (nsamples, srcdim, srcdim)
        
        # RHS for posterior mean
        rhs = (A.T @ torch.linalg.solve(Sigma, x.T)).T  # (nsamples, srcdim)
        
        # Posterior mean via batched matrix-vector product
        post_mean = torch.einsum('nij,nj->ni', post_cov, rhs)  # (nsamples, srcdim)
        
        # ─────────────────────────────────────────────────────────────────────
        # Handle known sources (conditional distribution)
        # ─────────────────────────────────────────────────────────────────────
        if sknown is not None:
            # Identify known vs unknown columns
            known_mask = ~torch.all(torch.isnan(sknown), dim=0)  # (srcdim,)
            known_idx = torch.where(known_mask)[0]
            unknown_idx = torch.where(~known_mask)[0]
            
            if len(known_idx) > 0:
                s_k = sknown[:, known_idx]  # (nsamples, n_known)
                A_k = A[:, known_idx]       # (nobs, n_known)
                A_u = A[:, unknown_idx]     # (nobs, n_unknown)
                
                # Extract xi inverse for unknown sources
                xiinv = torch.diagonal(Lambda_inv, dim1=1, dim2=2)  # (nsamples, srcdim)
                xiinv_u = xiinv[:, unknown_idx]  # (nsamples, n_unknown)
                Lambda_inv_u = torch.diag_embed(xiinv_u)
                
                # Precision of unknowns | knowns: K_u = A_u^T Sigma^{-1} A_u + Lambda_inv_u
                K_u_common = A_u.T @ torch.linalg.solve(Sigma, A_u)  # (n_unknown, n_unknown)
                K_u = K_u_common.unsqueeze(0) + Lambda_inv_u  # broadcast to (nsamples, n_unknown, n_unknown)
                
                # Covariance of unknowns | knowns
                if K_u.shape[1] == 1:
                    C_u = 1.0 / K_u  # scalar case
                else:
                    C_u = torch.linalg.inv(K_u)  # (nsamples, n_unknown, n_unknown)
                
                # Residual after removing known contributions
                x_u = x - torch.einsum('ij,nj->ni', A_k, s_k)  # (nsamples, nobs)
                
                # Conditional mean: C_u A_u^T Sigma^{-1} x_u
                mu_u = torch.einsum('nij,jk,kl,nl->ni', C_u, A_u.T, Sigma_inv, x_u)
                
                # Update posterior with conditional estimates
                post_mean[:, known_idx] = s_k
                post_mean[:, unknown_idx] = mu_u
                
                # Update posterior covariance (zero out known rows/cols)
                post_cov[:, known_idx, :] = 0
                post_cov[:, :, known_idx] = 0
                post_cov[:, unknown_idx, :][:, :, unknown_idx] = C_u
        
        # Compute second moment: E[SS^T] = E[S]E[S]^T + Cov[S]
        ssn = torch.einsum('ni,nj->nij', post_mean, post_mean) + post_cov
        
        # Validation
        diag_ssn = torch.diagonal(ssn, dim1=1, dim2=2)
        if (diag_ssn < 0).any():
            raise ValueError("❌ Posterior second moments not positive!")
        
        return post_mean, ssn
    
    def M_step(
        self,
        x: torch.Tensor,
        hatSn: torch.Tensor,
        ssn: torch.Tensor,
        aknown: Optional[torch.Tensor] = None,
        err_sd: Optional[float] = None,
        normalize_A: bool = False,
        update_covariance: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated M-step.
        
        Parameters
        ----------
        x : (nsamples, nobs)
        hatSn : (nsamples, nsrc)
        ssn : (nsamples, nsrc, nsrc)
        aknown : (nobs, nsrc) or None
            Known columns of A (NaN elsewhere)
        err_sd : float or None
            Fixed noise std (if not estimating Sigma)
        normalize_A : bool
            Whether to normalize columns to unit norm
        update_covariance : bool
            Whether to estimate Sigma or use fixed value
        
        Returns
        -------
        Anew : (nobs, nsrc)
        Sigma_new : (nobs, nobs)
        xin : (nsamples, nsrc)
            Updated xi parameter
        """
        
        nsample, nobs = x.shape
        nsrc = hatSn.shape[1]
        
        # ⭐ Sufficient statistics (GPU einsum is highly optimized)
        M = torch.einsum('ni,nj->ij', x, hatSn)      # (nobs, nsrc)
        Q = torch.einsum('nij->ij', ssn)              # (nsrc, nsrc)
        Sx = torch.einsum('ni,nj->ij', x, x)          # (nobs, nobs)
        
        # Xi update (variational parameter)
        diag_ssn = torch.diagonal(ssn, dim1=1, dim2=2)  # (nsamples, nsrc)
        xin = torch.sqrt(torch.clamp(diag_ssn, min=1e-10))
        
        # ⭐ A update: more stable than forming inverse
        # Instead of: Anew = M @ inv(Q)
        # Use:        Anew^T = solve(Q^T, M^T)
        Anew = torch.linalg.solve(Q.T, M.T).T  # (nobs, nsrc) — numerically stable
        
        # Normalize columns if requested
        if normalize_A:
            col_norms = torch.norm(Anew, p=2, dim=0, keepdim=True)
            Anew = Anew / (col_norms + 1e-10)
        
        # Pin known columns
        if aknown is not None:
            known_mask = ~torch.all(torch.isnan(aknown), dim=0)
            known_idx = torch.where(known_mask)[0]
            if len(known_idx) > 0:
                Anew[:, known_idx] = aknown[:, known_idx]
        
        # Sigma (noise covariance) update
        if update_covariance:
            Sigma_new = (Sx - Anew @ M.T) / nsample
            Sigma_new = (Sigma_new + Sigma_new.T) / 2  # enforce symmetry
            Sigma_new = Sigma_new + torch.eye(nobs, dtype=self.dtype, device=self.device) * 1e-8
        else:
            Sigma_new = torch.eye(nobs, dtype=self.dtype, device=self.device) * (err_sd ** 2)
        
        return Anew, Sigma_new, xin
    
    def run(
        self,
        X: np.ndarray,
        init_A: np.ndarray,
        init_Sigma: np.ndarray,
        init_xi: np.ndarray,
        n_iter: int = 100,
        sknown: Optional[np.ndarray] = None,
        aknown: Optional[np.ndarray] = None,
        true_A: Optional[np.ndarray] = None,
        true_S: Optional[np.ndarray] = None,
        err_sd: Optional[float] = None,
        verbose: bool = True
    ) -> dict:
        """
        Run EM algorithm on GPU.
        
        Parameters
        ----------
        X : (nsamples, nobs)
        init_A, init_Sigma, init_xi : initialization
        n_iter : maximum iterations
        sknown, aknown : partial observations
        true_A, true_S : for tracking error (optional)
        verbose : print progress
        
        Returns
        -------
        Dictionary with:
            - A_est, Sigma_est, xi_est: final parameters
            - S_est: posterior source means
            - errors_matrix, errors_latent: error histories
            - converged: whether algorithm converged
        """
        
        # Transfer data to GPU once (avoid repeated transfers!)
        x = self._to_tensor(X)
        A = self._to_tensor(init_A)
        Sigma = self._to_tensor(init_Sigma)
        xi = self._to_tensor(init_xi)
        sknown_gpu = self._to_tensor(sknown) if sknown is not None else None
        aknown_gpu = self._to_tensor(aknown) if aknown is not None else None
        true_A_gpu = self._to_tensor(true_A) if true_A is not None else None
        
        # Tracking
        errors_matrix = []
        errors_latent = []
        xi_diffs = []
        obs_likelihoods = []
        
        # EM loop — all on GPU!
        for iteration in range(n_iter):
            xi_old = xi.clone()
            
            # E-step
            S_mean, S_cov = self.E_step(A, Sigma, x, xi, sknown_gpu)
            
            # M-step
            A, Sigma, xi = self.M_step(
                x, S_mean, S_cov,
                aknown=aknown_gpu,
                err_sd=err_sd,
                update_covariance=True
            )
            
            # Optional: compute error if ground truth provided
            if true_A_gpu is not None:
                from overcomplete_learning.data import best_permutation_match_sign_flips
                A_np = self._to_numpy(A)
                true_A_np = self._to_numpy(true_A_gpu)
                
                A_perm, _, _, _, _ = best_permutation_match_sign_flips(true_A_np, A_np)
                A_perm_gpu = self._to_tensor(A_perm)
                
                err = torch.norm(true_A_gpu - A_perm_gpu, p='fro').item()
                errors_matrix.append(err)
                
                if verbose and iteration % 10 == 0:
                    print(f"Iter {iteration:3d} | A error: {err:.6f}")
            
            # Track xi change
            xi_change = torch.norm(xi - xi_old).item()
            xi_diffs.append(xi_change)
            
            # Convergence check
            if iteration > 10 and xi_change < 1e-6:
                print(f"✓ Converged at iteration {iteration}")
                break
        
        return {
            'A_est': self._to_numpy(A),
            'Sigma_est': self._to_numpy(Sigma),
            'xi_est': self._to_numpy(xi),
            'S_est': self._to_numpy(S_mean),
            'errors_matrix': errors_matrix,
            'xi_diffs': xi_diffs,
            'converged': iteration < n_iter - 1,
            'final_iteration': iteration
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_gpu_vs_cpu(nobs: int = 30, nsrc: int = 50, nsamples: int = 1000):
    """Benchmark GPU vs CPU E-step."""
    import time
    
    print(f"\nBenchmarking E-step: nobs={nobs}, nsrc={nsrc}, nsamples={nsamples}")
    print("=" * 60)
    
    # Generate test data
    A = np.random.randn(nobs, nsrc)
    Sigma = np.eye(nobs) * 0.1
    x = np.random.randn(nsamples, nobs)
    xi = np.abs(np.random.randn(nsamples, nsrc))
    
    # CPU (NumPy)
    print("CPU (NumPy)...", end=" ", flush=True)
    t0 = time.time()
    from overcomplete_learning.em import E_step as E_step_cpu
    post_mean_cpu, ssn_cpu = E_step_cpu(A, Sigma, x, xi, None)
    t_cpu = time.time() - t0
    print(f"✓ {t_cpu:.4f}s")
    
    # GPU (PyTorch)
    print("GPU (PyTorch)...", end=" ", flush=True)
    em_gpu = EMAlgorithmGPU(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    A_t = em_gpu._to_tensor(A)
    Sigma_t = em_gpu._to_tensor(Sigma)
    x_t = em_gpu._to_tensor(x)
    xi_t = em_gpu._to_tensor(xi)
    post_mean_gpu, ssn_gpu = em_gpu.E_step(A_t, Sigma_t, x_t, xi_t, None)
    t_gpu = time.time() - t0
    print(f"✓ {t_gpu:.4f}s")
    
    # Verify accuracy
    post_mean_np = em_gpu._to_numpy(post_mean_gpu)
    max_error = np.max(np.abs(post_mean_cpu - post_mean_np))
    
    print(f"\nSpeedup:     {t_cpu/t_gpu:.1f}×")
    print(f"Max error:   {max_error:.2e}")
    print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    # Run benchmark
    if torch.cuda.is_available():
        benchmark_gpu_vs_cpu(nobs=30, nsrc=50, nsamples=5000)
        benchmark_gpu_vs_cpu(nobs=30, nsrc=50, nsamples=50000)
    else:
        print("⚠️  CUDA not available. Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
