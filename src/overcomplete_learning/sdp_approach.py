import time
import numpy as np
import scipy.sparse.linalg as sp_linalg
import random
import overcomplete_learning.data as ol_data

# ===================================================================
# 1. Main Entry Point & Options
# ===================================================================

def set_default_opts():
    """Returns the default options for the OverICA algorithm."""
    return {
        's': 5,     # Number of generalized covariances multiplier
        't': None   # Scaling parameter
    }

def check_opts(opts):
    defaults = {'s': 5, 't': None}
    if opts is None: return defaults
    for k, v in defaults.items():
        opts.setdefault(k, v)
    return opts

def overica(X, k, rng, err_sd, opts=None, EM_iter=100, mu = 5, ):
    """
    Overcomplete ICA via SDP (Podosinnikova et al. 2019)
    With optimized Matrix-Vectorized FISTA for source estimation.
    """
    
    opts = check_opts(opts)
    n_samples, n_features = X.shape
    globtt = time.time()
    
    # --- 1. Subspace Estimation ---
    #print('Computing generalized covariances...')
    s = opts['s'] * k
    t_param = opts['t']
    if t_param is None:
        cov_matrix = np.cov(X, rowvar=False)
        t_param = 0.05 / np.sqrt(np.max(np.abs(cov_matrix)))
        
    C = estimate_gencovs(X, s, t_param)
    cum_time = time.time() - globtt
    
    # --- 2. SVD / Basis Selection ---
    #print('Computing SVD...')
    globtt_svd = time.time()
    if k >= min(C.shape) - 1:
        U, _, _ = np.linalg.svd(C, full_matrices=False)
        Hs = U[:, :k]
    else:
        U, S, _ = sp_linalg.svds(C, k=k)
        Hs = U[:, np.argsort(S)[::-1]] # Sort descending
        
    svd_time = time.time() - globtt_svd
    
    # --- 3. Adaptive Deflation (SDP) ---
    #print('Adaptive deflation...')
    globtt_sdp = time.time()
    ds_est, Ds_est = sdp_adaptive(Hs, k, mu=mu)
    sdp_time = time.time() - globtt_sdp
    
    # --- 4. Optimized Source Estimation (Reuse logic from previous step) ---
    #print('Estimating sources (Vectorized FISTA)...')
    # A_est is ds_est: (n_features, k)
    
    
    #adaptive estimation of the noise variance and use as hyperparameter We fixed the sources to have b = 1. Estiamte 
    
    #run iterative loop to estimate best noise level
    sigma_init = np.abs(rng.normal())
    # Replace original two-line block with:
    Shat, X_est, sigma_final, outer = fista_scaled_lasso(
        X=X,
        ds_est=ds_est,
        b=1,                     # we fix the scaling of S to 1 by assumption
        sigma_init=sigma_init,   # choosing a random init
        EM_iter=EM_iter,
        #outer_iter=100,
        tol=1e-6,
        err_sd=err_sd
        )   
    
    total_time = time.time() - globtt
    
    out = {
        'S_est': Shat,
        'A_est': ds_est,
        'X_est': X_est,
        'converged': True, 
        'convergence_iteration': outer,
        'Hs': Hs,
        'Ds_est': Ds_est,
        'times': {
            'total_time': total_time,
            'cum_time': cum_time,
            'svd_time': svd_time,
            'sdp_time': sdp_time
        },
    }
    return out


# ===================================================================
# 2. Subspace Estimation (Generalized Covariance)
# ===================================================================

def gencov(X, omega):
    """Hessian of the cumulant generating function at direction `omega`."""
    n, p = X.shape
    if np.isscalar(omega) or np.size(omega) == 1:
        omega = np.ones(p) * np.squeeze(omega) / p
        
    proj = X @ omega
    proj_max = np.max(proj)
    eproj = np.exp(proj - proj_max)
    sum_eproj = np.sum(eproj)
    
    Eomega = (X.T @ eproj) / sum_eproj
    C = (X.T * eproj) @ X / sum_eproj
    C = C - np.outer(Eomega, Eomega)
    
    # Enforce exact symmetry
    C = 0.5 * (C + C.T)
    return C.flatten(order='F')

def estimate_gencovs(X, s, t):
    """Constructs the differences of generalized covariances."""
    n, p = X.shape
    C = np.zeros((p**2, s))
    
    # Baseline at omega = 0
    G0_flat = gencov(X, 0.0) 
    
    for i in range(s):
        omega = np.random.randn(p)
        omega = t * omega
        C[:, i] = gencov(X, omega) - G0_flat
        
    return C


# ===================================================================
# 3. SDP Deflation Strategy (Adaptive)
# ===================================================================

def sdp_adaptive(Hs, k, mu):
    _, Fs = extract_basis(Hs, k)
    Ds_est = adaptive_deflation(Fs, k, mu)
    ds_est = approx_ds_from_Ds(Ds_est)
    return ds_est, Ds_est

def adaptive_deflation(Fs, k, mu, Ds_est=None):
    p = int(np.sqrt(Fs.shape[0]))
    
    if Ds_est is None:
        Ds_est = np.empty((Fs.shape[0], 0))

    k_current = Ds_est.shape[1]
    
    if k_current < k:
        kloc = k - k_current
        Fsloc = np.hstack((Fs, Ds_est)) if k_current > 0 else Fs.copy()

        # Corrected Deflation Loop Fragment
        for i in range(kloc):

            # Find the null space of the current constraints
            U, S, Vh = np.linalg.svd(Fsloc, full_matrices=True)
            # The columns of U corresponding to zero singular values form the basis Esloc
            # For a matrix of shape (p^2, current_constraints), take the trailing columns
            # Calculate how many constraint columns we currently have
            n_constraints = Fsloc.shape[1]
            ambient_dim = U.shape[1] # This is p^2
            
            if n_constraints < ambient_dim:
            # Standard Path: Extract the null space (leftover columns)
                Esloc = U[:, n_constraints:]
            else:
                # Edge Case Guard: Fsloc is full-rank and filling the space.
                # Fall back to using the tightest remaining unaligned noise dimensions
                Esloc = U[:, -1:]
            
            if Esloc.shape[1] == 0:
                print(f"Warning: Subspace exhausted at iteration {i}. Terminating deflation early.")
                break
            
            G = Esloc[:, 0].reshape((p, p), order='F')
            D = majorize_minimize(G, Fsloc, mu = mu)
            
            d_flat = D.flatten(order='F').reshape(-1, 1)
            Ds_est = np.hstack((Ds_est, d_flat))
            Fsloc = np.hstack((Fsloc, d_flat)) # Securely grow constraint matrix

    return Ds_est


# ===================================================================
# 4. Core Optimization (SDP / FISTA)
# ===================================================================

def majorize_minimize(G, Fs, mu, mm_tol=1e-5, nmmmax=100):
    p = int(np.sqrt(Fs.shape[0]))
    Dinit = np.eye(p) / p
    D = Dinit.copy()
    
    maxiter = 100
    tolerance = 1e-3
    iter_count = 1
    for iter_count in range(1, nmmmax + 1):
        D_old = D.copy() # Cache the previous state to calculate delta
        u, _ = extract_largest_eigenvector(G)
        Ginit = np.outer(u, u)
        D = solve_relaxation_mezcal_approx_fista(
            Fsbasis=Fs, G=Ginit, mu=mu, Dinit=Dinit, maxiter=maxiter, tolerance=tolerance
        )
        G = 0.5 * (D + D.T)
        matrix_delta = np.linalg.norm(D - D_old, 'fro')
        if matrix_delta < mm_tol:
            break
            
    return D

def fista_scaled_lasso(
    X,           # (n, nobs) observations
    ds_est,      # (nobs, nsrc) dictionary / mixing matrix estimate
    b,           # prior scale parameter — lambda = 2*sigma^2 / b
    sigma_init,  # initial noise level (fixed hyperparameter)
    err_sd,
    EM_iter=50,  # inner FISTA iterations
    #outer_iter=50,  # outer sigma update iterations
    tol=1e-4,    # convergence tolerance on sigma
):
    sigma = sigma_init
    Shat = None
    n_iter = 0
    #for outer in range(outer_iter):
    sigma_prev = sigma

        # Compute lambda from current sigma estimate
    lam = err_sd**2 / b

        # Inner FISTA solve: min ||X - ds_est @ Shat.T||_2^2 + lam * ||Shat||_1
        # ASSUMPTION: lam is passed as keyword argument to FISTA
    Shat,_ = FISTA_vectorized_inference(X, ds_est, niter=EM_iter, beta=lam)
    n_iter += _
    # Reconstruct X from current estimate
    X_est = (ds_est @ Shat.T).T  # (n, nobs)

    # Sun & Zhang (2012) sigma update: mean squared residual
    residual = X - X_est          # (n, nobs)
    n, nobs = X.shape
    sigma = np.sqrt(np.sum(residual**2) / (n * nobs))

            # Convergence check on sigma
    #    if abs(sigma - sigma_prev) < tol:
    #        print(f"Sigma converged at outer iteration {outer + 1}: sigma={sigma:.6f}")
    #        break
#
    #else:
    #    print(f"Sigma did not converge after {outer_iter} iterations. Final sigma={sigma:.6f}")

    return Shat, X_est, sigma, n_iter

def FISTA_vectorized_inference(X, A, niter=100, beta=1.0):
    """
    Solves: min 0.5 ||X - S A^T||^2_F + beta ||S||_1
    """
    n_samples, n_features = X.shape
    _, n_sources = A.shape
    
    # Precompute Lipschitz step size
    L = np.linalg.norm(A.T @ A, ord=2)
    step_size = 1.0 / L
    
    # Precompute constant parts of the gradient
    AtA = A.T @ A
    XAt = X @ A
    
    S = np.zeros((n_samples, n_sources))
    Y = S.copy()
    t_fista = 1.0
    
    for _ in range(niter):
        S_old = S.copy()
        
        # Matrix-matrix multiplication is much faster than per-sample loops
        grad = Y @ AtA - XAt
        
        # Proximal Step: Soft Thresholding
        arg = Y - step_size * grad
        
        S = np.sign(arg) * np.maximum(np.abs(arg) - beta * step_size, 0)
        
        # Nesterov Acceleration
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_fista**2)) / 2.0
        Y = S + ((t_fista - 1.0) / t_next) * (S - S_old)
        t_fista = t_next
        
        if np.linalg.norm(S-S_old)<1e-4:
            break
        
    return S, _


def solve_relaxation_mezcal_approx_fista(Fsbasis, G, mu, Dinit, maxiter, tolerance):
    d_sq, _ = Fsbasis.shape
    d = int(np.sqrt(d_sq))

    B = Dinit.copy()
    Y = B.copy()
    L = mu
    t = 1.0

    primal_vals = np.zeros(maxiter)
    dual_vals = np.zeros(maxiter)

    for i in range(maxiter):
        temp = Fsbasis.T @ Y.flatten(order='F')
        grad = -G + mu * (Fsbasis @ temp).reshape((d, d), order='F')
        Y = Y - (1.0 / L) * grad

        tnew = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t**2))
        Ysym = 0.5 * (Y + Y.T)

        e_vals, U = np.linalg.eigh(Ysym)
        e_vals = np.real(e_vals)
        U = np.real(U)

        eproj = proj_simplex(e_vals)
        Bnew = U @ np.diag(eproj) @ U.T
        Bnew = 0.5 * (Bnew + Bnew.T)

        Y = Bnew + ((t - 1.0) / tnew) * (Bnew - B)
        B = Bnew
        t = tnew

        if (i % 10) == 0:
            temp = Fsbasis.T @ B.flatten(order='F')
            grad = -G + mu * (Fsbasis @ temp).reshape((d, d), order='F')

            primal_vals[i] = -np.sum(G * B) + (mu / 2.0) * np.sum(temp**2)
            dual_vals[i] = np.min(np.linalg.eigvalsh(grad)) - (mu / 2.0) * np.sum(temp**2)

            dual_max = np.max(dual_vals[: i + 1 : 10])
            if (primal_vals[i] - dual_max) < tolerance:
                break

    return B

def proj_simplex(v):
    v = np.maximum(v, 0.0)
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    indices = np.arange(1, len(v) + 1)
    condition = u > (sv - 1.0) / indices
    rho = np.where(condition)[0][-1]  
    theta = max(0.0, (sv[rho] - 1.0) / (rho + 1.0))
    w = np.maximum(v - theta, 0.0)
    return w




# ===================================================================
# 5. Utilities & Subroutines
# ===================================================================

def extract_basis(Es, k):                               
    Q, _ = np.linalg.qr(Es, mode='complete')
    Esbasis = Q[:, :k] 
    Fsbasis = Q[:, k:] 
    return Esbasis, Fsbasis

def approx_ds_from_Ds(Ds_est):
    p = int(np.sqrt(Ds_est.shape[0]))
    k = Ds_est.shape[1]
    ds_est = np.zeros((p, k))
    
    for i in range(k):
        Di = Ds_est[:, i].reshape((p, p), order='F')
        u, e = extract_largest_eigenvector(Di)
        ds_est[:, i] = u * np.sqrt(max(e, 0.0))
        
    return ds_est

def extract_largest_eigenvector(D):
    D = 0.5 * (D + D.T)
    e_vals, e_vecs = np.linalg.eig(D)
    idx = np.argmax(np.abs(e_vals))
    u = np.real(e_vecs[:, idx])
    e = np.real(e_vals[idx])
    return u, e