import numpy as np

# ============================================================
# FastICA (Independent Component Analysis)
# ============================================================
# Mathematical reference: Hyvärinen & Oja (2000)
#
# Type Map:
#   X : Raw observed signals -> np.ndarray, shape (n_samples, n_features)
#   S : Latent sources -> np.ndarray, shape (n_samples, n_components)
#   W_final : Total Un-mixing matrix -> np.ndarray, shape (n_components, n_features)
#   A_hat : True Mixing matrix estimate -> np.ndarray, shape (n_features, n_components)
# ============================================================


def _get_contrast_functions(name):
    if name == 'logcosh':
        def g(u): return np.tanh(u)
        def g_prime(u): return 1 - np.tanh(u)**2
    elif name == 'exp':
        def g(u): return u * np.exp(-(u**2) / 2)
        def g_prime(u): return (1 - u**2) * np.exp(-(u**2) / 2)
    elif name == 'cube':
        def g(u): return u**3
        def g_prime(u): return 3 * u**2
    else:
        raise ValueError(f"Unknown contrast function: {name}")
    return g, g_prime

def whiten_canonical(X_T, n_components):
    """
    Standard Whitening/PCA reduction per Hyvärinen & Oja.
    X_T shape: (n_features, n_samples)
    return
    
    K: whitnetning matrix (nob, nobs)
    X_centred               (nobs, n)
    X_mean                  (nob, n)
    """
    # 1. Center
    X_mean = X_T.mean(axis=1, keepdims=True)
    X_centered = X_T - X_mean
    
    # 2. Covariance E[xx^T]
    cov = np.cov(X_centered)
    
    # 3. EVD
    d, E = np.linalg.eigh(cov)
    
    # 4. Sort and Truncate (Hyvärinen's recommendation for undercomplete)
    # We take the m largest eigenvalues to separate signal from noise
    idx = np.argsort(d)[::-1]
    d = d[idx][:n_components]
    E = E[:, idx][:, :n_components]
    
    # 5. Whitening Matrix K = D^{-1/2} E^T
    K = np.diag(1.0 / np.sqrt(d + 1e-12)) @ E.T
    
    return K @ X_centered, K, X_mean

def fast_ica(X, n_components=None, approach='parallel', contrast='logcosh', 
             max_iter=200, tol=1e-6, whiten=True, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X_T = X.T
    if n_components is None:
        n_components = X_T.shape[0]
    n_features, n_samples = X_T.shape
    if whiten:
        X_proc, K, X_mean = whiten_canonical(X.T, n_components)
    else:
        X_mean = X_T.mean(axis=1, keepdims=True)
        X_proc = X_T - X_mean
        K = np.eye(n_features)
    
    g, g_prime = _get_contrast_functions(contrast)

    W_rot = rng.standard_normal((n_components, n_components))
    
    converged = False
    total_iters = 0

    if approach == 'parallel':
        # Initial symmetric orthogonalization
        u, s, vh = np.linalg.svd(W_rot, full_matrices=False)
        W_rot = u @ vh
        
        for i in range(1, max_iter + 1):
            total_iters = i
            W_old = W_rot.copy()
            dot_product = W_rot @ X_proc
            
            term1 = (g(dot_product) @ X_proc.T) / n_samples
            term2 = np.mean(g_prime(dot_product), axis=1, keepdims=True) * W_rot
            W_rot = term1 - term2
            
            u, s, vh = np.linalg.svd(W_rot, full_matrices=False)
            W_rot = u @ vh
            
            if np.max(1 - np.abs(np.diag(W_rot @ W_old.T))) < tol:
                converged = True
                break
                
    elif approach == 'deflation':
        iters_per_comp = []
        for p in range(n_components):
            w = W_rot[p, :].reshape(-1, 1)
            w /= np.linalg.norm(w)
            comp_converged = False
            
            for i in range(1, max_iter + 1):
                w_old = w.copy()
                dot_product = w.T @ X_proc
                w = (X_proc @ g(dot_product).T) / n_samples - np.mean(g_prime(dot_product)) * w
                
                if p > 0:
                    W_prev = W_rot[:p, :]
                    w -= W_prev.T @ (W_prev @ w) 
                
                w /= np.linalg.norm(w)
                if 1 - np.abs(w.T @ w_old) < tol:
                    comp_converged = True
                    iters_per_comp.append(i)
                    break
            
            W_rot[p, :] = w.flatten()
        
        converged = len(iters_per_comp) == n_components
        total_iters = np.sum(iters_per_comp) # Return list for deflation

    W_total = W_rot @ K
    S = (W_total @ (X_T - X_mean)).T
    A_hat = np.linalg.pinv(W_total)
    
    # [Fix 3]: Add back the mean for true reconstruction
    X_reconstructed = (A_hat @ S.T).T + X_mean.T

    return {
        'A_est': A_hat,
        'S_est': S, 
        'X_est': X_reconstructed,
        'converged': converged,
        'convergence_iteration': total_iters
    }
