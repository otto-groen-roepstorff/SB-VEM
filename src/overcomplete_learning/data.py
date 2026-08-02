#loading packages
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy
import pandas as pd
from dataclasses import dataclass
import overcomplete_learning.data as ol_data
import glob

#----- DATA GENERATION-----
#create quasi orthogonal matricies
def quasi_orthogonal_matrix(
    dobs: int,
    dsrc: int,
    Iter: int,
    dd1: float,
    dd2: float,
    thres: float,
    rng,
    verbose: bool = False,
) -> np.ndarray:
    """
    Faithful Python translation of the provided MATLAB code.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Initialization ---
    D = rng.standard_normal((dobs, dsrc))
    # Normalize columns
    D /= np.linalg.norm(D, axis=0)
    # Gram matrix
    G = D.T @ D
    # Welch bound
    if dsrc <=1:
        return D

    mu = np.sqrt((dsrc - dobs) / (dobs * (dsrc - 1))) 
    
    # Precompute index used in MATLAB:
    matlab_index = int(round(dd1 * (dsrc * dsrc - dsrc))) #finding top dd1% greatest value
    python_index = max(matlab_index, 0) #doing python indexing
    iteration_list_max = []
    iteration_list_means = []
    for k in range(Iter):
        # ---- Shrink large inner products ----
        absG_flat = np.abs(G).flatten()
        gg = np.sort(absG_flat)
        threshold = gg[python_index] 
        #finding entries with entries greater than dd1% while still not being colinear
        off_diag = ~np.eye(dsrc, dtype=bool)
        mask = (np.abs(G) >= threshold) & off_diag
        G[mask] *= dd2 #multiply big off-diagonals by dd2

        # ---- Rank projection ----
        U, S, Vt = np.linalg.svd(G, full_matrices=False)
        S[dobs:] = 0  # enforce rank N
        G = U @ np.diag(S) @ Vt

        # ---- Normalize diagonal to 1 ----
        #G = normalize_columns(G)
        diag_vals = np.sqrt(np.diag(G))
        G = (G / diag_vals).T / diag_vals
        G = G.T  # clean symmetric normalization

        # ---- Status output ----
        absG_flat = np.abs(G).flatten()
        #gg = np.sort(absG_flat)
        #threshold = gg[python_index]

        mean_corr = np.mean(np.abs(G[off_diag]))
        off_diag_elements = np.abs(G[off_diag])
        max_corr = np.max(off_diag_elements) if off_diag_elements.size > 0 else 0.0
        #max_corr = np.max(np.abs(G[off_diag]))
        
        iteration_list_max.append(max_corr)
        iteration_list_means.append(mean_corr)
        if(verbose):
            print(f"Iter {k+1}: Welch={mu:.6f}, "f"mean={mean_corr:.6f}, max={max_corr:.6f}")
        if (k>10 and (max(
            abs((iteration_list_max[k-1]-iteration_list_max[k])),
            abs((iteration_list_means[k-1]-iteration_list_means[k])))<thres)):
            break
        #if not np.any(mask):
        #    break

    # ---- Final factorization ----
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    D_final = np.sqrt(np.diag(S[:dobs])) @ U[:, :dobs].T
    D_final = normalize_columns(D_final)
    return D_final

#--- generate S  
def generate_laplace(dsource: int, n_samples: int, seed, scale:float = 1):
    '''
    Generates mean-0, variance 2 laplace observations and returns (dsource, n_samples) tuple
    dsource: number of independent sources
    out: a function that takes n 
    '''
    np.random.seed(seed=seed)
    return np.random.laplace(loc = 0, scale = scale, size =(n_samples,dsource)) #shape = (dsource, n_samples)

#--- generate X given A and S
def generate_X(A, S, noisy_flag: bool, non_sparsity=None):
    mutual_coherence_lower_bound = maximal_non_sparsity(A, verbose=False)
    N, nsrc = S.shape

    # Default behaviour: [1, 2, 3] multiples of the coherence bound
    if non_sparsity is None:
        non_sparsity = [1, 2, 3]

    # Resolve each entry as an absolute number of non-zeros
    k_values = [min(k, nsrc - 1) for k in non_sparsity]
    #[min(k * mutual_coherence_lower_bound, nsrc - 1)
    #            for k in non_sparsity]

    out_dictionary = {
        'x_all':  (A @ S.T).T,
        's_all':  S,
        'nonzero':   mutual_coherence_lower_bound
    }

    for mult, k in zip(non_sparsity, k_values):
        S_sparse = np.zeros_like(S)
        for i in range(N):
            idx = np.random.choice(nsrc, size=k, replace=False) #only allowing for k non-zero values
            S_sparse[i, idx] = S[i, idx]
        X_sparse = (A @ S_sparse.T).T

        # Keys: x_1.nonsparse, s_1.nonsparse, x_2.nonsparse, s_2.nonsparse, ...
        suffix = f'{mult}.nonsparse'
        out_dictionary[f'x_{suffix}'] = X_sparse
        out_dictionary[f's_{suffix}'] = S_sparse

    return out_dictionary

#generating the model with noise. err_sd is the standard deviation of the noise and controls the SNR.
def generate_noisy_model(nsrc: int, nobs: int, nreps: int, seed: int, Iter, dd1, dd2:float, verbose, thres, err_sd, A_scale, S_scale = 1):
    S = generate_laplace(dsource= nsrc, n_samples=nreps, seed = seed, scale= S_scale)  #generate laplace (nreps, nsrc) 
    A = quasi_orthogonal_matrix(dobs=nobs, dsrc=nsrc,  Iter = Iter, dd1=dd1, dd2=dd2, verbose=verbose,thres = thres, seed=seed)*A_scale #generate quasiorthgonal matrix #(nobs, nreps)
    if(err_sd ==0):
        eps = np.zeros(shape=(nreps, nobs))
    else:
        eps = np.random.normal(loc = 0, scale = err_sd, size = (nreps, nobs))
    
    X = (A @ S.T).T + eps
    data = {}
    data['S'] = S
    data['A'] = A
    data['eps'] = eps
    data['X'] = X
    return(data)


@dataclass #defining a class for the EM data to store the generated data and the true values of A, S and X for later evaluation and debugging
class EMData:
    X     : np.ndarray   # (nreps, nobs)
    A     : np.ndarray   # (nobs, nsrc)
    S     : np.ndarray   # (nreps, nsrc)
    eps   : np.ndarray   # (nreps, nobs)

# -- GENERATE DATA: Essentially the same as ol_em.data.generate_noisy_data, just with different output format (EMData dataclass) and more parameters exposed for flexibility in the study.

#--- DATA STRUCTURES-----

def generate_correlated_errors(nreps, nobs, err_sd, correlation_type='random', rng=None):
    """
    Generates correlated errors by multiplying i.i.d. noise with an
    orthonormal matrix Q, giving eps ~ N(0, err_sd^2 * Q Q^T) = N(0, err_sd^2 * I)
    
    Note: Q orthonormal preserves variance (Q Q^T = I) but introduces
    correlation structure BETWEEN observations within each sample.
    
    If you want correlation ACROSS the nobs dimension with non-identity
    covariance, use a non-orthonormal matrix L (Cholesky factor) instead.

    Parameters
    ----------
    nreps            : int    — number of samples
    nobs             : int    — observation dimension  
    err_sd           : float  — noise standard deviation
    correlation_type : str    — 'random', 'banded', or 'toeplitz'
    seed             : int    — random seed

    Returns
    -------
    eps : (nreps, nobs) — correlated errors
    Q   : (nobs, nobs)  — orthonormal mixing matrix
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if err_sd == 0:
        return np.zeros((nreps, nobs)), np.eye(nobs)

    # ── Construct orthonormal Q ───────────────────────────────────────────────
    if correlation_type =='iid':
        Q = np.identity(nobs)
        
    elif correlation_type == 'random':
        # Random orthonormal matrix via QR decomposition
        M = rng.standard_normal(size=(nobs, nobs))
        Q, _ = np.linalg.qr(M)                          # (nobs, nobs) orthonormal

    elif correlation_type == 'banded':
        # Banded correlation — nearby observations are correlated
        # Construct symmetric PD matrix then orthogonalise
        M = np.zeros((nobs, nobs))
        for i in range(nobs):
            for j in range(nobs):
                M[i, j] = np.exp(-abs(i - j))           # exponential decay
        Q, _ = np.linalg.qr(M)

    elif correlation_type == 'toeplitz':
        # Toeplitz structure — stationary correlation
        from scipy.linalg import toeplitz
        rho  = 0.5
        col  = rho ** np.arange(nobs)
        M    = toeplitz(col)
        Q, _ = np.linalg.qr(M)
    else:
        raise ValueError(f'Unknown correlation_type: {correlation_type}')

    # ── Verify orthonormality ─────────────────────────────────────────────────
    assert np.allclose(Q @ Q.T, np.eye(nobs), atol=1e-10), 'Q is not orthonormal'

    # ── Generate i.i.d. errors and mix ───────────────────────────────────────
    eps_iid = np.random.normal(0, err_sd, (nreps, nobs))  # (nreps, nobs)
    eps     = eps_iid @ Q.T                                # (nreps, nobs)

    return eps, Q

def generate_data(nreps, nobs, nsrc, err_sd, S_scale=1.0,
                  dd1=0.9, dd2=0.9, thres=1e-4, Iter=1000,
                  A_scale=1, correlation_type = 'iid', rng=None) -> EMData:
    if rng is None:
        rng = np.random.default_rng()
    S   = rng.laplace(loc=0, scale=S_scale, 
                                    size=(nreps, nsrc))
    #A   = rng.normal(size = (nobs, nsrc))
    A = normalize_columns(rng.normal(size = (nobs, nsrc)))
    #A = ol_data.quasi_orthogonal_matrix(
    #          dobs=nobs, dsrc=nsrc, Iter=Iter,
    #          dd1=dd1, dd2=dd2, thres=thres, rng = rng,
    #      ) * A_scale
    eps, Q = generate_correlated_errors(nreps=nreps, nobs = nobs, err_sd=err_sd, correlation_type=correlation_type, rng=rng)
    
    X   = (A @ S.T).T + eps
    return EMData(X=X, A=A, S=S, eps=eps)


#initialized parameters for EM
def initialise_parameters(nobs, nsrc, nreps, rng=None):
    """Returns (A_init, Sigma_init, xi_init)."""
    rng = np.random.default_rng(rng)
    A_init     = normalize_columns(rng.normal(loc=0, scale=1, size = (nobs, nsrc)))
    #_          = rng.normal(loc=1, scale=0, size = (nobs, nobs))
    Sigma_init = np.eye(nobs) * 1 #+ _ @ _.T 
    xi_init    = np.abs(
        rng.laplace(loc=0, scale=1, 
        #rng.normal(loc=1, scale=1, 
                    size = (nreps, nsrc))) + 1e-3
        #
    
    return A_init, Sigma_init, xi_init

def make_sknown(S, n_known_src, noise_level = 0, rng = None) -> np.ndarray:
    """Returns sknown mask — NaN for unknown, true value for known."""
    nreps = S.shape[0]
    sknown = np.full(S.shape, np.nan)
    if n_known_src > 0:
        sknown[:, :n_known_src] = S[:, :n_known_src]
        if noise_level >0:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(loc=0.0, scale=noise_level, size=(nreps, n_known_src))
            sknown[:,:n_known_src] += noise
            
    return sknown

def whiten_canonical(X, n_components):
    """
    Standard Whitening/PCA reduction for sample-row data architectures.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data matrix of shape (nreps, nobs) 
    n_components : int
        Target reduced source dimension (nout)
        
    Returns:
    --------
    X_whitened : np.ndarray of shape (nreps, nout)
        The dimensionality-reduced and sphericalized data matrix
    K : np.ndarray of shape (nobs, nout)
        The right-multiplying whitening projection matrix
    X_mean : np.ndarray of shape (1, nobs)
        The empirical mean vector removed during centering
    """
    
    # 1. Center across samples (axis=0)
    # X_mean shape: (1, nobs)
    X_mean = X.mean(axis=0, keepdims=True)
    # X_centered shape: (nreps, nobs)
    X_centered = X - X_mean
    
    # 2. Covariance Matrix E[x^T x]
    # rowvar=False ensures we compute covariance between features (nobs)
    # cov shape: (nobs, nobs)
    cov = np.cov(X_centered, rowvar=False)
    
    # 3. Eigenvalue Decomposition
    # d shape: (nobs,), E shape: (nobs, nobs)
    d, E = np.linalg.eigh(cov)
    
    # 4. Sort and Truncate
    idx = np.argsort(d)[::-1]
    # d truncated shape: (nout,) nout = min(nobs, n_components)
    
    d = d[idx][:n_components]
    # E truncated shape: (nobs, nout)
    E = E[:, idx][:, :n_components]
    
    # 5. Compute Right-Multiplying Whitening Matrix K
    # E shape: (nobs, nout), diag shape: (nout, nout)
    # K shape: (nobs, nout)
    K = E @ np.diag(1.0 / np.sqrt(d + 1e-12))
    
    # X_whitened shape: (nreps, nobs) @ (nobs, nout) -> (nreps, nout)
    X_whitened = X_centered @ K
    
    return X_whitened, K, X_mean

#---- MATRIX PROPERTIES----
def normalize_columns(D: np.ndarray) -> np.ndarray:
    """
    Normalize columns of matrix to unit norm.
    """
    norms = np.linalg.norm(D, axis=0)
    return D / norms

#coheerence of matrix calculation
def coherence(A: np.ndarray) -> float:
    """
    Compute matrix coherence.
    A: matrix to find coherence from
    """
    A = normalize_columns(A)
    G = A.T @ A
    np.fill_diagonal(G, 0)
    return np.max(np.abs(G))
#finding the maximum number of non-sparse elements we can guarentee to reconstruct based on the coherence of the matrix
def maximal_non_sparsity(A, verbose = False):
    dobs, dsrc = A.shape
    
    wbound = np.sqrt((dsrc - dobs) / (dobs * (dsrc - 1)))

    mu = coherence(A)
    mutual_coherence_lower_bound = int(np.floor((1/2)*(1+1/mu)))
    if verbose:
        print(f'coherence of matrix is',mu)
        print(f'Welch bound is',wbound)
        print(f'lowest possible number elements in s that can be discovered:', mutual_coherence_lower_bound)
    return mutual_coherence_lower_bound


#---- MATRIX COMPARISON AND CALIBRATION ----
def best_permutation_match(A, B):
    '''
    compare matrices to find the permutations that match the best. Does not account for sign flips or scaling, ONLY PERMUTATION.
    A: True matrix
    B: Estimated matrix 
    
    Returns
    -------
    B_perm : The permuted B matrix
    col_ind: the indicies that cause the permutation
    err : The Frobenius error between A and B
    '''
    
    # Build cost matrix: cost[i,j] = norm of col i of A vs col j of B
    cost = np.array([[np.linalg.norm(A[:, i] - B[:, j]) 
                      for j in range(B.shape[1])] 
                     for i in range(A.shape[1])])
    
    row_ind, col_ind = linear_sum_assignment(cost)    
    B_perm = B[:, col_ind]  # permute columns of B to best match A
    err =  np.linalg.norm(A - B_perm, 'fro')
    return B_perm, col_ind,err

def best_permutation_match_sign_flips(A, B):
    """
    Finds the best permutation, sign, AND scale of columns of B to match A.

    Returns
    -------
    B_perm  : (obsdim, srcdim) — permuted, sign-corrected, and scaled B
    col_ind : (srcdim,)        — permutation indices
    signs   : (srcdim,)        — array of +1 or -1 for each matched column
    scales  : (srcdim,)        — optimal scale factor per column
    err     : float            — Frobenius norm between A and B_perm
    """
    # ── Precompute all dot products at once ───────────────────────────────────
    # AtB[i, j] = A[:, i] . B[:, j]  →  (srcdim, srcdim)
    # BtB[j]    = B[:, j] . B[:, j]  →  (srcdim,) the column inner products
    # AtA[i]    = A[:, i] . A[:, i]  →  (srcdim,)
    AtB = A.T @ B                                        # (srcdim_A, srcdim_B) #finding all column inner products
    BtB = np.einsum('ij,ij->j', B, B)                   # (srcdim_B,) #column norms squared 
    AtA = np.einsum('ij,ij->j', A, A)                   # (srcdim_A,) #column norms squared 

    # ── Optimal scales for +B and -B ─────────────────────────────────────────
    # alpha_pos[i,j] = (a_i . b_j)  / (b_j . b_j)
    # alpha_neg[i,j] = (-a_i . b_j) / (b_j . b_j)  =  -alpha_pos[i,j]
    denom      = np.where(BtB < 1e-10, 1.0, BtB)        # (srcdim_B,) safe denom
    alpha_pos  =  AtB / denom[np.newaxis, :]             # (srcdim_A, srcdim_B)
    alpha_neg  = -alpha_pos                              # (srcdim_A, srcdim_B)

    # ── Vectorized cost: ||a_i - alpha * b_j||^2-------------------------------
    # Using ||a - alpha*b||^2 = ||a||^2 - 2*alpha*(a.b) + alpha^2*||b||^2
    def residual_sq(alpha):
        # alpha: (srcdim_A, srcdim_B)
        return (AtA[:, np.newaxis]
                - 2 * alpha * AtB
                + alpha**2 * BtB[np.newaxis, :])        # (srcdim_A, srcdim_B)

    cost_pos = residual_sq(alpha_pos)                    # (srcdim_A, srcdim_B)
    cost_neg = residual_sq(alpha_neg)                    # (srcdim_A, srcdim_B)

    # Take elementwise min and sqrt for the cost matrix
    cost = np.sqrt(np.maximum(np.minimum(cost_pos, cost_neg), 0))

    # ── Hungarian algorithm ───────────────────────────────────────────────────
    row_ind, col_ind = linear_sum_assignment(cost)

    # ── Apply permutation ─────────────────────────────────────────────────────
    B_perm = B[:, col_ind].copy()                        # (obsdim, srcdim)

    # ── Vectorized sign and scale assignment ──────────────────────────────────
    # Extract the matched (i, j) alpha and cost values
    cp = cost_pos[row_ind, col_ind]                      # (srcdim,)
    cn = cost_neg[row_ind, col_ind]                      # (srcdim,)
    ap = alpha_pos[row_ind, col_ind]                     # (srcdim,)
    an = alpha_neg[row_ind, col_ind]                     # (srcdim,)

    use_neg      = cn < cp                               # (srcdim,) bool mask
    signs        = np.where(use_neg, -1.0,  1.0)        # (srcdim,)
    scales       = np.where(use_neg, an, ap)             # (srcdim,)

    # Apply scale and sign to permuted columns
    B_perm *= scales[np.newaxis, :]                      # broadcast over obsdim

    return B_perm, col_ind, signs, scales, np.linalg.norm(A - B_perm, 'fro')


def scale_permute_src(srcest, scaling = None, permutation = None, true_mat = None, est_mat = None):
    '''
    This is a function that given a known scaling and permutation estimates the sources. It is supposed to make sure the permutation and scaling indeterminancies do not inflate the errors. 
    It either requires the true and the estimated mixing matrix, or it requires the ncessary scaling and permutation
    
    #srcest (nsamples, nobs)
    #scaling and permutation is requires to BE ESTIMATED FROM COMPARING THE TRUE A WITH THE ESTIMATED A
    '''
    
    if not (scaling is None or permutation is None):
        out = (srcest[:, permutation] / scaling[np.newaxis, :]).copy()
    else:
        if not (true_mat is None or est_mat is None):
            Aperm, col_ind, perm_sign, perm_scale, matrix_err = best_permutation_match_sign_flips(A=true_mat, B=est_mat)
            out = (srcest[:, col_ind] / perm_scale[np.newaxis, :]).copy()
            
        else:
            print('Please provide either (the true and the esimated matrix) or (the scaling and permutation). ')
            out= 'error'
    # Calibration — permutation, sign, scale
    return out

def remove_known_sources(true, estimate, sknown):
    # removing known columns
    if sknown is not None:
        col_indices = np.where(np.all(np.isnan(sknown), axis=0))[0]
        true        = true[:, col_indices].copy()      # (nsamples, n_unknown)
        estimate    = estimate[:, col_indices].copy()  # (nsamples, n_unknown)
    
    return true, estimate    

def enforce_floor_A(A, floor_ratio=0.01):
    """Ensures no column drops below a healthy fraction of the average column size."""
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    avg_norm = np.mean(norms)
    min_allowable_norm = avg_norm * floor_ratio
    
    # Identify which columns have collapsed
    collapsed_mask = norms < min_allowable_norm
    
    if np.any(collapsed_mask):
        # Scale up only the collapsed columns to the safety floor
        A[:, collapsed_mask.squeeze()] *= (min_allowable_norm / np.maximum(norms[:, collapsed_mask.squeeze()], 1e-12))
        
    return A

#---- CALCULATE LIKELIHOODS ----
def calculate_x_likelihood_variational(x, xi, A, Sigma):
    nreps = x.shape[0]
    nobs,    = A.shape
    xi = np.abs(xi)
    LambdaN = np.apply_along_axis(np.diag, 1, np.abs(xi))
    
    est_variances = Sigma[np.newaxis,:,:]+np.einsum('ij,njk,kl->nil', A, LambdaN, A.T)
    dets= np.array([np.linalg.det(est_variances[i,:,:]) for i in range(nreps)] )
    if(any(dets<=0)):
        print(dets)
        print(est_variances)
        
    lKxi = np.log(2)*(-nsrc)-0.5*np.sum(xi, axis = 1)+ 0.5*np.sum(np.log(xi), axis = 1) + nsrc*0.5*np.log(2*np.pi) #(nreps)
    mu = np.zeros(shape = (nreps, nobs))
    
    norm_dens = np.array([scipy.stats.multivariate_normal.pdf(x = x[i,:], mean = mu[i,:], cov = est_variances[i,:,:]) for i in range(nreps)] )
    res = np.sum(lKxi +np.log(norm_dens))
    return res
    
def calculate_x_likelihood_true(x, A, s,Sigma):
    nreps = x.shape[0]
    nobs, nsrc = A.shape
    mu = np.einsum('ij,nj->ni', A, s)
    norm_dens = np.array([scipy.stats.multivariate_normal.pdf(x = x[i,:], mean = mu[i,:], cov = Sigma) for i in range(nreps)] )
    res = np.sum(np.log(norm_dens))
    return res    

def get_X_tilde(X, n_known, S):
    """
    Alternative EM algorithm for overcomplete learning.

    Parameters:
    X : array-like, shape (n_samples, nobs)
        The input data.
    n_known : int
        The number of known components.
    S : array-like, shape (n_samples, nsrc)
    
    Returns
    tildeX = X-A_k S_k
    A_k
    """
    X = X.copy()
    if n_known == 0:
        return X, np.empty(shape=(X.shape[1], 0))
    
    S_known = make_sknown(S=S, n_known_src=n_known)
    Sk = S_known[:, :n_known]
    
    A_k = np.linalg.lstsq(Sk, X, rcond=None)[0].T #Solving ||X - Sk @ A_k.T||_F^2 for A_k
    tildeX = X - Sk @ A_k.T
    return tildeX, A_k

def records_to_df(records):
    """
    Convert list of dictionaries to a pandas DataFrame
    and replace NaNs with 0.
    """
    df = pd.DataFrame(records)

    # Convert numpy scalars to native floats (optional but clean)
    df = df.astype(float)

    # Replace NaNs with 0
    df = df.fillna(0.0)

    return df

#--- Estimate sources

def FISTA_estimate_sources(A_est, Xobs, niter, beta=1.0, tol=1e-7):
    """
    Optimized Matrix-Vectorized FISTA for Sparse Coding.
    Solves: min 0.5 ||X - S A^T||^2_F + beta ||S||_1
    
    A_est: (n_features, n_sources)
    Xobs:  (n_samples, n_features)
    """
    n_samples, n_features = Xobs.shape
    n_features, n_sources = A_est.shape
    
    # 1. Precompute Step Size (1/L)
    # L is the spectral norm of A^T A (largest eigenvalue)
    L = np.linalg.norm(A_est.T @ A_est, ord=2)
    step_size = 1.0 / L
    
    # 2. Initializations
    S = np.zeros((n_samples, n_sources))
    Y = S.copy()
    t = 1.0
    
    # Precompute constant matrices to save time in loop
    # Gradient of 0.5||X - SA^T||^2 w.r.t S is (S A^T - X) @ A
    # We can rewrite this as S @ (A^T A) - (X @ A)
    AtA = A_est.T @ A_est
    XAt = Xobs @ A_est
    
    for i in range(niter):
        S_old = S.copy()
        
        # [Step]: Gradient Step (Vectorized across all samples)
        # grad = Y @ AtA - XAt
        grad = Y @ AtA - XAt
        
        # [Step]: Proximal Step (Soft Thresholding)
        arg = Y - step_size * grad
        # Apply L1 penalty (Laplace prior)
        S = np.sign(arg) * np.maximum(np.abs(arg) - beta * step_size, 0)
        
        # [Step]: Acceleration update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        Y = S + ((t - 1.0) / t_next) * (S - S_old)
        t = t_next
        
        # Optional: Early exit if S stabilizes
        if i > 5 and np.linalg.norm(S - S_old) / (np.linalg.norm(S) + 1e-9) < tol:
            break
            
    return S


#---- legacy code
#---- one line code for generating data with different levels of non-sparsity, and for generating noisy data with different levels of noise. ----
#def generate_data(nsrc: int, nobs: int, nreps: int, seed: int, Iter, dd1, dd2:float, verbose, thres, nonsparsity_levels, scale = 1):
#    S = generate_laplace(dsource= nsrc, n_samples=nreps, seed = seed, scale=scale)  #generate laplace
#    A = quasi_orthogonal_matrix(dobs=nobs, dsrc=nsrc,  Iter = Iter, dd1=dd1, dd2=dd2, verbose=verbose,thres = thres, seed=seed) #generate quasiorthgonal matrix
#    maximal_non_sparsity(A, verbose=verbose) #print the maximum number of nonsparse element we can guarentee to reconstruct
#    if verbose:
#        ol_plot.plot_gram_matrix(A) #plot quasiorthogonal matrix
#    data = generate_X(A = A, S = S, noisy_flag=False, non_sparsity=nonsparsity_levels) #generate data
#    data['A'] = A
#    return data


#def extract_nonzero_padded(coef_list, k):
#    """Extract non-zero coefficients, padded to length k."""
#    result = []
#    for x in coef_list:
#        nz = sorted(x[x != 0])
#        # Pad with zeros if fewer than k non-zero elements returned
#        padded = nz + [0.0] * (k - len(nz))
#        result.append(padded[:k])  # truncate if somehow longer
#    return np.array(result)

def filter_dataframe(df, filter):
    df = df.copy()
    for col, val in filter.items():
        df = df[df[col] == val]
    return df
    
def filter_known_plots(df, modulo, start_modulo_from, col = 'n_unknown', col_max = np.inf):
    under_start = df[col] <= start_modulo_from
    matches_modulo = (df[col] % modulo == 0) & (df[col] > start_modulo_from)
    under_max = df[col] <= col_max
    mask = (under_start | matches_modulo) & under_max
    return df[mask]

def remove_false_ICA_data(df):
    method_mask = df['method'] == 'fast_ICA'
    overcomplete_mask = df['n_unknown'] > df['nobs']
    mask = method_mask & overcomplete_mask
    return df[~mask]

def load_data(RESULTS_DIR):
    files = glob.glob(f"{RESULTS_DIR}/*.csv")

    data_frame = pd.concat((pd.read_csv(f) for f in files),ignore_index=True,)

    data_frame["n_unknown"] = (
        data_frame["nsrc"] - data_frame["n_known_src"]
    ).astype(int)

    data_frame = ol_data.remove_false_ICA_data(data_frame)

    # Normalize angle errors to [0,1]
    for col in ["A_err_angle_max", "A_err_angle_mean"]:
        data_frame[col] /= (np.pi / 2)

    print(f"Loaded {len(data_frame):,} rows from {len(files)} files")
    return remove_false_ICA_data(data_frame)
