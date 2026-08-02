import numpy as np
import overcomplete_learning.data as ol_data
import overcomplete_learning.metrics as ol_metric
import overcomplete_learning.plotting as ol_plot
import overcomplete_learning.fastICA as ol_fastICA
import overcomplete_learning.sdp_approach as ol_sdp
from functools import partial
import pandas as pd

def _resolve_known(S_known, nsrc):
    """Return (known_idx, unknown_idx, s_k). S_known=None -> fully blind."""
    if S_known is None:
        return np.array([], dtype=int), np.arange(nsrc), None
    known_mask  = ~np.all(np.isnan(S_known), axis=0)
    known_idx   = np.where(known_mask)[0]
    unknown_idx = np.where(~known_mask)[0]
    s_k = S_known[:, known_idx] if known_idx.size else None
    return known_idx, unknown_idx, s_k
def woodbury_posterior(A_sub, var_sub, Sigma, X_eff):
    """
    Variational Gaussian posterior N(mean, cov) over the *sub*-set of sources,
    via Woodbury so the only inverse is obsdim x obsdim (cheap when nsrc > nobs).

        prior  s ~ N(0, Lambda),  Lambda = diag(var_sub)      (per sample)
        model  x_eff = A_sub s + eps,   eps ~ N(0, Sigma)

        Omega^-1 = A_sub Lambda A_sub^T + Sigma               (n, nobs, nobs)
        mean = Lambda A_sub^T Omega x_eff
        cov  = Lambda - Lambda A_sub^T Omega A_sub Lambda
    """
    n, nobs = X_eff.shape
    Omega_inv = np.einsum('ij,nj,kj->nik', A_sub, var_sub, A_sub) + Sigma
    # one batched solve with two RHS (x_eff, A_sub) -- more stable than an explicit inverse
    Omega_x = np.linalg.solve(Omega_inv, X_eff[..., None])[..., 0]                 # (n, nobs)
    Omega_A = np.linalg.solve(Omega_inv, np.broadcast_to(A_sub, (n,) + A_sub.shape).copy())
    mean = var_sub * np.einsum('ni,ij->nj', Omega_x, A_sub)                        # Λ Aᵀ Ω x
    AT_Om_A = np.einsum('ij,nik->njk', A_sub, Omega_A)                             # Aᵀ Ω A
    cov = -np.einsum('ni,nij,nj->nij', var_sub, AT_Om_A, var_sub)                  # -Λ AᵀΩA Λ
    d = A_sub.shape[1]
    cov[:, np.arange(d), np.arange(d)] += var_sub                                  # +Λ
    return mean, cov


def E_step(X, A, Sigma, xi, S_known, scale_source, iid_noise = False, post_cov_scaling:float = 1, ):
    """
    Consolidated E-step of the variational EM algorithm.

    Parameters
    ----------
    X         : (nsamples, obsdim)          — Observations
    A         : (obsdim, srcdim)            — Mixing matrix
    Sigma     : (obsdim, obsdim) or float   — Noise covariance matrix (if iid_noise=False) 
                                              or scalar variance (if iid_noise=True)
    xi        : (nsamples, srcdim)          — Variational parameters
    S_known    : (nsamples, srcdim)         — Known observed latent variables (mostly NaN)
    n_known     :                           - known columns in S
    iid_noise : bool                        — Toggle to use i.i.d. scalar noise optimization
    
    Returns
    ------------
    post_mean, SS_est, post_cov
    """
    nsamples, srcdim = X.shape[0], A.shape[1]
    known_idx, unknown_idx, s_k = _resolve_known(S_known, srcdim)

    
    W = A / Sigma if iid_noise else np.linalg.inv(Sigma) @ A
    # Compute 1/|xi| safely — where xi=0, set to 0 instead of inf
    
    
    b = np.mean(np.abs(s_k)) if (scale_source and s_k is not None) else 1.0
    xi = np.clip(xi, a_min=1e-24, a_max=None)
    variance_xi = b*xi
    lambda_inv_scalar = 1.0 / variance_xi
    
    post_mean = np.zeros((nsamples, srcdim))
    post_cov  = np.zeros((nsamples, srcdim, srcdim))
    #np.where(np.abs(xi) < 1e-20, 0.0, 1.0 / np.abs(xi))  # (nsamples, srcdim)
    
    #alternative calculation
    if True:    
        a = 3+1
    
    else:
        #Only calculating ASigma^{-1} once
        innerA = A.T @ W

        rhs = X @ W



        
        Lambda_inv = np.zeros((nsamples, srcdim, srcdim))
        Lambda_inv[:, np.arange(srcdim), np.arange(srcdim)] = lambda_inv_scalar

        K = innerA[np.newaxis, :, :] + Lambda_inv #precision matrix
        post_cov = np.linalg.inv(K)             # (nsamples, srcdim, srcdim) C_s: an estimated covariance matrix for the latent variables
        post_mean  = np.einsum('nij,nj->ni', post_cov, rhs)                      # (nsamples, srcdim) #mu_s
    
    known_idx, unknown_idx, s_k  = _resolve_known(S_known, srcdim)

    #Handle partially observed latent variables
    #if S_known is not None:
    #    known_mask   = ~np.all(np.isnan(S_known), axis=0)         # (srcdim,) bool
    #    known_idx    = np.where( known_mask)[0]                   # indices of known sources
    #    unknown_idx  = np.where(~known_mask)[0]                   # indices of unknown sources            
    #    
    if  known_idx.size == 0:
        Sigma = np.identity(A.shape[0])*Sigma if iid_noise else Sigma
        A_Lambda_AT = np.einsum('ij,nj,kj->nik', A, variance_xi, A) # (nsamples, obsdim, obsdim)
        Omega_inv = A_Lambda_AT + Sigma[np.newaxis, :, :]
        Omega = np.linalg.inv(Omega_inv) # (nsamples, obsdim, obsdim)

        #Calculate Posterior Mean: mu = xi * A.T @ Omega @ x
        # Omega @ x:
        Omega_x = np.einsum('nij,nj->ni', Omega, X) # (nsamples, obsdim)
        post_mean = variance_xi * (Omega_x @ A) # (nsamples, srcdim)

        # 5. Calculate Posterior Covariance: Cov = Lambda - Lambda @ A.T @ Omega @ A @ Lambda
        AT_Omega_A = np.einsum('ji,njk,kl->nil', A, Omega, A)

        # Multiply by Lambda (xi) on left and right sides
        # post_cov_ij = - xi_i * AT_Omega_A_ij * xi_j
        post_cov = - np.einsum('ni,nij,nj->nij', variance_xi, AT_Omega_A, variance_xi)

        # Add the baseline diagonal Lambda matrix
        # (indexing to add xi to the diagonals)
        idx = np.arange(srcdim)
        post_cov[:, idx, idx] += variance_xi
    
    if known_idx.size > 0:
        # Observed values for known sources
        
        
        A_k, A_u = A[:, known_idx], A[:, unknown_idx]

        W_u = W[:, unknown_idx]

        
        Lambda_inv_u = np.zeros((nsamples, len(unknown_idx), len(unknown_idx)))
        Lambda_inv_u[:, np.arange(len(unknown_idx)), np.arange(len(unknown_idx))] = lambda_inv_scalar[:, unknown_idx]

        #estimating conditional precision matrix
        K_u = (A_u.T @ W_u)[np.newaxis, :, :] + Lambda_inv_u

        #estimating conditional covariance matrix
        C_u = 1 / K_u if K_u.shape[1] == 1 else np.linalg.inv(K_u)

        x_u = X - s_k @ A_k.T  # residual data            

        # Map across sample batch to get final conditional means
        mu_u = np.einsum('nij,nj->ni', C_u, x_u @ W_u)

        # Apply corrections to unknown sources
        post_mean[:, known_idx]       = s_k             # pin known means exactly
        post_mean[:, unknown_idx]     = mu_u            # assigning unknown means

        post_cov[:, known_idx, :]     = 0
        post_cov[:, :, known_idx]     = 0
        post_cov[:, unknown_idx[:, np.newaxis], unknown_idx[np.newaxis, :]] = post_cov_scaling*C_u +(1-post_cov_scaling)*post_cov[:, unknown_idx[:, np.newaxis], unknown_idx[np.newaxis, :]]
                        
    SS_est = np.einsum('ni,nj->nij', post_mean, post_mean) + post_cov   #(nsamples, srcdim, srcdim)
    
    if((np.diagonal(SS_est, axis1 = 1, axis2=2)<0).any()):
        raise ValueError('The second moments are not positive!')
    return post_mean, SS_est, post_cov, b

def M_step(X, S_est, SS_est, whiten_data, normalize_A_col, S_known = None, aknown=None, iid_noise = False, known_noise_level = None):
    """
    M-step of the variational EM algorithm (Girolami 2001).

    Parameters
    ----------
    X            : (nsamples, nobs)           — observations
    S_est        : (nsamples, nsrc)           — posterior means E[s|x]
    SS_est       : (nsamples, nsrc, nsrc)     — posterior second moments E[ss^T|x]s
    aknown       : (nobs, nsrc) or None       — known columns of A, NaN elsewhere
    err_sd       : float or None              — fixed noise std if not updating Sigma
    normalize_A  : bool                       — normalise columns of A after update
    update_covariance : bool                  — estimate Sigma or use fixed err_sd*I
    known_noise_level : float or none         - the true noise level used as a regularizer. Only allows floarts   
    
    return Anew, Sigma_new, xin

    """
    
    nsample, nobs = X.shape                  # scalars
    nsrc          = S_est.shape[1]           # scalar

    if known_noise_level:
        assert isinstance(known_noise_level, (int, float)), 'Please provide a float or an integer.'

    # ── Sufficient statistics ─────────────────────────────────────────────────
    M  = X.T @ S_est   # (nobs,  nsrc) — x^T hatS = sum_n x_n E[s_n^T | x_n]   (Estimated covariance of X and S)
    Q  = np.sum(SS_est, axis=0)         # (nsrc,  nsrc) — sum_n E[s_n s_n^T | x_n]             (estimated second moment of S)
 
    # ── xi update (variational parameter) ────────────────────────────
    # (nsamples, nsrc) — E[s_l^2|x_n] (the conditional second moment ~ conditional variance)
    
    xi_est      = np.sqrt(np.diagonal(SS_est, axis1=1, axis2=2))        # (nsamples, nsrc) — sqrt(E[s_l^2|x_n])  (~conditional standard deviation)

    # ── A update: A_new = (sum_n x_n E[s|x_n]^T)(sum_n E[ss^T|x_n])^{-1} ──
    Anew = np.linalg.solve(Q, M.T).T                            # (nobs, nsrc)

    if aknown is not None:
        col_indices = np.where(~np.all(np.isnan(aknown), axis=0))[0]  # (n_known,)
        if len(col_indices) > 0:
            Anew[:, col_indices] = aknown[:, col_indices]  # pin known columns

    # ── Sigma update (eq. 3.8): Sigma = (1/N)(sum x_n x_n^T - A_new sum x_n E[s|x_n]^T) ──
    
    if known_noise_level:
        if iid_noise:
            Sigma_new = known_noise_level
        else:
            Sigma_new = known_noise_level*np.identity(nobs)
    else:
        if iid_noise:
            tr_Sx = np.sum(X ** 2)
            tr_A_M = np.sum(Anew * M)  # Fast trace calculation shortcut
            Sigma_new = max((tr_Sx - tr_A_M) / (nsample * nobs), 0.0) + 1e-8
        else:
            Sigma_new = (X.T @ X - Anew @ M.T) / nsample
            Sigma_new = (Sigma_new + Sigma_new.T) / 2 + np.identity(nobs) * 1e-8

    # Calculate current column L2 norms
    if normalize_A_col:
        #Anew = ol_data.enforce_floor_A(A = Anew)
        
        norms = np.linalg.norm(Anew, axis=0)
        norms = np.where(norms < 1e-12, 1.0, norms)  # Prevent division by zero
        #
        ## Determine which columns are safe to normalize 
        ## (We must skip columns that correspond to known sources or pinned matrix values)
        to_normalize = np.ones(nsrc, dtype=bool)
       
       # if S_known is not None:
       #     known_mask   = ~np.all(np.isnan(S_known), axis=0)         # (srcdim,) bool
       #     known_idx    = np.where( known_mask)[0]                   # indices of known sources
       #     to_normalize[known_idx] = False
       ##     
       ## # 1. Scale down the columns of the mixing matrix
        Anew[:, to_normalize] /= norms[to_normalize]
    if whiten_data:
        norms = np.linalg.norm(Anew, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        Anew /= norms[:, np.newaxis]


    
    return Anew, Sigma_new, xi_est


def M_step_hybrid_A(x, hatSn, ssn, sknown, A, iid_noise = False):
    """
    M-step of the variational EM algorithm (Girolami 2001).

    Parameters
    ----------
    x            : (nsamples, nobs)           — observations
    hatSn        : (nsamples, nsrc)           — posterior means E[s|x]
    ssn          : (nsamples, nsrc, nsrc)     — posterior second moments E[ss^T|x]
    """
    nsample, nobs = x.shape                  # scalars
    nsrc          = hatSn.shape[1]           # scalar

    # ── Sufficient statistics ─────────────────────────────────────────────────
    M  = x.T @ hatSn   # (nobs,  nsrc) — x^T hatS = sum_n x_n E[s_n^T | x_n]   (Estimated covariance of X and S)
    Q  = np.sum(ssn, axis=0)         # (nsrc,  nsrc) — sum_n E[s_n s_n^T | x_n]             (estimated variance of S)
 
    # ── xi update (variational parameter) ────────────────────────────
    # (nsamples, nsrc) — E[s_l^2|x_n] (the conditional second moment ~ conditional variance)
    xin      = np.sqrt(np.maximum(np.diagonal(ssn, axis1=1, axis2=2), 1e-10))        # (nsamples, nsrc) — sqrt(E[s_l^2|x_n])  (~conditional standard deviation)

    # ── A update: A_new = (sum_n x_n E[s|x_n]^T)(sum_n E[ss^T|x_n])^{-1} ──
    # Handle partially observed latent variables
    if sknown is not None:
        known_mask   = ~np.all(np.isnan(sknown), axis=0)         # (srcdim,) bool
        known_idx    = np.where( known_mask)[0]                   # indices of known sources
        unknown_idx  = np.where(~known_mask)[0]                   # indices of unknown sources
        
        if len(known_idx) > 0:
            # Observed values for known sources
            Qkk = Q[np.ix_(known_idx,   known_idx)]    # (n_known, n_known)
            Quu = Q[np.ix_(unknown_idx, unknown_idx)]  # (n_unknown, n_unknown)
            s_k, s_u, A_k, A_u = sknown[:, known_idx], hatSn[:,unknown_idx], A[:, known_idx], A[:, unknown_idx]
            #update Ak then Au
            x_k = x-(A_u@s_u.T).T
            Aknew = (x_k.T@s_k)@np.linalg.inv(Qkk)

            
            x_u = x-(Aknew@s_k.T).T
            Aunew = (x_u.T@s_u)@np.linalg.inv(Quu)
            
            Anew = np.empty_like(A)
            Anew[:, known_idx]   = Aknew
            Anew[:, unknown_idx] = Aunew
        else:
            Anew = np.linalg.solve(Q, M.T).T   # no known sources — standard update
            
    else:
        Anew = np.linalg.solve(Q, M.T).T                            # (nobs, nsrc)
    
    


    # ── Sigma update (eq. 3.8): Sigma = (1/N)(sum x_n x_n^T - A_new sum x_n E[s|x_n]^T) ──
    if iid_noise:
        tr_Sx = np.sum(x ** 2)
        tr_A_M = np.sum(Anew * M)  # Fast trace calculation shortcut
        Sigma_new = max((tr_Sx - tr_A_M) / (nsample * nobs), 0.0) + 1e-8
    else:
        Sigma_new = (x.T @ x - Anew @ M.T) / nsample
        Sigma_new = (Sigma_new + Sigma_new.T) / 2 + np.identity(nobs) * 1e-8
    
    return Anew, Sigma_new, xin
####################
#   WRAPPERS        
####################

#STANDARD IMPLEMENTATION OF EM-ALGORTIHM PROPOSED BY GIROLAMI
def run_EM_extended(X, rng,  normalize_A_col, whiten_data, scale_source, S_known = None, EM_iter: int = 1_000, err_tolerance: float = 1e-4, nsrc: int = 0, iid_noise: bool = False, post_cov_scaling:float = 1, known_noise_level = None, fix_Sigma = False, **kwargs):
    '''
    Wrapper function for running the EM-algorithm with jointly estimated covariance
    
    ----------
    Parameters
    ----------
    
    X                     : (n, nobs)                  — the observed values
    rng                  : int                          — random generator for reproducibility
    S_known               : (n, nsrc)                  — known sources; first n_known columns, NA elsewhere
    EM_iter               : int                        — maximum number of iterations allowed for convergence
    err_tolerance         : float                      — convergence tolerance threshold to stop EM loop
    nsrc                  : int                        — number of sources to estimate (optional if S is given)
    iid_noise             : bool                       — flag for using MLE for Sigma estimation
    post_cov_scaling      : float                      — covariance scaling adjustment (1 = analytical, 0 = empirical)
    normalize_A           : bool                       - flag for normalizing the columns of A for the columns corresponding to unknown sources

    -------
    Returns
    -------
    out : dict
        A dictionary containing the following tracking structures:
    S_est                 : (n, nsrc)                  — estimated posterior means of the sources
    SS_est                : (n, nsrc, nsrc)            — estimated posterior second moments of sources
    A_est                 : (nobs, nsrc)               — estimated value of the mixing matrix
    Sigma_est             : (nobs, nobs)               — estimated noise covariance matrix
    xi_est                : (n, nsrc)                  — estimated values of variational parameter
    X_est                 : (n, nobs)                  — reconstruction/estimated values of X
    post_cov              : (n, n_unknown, n_unknown)  — estimated posterior covariance of unknown sources
    converged             : bool                       — boolean flag indicating if EM reached convergence
    convergence_iteration : int                        — iteration index when the algorithm converged
    elbo_history          : list                       — history of ELBO values per iteration
    '''
    
    nreps, nobs = X.shape
    elbo_history = []
    prev_elbo = -np.inf
    X = X.copy()
       

    #create S
    if S_known is None:
        assert nsrc != 0, 'Please provide a valid number of sources, when the true sources are unknown'
        S_est = np.full(shape= (nreps, nsrc), fill_value=np.nan)
        n_known = 0
    else:
        nsrc = S_known.shape[1]
        S_est = np.full(shape= (nreps, nsrc), fill_value=np.nan)
        n_known = np.sum(~np.any(np.isnan(S_known), axis=0))
    
    A_new, Sigma_new, xi_new = ol_data.initialise_parameters(nobs=nobs, nsrc=nsrc, nreps=nreps, rng = rng)
    Sigma_new = Sigma_new if not iid_noise else Sigma_new[0,0]
    
    # Warm-start overrides for iterative loops
    if 'A_init' in kwargs and kwargs['A_init'] is not None:
        A_new = kwargs['A_init'].copy()
    if 'Sigma_init' in kwargs and kwargs['Sigma_init'] is not None:
        Sigma_new = np.array(kwargs['Sigma_init']) 
        
    if 'xi_init' in kwargs and kwargs['xi_init'] is not None:
        xi_new = kwargs['xi_init'].copy()
    # ---------------------------------------------------------------
    A_init = A_new
    i = 0
    SS_est = None
    post_cov = None
    tau = 1
    #EM - LOOP
    while i < EM_iter:
        #run one iteration
        S_est, SS_est, post_cov, b = E_step(X = X, A = A_new,  Sigma=Sigma_new, xi = xi_new, S_known = S_known, scale_source=scale_source, iid_noise=iid_noise, post_cov_scaling = post_cov_scaling)
        A_new, Sigma_new, xi_new = M_step(X = X, S_est = S_est, SS_est = SS_est, iid_noise=iid_noise, S_known=S_known, normalize_A_col=normalize_A_col, whiten_data=whiten_data)
        #scale everything to look nicer
        
        if fix_Sigma:
            Sigma_new = np.array(kwargs['Sigma_init']) 
        #get elbo values
        current_elbo = compute_total_elbo(
            X = X, S = S_est, SS = SS_est, post_cov = post_cov[:,n_known:,n_known:], A = A_new, Sigma=Sigma_new, xi_all = xi_new, k=n_known, b = b)
        elbo_history.append(current_elbo)
        
        if i > 10:
            # The EM algorithm mathematically guarantees monotonic increases. We only allow for convergence after 10 steps
            window_delta = np.max(np.abs(np.diff(elbo_history[-11:])))/(np.abs(elbo_history[-1]))
            #if i%10 == 0:
            #    print(window_delta/nobs)
            ##Convergence break condition: If the error does not change by err_tolerance percentage per sample point and observations, then we stop
            if abs(window_delta)/nobs < err_tolerance:
                break
                    #print(f"Algorithm successfully converged at iteration {i}.")    
        i+=1
    x_final = (A_new@S_est.T).T #reconstruction X
    
    out = {
        'S_est'     : S_est,
        'SS_est'     : SS_est,
        'A_est'     : A_new,
        'Sigma_est' : Sigma_new,
        'xi_est'    : xi_new,
        'X_est'     : x_final,
        'post_cov'  : post_cov,
        'converged':             i < EM_iter - 1,
        'convergence_iteration': i,
        'elbo_history': elbo_history,
        'n_known'   :   n_known,
        'A_init'    :   A_init
    }       
    
    return out 

#i.i.d. variation
def run_EM_extended_iid():
    '''
    Simple wrapper function for running the EM-algorithm with iid esimates. Should be called and then all inputs passed into the output
    
    ------
    OUTPUT
    ------
    
    run_EM_extended         : see run_EM_Extended 
    '''
    return(partial(run_EM_extended, iid_noise = True))
    

#model misspecification 
def run_EM_extended_wrong():
    '''
    Simple wrapper function for running the EM-algorithm without changing posterior covariance estimate, 
    
    ------
    OUTPUT
    ------
    
    run_EM_extended         : see run_EM_Extended with parameter post_cov_scaling fixed to 0
    '''
    return(partial(run_EM_extended, iid_noise = False, post_cov_scaling = 0))

#DIMENSION REDUCTION

def run_EM_OLS(X, rng, S_known, EM_iter: int = 1_000, err_tolerance: float = 1e-4, 
               iid_noise: bool = False, post_cov_scaling: float = 1.0, outer_iters: int = 1,outer_tol = 1e-4, normalize_A_col = False, **kwargs
):
    """
    Regress away the known source components via OLS and execute the extended 
    variational EM algorithm on the remaining residual space.

    Parameters
    ----------
    X                 : (nsamples, nobs)           — observations
    rng              : int                        — random generator for reproducibility
    S_known           : (nsamples, nsrc)           — an array of the known sources
    EM_iter           : int                        — maximum iterations allowed for EM convergence
    err_tolerance     : float                      — condition for stopping EM loop
    iid_noise         : bool                       — boolean flag for using MLE for Sigma estimation
    post_cov_scaling  : float                      — empirical scaling multiplier for posterior covariance
    """
    nsamples, nsrc = S_known.shape
    n_known = np.sum(~np.any(np.isnan(S_known), axis=0))
    n_unknown = nsrc - n_known
    
    out = None #placeholder for output
    
    # Isolate residual variance by regressing away known mixing channels
    tilde_X, A_k = ol_data.get_X_tilde(X=X, S=S_known, n_known=n_known)
    
    # Edge Case: No unknown components left to estimate
    if n_unknown == 0:
        X_est = (A_k @ S_known.T).T
        #tilde_X are the residuals
        Sigma_est = (tilde_X.T@tilde_X)/nsamples
        SS_est = np.einsum('ni,nj->nij', S_known, S_known)
        xi_est = np.abs(S_known)
        final_ELBO = compute_final_likelihood(X = X, S = S_known, A = A_k)

        out =  {
            'A_est': A_k,
            'S_est': S_known, 
            'SS_est': SS_est,
            'X_est': X_est,
            'Sigma_est' : Sigma_est,
            'xi_est'    :   xi_est,
            'converged': True,
            'convergence_iteration': 0,
            'elbo_final': final_ELBO
        }
        return out
    # Pre-compute static projection structures if known data exists
    if n_known > 0:
        S_k = S_known[:, :n_known]
        S_k_mp = np.linalg.pinv(S_k)
        b = np.mean(np.abs(S_k))
    
    # Run extended blind EM algorithm on the OLS-cleaned residual observations    
    A_old = None
    A_k_current = A_k.copy()
    
    # Before the outer loop
    full_elbo_history = []
    iteration_of_convergence =  0
    # --- Structural Outer Optimization Loop ---
    for step in range(outer_iters):
        if (step == 0):
            # First pass executes on the raw OLS residual matrix
            tilde_X_current = tilde_X
            out = run_EM_extended(
                X=tilde_X_current, rng=rng, EM_iter=EM_iter, err_tolerance=err_tolerance, 
                nsrc=n_unknown, iid_noise=iid_noise, post_cov_scaling=post_cov_scaling, normalize_A_col=normalize_A_col, **kwargs
            )
            if 'elbo_history' in out:
                full_elbo_history.extend(out['elbo_history'])
            iteration_of_convergence += out['convergence_iteration']
            if (n_known == 0):
                break
        else:
            # Subsequent passes compute bias correction and warm-start the inner EM
            A_u = out['A_est']
            S_u = out['S_est']
            Sigma = out['Sigma_est']
            xi = out['xi_est']
            
            # Compute structural mixing matrix correction 
            correction = A_u @ S_u.T @ S_k_mp.T
            A_k_current = A_k - correction
            
            # Recalculate adjusted residual matrix using corrected tracking paths
            tilde_X_current = X - (A_k_current @ S_k.T).T
            
            # Warm-started inner EM execution loop
            out = run_EM_extended(
                X=tilde_X_current, rng=rng, EM_iter=EM_iter, err_tolerance=err_tolerance, 
                nsrc=n_unknown, iid_noise=iid_noise, post_cov_scaling=post_cov_scaling,normalize_A_col=normalize_A_col,
                A_init=A_u, Sigma_init=Sigma, xi_init=xi, 
                **kwargs
            )
            iteration_of_convergence += out['convergence_iteration']
            if 'elbo_history' in out:
                full_elbo_history.extend(out['elbo_history'])
                
        # Reconstruct full mixing matrix state for this step to assess convergence
        if n_known > 0:
            if outer_iters>1:
                A_k_current = A_k - (out['A_est'] @ out['S_est'].T @ S_k_mp.T)
            A_current_full = np.concatenate([A_k_current, out['A_est']], axis=1)
        else:
            A_current_full = out['A_est']
        
        # Check for matrix convergence across outer loop updates
        if A_old is not None:
            relative_param_change = np.linalg.norm(A_current_full - A_old) / (np.linalg.norm(A_old) + 1e-12)

            if relative_param_change < outer_tol:
                break
        A_old = A_current_full.copy()
    
    if n_known > 0:   
        A_new = out['A_est']
        S_new = out['S_est']
        C_u_new = out['post_cov']  # Posterior covariance block from E-step: (nsamples, du, du)
        S_k = S_known[:, :n_known]
        # Standardize full space parameters by joining known and estimated blocks
        A_final = np.concatenate([A_k_current, A_new], axis=1)
        S_final = np.concatenate([S_k, S_new], axis=1)
        
        # Vectorized assembly of the expected joint second moment matrix (SS)
        SSn_final = np.einsum('ni,nj->nij', S_final, S_final)
        SSn_final[:, n_known:, n_known:] += C_u_new 
        
        # Extract variational components for the unobserved features
        xi_unknown = out['xi_est']
        xi_final = np.concatenate([np.abs(S_known[:, :n_known]), xi_unknown], axis=1)
        b = np.mean(np.abs(S_k))
        # --- MODIFIED: Adjusted interface mapping to match your new ELBO signature ---
        elbo_final = compute_total_elbo(
            X=X,                  # Raw complete observations matrix
            S=S_final,            # Full combined posterior means matrix
            SS=SSn_final,          # Full combined expected second moments matrix
            post_cov=C_u_new,     # Covariances of unknown sources: (nsamples, du, du)
            A=A_final,            # Unified mixing matrix
            Sigma=out['Sigma_est'], 
            xi_all=xi_final,      # Variational parameters matrix: (nsamples, ds)
            k=n_known,             # Boundary split index
            b = b
        )
        
        # Append known parameter log prior penalty structures back into tracking logs
        log_prior_known = np.mean(-np.log(2.0) - np.abs(S_known[:, :n_known])) #equivalent to true log density of Laplace because |xi|=|s|
        if 'elbo_history' in out:
            out['elbo_history'] = [e + log_prior_known for e in out['elbo_history']]
            out['elbo_history'].append(elbo_final)
        
        
        if n_known > 0 and len(full_elbo_history) > 0:
            out['elbo_history'] = [e + log_prior_known for e in full_elbo_history]
            out['elbo_history'].append(elbo_final)
        
        
        # Package finalized configurations
        out['A_est'] = A_final
        out['S_est'] = S_final
        out['X_est'] = (A_final @ S_final.T).T
        out['SS_est'] = SSn_final
        out['elbo_final'] = elbo_final
        out['xi_est']   = xi_final
    out['convergence_iteration'] = iteration_of_convergence
    out['n_known'] = n_known
    return out

def run_EM_OLS_oracle(X, rng, S_known, A_true, Sigma=None,
                      EM_iter: int = 1_000, err_tolerance: float = 1e-4,
                      iid_noise: bool = False, post_cov_scaling: float = 1.0,
                      outer_iters: int = 1, outer_tol=1e-4,   # accepted for API compatibility; ignored
                      normalize_A_col=False, **kwargs):
    """
    Oracle ablation variant of run_EM_OLS: residualize with the TRUE known mixing
    block and run the blind variational EM on the residuals. The known block is
    never estimated, corrected, or updated — so any n_known-dependence of the
    unknown-source recovery observed with this function falsifies the
    estimation-error mechanism.

    Differences from run_EM_OLS:
      * A_k is taken from the true mixing matrix `A`, not estimated by OLS.
      * No outer/debias loop: the correction targets estimation error in A_k,
        which does not exist here. (`outer_iters`, `outer_tol` are ignored.)
      * If `Sigma` is given, the inner EM is asked to keep it fixed
        (requires `fix_Sigma` support in run_EM_extended; see note below).

    Parameters
    ----------
    X        : (nsamples, nobs)   — observations
    rng      : Generator/int      — random generator for reproducibility
    S_known  : (nsamples, nsrc)   — sources; unknown columns are NaN-filled
    A        : (nobs, nsrc)       — TRUE mixing matrix (only A[:, :n_known] is used)
    Sigma    : (nobs, nobs)/None  — TRUE noise covariance; if given, clamped
    """
    
    A = A_true
    if A is None:
        raise ValueError("run_EM_OLS_oracle requires the true mixing matrix A.")
    nsamples, nsrc = S_known.shape
    n_known = int(np.sum(~np.any(np.isnan(S_known), axis=0)))
    n_unknown = nsrc - n_known

    # --- Oracle residualization: true known block, true residuals -------------
    A_k = A[:, :n_known]                       # (nobs, n_known)
    S_k = S_known[:, :n_known]                 # (nsamples, n_known)
    tilde_X = X - S_k @ A_k.T                  # (nsamples, nobs)

    # --- Edge case: everything known -----------------------------------------
    if n_unknown == 0:
        X_est = S_known @ A_k.T
        Sigma_est = Sigma if Sigma is not None else (tilde_X.T @ tilde_X) / nsamples
        return {
            'A_est': A_k,
            'S_est': S_known,
            'SS_est': np.einsum('ni,nj->nij', S_known, S_known),
            'X_est': X_est,
            'Sigma_est': Sigma_est,
            'xi_est': np.abs(S_known),
            'converged': True,
            'convergence_iteration': 0,
            'elbo_final': compute_final_likelihood(X=X, S=S_known, A=A_k),
            'n_known': n_known,
            'oracle': True,
        }

    if Sigma is not None:
        assert np.isscalar(Sigma), "Pass the scalar noise variance sigma^2."
        sigma2 = float(Sigma)
        kwargs['Sigma_init'] = sigma2 if iid_noise else sigma2 * np.identity(X.shape[1])
        kwargs['fix_Sigma'] = True

    # --- Single blind EM pass on the oracle residuals -------------------------
    out = run_EM_extended(
        X=tilde_X, rng=rng, EM_iter=EM_iter, err_tolerance=err_tolerance,
        nsrc=n_unknown, iid_noise=iid_noise, post_cov_scaling=post_cov_scaling,
        normalize_A_col=normalize_A_col, **kwargs,
    )
    
    if n_known == 0:
        out['n_known'] = 0
        out['oracle'] = True
        return out

    # --- Package: join TRUE known block with estimated unknown block ----------
    A_u = out['A_est']
    S_u = out['S_est']
    C_u = out['post_cov']                       # (nsamples, du, du)

    A_final = np.concatenate([A_k, A_u], axis=1)          # A_k untouched
    S_final = np.concatenate([S_k, S_u], axis=1)

    SSn_final = np.einsum('ni,nj->nij', S_final, S_final)
    SSn_final[:, n_known:, n_known:] += C_u

    xi_final = np.concatenate([np.abs(S_k), out['xi_est']], axis=1)
    b = np.mean(np.abs(S_k))

    elbo_final = compute_total_elbo(
        X=X, S=S_final, SS=SSn_final, post_cov=C_u, A=A_final,
        Sigma=out['Sigma_est'], xi_all=xi_final, k=n_known, b=b,
    )

    log_prior_known = np.mean(-np.log(2.0) - np.abs(S_k))
    if 'elbo_history' in out:
        out['elbo_history'] = [e + log_prior_known for e in out['elbo_history']]
        out['elbo_history'].append(elbo_final)

    out.update({
        'A_est': A_final,
        'S_est': S_final,
        'X_est': S_final @ A_final.T,
        'SS_est': SSn_final,
        'xi_est': xi_final,
        'elbo_final': elbo_final,
        'n_known': n_known,
        'oracle': True,
    })
    return out


#i.i.d. variation
def run_EM_OLS_iid():
    return(
        partial(run_EM_OLS, iid_noise = True)
    )

#debiasing the OLS to make it similar run_extended
def run_EM_OLS_debias(outer_iters, outer_tol):
    return(
        partial(run_EM_OLS, outer_iters = outer_iters, outer_tol = outer_tol)
    )

    

def run_EM_OLS_debias_iid(outer_iters, outer_tol):
    return(
        partial(run_EM_OLS, outer_iters = outer_iters, outer_tol = outer_tol, iid_noise = True)
    )
    
#ALTERNATIVE METHOD
def run_EM_FastICA(X, rng, S_known, EM_iter:int, err_tolerance: float,  **kwargs):
    """
    Regress away the known S and run FastICA on the rest if we reduce to undercomplete case. Only feasible, if nsrc unknown <= nobs
    
    Parameter:
    X: array, shape (nreps, nobs)
        The observed values
    Sknown: array, shape (nreps, nsrc)
        An array of the known sources
    """
    nreps, nsrc = S_known.shape
    nobs        = X.shape[1]
    n_known = np.sum(~np.any(np.isnan(S_known), axis=0))
    
    n_unknown = nsrc - n_known
    
    if n_unknown>nobs:
        print('Please ensure we are in the complete or undercomplete case! Generating Random Gaussian Estimate w. sd = 1')
        out = {'A_est'                 :  np.full(shape=(nobs, nsrc), fill_value=np.nan),
                'S_est'                 : np.full(shape=(nreps, nsrc), fill_value=np.nan), 
                'X_est'                 : np.full(shape=(nreps, nobs), fill_value=np.nan),
                'converged'             : True,
                'convergence_iteration' : int(0)}
        return out
        
    tilde_X, A_k = ol_data.get_X_tilde(X = X, S = S_known, n_known= n_known)
    #combining the known estimates with the new
    if(n_unknown == 0):
        Xest = (A_k@S_known.T).T
        return {'A_est'                 : A_k,
                'S_est'                 : S_known, 
                'X_est'                 : Xest,
                'converged'             : True,
                'convergence_iteration' : int(0)}
    
    out = ol_fastICA.fast_ica(
            X=tilde_X, n_components=n_unknown, approach=kwargs.get('approach', 'deflation'), 
            contrast=kwargs.get('contrast', 'logcosh'), max_iter=EM_iter, 
            tol=err_tolerance, whiten=True, rng=rng)
    
    #update outputs with the regressed data
    if n_known>0:    
        A_new = out['A_est']
        S_new = out['S_est']
        A_final = np.concatenate([A_k, A_new], axis=1)
        S_final = np.concatenate([S_known[:,:n_known], S_new], axis = 1)
        out['A_est'] = A_final
        out['S_est'] = S_final
        out['X_est'] = (A_final@S_final.T).T
        
    out['elbo_final'] = compute_final_likelihood(X = X, S = out['S_est'], A = out['A_est'])
    out['n_known'] = n_known
    return(out)

def run_EM_OverICA(X, rng, err_sd, EM_iter:int, S_known, mu, **kwargs):
    """
    Regress away the known S and run sdp
    
    ----------
    Parameters
    ----------
    X: array, shape (nreps, nobs)
        The observed values
    n_known: 
        The number of known components
    Sknown: array, shape (nreps, nsrc)
        An array of the known sources
    """
    nsamples, nsrc = S_known.shape
    n_known = np.sum(~np.any(np.isnan(S_known), axis=0))
    n_unknown = nsrc - n_known

    #If some are known, we regress away and run the method
    tilde_X, A_k = ol_data.get_X_tilde(X = X, S = S_known, n_known= n_known)
    #combining the known estimates with the new
    if(n_unknown == 0):
        Xest = (A_k@S_known.T).T
        out = {'A_est'                 : A_k,
                'S_est'                 : S_known, 
                'X_est'                 : Xest,
                'converged'             : True,
                'convergence_iteration' : 0}
    else:
        out = ol_sdp.overica(X = tilde_X, k = n_unknown, opts=None, rng = rng, EM_iter = EM_iter, mu=mu, err_sd=err_sd)
    
    if n_known>0:
        A_new = out['A_est']
        S_new = out['S_est']
        A_final = np.concatenate([A_k, A_new], axis=1)
        S_final = np.concatenate([S_known[:,:n_known], S_new], axis = 1)
        out['A_est'] = A_final
        out['S_est'] = S_final
        out['X_est'] = (A_final@S_final.T).T
    
    out['elbo_final'] = compute_final_likelihood(X = X, S = out['S_est'], A = out['A_est'])
    out['n_known'] = n_known
    return(out)
        
#def run_EM_alternate(X, seed: int, EM_iter:int, nsrc, err_tolerance: float, Sknown, iid_noise, **kwargs):
#    '''
#    Simple wrapper function for running the EM-algorithm
#    
#    ----INPUT----
#    
#    X: array (nreps, nobs)
#        the observed values 
#    seed: int
#        random seed for reproducibility
#    EM_iter: int
#        the maximum number of iteration allowed for convergence
#    err_tolerance: float
#        The toleracne for difference in X-values
#    nsrc: int 
#        the number of unknown source. Is not necessary to provide if S is given
#    Sknown: (nreps, nsrc)
#        The known sources. If sources are known, the corresponding columns should be not Na
#    
#    ----OUTPUT----
#    
#    Shat: 
#        Estimated values of the sources
#        
#    SSn:
#        Estimated second moments of the sources
#        
#    A_new: 
#        Estimated value of the mixing matrix
#        
#    Sigma_new:
#        Estimated noise covariance matrix
#        
#    xi_new: 
#        Estimated values of variational parameter
#    '''
#    
#    #getting fundamental information
#    nreps, nobs = X.shape
#    #create S
#    if Sknown is None:
#        assert nsrc != 0, 'Please provide a valid number of sources, when the true sources are unknown'
#        Sest = np.full(shape= (nreps, nsrc), fill_value=np.nan)
#    else:
#        nsrc = Sknown.shape[1]
#        Sest = Sknown
#    
#    x_new = np.empty(shape=X.shape)
#        
#    A_new, Sigma_new, xi_new = ol_data.initialise_parameters(nobs=nobs, nsrc=nsrc, nreps=nreps, seed=seed)
#    Sigma_new = Sigma_new if not iid_noise else Sigma_new[0,0]
#
#    i = 0
#    Shat = np.full(shape=Sest.shape, fill_value=np.nan)
#    SSn = np.full(shape =(nreps, nsrc, nsrc), fill_value=np.nan)
#    while i < EM_iter:
#        Shat, SSn = E_step(x = X, A = A_new,  Sigma=Sigma_new, xi = xi_new, sknown= Sest, iid_noise=iid_noise)
#        A_new, Sigma_new, xi_new = M_step_hybrid_A(x = X, hatSn=Shat, ssn = SSn, iid_noise=iid_noise, A=A_new, sknown=Sest)
#
#        if i>1:
#            x_old = x_new.copy()
#            x_new = (A_new@Shat.T).T
#            deltaX = np.linalg.norm(x_old- x_new, ord = 'fro')
#            if deltaX<err_tolerance:
#            #    print(f'converged at iteration {i}')
#                break
#        i+=1
#    
#    out = {
#        'S_est'     : Shat,
#        'Ssn'       : SSn,
#        'A_est'     : A_new,
#        'Sigma_est' : Sigma_new,
#        'xi_est'    : xi_new,
#        'X_est'     : x_new,
#        'converged':             i < EM_iter - 1,
#        'convergence_iteration': i,
#    }       
#    
#    return out 

#RANDOM BASELINES
def run_EM_random(X, rng, EM_iter:int, err_tolerance: float, S_known, **kwargs):
    nreps, nsrc = S_known.shape
    nobs        = X.shape[1]
    n_known = np.sum(~np.any(np.isnan(S_known), axis=0))

    n_unknown = nsrc - n_known
    tilde_X, A_k = ol_data.get_X_tilde(X = X, S = S_known, n_known= n_known)
    #combining the known estimates with the new
    if(n_unknown == 0):
        Xest = (A_k@S_known.T).T
        return {'A_est'                 : A_k,
                'S_est'                 : S_known, 
                'X_est'                 : Xest,
                'converged'             : True,
                'convergence_iteration' : 0}
    
    rng = np.random.default_rng(rng)    
    
    A_est = rng.standard_normal(size = (nobs, n_unknown))
    A_est = ol_data.normalize_columns(A_est)
    
    S_est = (A_est.T @ tilde_X.T).T #np.random.normal(scale=1, size = (nreps, n_unknown))
    X_est = (A_est@S_est.T).T
        
    out = {
        'A_est': A_est,
        'X_est': X_est,
        'S_est': S_est,
        'converged':             True,
        'convergence_iteration': 0
    }

    if n_known>0:    
        A_new = out['A_est']
        S_new = out['S_est']
        A_final = np.concatenate([A_k, A_new], axis=1)
        S_final = np.concatenate([S_known[:,:n_known], S_new], axis = 1)
        out['A_est'] = A_final
        out['S_est'] = S_final
        out['X_est'] = (A_final@S_final.T).T
    
    return(out)    

def run_EM_random_quasiorthogonal(X, rng, EM_iter:int, err_tolerance: float, S_known, **kwargs):
    nreps, nsrc = S_known.shape
    n_known = np.sum(~np.any(np.isnan(S_known), axis=0))
    nobs        = X.shape[1]
    n_unknown = nsrc - n_known
    tilde_X, A_k = ol_data.get_X_tilde(X = X, S = S_known, n_known= n_known)
    #assert n_unknown<=nobs, 'Please ensure we are in the complete or undercomplete case!'
    #combining the known estimates with the new
    if(n_unknown == 0):
        Xest = (A_k@S_known.T).T
        return {'A_est'                 : A_k,
                'S_est'                 : S_known, 
                'X_est'                 : Xest,
                'converged'             : True,
                'convergence_iteration' : 0}
    
    A_est = ol_data.quasi_orthogonal_matrix(dobs = nobs, dsrc = nsrc, Iter=EM_iter,dd1 = 0.9, dd2=0.9, thres=1e-4, rng = rng)

    
    S_est = tilde_X @ A_est
    X_est = (A_est@S_est.T).T
        
    out = {
        'A_est': A_est,
        'X_est': X_est,
        'S_est': S_est,
        'converged':             True,
        'convergence_iteration': 0
    }

    if n_known>0:    
        A_new = out['A_est']
        S_new = out['S_est']
        A_final = np.concatenate([A_k, A_new], axis=1)
        S_final = np.concatenate([S_known[:,:n_known], S_new], axis = 1)
        out['A_est'] = A_final
        out['S_est'] = S_final
        out['X_est'] = (A_final@S_final.T).T
    
    return(out)    


 
#def run_EM(x, init_Sigma, init_xi, init_A, sknown, aknown, err_sd, logger, Iter, trueA = None, trueS = None, truncated = False, n_nonzero = 0, adjust_err_mat = False, adjust_err_src = False, normalize_A = False):
    
    errors_matrix = []
    err_matrix, err_latent, coherence = None, None, None
    errors_latent_variables = []
    coherence_series = []
    xi_diff = []
    Anew, Sigma_new, xi = init_A, init_Sigma, init_xi
    
    for i in range(Iter):
        try:
            xi_old = xi
            if(truncated):
                e_step_fn = lambda **kwargs: E_step_truncate(**kwargs, n_nonzero=n_nonzero)
            else:
                e_step_fn = E_step
            hatSn, ssn = e_step_fn(A=Anew, Sigma=Sigma_new, x=x, xi=xi, sknown=sknown)
            Anew, Sigma_new, xi = M_step(x=x, hatSn=hatSn, ssn=ssn,
                                           aknown=aknown, err_sd=err_sd, normalize_A=normalize_A)
            
            #finding the newest estiate
            shat = hatSn#ol_eval.fit_OMP_to_sample(A = Anew, X = X, n_nsrc=3)
            coherence_series.append(ol_data.coherence(Anew))
            
            if trueA is not None:
                if adjust_err_mat:
                    err_f_matrix = lambda **kwargs: ol_metric.frobenius_err_remove_known(**kwargs, aknown=aknown)
                else:
                    err_f_matrix = ol_metric.frobenius_err_remove_known

                    
                err_matrix,name_matrix = err_f_matrix(true=trueA, estimate=Anew)
                errors_matrix.append(err_matrix)
                if i % 5 == 0:
                    logger.info(f'Iter {i:03d} | {name_matrix}: {err_matrix:.6f}')
                # Log a warning if error is increasing
                if len(errors_matrix) > 1 and errors_matrix[-1] > errors_matrix[-2]:
                    logger.warning(f'Iter {i:03d} | Error increased: '
                                   f'{errors_matrix[-2]:.6f} -> {errors_matrix[-1]:.6f}')
            
            if trueS is not None:
                if adjust_err_src:
                    err_f_src = lambda **kwargs: ol_metric.mse_weighted_remove_known(**kwargs, sknown=sknown)
                else:
                    err_f_src = ol_metric.mse_weighted
                err_latent, name_latent = err_f_src(true = trueS, estimate=shat)
                errors_latent_variables.append(err_latent)
                if i % 5 == 0:
                    logger.info(f'Iter {i:03d} | {name_latent}: {err_latent:.6f}')
            
            #logging errors
            if(i>0):
                xi_change, xi_name = ol_metric.mse_weighted(true = xi_old, estimate = xi)
                xi_diff.append(xi_change)
            else:
                xi_name = 'xi_difference'
                xi_change = 0

            # Log progress every 10 iterations
            if i % 5 == 0:
                logger.info(f'Iter {i:03d} | xi_change: {xi_change:.6f}')

        except Exception as e:
            logger.error(f'Iter {i:03d} | Step failed: {type(e).__name__}: {e}')
            break
        matrix_coherence = ol_data.coherence(Anew)
    return Anew, Sigma_new, xi, hatSn, errors_latent_variables, errors_matrix, xi_diff, matrix_coherence, coherence_series    


class EM:
    def __init__(self, nreps, nobs, nsrc, err_sd, EM_convergence_threshold, seed):
        assert nobs <= nsrc, "For overcomplete learning, the number of sources must be greater than the number of observations."
        self.nreps = nreps
        self.nobs = nobs
        self.nsrc = nsrc
        self.err_sd = err_sd
        self.EM_convergence_threshold = EM_convergence_threshold
        self.seed = seed
    
    
    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def generate_data(self, S_scale=1.0, dd1=0.9, dd2=0.9, verbose=False, thres=1e-4, Iter=1000, A_scale=1): #generating data with specified parameters and storing the true A, S and X for later evaluation
        S = np.random.laplace(loc=0, scale=S_scale, size = (self.nreps, self.nsrc))
        A = ol_data.quasi_orthogonal_matrix(dobs=self.nobs, dsrc=self.nsrc,  Iter = Iter, dd1=dd1, dd2=dd2, verbose=verbose,thres = thres, seed=None)*A_scale #generate quasiorthgonal matrix #(nobs, nreps)
        if(self.err_sd ==0):
            eps = np.zeros(shape=(self.nreps,   self.nobs))
        else:
            eps = np.random.normal(loc = 0, scale = self.err_sd, size = (self.nreps, self.nobs))
    
        X = (A @ S.T).T + eps
        #storing the true values of A, S and X for EM_algorithm to evaluate the estimation performance after running
        self.X_true = X
        self.A_true = A
        self.S_true = S
        
        #storing the generated data in a dictionary for later use and debugging
        data = {}
        data['S'] = S
        data['A'] = A
        data['eps'] = eps
        data['X'] = X
        self.data = data
    
    def initialize_parameters(self): #random initialisation of A, Sigma and xi
        self.A_est, self.Sigma_est, self.xi_est = ol_data.initialise_parameters(nobs=self.nobs, nsrc=self.nsrc, nreps=self.nreps)

    def set_known_variables(self, n_known_src): #setting the known latent variables based on the number of known sources specified
        self.n_known_src = n_known_src
        self.sknown = np.full(self.S_true.shape, np.nan)
        if n_known_src > 0:
            self.sknown[:, :n_known_src] = self.S_true[:, :n_known_src]
    
    def visualize_A_true(self):
        ol_plot.plot_gram_matrix(self.A_true, title='True A Gram Matrix')
            
    def E_step(self): #performing the E-step using the current estimates of A, Sigma and xi, and the observed data X_true. The sknown variable is also passed to incorporate any known latent variables into the estimation.
        self.S_est, self.SS_est = E_step(A=self.A_est, Sigma=self.Sigma_est, x=self.X_true, xi=self.xi_est, sknown=self.sknown)
    
    def M_step(self):
        self.A_est, self.Sigma_est, self.xi_est = M_step(x=self.X_true, hatSn=self.S_est, ssn=self.SS_est)
        
    def estimate_X(self):
        self.X_est = (self.A_est @ self.S_est.T).T
        
    def run_EM_single(self, EM_iters):
        self.i = 0
        while self.i < EM_iters:
            if self.i>0:
                old_A_est = self.A_est.copy()
                old_S_est = self.S_est.copy()
                old_X_est = self.X_est.copy()
            self.E_step()
            self.M_step()
            self.estimate_X()
            # Check for convergence (you can adjust the threshold as needed)
            if self.i>0:
                A_change = np.linalg.norm(self.A_est - old_A_est)
                S_change = np.linalg.norm(self.S_est - old_S_est)
                X_change = np.linalg.norm(self.X_est - old_X_est)
                if A_change < self.EM_convergence_threshold and S_change < self.EM_convergence_threshold and X_change < self.EM_convergence_threshold:
                    print(f'Converged at iteration {self.i}')
                    break
        
            if self.i == EM_iters - 1:
                print('did not converge after {} iterations'.format(EM_iters))    
            self.i += 1
        self.A_err = self.evaluate_A()
        self.S_err = self.evaluate_S()
        self.X_err = self.evaluate_X()
    
    def evaluate_S(self):
        #returning the sources scalaed and permuted according to the best permutation and scaling of A_est compared to A_true. We calculate the MSE after applying these corrections
        S_perm = ol_data.scale_permute_src(srcest=self.S_est, true_mat=self.A_true, est_mat=self.A_est) 
        self.final_S_est = S_perm #storing the final estimate of S after permutation and scaling 
        #remove the known sources from the error calculation
        S_err = ol_metric.mse_weighted_remove_known(true=self.S_true, estimate=self.final_S_est, sknown=self.sknown)[0] #
        return S_err
    
    def evaluate_A(self):    
        estimate_perm, col_ind, signs, scales, _ = ol_data.best_permutation_match_sign_flips(A=self.A_true, B=self.A_est)
        # Normalise true columns — error is purely directional after scale correction
        col_norms  = np.linalg.norm(self.A_true, axis=0)
        col_norms  = np.where(col_norms < 1e-10, 1.0, col_norms)  # guard zero cols
        true_normed = self.A_true / col_norms[np.newaxis, :]

        # estimate_perm is already optimally scaled — normalise by same norms
        est_normed  = estimate_perm / col_norms[np.newaxis, :]
        self.final_A_est = estimate_perm
        nsrc = np.maximum(self.A_true.shape[1], 1)
        err  = np.linalg.norm(true_normed - est_normed, 'fro') / np.sqrt(nsrc)
        return err
    
    def evaluate_X(self):
        X_err = ol_metric.mse(true=self.X_true, estimate=self.X_est)[0]
        return X_err
    
    def run_study(self, known_src_range, n_inits, EM_iters, save_csv=True, results_dir='results', csv_name='em_known_src_study.csv'):
        records = []
        for n_known_src in known_src_range:
            self.set_known_variables(n_known_src)
            for init in range(n_inits):
                np.random.seed(init)
                self.initialize_parameters()
                self.run_EM_single(EM_iters)
                records.append({
                    'n_known_src': n_known_src,
                    'init': init,
                    'A_err': self.A_err,
                    'S_err': self.S_err,
                    'X_err': self.X_err,
                    'converged': self.i < EM_iters - 1,
                    'convergence_iteration': self.i,
                })
        df = pd.DataFrame(records)
        df['nsrc'] = self.nsrc
        df['nobs'] = self.nobs
        df['n_inits'] = n_inits
        df['sd_err'] = self.err_sd
        df['EM_convergence_threshold'] = self.EM_convergence_threshold
        df['seed'] = self.seed
        return df

#legacy

'''
def run_EM(x, init_Sigma, init_xi, init_A, sknown, aknown, err_sd, logger, Iter, trueA=None, trueS=None, truncated=False, n_nonzero=0,
           adjust_err_mat=False, adjust_err_src=False, normalize_A=False, update_covariance = True, store_all = False):

    

    # ── Resolve step functions once before the loop ───────────────────────────
    e_step_fn = partial(E_step_truncate, n_nonzero=n_nonzero) if truncated else E_step

    err_f_matrix = partial(ol_metric.frobenius_err_remove_known, aknown=aknown) \
                   if adjust_err_mat else ol_metric.frobenius_err
    err_f_src    = partial(ol_metric.mse_weighted_remove_known, sknown=sknown) \
                   if adjust_err_src else ol_metric.mse_weighted

    # ── Initialise state ──────────────────────────────────────────────────────
    Anew, Sigma_new, xi = init_A, init_Sigma, init_xi
    hatSn             = None
    matrix_coherence  = None
    errors_matrix, errors_latent_variables, coherence_series, xi_diff = [], [], [], []
    err_matrix, err_latent = None, None
    obs_likelihood = []
    if store_all:
        hatSn_history = []
        xi_history = []
        
    # ── EM loop ───────────────────────────────────────────────────────────────
    for i in range(Iter):
        try:
            xi_old = xi.copy()

            hatSn, ssn       = e_step_fn(A=Anew, Sigma=Sigma_new, x=x, xi=xi, sknown=sknown)
                
            Anew, Sigma_new, xi = M_step(x=x, hatSn=hatSn, ssn=ssn,
                                         aknown=aknown, err_sd=err_sd, normalize_A=normalize_A, update_covariance = update_covariance)
            if store_all:
                hatSn_history.append(hatSn)
                xi_history.append(xi)
            shat             = hatSn
            matrix_coherence = ol_data.coherence(Anew)
            coherence_series.append(matrix_coherence)
            obs_likelihood.append(ol_data.calculate_x_likelihood_variational(x = x, xi = xi, A = Anew, Sigma=Sigma_new))
            if trueA is not None:
                err_matrix, name_matrix = err_f_matrix(true=trueA, estimate=Anew)
                errors_matrix.append(err_matrix)
                if i % 5 == 0:
                    logger.info(f'Iter {i:03d} | {name_matrix}: {err_matrix:.6f}')
                if len(errors_matrix) > 1 and errors_matrix[-1] > errors_matrix[-2]:
                    logger.warning(f'Iter {i:03d} | Error increased: '
                                   f'{errors_matrix[-2]:.6f} -> {errors_matrix[-1]:.6f}')

            if trueS is not None:
                err_latent, name_latent = err_f_src(true=trueS, estimate=shat)
                errors_latent_variables.append(err_latent)
                if i % 5 == 0:
                    logger.info(f'Iter {i:03d} | {name_latent}: {err_latent:.6f}')

            xi_change = 0
            if i > 0:
                xi_change, xi_name = ol_metric.mse_weighted(true=xi_old, estimate=xi)
                xi_diff.append(xi_change)
            if i % 5 == 0:
                logger.info(f'Iter {i:03d} | xi_change: {xi_change:.6f}')

        except Exception as e:
            logger.error(f'Iter {i:03d} | Step failed: {type(e).__name__}: {e}')
            break
        
    if store_all:
        hatSn = hatSn_history
        xi = xi_history
    return Anew, Sigma_new, xi, hatSn, errors_latent_variables, errors_matrix, xi_diff, matrix_coherence, coherence_series, obs_likelihood
'''
  
def compute_total_elbo(X, S, SS, post_cov, A, Sigma, xi_all, k, b):
    """
    Computes the formal joint data likelihood lower bound (ELBO) 
    
    Parameters:
    -----------
    X : ndarray of shape (n, nobs)
        The observational matrix (columns are sample vectors).
    S : ndarray of shape (n, nsrc)
        The matrix the estimated values of S= Eq[S]. First k columns should consist of known values, last nsrc - k columns the posterior means.
    S : ndarray of shape (n, nsrc)
        The matrix of estimated values of Eq[SS]. 
    post_cov : ndarray of shape (n, nsrc - k, nsrc - k)
        Posterior covariances of unknown sources for all samples
    A : ndarray of shape (nobs, nsrc)
        The currently estimated joint mixing matrix.
    Sigma : ndarray of shape (nobs, nobs)
        The currently estimated noise covariance matrix. Scalar value may also be passed for isotropic noise.
    xi_all : ndarray of shape (n, nsrc)
        The variational bounds parameters matrix for all samples.
    k : int
        Number of known/observed source dimensions.
        
    Returns:
    --------
    total_elbo : float
        The calculated unified scalar variational lower bound value.
    """
    n, dx = X.shape
    ds = A.shape[1]
    du = ds - k #number of unknown sources
    
    
    if Sigma.shape == (): #if Sigma is passed as scalar we assume isotropic noise.
        Sigma = np.identity(dx)*Sigma
    # 1. Numerical protection for xi
    xi_abs = np.abs(xi_all) + 1e-12
    
    # --- 1. Variational Laplace Constant: log(K_xi) ---
    # Vectorized computation over all entries simultaneously
    log_K_xi_total = np.sum(-np.log(2.0*b) - 0.5 * xi_abs/b + 0.5 * np.log(2.0 * np.pi * xi_abs * b))
    
    # --- 2. Observation Likelihood Expectation: E_q[log p(x|s)] ---
    inv_Sigma = np.linalg.inv(Sigma)
    _, logdet_Sigma = np.linalg.slogdet(Sigma)
    
    # Vectorized Mahalanobis distance (Quad X)
    diff = X - S @ A.T  # shape: (n, dx)
    quad_x_total = np.sum((diff @ inv_Sigma) * diff)
    
    if du > 0:
        mu_u = S[:, k:]
        A_u = A[:, k:]
        
        # Vectorized posterior covariance derivation
        C_u_all = post_cov.copy()
            
        # sum_l tr(inv_Sigma @ A_u @ C_l @ A_u.T) == tr(A_u.T @ inv_Sigma @ A_u @ sum_l(C_l))
        M = A_u.T @ inv_Sigma @ A_u            # shape: (du, du)
        C_u_sum = np.sum(C_u_all, axis=0)      # shape: (du, du)
        tr_x_total = np.sum(M * C_u_sum)       # Hadamard product sum equals trace
    else:
        tr_x_total = 0.0

    const_x = -0.5 * dx * np.log(2.0 * np.pi ) - 0.5 * logdet_Sigma
    E_log_p_x_given_s_total = n * const_x - 0.5 * (quad_x_total + tr_x_total)
    
    # --- 3. Latent Prior Expectation: E_q[log p(s)] ---
    # OPTIMIZATION: Q_l = Cov_s_l + mu_s_l * mu_s_l.T is mathematically identical 
    # to the joint second moment matrix SSn. We just need its diagonal!
    ss_diag = np.diagonal(SS, axis1=1, axis2=2)  # shape: (n, ds)
    
    quad_s_total = np.sum(ss_diag / (xi_abs*b))
    log_det_Lambda_total = np.sum(np.log(xi_abs*b))
    
    const_s = -0.5 * ds * np.log(2.0 * np.pi)
    E_log_p_s_total = n * const_s - 0.5 * log_det_Lambda_total - 0.5 * quad_s_total
    
    
    # --- 4. Posterior Entropy: H(q(s_u)) ---
    if du > 0:
        # np.linalg.slogdet natively processes batches of matrices (shape: n, du, du)
        _, logdet_Cu_all = np.linalg.slogdet(C_u_all)
        const_q = 0.5 * du * np.log(2.0 * np.pi * np.e)
        entropy_q_total = n * const_q + 0.5 * np.sum(logdet_Cu_all)
    else:
        entropy_q_total = 0.0
        
    # Sum all bound components
    total_elbo = log_K_xi_total + E_log_p_x_given_s_total + E_log_p_s_total + entropy_q_total
    #scale the elbo to be per observation*n
    total_elbo/= n
    
    return total_elbo

def compute_final_likelihood(X, S, A):
    n, nsrc = S.shape

    res = X-(A@S.T).T
    Sigma_est = (res.T @ res) / n
    
    SS_est = np.einsum('ni,nj->nij', S, S)
    xi_est = np.abs(S)    
    b = np.mean(np.abs(S))
    final_ELBO = compute_total_elbo(X = X, S = S, SS = SS_est, post_cov=None, A = A, Sigma= Sigma_est, k = nsrc, xi_all=xi_est, b= b)
    return final_ELBO