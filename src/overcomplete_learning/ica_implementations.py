import numpy as np

def g(u, nonlinearity='logcosh'):
    """
    Non-linearity g = G' and its derivative g' = G''
    Hyvarinen 2001 recommends logcosh or exp.
    """
    if nonlinearity == 'logcosh':
        g_val  = np.tanh(u)                      # g(u)  = tanh(u)
        gp_val = 1.0 - np.tanh(u) ** 2           # g'(u) = 1 - tanh^2(u)
    elif nonlinearity == 'exp':
        exp_u  = np.exp(-0.5 * u ** 2)
        g_val  = u * exp_u                        # g(u)  = u * exp(-u^2/2)
        gp_val = (1.0 - u ** 2) * exp_u          # g'(u) = (1-u^2) * exp(-u^2/2)
    elif nonlinearity == 'cube':
        g_val  = u ** 3                           # g(u)  = u^3
        gp_val = 3.0 * u ** 2                     # g'(u) = 3u^2
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
    return g_val, gp_val


def whiten(X):
    """
    Whitens the data matrix X.
    X : (nsamples, nobs)
    Returns whitened data and the whitening matrix.
    """
    X_centered  = X - X.mean(axis=0)                          # centre
    cov         = np.cov(X_centered.T)                        # (nobs, nobs)
    eigvals, eigvecs = np.linalg.eigh(cov)                    # sorted ascending
    eigvals     = np.maximum(eigvals, 1e-10)                  # numerical guard
    W_whiten    = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    X_white     = np.einsum('ij,nj', W_whiten, X_centered)                     # (nsamples, ndim)
    return X_white, W_whiten


def fastica_single(X_white, w_init, nonlinearity='logcosh',
                   max_iter=1000, tol=1e-8):
    """
    FastICA fixed-point algorithm for a SINGLE component.
    Extracts one independent component.

    Parameters
    ----------
    X_white     : (nsamples, ndim) — whitened data
    w_init      : (ndim,)          — initial weight vector
    nonlinearity: 'logcosh', 'exp', or 'cube'
    max_iter    : maximum number of iterations
    tol         : convergence threshold on |w_new . w_old - 1|

    Returns
    -------
    w_final     : (ndim,)          — converged weight vector
    w_history   : (n_iters, ndim)  — weight vector at every iteration
    n_iters     : int              — number of iterations until convergence
    converged   : bool
    """
    nsamples, ndim = X_white.shape

    # Normalise initial vector
    w        = w_init / np.linalg.norm(w_init)
    w_history = [w.copy()]

    converged = False
    for i in range(max_iter):

        # Project data onto current w: (nsamples,)
        u = X_white @ w                                        # u = W^T x

        # Evaluate nonlinearity
        g_val, gp_val = g(u, nonlinearity)                    # both (nsamples,)

        # Fixed-point update (eq. 11 in Hyvarinen 2001):
        # w_new = E[x g(w^T x)] - E[g'(w^T x)] w
        w_new  = (X_white.T @ g_val) / nsamples               # E[x g(w^T x)]
        w_new -= gp_val.mean() * w                             # - E[g'(w^T x)] w

        # Normalise
        w_new /= np.linalg.norm(w_new)

        # Store this iteration's estimate
        w_history.append(w_new.copy())

        # Convergence: |w_new . w_old| should approach 1
        # (accounts for sign ambiguity)
        delta = np.abs(np.abs(w_new @ w) - 1.0)

        w = w_new

        if delta < tol:
            converged = True
            break

    return w, np.array(w_history), i + 1, converged


def fastica(X, n_components=None, nonlinearity='logcosh',
            max_iter=1000, tol=1e-8, seed=0):
    """
    FastICA with deflationary orthogonalisation (Hyvarinen 2001, Algorithm 1).
    Extracts n_components independent components sequentially,
    orthogonalising each new component against previously found ones.

    Parameters
    ----------
    X            : (nsamples, ndim)  — observed mixed signals
    n_components : int               — number of ICs to extract (default: ndim)
    nonlinearity : 'logcosh', 'exp', or 'cube'
    max_iter     : max iterations per component
    tol          : convergence threshold
    seed         : random seed for initial weight vectors

    Returns
    -------
    S_hat       : (nsamples, n_components) — estimated independent components
    W_hat       : (n_components, ndim)     — estimated unmixing matrix (in whitened space)
    W_whiten    : (ndim, ndim)             — whitening matrix
    histories   : list of (n_iters, ndim) arrays — w_history per component
    diagnostics : list of dicts            — convergence info per component
    """
    nsamples, ndim = X.shape
    n_components   = n_components or ndim

    rng = np.random.default_rng(seed)

    # ── Step 1: Whiten ────────────────────────────────────────────────────────
    X_white, W_whiten = whiten(X)

    # ── Step 2: Deflationary extraction ──────────────────────────────────────
    W_hat      = np.zeros((n_components, ndim))  # rows = unmixing vectors
    histories   = []
    diagnostics = []

    for c in range(n_components):

        # Random initialisation
        w_init = rng.standard_normal(ndim)

        # Deflationary orthogonalisation against already-found components
        # (Gram-Schmidt step before starting iterations for this component)
        for prev in range(c):
            w_init -= (w_init @ W_hat[prev]) * W_hat[prev]
        if np.linalg.norm(w_init) < 1e-10:
            w_init = rng.standard_normal(ndim)   # re-draw if degenerate
        w_init /= np.linalg.norm(w_init)

        # Run fixed-point algorithm
        w_final, w_history, n_iters, converged = fastica_single(
            X_white, w_init,
            nonlinearity=nonlinearity,
            max_iter=max_iter,
            tol=tol
        )

        # Deflate: orthogonalise w_final against all previous components
        for prev in range(c):
            w_final -= (w_final @ W_hat[prev]) * W_hat[prev]
        w_final /= np.linalg.norm(w_final)

        W_hat[c]    = w_final
        histories.append(w_history)
        diagnostics.append({
            'component':  c,
            'n_iters':    n_iters,
            'converged':  converged,
        })

        status = 'converged' if converged else 'DID NOT CONVERGE'
        print(f'Component {c:02d} | {status} in {n_iters} iterations')

    # ── Step 3: Recover sources ───────────────────────────────────────────────
    S_hat = X_white @ W_hat.T                    # (nsamples, n_components)

    return S_hat, W_hat, W_whiten, histories, diagnostics


def fastica_parallel(X, n_components=None, nonlinearity='logcosh',
                     max_iter=1000, tol=1e-8, seed=0):
    """
    FastICA with parallel symmetric orthogonalisation (Hyvarinen 2001, Algorithm 2).
    All components are updated simultaneously and orthogonalised via
    the symmetric decorrelation W <- (W W^T)^{-1/2} W after each step.
    
    Supports the overcomplete case where n_components > ndim.

    Parameters
    ----------
    X            : (nsamples, ndim)  — observed mixed signals
    n_components : int               — number of ICs to extract (can exceed ndim)
    nonlinearity : 'logcosh', 'exp', or 'cube'
    max_iter     : max iterations
    tol          : convergence threshold on max|w_new . w_old - 1| across components
    seed         : random seed

    Returns
    -------
    S_hat       : (nsamples, n_components) — estimated independent components
    W_hat       : (n_components, ndim)     — unmixing matrix (whitened space)
    W_whiten    : (ndim, ndim)             — whitening matrix
    W_history   : (n_iters, n_components, ndim) — W at every iteration
    diagnostics : dict                     — convergence info
    """
    nsamples, ndim = X.shape
    n_components   = n_components or ndim

    rng = np.random.default_rng(seed)

    # ── Step 1: Whiten ────────────────────────────────────────────────────────
    X_white, W_whiten = whiten(X)                              # (nsamples, ndim)

    # ── Step 2: Initialise all weight vectors ─────────────────────────────────
    W = rng.standard_normal((n_components, ndim))              # (n_components, ndim)

    # Initial symmetric orthogonalisation
    W = sym_orth(W)

    W_history = [W.copy()]

    # ── Step 3: Parallel fixed-point iterations ───────────────────────────────
    converged = False
    for i in range(max_iter):

        W_old = W.copy()

        # Project all components at once
        # U[n, c] = w_c^T x_n  →  shape (nsamples, n_components)
        U = X_white @ W.T                                      # (nsamples, n_components)

        # Evaluate nonlinearity for all components simultaneously
        g_val, gp_val = g(U, nonlinearity)                    # both (nsamples, n_components)

        # Parallel fixed-point update for all w_c simultaneously:
        # w_c_new = E[x g(w_c^T x)] - E[g'(w_c^T x)] w_c
        # Vectorised over all components:
        # W_new = (1/N) * g(U)^T @ X_white  -  diag(E[g'(U)]) @ W
        W_new  = (g_val.T @ X_white) / nsamples               # (n_components, ndim)
        W_new -= gp_val.mean(axis=0)[:, np.newaxis] * W       # (n_components, ndim)

        # Symmetric orthogonalisation — key difference from deflationary
        W = sym_orth(W_new)

        W_history.append(W.copy())

        # Convergence: max change in |w_new . w_old| across all components
        # Each dot product should approach ±1
        delta = np.max(np.abs(np.abs(np.einsum('ci,ci->c', W, W_old)) - 1.0))

        if delta < tol:
            converged = True
            break

    n_iters = i + 1
    status  = 'converged' if converged else 'DID NOT CONVERGE'
    print(f'FastICA parallel | {status} in {n_iters} iterations')

    # ── Step 4: Recover sources ───────────────────────────────────────────────
    S_hat = np.einsum('ij,nj->ni', W.T, X_white)
    #X_white @ W.T                                      # (nsamples, n_components)

    diagnostics = {
        'n_iters':   n_iters,
        'converged': converged,
        'delta':     delta,
    }

    return S_hat, W, W_whiten, np.array(W_history), diagnostics


def sym_orth(W):
    """
    Symmetric orthogonalisation: W <- (W W^T)^{-1/2} W
    Hyvarinen 2001 eq. 45.
    Works for both square (W: n x n) and overcomplete (W: p x n, p > n) cases.

    Parameters
    ----------
    W : (n_components, ndim)

    Returns
    -------
    W_orth : (n_components, ndim)
    """
    WWt           = W @ W.T                                    # (n_components, n_components)
    eigvals, eigvecs = np.linalg.eigh(WWt)
    eigvals       = np.maximum(eigvals, 1e-10)                 # numerical guard
    W_orth        = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T @ W
    return W_orth