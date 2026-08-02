"""
semiblind_em.py
===============

Faithful, self-contained implementation of

    Q. Lin, N. Xu and H. Liang, "A Semi-Blind EM Algorithm for Overcomplete ICA",
    ICASSP 2009, pp. 1733-1736.

The paper extends the approximate EM algorithm of Zhong et al. (2004) -- the same
noisy-overcomplete-ICA model that ``em.py`` implements -- by incorporating
*waveform / reference* information about **some** of the sources (the "semi-blind"
setting).  The extra information enters through a *closeness measure* ``G(s_hat)``
that is added to the original MAP cost, biasing the source estimates towards a set
of reference signals and, as a by-product, removing the permutation ("order")
ambiguity for the sources with prior information.

Model (paper eq. 1, in this code base's shape conventions)
----------------------------------------------------------
    X = S @ A.T + noise
        A : (nobs, nsrc)      mixing matrix          (N x M, overcomplete M > N)
        X : (nsamples, nobs)  observations           (T x N)
        S : (nsamples, nsrc)  sources                (T x M)
        Sigma : (nobs, nobs)  noise covariance       (N x N)

Sources are Laplacian (paper eq. 2):   p(s) = (sqrt2)^-M  prod_i exp(-sqrt2 |s_i|)

Algorithm implemented here
--------------------------
E-step (learn S)  -- gradient ascent of the extended cost J = L + G (paper eq. 10):

    s <- s + eta * ( A^T Sigma^-1 (x - A s) + grad_phi(s) ) + grad_G(s)

    grad_phi(s) = -sqrt2 * sign(s)                              (Laplacian prior)
    grad_G(s)  ~= 2 * rho_i * ( R_ref @ g_i )                   (paper eq. 10 approx.)

  where, for each source i, g_i = [ corr(s_i, r_1), ..., corr(s_i, r_M) ] are the
  (normalised) correlations of the current estimate with every reference, and the
  per-source weight rho_i is learned as (paper eq. 9)

    rho_i = lam * max_j g_i[j]     if max_j g_i[j] >= xi_thr
          = lam * min_j g_i[j]     otherwise.

M-step (learn A) -- paper eq. 5 (Zhong et al.):

    A <- ( sum_t x_t s_t^T ) ( sum_t ( H(s_t)^-1 + s_t s_t^T ) )^-1

  with the (negative) Hessian of the per-sample MAP cost

    H(s_t) = A^T Sigma^-1 A + diag( sqrt2 / |s_t| )

  the diagonal term being the standard Gaussian (variational) approximation of the
  Laplacian curvature -- it is what makes the otherwise rank-deficient
  ``A^T Sigma^-1 A`` (M x M, rank N < M) invertible, exactly as the variational
  ``xi`` does in ``em.py``.

Order correction (paper eq. 11): after convergence the columns of S (and A) are
permuted so that estimate i is aligned with reference i.  Because the downstream
evaluation is permutation/sign invariant this does not change the reported error,
so it is applied only cosmetically and can be switched off.

Adaptation of the semi-blind prior to this code base
----------------------------------------------------
The paper builds ``L`` rough references for the ``L`` sources with prior
information and random references for the rest.  Here the prior information is the
``S_known`` matrix (columns that are known *exactly*, NaN elsewhere), so -- per the
chosen configuration -- the reference for a known slot is the known waveform
itself and the reference for an unknown slot is a random signal (as in the paper).
Known columns are additionally pinned to their true values during the E-step, which
is consistent with every other ``run_EM_*`` baseline and with the exact-known
semantics of ``S_known``.  ``build_references`` is factored out so a coarser
reference construction can be swapped in later.
"""

from __future__ import annotations

import numpy as np

import overcomplete_learning.data as ol_data


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
_SQRT2 = np.sqrt(2.0)


def _known_unknown_indices(S_known, nsrc):
    """Return (known_idx, unknown_idx) column indices from an ``S_known`` matrix."""
    if S_known is None:
        return np.array([], dtype=int), np.arange(nsrc)
    known_mask = ~np.all(np.isnan(S_known), axis=0)      # (nsrc,) bool
    known_idx = np.where(known_mask)[0]
    unknown_idx = np.where(~known_mask)[0]
    return known_idx, unknown_idx


def _unit_norm_columns(M, eps=1e-12):
    """L2-normalise every column; zero columns are left as zero."""
    norms = np.linalg.norm(M, axis=0, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return M / norms


def build_references(S_known, nsamples, nsrc, rng):
    """
    Construct the reference matrix ``R_ref`` (nsamples, nsrc).

    Column ``i`` is the reference for source-slot ``i``:
      * known slot   -> the exact known waveform (paper: "waveform information"),
      * unknown slot -> a random signal (paper: random references for s_{L+1..M}).

    References are returned with unit-L2-norm columns so that the correlations used
    for ``rho`` (paper eq. 9) and ``G`` live on a comparable, scale-free footing and
    the threshold ``xi_thr`` in (0, 1) is meaningful.
    """
    rng = np.random.default_rng(rng)
    R = rng.standard_normal((nsamples, nsrc))
    known_idx, _ = _known_unknown_indices(S_known, nsrc)
    if known_idx.size:
        R[:, known_idx] = S_known[:, known_idx]
    return _unit_norm_columns(R)


def _rho(g_corr, lam, xi_thr):
    """
    Per-source correction factor rho_i (paper eq. 9).

    g_corr : (nsrc, nsrc) matrix of correlations, row i = g_i(s_hat_i, r).
    Returns rho : (nsrc,).
    """
    row_max = g_corr.max(axis=1)
    row_min = g_corr.min(axis=1)
    return lam * np.where(row_max >= xi_thr, row_max, row_min)


def _objective(X, S, A, Sigma_inv, g_corr, rho):
    """Extended cost  J = L(s) + G(s)  (paper eqs. 3, 6, 8)."""
    R = X - S @ A.T                                   # residual (T, N)
    quad = -0.5 * np.einsum('ti,ij,tj->', R, Sigma_inv, R)
    prior = -_SQRT2 * np.abs(S).sum()
    L = quad + prior
    G = float(np.sum(rho * np.sum(g_corr ** 2, axis=1)))
    return L + G


# --------------------------------------------------------------------------- #
#  E-step  (gradient ascent on the extended cost, paper eq. 10)
# --------------------------------------------------------------------------- #
def _soft_threshold(z, t):
    """Prox of the L1 term: sign(z) * max(|z| - t, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def E_step_semiblind(X, A, Sigma_inv, S, R_ref, known_idx, known_vals,
                     lam, xi_thr, eta_scale, n_inner):
    """
    Refine the source estimate ``S`` by ``n_inner`` steps of the extended learning
    rule (paper eq. 10), maximising  J = L(s) + G(s).

    The per-sample data cost + Laplacian prior is exactly the LASSO objective
    ``0.5 ||x - A s||^2_Sigma + sqrt2 ||s||_1``.  The paper writes the learning rule
    with the sub-gradient ``grad_phi = -sqrt2 sign(s)``; that sub-gradient chatters
    around zero and converges slowly.  We realise the *same* objective with a
    proximal (ISTA) step -- an explicit ascent step on the smooth part
    (data term + closeness term ``G``) followed by soft-thresholding for the L1
    prior -- which optimises the identical cost but converges reliably.  Known
    columns are pinned to their exact values.

    Returns
    -------
    S       : (nsamples, nsrc) refined estimate (known columns pinned)
    g_corr  : (nsrc, nsrc) final reference-correlation matrix
    rho     : (nsrc,) final per-source weights
    """
    S = S.copy()
    AtSA = A.T @ Sigma_inv @ A                        # (M, M), data-term curvature
    XSinvA = (X @ Sigma_inv) @ A                      # (T, M), data pull, constant
    nsamples = X.shape[0]
    lip = np.linalg.norm(AtSA, 2) + 1e-12             # Lipschitz const of smooth grad
    step = eta_scale / lip

    g_corr = np.zeros((A.shape[1], A.shape[1]))
    rho = np.zeros(A.shape[1])

    for _ in range(max(1, n_inner)):
        # --- gradient of the smooth part: data term  A^T Sigma^-1 (x - A s) ---
        grad_data = XSinvA - S @ AtSA

        # --- closeness / reference gradient (paper eqs. 7-10) ---
        S_unit = _unit_norm_columns(S)                # cosine-style correlations
        g_corr = S_unit.T @ R_ref                     # (M, M), row i = g_i
        rho = _rho(g_corr, lam, xi_thr)               # (M,)
        # grad_G[:, i] = 2 * rho_i * (R_ref @ g_i)  (g_i = i-th row of g_corr)
        grad_G = 2.0 * (R_ref @ g_corr.T) * rho[np.newaxis, :] / nsamples

        # --- ascent step on the smooth part, then prox of the L1 prior ---
        S_half = S + step * (grad_data + grad_G)
        S = _soft_threshold(S_half, step * _SQRT2)

        if known_idx.size:                            # pin known sources each step
            S[:, known_idx] = known_vals

    return S, g_corr, rho


# --------------------------------------------------------------------------- #
#  M-step  (paper eq. 5)
# --------------------------------------------------------------------------- #
def _condition_floor(Sigma, abs_floor, cond_ratio=1e-2):
    """
    Floor the eigenvalues of a symmetric PSD noise covariance.

    An overcomplete model can reconstruct the observations almost exactly along
    some direction, driving the residual covariance towards zero; ``Sigma^-1`` then
    explodes and the E-step step size collapses (or the estimate overfits).  We
    apply both an *absolute* floor ``abs_floor`` (a minimum noise level, a small
    fraction of the observation power) and a *relative* one that caps the condition
    number at ``1 / cond_ratio`` -- this keeps ``Sigma^-1`` bounded and well-scaled
    throughout, consistent with the paper's assumption of a known, bounded noise
    covariance.
    """
    Sigma = (Sigma + Sigma.T) / 2
    w, V = np.linalg.eigh(Sigma)
    floor = max(abs_floor, cond_ratio * float(w.max()))
    w = np.clip(w, floor, None)
    return (V * w) @ V.T


def M_step_semiblind(X, S, A, Sigma_inv, normalize_A_col, unknown_idx,
                     iid_noise=True, ridge=1e-6):
    """
    Update the mixing matrix by paper eq. 5 and re-estimate the noise covariance.

        A <- ( sum_t x_t s_t^T )( sum_t ( H(s_t)^-1 + s_t s_t^T ) )^-1
        H(s_t) = A^T Sigma^-1 A + diag( sqrt2 / |s_t| )
    """
    nsamples, nsrc = S.shape
    AtSA = A.T @ Sigma_inv @ A                                    # (M, M)

    # Per-sample precision (Hessian) with Laplacian curvature on the diagonal.
    diag_prior = _SQRT2 / np.clip(np.abs(S), 1e-6, None)          # (T, M)
    H = AtSA[np.newaxis, :, :] + \
        diag_prior[:, :, np.newaxis] * np.eye(nsrc)[np.newaxis, :, :]
    H_inv = np.linalg.inv(H)                                      # (T, M, M)

    sum_H_inv = H_inv.sum(axis=0)                                 # (M, M)
    gram = S.T @ S                                                # sum_t s_t s_t^T
    denom = sum_H_inv + gram + ridge * np.eye(nsrc)
    numer = X.T @ S                                               # sum_t x_t s_t^T

    A_new = numer @ np.linalg.inv(denom)                         # (N, M)

    if normalize_A_col:
        # Keep the physical scale of known columns; normalise only unknown ones.
        cols = A_new.copy()
        if unknown_idx.size:
            cols[:, unknown_idx] = ol_data.normalize_columns(A_new[:, unknown_idx])
        A_new = cols

    # Noise covariance from the reconstruction residual, floored (absolute +
    # relative) so the overcomplete fit cannot drive Sigma singular / to zero.
    resid = X - S @ A_new.T                                       # (T, N)
    abs_floor = 1e-3 * float(np.mean(X ** 2))                     # min noise level
    if iid_noise:
        # Isotropic noise: far more stable than a free covariance, which tends to
        # develop a near-singular direction the overcomplete model overfits.
        var = max(float(np.mean(resid ** 2)), abs_floor)
        Sigma_new = var * np.eye(A_new.shape[0])
    else:
        Sigma_new = (resid.T @ resid) / nsamples
        Sigma_new = _condition_floor(Sigma_new, abs_floor=abs_floor)
    return A_new, Sigma_new


# --------------------------------------------------------------------------- #
#  Order correction (paper eq. 11)
# --------------------------------------------------------------------------- #
def order_correction(S, A, g_corr, known_idx):
    """
    Permute columns so estimate ``i`` aligns with reference ``i`` (paper eq. 11).

    Uses the reference-correlation matrix ``g_corr`` (row = estimate,
    col = reference) and a greedy / Hungarian assignment.  Purely cosmetic w.r.t.
    the permutation-invariant evaluation, hence failures fall back to identity.
    """
    nsrc = S.shape[1]
    try:
        from scipy.optimize import linear_sum_assignment
        # maximise total |correlation|  ->  minimise its negative
        est_for_ref, ref = linear_sum_assignment(-np.abs(g_corr.T))
        perm = np.empty(nsrc, dtype=int)
        perm[ref] = est_for_ref                      # column that fills ref slot
    except Exception:
        return S, A
    # Do not disturb pinned known slots.
    if known_idx.size:
        perm[known_idx] = known_idx
    if len(set(perm.tolist())) != nsrc:              # not a valid permutation
        return S, A
    return S[:, perm], A[:, perm]


# --------------------------------------------------------------------------- #
#  Baseline entry point  (matches the run_EM_* contract used by run_study)
# --------------------------------------------------------------------------- #
def run_EM_semiblind(X, rng, normalize_A_col=True, whiten_data=False,
                     scale_source=False, S_known=None,
                     EM_iter: int = 1_000, err_tolerance: float = 1e-4,
                     nsrc: int = 0,
                     lam: float = 0.01, xi_thr: float = 0.4,
                     eta_scale: float = 0.9, n_inner: int = 50,
                     iid_noise: bool = True,
                     order_correct: bool = True, **kwargs) -> dict:
    """
    Semi-blind EM algorithm (Lin, Xu & Liang, ICASSP 2009) as an overcomplete-ICA
    baseline.

    Parameters mirror the other ``run_EM_*`` wrappers.  Paper-specific knobs:

    lam        : lambda in (0, 1), overall strength of the reference term (eq. 9).
    xi_thr     : xi in (0, 1), correlation threshold selecting max vs. min in eq. 9.
    eta_scale  : (0, 1] fraction of the inverse Lipschitz constant used as the
                 gradient step size.
    n_inner    : number of gradient-ascent inner steps per EM iteration (eq. 10).
    iid_noise  : model the noise covariance as isotropic (scalar * I).  Strongly
                 recommended -- a free covariance tends to develop a near-singular
                 direction that the overcomplete model overfits, destabilising EM.
    order_correct : apply the eq.-11 permutation to align estimates with references.

    Returns the standard result dict (A_est, S_est, X_est, Sigma_est, converged,
    convergence_iteration, elbo_history, n_known, A_init).
    """
    X = np.asarray(X, dtype=float).copy()
    nsamples, nobs = X.shape

    # ---- source dimension & known bookkeeping -----------------------------
    if S_known is None:
        assert nsrc != 0, 'Provide nsrc when S_known is None.'
    else:
        nsrc = S_known.shape[1]
    known_idx, unknown_idx = _known_unknown_indices(S_known, nsrc)
    n_known = int(known_idx.size)

    # ---- initialisation ---------------------------------------------------
    rng = np.random.default_rng(rng)
    A, Sigma, _xi = ol_data.initialise_parameters(
        nobs=nobs, nsrc=nsrc, nreps=nsamples, rng=rng)
    A_init = A.copy()

    # Warm-start overrides (parity with run_EM_extended)
    if kwargs.get('A_init') is not None:
        A = kwargs['A_init'].copy()
        A_init = A.copy()
    if kwargs.get('Sigma_init') is not None:
        Sigma = kwargs['Sigma_init'].copy()

    # Reference signals (semi-blind prior)
    R_ref = build_references(S_known, nsamples, nsrc, rng)

    # Initial source estimate: least-squares back-projection, known columns pinned.
    S = X @ np.linalg.pinv(A).T                       # (T, M)
    if n_known:
        S[:, known_idx] = S_known[:, known_idx]

    Sigma_inv = np.linalg.inv(Sigma)
    known_vals = S_known[:, known_idx] if n_known else None

    # ---- EM loop ----------------------------------------------------------
    obj_history = []
    i = 0
    g_corr = np.zeros((nsrc, nsrc))
    rho = np.zeros(nsrc)
    while i < EM_iter:
        S, g_corr, rho = E_step_semiblind(
            X, A, Sigma_inv, S, R_ref, known_idx, known_vals,
            lam=lam, xi_thr=xi_thr, eta_scale=eta_scale, n_inner=n_inner)
        if n_known:
            S[:, known_idx] = S_known[:, known_idx]   # keep known sources exact

        A, Sigma = M_step_semiblind(
            X, S, A, Sigma_inv, normalize_A_col, unknown_idx, iid_noise=iid_noise)
        Sigma_inv = np.linalg.inv(Sigma)

        obj = _objective(X, S, A, Sigma_inv, g_corr, rho)
        obj_history.append(obj)

        if i > 10:
            window_delta = np.max(np.abs(np.diff(obj_history[-11:])))
            if abs(window_delta) / nobs < err_tolerance:
                break
        i += 1

    # ---- order correction (paper eq. 11) ----------------------------------
    if order_correct:
        S, A = order_correction(S, A, g_corr, known_idx)
        if n_known:
            S[:, known_idx] = S_known[:, known_idx]

    X_est = S @ A.T

    return {
        'S_est': S,
        'SS_est': None,
        'A_est': A,
        'Sigma_est': Sigma,
        'xi_est': None,
        'X_est': X_est,
        'post_cov': None,
        'converged': i < EM_iter - 1,
        'convergence_iteration': i,
        'elbo_history': obj_history,
        'n_known': n_known,
        'A_init': A_init,
    }
