import numpy as np
from scipy.linalg import eigh, svd

def overica(X, k, mu=1.0, s_factor=10, max_iter_sdp=100):
    """
    Overcomplete ICA via SDP (Algorithm 1).
    
    Parameters:
    X : ndarray of shape (p, n) - Observations (p sensors, n samples).
    k : int - Latent dimension (number of sources).
    mu : float - Regularization parameter for the SDP relaxation[cite: 2231].
    s_factor : int - Multiple of k for number of Hessians to estimate[cite: 2189].
    """
    X = X.T
    p, n = X.shape
    s = s_factor * k
    
    # Pre-centering (Assumption 2.2) [cite: 2155]
    X = X - np.mean(X, axis=1, keepdims=True)
    
    # --- STEP I: Subspace Estimation [cite: 2119, 2156] ---
    # 1. Sample vectors t_j from Gaussian distribution [cite: 2190]
    # We scale t to avoid numerical overflow in exp(t'x)
    std_x = np.std(X)
    T_vecs = np.random.randn(p, s) / (std_x * np.sqrt(p))
    
    H_list = []
    print("Step I: Estimating Hessians...")
    for j in range(s):
        H_j = compute_hessian(X, T_vecs[:, j])
        H_list.append(H_j.flatten())
    
    # 2. Extract orthonormal basis for W using SVD [cite: 2188]
    # W is the span of atoms d_i d_i^T.
    H_matrix = np.array(H_list).T # (p*p, s)
    U, S, _ = svd(H_matrix, full_matrices=False)
    
    # Basis for the subspace W (first k singular vectors)
    # Basis for the null space N(W) (remaining vectors)
    # Total symmetric dimension m = p(p+1)/2 [cite: 2154]
    # For simplicity, we use the p*p basis and handle symmetry later
    W_basis_vecs = U[:, :k] 
    N_W_basis_vecs = U[:, k:p*p] # F_j basis [cite: 2220]
    
    # Convert N_W_basis back to matrix form for trace operations
    F_matrices = [v.reshape(p, p) for v in N_W_basis_vecs.T]

    # --- STEP II: Estimation of Atoms (Deflation) [cite: 2120, 2197] ---
    D_hat = np.zeros((p, k))
    current_F = list(F_matrices)
    
    print("Step II: Solving SDP for atoms...")
    for i in range(k):
        # 1. Choose G (randomly from W or standard normal) [cite: 2206, 2242]
        G = np.random.randn(p, p)
        G = (G + G.T) / 2 # Ensure symmetry
        
        # 2. Solve SDP Relaxation via FISTA (Algorithm 2) [cite: 2074, 2232]
        B_star = solve_sdp_fista(G, current_F, mu, max_iter=max_iter_sdp)
        
        # 3. Estimate mixing component d_i (largest eigenvector) [cite: 2142]
        eigvals, eigvecs = eigh(B_star)
        d_i = eigvecs[:, -1]
        D_hat[:, i] = d_i
        
        # Adaptive Deflation: Add found atom to the null space basis 
        atom_vec = np.outer(d_i, d_i)
        current_F.append(atom_vec)
        
    return D_hat

def compute_hessian(X, t):
    """
    Estimates the Hessian of the CGF (Equation 5).
    C_x(t) = E[xx' exp(t'x)] / E[exp(t'x)] - E[x exp(t'x)]E[x exp(t'x)]' / (E[exp(t'x)]^2)
    """
    tx = t @ X # (n,)
    exp_tx = np.exp(tx - np.max(tx)) # Robust softmax-style scaling
    
    w = exp_tx / np.sum(exp_tx)
    
    # Weighted mean: E_x(t)
    ex_t = X @ w # (p,)
    
    # Weighted covariance-like term
    # Weighted sum of xx'
    # Equivalent to (X * w) @ X.T
    H = (X * w) @ X.T - np.outer(ex_t, ex_t)
    return (H + H.T) / 2

def solve_sdp_fista(G, F_list, mu, max_iter):
    """
    Projected Accelerated Gradient Descent (FISTA).
    Objective: max <G, B> - (mu/2) * sum(<B, Fj>^2)
    Equivalent to: min f(B) = -<G, B> + (mu/2) * sum(<B, Fj>^2)
    Constraints: B >= 0, Tr(B) = 1.
    """
    p = G.shape[0]
    B = np.eye(p) / p
    Z = B.copy()
    t = 1.0
    L = mu # Lipschitz constant [cite: 1934]
    
    for _ in range(max_iter):
        B_old = B.copy()
        
        # 1. Gradient of f(B) [cite: 1934]
        # grad = -G + mu * sum( Tr(Fj * Z) * Fj )
        grad = -G.copy()
        for F in F_list:
            tr_fb = np.sum(F * Z) # Trace of product for symmetric matrices
            grad += mu * tr_fb * F
            
        # 2. Gradient Step
        B_step = Z - (1.0 / L) * grad
        
        # 3. Proximal Operator: Project onto {B >= 0, Tr(B) = 1}
        B = project_to_psd_trace_one(B_step)
        
        # 4. Acceleration
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        Z = B + ((t - 1.0) / t_next) * (B - B_old)
        t = t_next
        
    return B

def project_to_psd_trace_one(M):
    """
    Proximal operator for the set K = {B >= 0, Tr(B) = 1}[cite: 2210].
    This is a projection of eigenvalues onto the unit simplex.
    """
    # Symmetric Eigendecomposition
    vals, vecs = eigh(M)
    
    # Project eigenvalues onto the unit simplex
    # (Algorithm for projection onto simplex)
    v_sorted = np.sort(vals)[::-1]
    rho = 0
    for j in range(len(v_sorted)):
        if v_sorted[j] + (1.0 / (j + 1)) * (1.0 - np.sum(v_sorted[:j+1])) > 0:
            rho = j + 1
    theta = (1.0 / rho) * (1.0 - np.sum(v_sorted[:rho]))
    new_vals = np.maximum(vals + theta, 0)
    
    return vecs @ np.diag(new_vals) @ vecs.T

# --- Validation Logic ---
if __name__ == "__main__":
    p, k = 3, 4 # Overcomplete case p < k
    n = 5000
    print(f"Simulating OverICA: p={p} sensors, k={k} sources...")
    
    # 1. Generate Non-Gaussian Sources (Laplacian)
    S = np.random.laplace(0, 1, (k, n))
    
    # 2. Mixing Matrix D (unit norm columns) [cite: 2150]
    D_true = np.random.randn(p, k)
    D_true /= np.linalg.norm(D_true, axis=0)
    
    # 3. Observations
    X = D_true @ S
    
    # 4. Run OverICA
    D_est = overica(X, k)
    
    print("\nTrue Mixing Matrix D:\n", D_true)
    print("\nEstimated Mixing Matrix D (Columns may be permuted/flipped):\n", D_est)