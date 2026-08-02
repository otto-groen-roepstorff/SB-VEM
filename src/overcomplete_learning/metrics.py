import numpy as np
import overcomplete_learning.data as ol_data
from scipy.optimize import linear_sum_assignment
from overcomplete_learning.data         import EMData  
#-------------------------------------
#----- SOURCE METRIC EVALUATIONS -----
#-------------------------------------

def mse(true, estimate):
    return np.mean((true-estimate) ** 2), 'MSE'

def mse_weighted(true,estimate):
    num = np.linalg.norm(true-estimate, ord = 2, axis = 1)
    den = np.linalg.norm(true, ord = 2, axis = 1)
    return np.mean(num/den), 'Relative MSE'

def mse_weighted_remove_known(true, estimate, sknown=None):
    #removing the known soruces
    true, estimate = ol_data.remove_known_sources(true=true, estimate=estimate, sknown=sknown)
        
    if true.shape[1] == 0:    # all sources were known — nothing to evaluate
        return 0, 'Relative L2 Error per Source'

    
    # Column-wise norms — each column is one source signal over all samples
    num = np.linalg.norm(true - estimate, ord=2, axis=0)  # (n_unknown,)
    den = np.linalg.norm(true,            ord=2, axis=0)  # (n_unknown,)
    den = np.where(den == 0, 1, den) #numerical guard to avoid numeric errors at 0's

    # Per-source relative error, averaged over unknown sources
    per_source_err = num / den                            # (n_unknown,)
    return np.mean(per_source_err), 'Relative L2 Error per Source'

def MCC_remove_known(true, estimate, sknown):
    """
    Computes MCC assuming columns are already permuted according to mixing matrix estimate

    Parameters
    ----------
    true     : (nsamples, nsrc)
    estimate : (nsamples, nsrc) — already column-permuted to match true
    sknown   : (nsamples, nsrc) or None
    
    Returns
    -------
    (error, name of error)
    """
    
    #removing known sources
    true, estimate = ol_data.remove_known_sources(true=true, estimate=estimate, sknown=sknown)

    if true.shape[1] == 0:    # all sources were known — nothing to evaluate
        return 1.0, 'MCC'
    
    nsrc = true.shape[1]

    # corrcoef expects (nvars, nsamples) — transpose since rowvar=False
    # returns (2*nsrc, 2*nsrc) — extract cross block
    C          = np.abs(np.corrcoef(true.T, estimate.T))  # (2*nsrc, 2*nsrc)
    cross_corr = C[:nsrc, nsrc:]                           # (nsrc, nsrc)
    d = np.nan_to_num(cross_corr, nan=0.0)     # constant/collapsed source -> corr 0

    # Diagonal = matched pair correlations (permutation already applied)
    d = np.diag(d)
    return np.mean(d), 'MCC'


def src_error_remove_known(true, estimate, sknown = None):
    estimate, col_ind, signs, scales, err = ol_data.best_permutation_match_sign_flips(A = true, B= estimate) #using hungarian algorithm for matching columns
    
    true, estimate = ol_data.remove_known_sources(true=true, estimate=estimate, sknown=sknown)
    
    if true.shape[1] == 0:    # all sources were known — nothing to evaluate
        return 0.0, 'Calibrated Relative MSE'
    
    num = np.linalg.norm(true-estimate, ord = 2, axis = 1)
    den = np.linalg.norm(true, ord = 2, axis = 1)
    den = np.where(den == 0, 1, den)
    #decide whether or not to scale the errors!
    return np.mean(num), 'Calibrated Relative MSE'

# ----------------------------------
# ----- MATRIX ERROR ASSESMENT -----
# ----------------------------------

def frobenius_err(true, estimate):
    estimate_perm, col_ind, signs, scales, _ = ol_data.best_permutation_match_sign_flips(
        A=true, B=estimate
    )
    # Normalise true columns — error is purely directional after scale correction
    col_norms  = np.linalg.norm(true, axis=0)
    col_norms  = np.where(col_norms < 1e-10, 1.0, col_norms)  # guard zero cols
    true_normed = true / col_norms[np.newaxis, :]

    # estimate_perm is already optimally scaled — normalise by same norms
    est_normed  = estimate_perm / col_norms[np.newaxis, :]

    nsrc = np.maximum(true.shape[1], 1)
    err  = np.linalg.norm(true_normed - est_normed, 'fro') / np.sqrt(nsrc)
    return err, 'Frobenius'

def matrix_angle_error(A_true, A_est, degrees=False):
    """
    Computes the column-wise angle error between two matrices,
    resolving permutation and sign ambiguities.
    """
    # 1. Normalize columns to unit length
    A_t = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
    A_e = A_est / np.linalg.norm(A_est, axis=0, keepdims=True)
    
    # 2. Compute the absolute correlation/distance matrix
    # entry (i, j) is the absolute cosine similarity between true column i and est column j
    corr = np.abs(A_t.T @ A_e)
    
    # 3. Hungarian matching to solve permutation (maximize correlation)
    true_ind, est_ind = linear_sum_assignment(-corr)
    
    # Reorder the estimated matrix columns to match the true matrix
    A_e_aligned = A_e[:, est_ind]
    
    # 4. Calculate individual dot products along the paired columns
    # Using element-wise multiplication and summing down rows
    dot_products = np.sum(A_t[:, true_ind] * A_e_aligned, axis=0)
    
    # Take absolute value because sign flip handles orientation 
    # (cos(theta) is identical for a vector and its negative counterpart up to 180 deg)
    dot_products = np.clip(np.abs(dot_products), 0.0, 1.0)
    
    # 5. Compute angles
    angles = np.arccos(dot_products)
    
    if degrees:
        angles = np.degrees(angles)
        
    return {
        'column_angles': angles,
        'mean_error': np.mean(angles),
        'max_error': np.max(angles),
        'permutation': est_ind
    }

def amari_distance(true, estimate):
    """
    Compute the Amari distance between two square matrices A and B.
    
    Parameters:
        A, B: numpy arrays of shape (n, n)
    
    Returns:
        scalar Amari distance
    """
    
    A = np.asarray(true)
    B = np.asarray(estimate)
    
    assert A.shape == B.shape, "Matrices must have the same shape"
    n = A.shape[0]
    
    # Compute P = A^{-1} B
    P = np.linalg.solve(A, B)
    P = np.abs(P)
    
    # Row terms
    row_sums = np.sum(P, axis=1)
    row_max = np.max(P, axis=1)
    row_term = np.sum(row_sums / row_max - 1)
    
    # Column terms
    col_sums = np.sum(P, axis=0)
    col_max = np.max(P, axis=0)
    col_term = np.sum(col_sums / col_max - 1)
    err = (row_term + col_term) / (2 * n * (n - 1))
    return err, 'Amari'

def generalized_correlation_matching(true, estimate):
    '''
    Finds the Amari distance between the correlation matrices of the matrices
    
    INPUTS
    
    true: the true matrix
    estimate: the ALREADY PERMUTED estimated matrix
    
    OUTPUTS
    err
    
    '''
    #get correlation matrix
    if true.shape[1] == 0:    # not inputting a matrix
        return 1.0, 'MCC'
    nsrc = true.shape[1]
    
    true_norm = ol_data.normalize_columns(true)
    estimate_norm = ol_data.normalize_columns(estimate)
    
    # corrcoef expects (nvars, nsamples) — transpose since rowvar=False
    # returns (2*nsrc, 2*nsrc) — extract cross block
    C          = np.abs(np.corrcoef(true.T, estimate.T))  # (2*nsrc, 2*nsrc)
    corr_true = C[nsrc:, nsrc:]                           # (nsrc, nsrc)
    corr_estimate = C[:nsrc, :nsrc]                           # (nsrc, nsrc)
    err = amari_distance(true=corr_true, estimate=corr_estimate)
    return err, 'correlation_amari'

#final evaluation code for evaluating outcomes
def evaluate(data: EMData, results: dict, sknown) -> dict:
    """
    Evaluates EM results against ground truth. Requies dictionary with Aest, Xest and Sest
    Takes data and results, returns error metrics.
    """
    #--------------------------------------
    #----- mixing matrix, A, error --------
    #--------------------------------------
    
    A_perm, permutation, _, scales, _ = ol_data.best_permutation_match_sign_flips(
        A=data.A, B=results['A_est']
    )
    
    #column norms of the true matrix
    col_norms = np.linalg.norm(data.A, axis=0)
    col_norms = np.maximum(col_norms, 1e-10)
    
    #scaled Frobenius norm error with scaled entries
    A_f_err = np.linalg.norm(data.A / col_norms - A_perm / col_norms, 'fro') / np.sqrt(data.A.shape[1])
    
    #recovering different matrix scoring metric
    A_err_angles = matrix_angle_error(A_true=data.A, A_est=results['A_est'])
    
    if np.all(A_err_angles['permutation'] != permutation):
        print('Two different permutations for validating A has been used')
        print(permutation)
        print(A_err_angles['permutation'])
    
    #--------------------------------------
    #-----    sources, S, error    --------
    #--------------------------------------
    
    #S permute according to the matrix permutation
    S_perm = results['S_est'][:, permutation] 
    
    # S permute and scale sources consistently with A permutation
    S_perm_scale = ol_data.scale_permute_src(
        srcest=results['S_est'], scaling=scales, permutation=permutation)
    
    #metrics
    S_err_MCC = MCC_remove_known(                 #permutation mean correlation coefficient
        true=data.S, estimate=S_perm, sknown=sknown
    )[0]
    
    S_err_MSE = mse_weighted_remove_known(        #mean squared error
        true=data.S, estimate=S_perm_scale, sknown=sknown
    )[0]

    #--------------------------------------
    #-----   observations, X, error   -----
    #--------------------------------------
    
    X_err = mse(true=data.X, estimate=results['X_est'])[0]

    return {
        'A_err': A_f_err, 'A_err_angle_mean': A_err_angles['mean_error'], 'A_err_angle_max': A_err_angles['max_error'],
        'S_err_MSE': S_err_MSE, 'S_err_MCC':S_err_MCC, 
        'X_err': X_err, 
        }
    

def calculate_iqr(x):
    """
    Custom function to calculate the Interquartile Range (IQR).
    """
    return x.quantile(0.75) - x.quantile(0.25)
    
#--- legacy code

#def frobenius_err_remove_known(true, estimate, aknown = None):
#    estimate_perm, col_ind, signs, scales, _ = ol_data.best_permutation_match_sign_flips(A = true, B= estimate) #using hungarian algorithm for matching columns
#    
#    if not aknown is None:
#        #removing the known columns
#        # Get the column indices that are NOT known (i.e. where aknown is nan)
#        col_indices = np.where(np.all(np.isnan(aknown), axis=0))[0] 
#        true = true[:, col_indices]  # (nsamples, n_unknown)
#        estimate_perm = estimate_perm[:,col_indices]
#        
#        err = np.linalg.norm(true - estimate_perm) 
#    n = np.maximum(estimate_perm.shape[1], 1)
#    err = err/np.sqrt(n) #error per column
#    #print(f'Frobenius difference is ',err)
#    return err, 'Frobenius Calibrated'
#
#import numpy as np
