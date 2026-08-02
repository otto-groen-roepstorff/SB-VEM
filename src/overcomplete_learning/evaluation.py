import numpy as np    
#from scikit-learn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
import warnings
import pandas as pd
import overcomplete_learning.data  as ol_data
import overcomplete_learning.metrics  as ol_metrics
'''
def extract_nonzero_padded(coef_list, k):
    """Extract non-zero coefficients, padded to length k."""
    result = []
    for x in coef_list:
        nz = sorted(x[x != 0])
        # Pad with zeros if fewer than k non-zero elements returned
        padded = nz + [0.0] * (k - len(nz))
        result.append(padded[:k])  # truncate if somehow longer
    return np.array(result)

def fit_OMP_to_sample(A, X, n_nsrc=None):
    if n_nsrc is None:
        omp = OrthogonalMatchingPursuitCV()
    else:
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nsrc)
    nreps = X.shape[0]
    shat = [omp.fit(X = A, y = X[i,]).coef_ for i in range(nreps)]
    return shat

#matrix handling and evaluation    
def evaluate_matrix(A, data, nonsparsity_factors= None, eval_all=False, known_n_src= False):
    k0 = data['nonzero']
    nreps, nsrc = data['s_all'].shape
    nobs = data['x_all'].shape[1]
    
    if nonsparsity_factors is None:
        nonsparsity_factors = np.array([1, 2, 3])
    else:
        nonsparsity_factors = np.asarray(nonsparsity_factors)
    
    
    nonsparsity_factors = nonsparsity_factors[nonsparsity_factors < nsrc]
    #fit on full data set
    outdictionary = {}
    
    if len(nonsparsity_factors) == 0:
        print('Warning: all nonsparsity_factors exceed nsrc. Nothing to evaluate.')
        return outdictionary
    
    if eval_all:
        omp_all = OrthogonalMatchingPursuitCV()
        shat_all = extract_nonzero_padded(
            [omp_all.fit(X = A, y = data['x_all'][i,]).coef_ for i in range(nreps)], k = nsrc) 
        #shat_all = np.array([sorted(x[x!=0]) for x in shathall ])
        true_all = extract_nonzero_padded(np.array([sorted(x[x!=0]) for x in data['s_all']]), k=nsrc)
        outdictionary['nonsparse'] = np.mean((shat_all-true_all)**2)
        print(f'nonsparse data evaluation:',outdictionary['nonsparse'])
    
    if nonsparsity_factors is None:
        nonsparsity_factors = [1,2,3]
    
    for factor in nonsparsity_factors:
        #finding the number of sources we search for
        ki = int(factor)
        x_key  = f'x_{factor}.nonsparse'
        s_key  = f's_{factor}.nonsparse'
        
        if x_key not in data:
            print(f'Warning: key {x_key} not found in data, skipping.')
            continue
        if known_n_src:
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=ki)
        else:
            omp = OrthogonalMatchingPursuitCV()
        #fitting sparse fit and plotting results
        #fitting orthogonal matching pursuit in sparse setting
        
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            shat = extract_nonzero_padded(
                [omp.fit(X=A, y=data[x_key][i]).coef_ for i in range(nreps)],
                k=ki)

        if len(w) > 0:
            print(f'Warning: OMP ended prematurely for factor {factor}x '
            f'({len(w)}/{nreps} samples affected)')
        
        trueS = extract_nonzero_padded(data[s_key], k=ki)
        
        #making the error function easy to change
        err_fun  = ol_metrics.mse_weighted
        err, err_name  = err_fun(true = trueS, estimate=shat)
        
        outdictionary[f'{factor}_nonsparse'] = err
        outdictionary[f'{factor}_nonsparse_k'] = ki
        outdictionary[f'{factor}_nonsparse_warnings'] = len(w)
        
        print(f'{factor}_nonsparse {err_name}: {err}, nonzero_elements: {ki}')
    #storing the initial information
    outdictionary['nsrc'] = nsrc
    outdictionary['nobs'] = nobs
    outdictionary['coherence'] = ol_data.coherence(A)
    outdictionary['recovery_guarantee'] = ol_data.maximal_non_sparsity(A)
    outdictionary['known_n_src'] = known_n_src
    return outdictionary



def dict_to_df(result_dictionary):
    """Convert results to  tidy DataFrame."""
    
    nsrc = result_dictionary['nsrc']
    nobs = result_dictionary['nobs']
    k0 = result_dictionary['recovery_guarantee']
    rows = []
    for key, val in result_dictionary.items():
        if key.endswith('_nonsparse') and not key.endswith('_k'):
            factor = key.replace('_nonsparse', '')
            k_key  = f'{factor}_nonsparse_k'
            w_key  = f'{factor}_nonsparse_warnings'
            rows.append({
                'mse':          float(val),
                'k_times_limit':  int(factor),
                'n_nonzeroes':            int(result_dictionary[k_key]),
                'w':            int(result_dictionary[w_key]),
                'nsrc':         int(nsrc),
                'nobs':         int(nobs),
                'k0' :          int(k0),
                'known_n_src':  result_dictionary['known_n_src']
            })
    return pd.DataFrame(rows)
    
'''