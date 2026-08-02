import numpy as np


#---- SDP FUNCTIONS
def extract_largest_eigenvector(D):
    ''' Extract the largest eigenvector and eigenvalue from a symmetric matrix D. '''
    D = (D + D.T) / 2  # Ensure D is symmetric
    eigenvalues, eigenvectors = np.linalg.eig(D)
    max_index = np.argmax(np.abs(eigenvalues))
    u = np.real(eigenvectors[:, max_index])
    e = np.abs(eigenvalues[max_index])
    return u, e


def OverICA(X, nsrc):
    '''
    Perform overcomplete ICA using a semidefinite programming approach based on the method described in "Overcomplete Independent Component Analysis via Semidefinite Programming" by Podosinnokova et al. (2019).
    X: The input data matrix.                                   (nsample, nobs)
    nsrc (int): The number of source signals to separate.       
    '''
    
    
    
    
