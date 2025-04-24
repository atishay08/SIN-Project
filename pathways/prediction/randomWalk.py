import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve

def random_walk_scores(adjacency_matrix, assoc_gene_vector, ratio=0.85, max_iter=100, tol=1e-6):
    """
    Compute random walk scores using sparse matrix operations
    
    Parameters:
    -----------
    adjacency_matrix : scipy.sparse matrix
        Sparse adjacency matrix of the network
    assoc_gene_vector : numpy.ndarray
        Binary vector indicating disease-associated genes
    ratio : float
        Restart probability (default: 0.85)
    max_iter : int
        Maximum number of iterations (default: 100)
    tol : float
        Convergence tolerance (default: 1e-6)
    """
    # Ensure input is sparse
    if not issparse(adjacency_matrix):
        adjacency_matrix = csr_matrix(adjacency_matrix)
    
    # Initialize vectors
    n = adjacency_matrix.shape[0]
    p0 = assoc_gene_vector / np.sum(assoc_gene_vector)
    old_vector = p0.copy()
    
    # Iterative computation with sparse operations
    for _ in range(max_iter):
        # Sparse matrix-vector multiplication
        new_vector = (1-ratio) * adjacency_matrix.dot(old_vector) + ratio * p0
        
        # Check convergence using sparse operations
        if np.linalg.norm(new_vector - old_vector, ord=1) < tol:
            break
            
        old_vector = new_vector
    
    return new_vector