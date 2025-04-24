import numpy as np
from scipy.sparse import issparse, csr_matrix

def diamond_scores(adjacency_matrix, assoc_gene_vector, n_iterations=200):
    """
    Compute DIAMOnD scores using sparse matrix operations
    
    Parameters:
    -----------
    adjacency_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the network
    assoc_gene_vector : numpy.ndarray
        Binary vector indicating disease-associated genes
    n_iterations : int
        Number of iterations to run (default: 200)
    """
    # Ensure input is sparse
    if not issparse(adjacency_matrix):
        adjacency_matrix = csr_matrix(adjacency_matrix)
    
    n_nodes = adjacency_matrix.shape[0]
    seed_genes = np.where(assoc_gene_vector > 0)[0]
    
    # Initialize scores
    scores = np.zeros(n_nodes, dtype=np.float32)
    scores[seed_genes] = 1
    
    # Keep track of candidates
    candidates = np.ones(n_nodes, dtype=bool)
    candidates[seed_genes] = False
    
    # Initialize connections to seed genes
    connections_to_seeds = np.zeros(n_nodes, dtype=np.float32)
    for seed in seed_genes:
        # Get column slice using proper indexing
        col = adjacency_matrix[:, [seed]].toarray().flatten()
        connections_to_seeds += col
    
    for iteration in range(n_iterations):
        if not np.any(candidates):
            break
            
        # Get candidate scores
        candidate_scores = np.zeros(n_nodes, dtype=np.float32)
        candidate_scores[candidates] = connections_to_seeds[candidates]
        
        # Find best candidate
        best_cand = np.argmax(candidate_scores)
        scores[best_cand] = 1.0 - (iteration / n_iterations)
        candidates[best_cand] = False
        
        # Update connections using proper column indexing
        col = adjacency_matrix[:, [best_cand]].toarray().flatten()
        connections_to_seeds += col
    
    return scores