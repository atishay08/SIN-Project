import numpy as np
from scipy.sparse import issparse, csr_matrix

def neighborhood_scores(adjacency_matrix, assoc_gene_vector, dtype=np.float32):
    """
    Compute neighborhood-based scores using sparse matrix operations
    
    Parameters:
    -----------
    adjacency_matrix : scipy.sparse.csr_matrix
        Sparse adjacency matrix of the network
    assoc_gene_vector : numpy.ndarray
        Binary vector indicating disease-associated genes
    dtype : numpy.dtype
        Data type for computations (default: np.float32)
    """
    # Ensure input is sparse and correct dtype
    if not issparse(adjacency_matrix):
        adjacency_matrix = csr_matrix(adjacency_matrix, dtype=dtype)
    else:
        adjacency_matrix = adjacency_matrix.astype(dtype)
    
    # Convert association vector to correct dtype
    assoc_gene_vector = assoc_gene_vector.astype(dtype)
    
    try:
        # Compute neighborhood scores using sparse matrix multiplication
        scores = adjacency_matrix.dot(assoc_gene_vector)
        
        # Normalize scores if needed (optional)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
            
        return scores
        
    except Exception as e:
        print(f"Error in neighborhood calculation: {str(e)}")
        print(f"Matrix shape: {adjacency_matrix.shape}")
        print(f"Vector shape: {assoc_gene_vector.shape}")
        raise