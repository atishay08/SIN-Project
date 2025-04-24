import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

def getCommunityScores(nx_graph, assoc_gene_vector):
    """
    Calculate community-based scores for the network
    
    Parameters:
    -----------
    nx_graph : networkx.Graph
        The protein-protein interaction network
    assoc_gene_vector : numpy.ndarray
        Binary vector indicating disease-associated genes
    """
    # Get adjacency matrix using correct NetworkX function
    adj_matrix = nx.adjacency_matrix(nx_graph)
    
    # Convert to CSR format for efficient operations
    adj_matrix = adj_matrix.astype(np.float32)
    
    # Calculate basic community metrics
    community_props = {}
    
    # Number of disease proteins
    n_disease = np.sum(assoc_gene_vector > 0)
    community_props['disease_proteins'] = int(n_disease)
    
    # Calculate clustering coefficient for disease proteins
    disease_nodes = [node for i, node in enumerate(nx_graph.nodes()) if assoc_gene_vector[i] > 0]
    disease_subgraph = nx_graph.subgraph(disease_nodes)
    clustering = nx.average_clustering(disease_subgraph)
    community_props['disease_clustering'] = float(clustering)
    
    # Calculate connectivity metrics
    try:
        # Average degree of disease proteins
        disease_degrees = [nx_graph.degree(node) for node in disease_nodes]
        avg_degree = np.mean(disease_degrees) if disease_degrees else 0
        community_props['avg_disease_degree'] = float(avg_degree)
        
        # Largest connected component size
        connected_components = list(nx.connected_components(disease_subgraph))
        largest_cc_size = len(max(connected_components, key=len)) if connected_components else 0
        community_props['largest_component_size'] = int(largest_cc_size)
        
    except Exception as e:
        print(f"Warning: Error calculating some community metrics: {str(e)}")
        
    return community_props