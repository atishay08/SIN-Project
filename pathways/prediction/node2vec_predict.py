import numpy as np
from sklearn.linear_model import LogisticRegression
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def simplified_random_walks(G, nodes, num_walks=3, walk_length=30):
    """Generate simplified random walks with proper node mapping"""
    walks = []
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    print(f"Generating {num_walks} walks of length {walk_length}")
    for _ in tqdm(range(num_walks), desc="Walk iteration"):
        for node in tqdm(nodes, desc="Processing nodes", leave=False):
            walk = [node_to_idx[node]]
            curr_node = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(curr_node))
                if not neighbors:
                    break
                next_node = np.random.choice(neighbors)
                walk.append(node_to_idx[next_node])
                curr_node = next_node
            walks.append(walk)
    
    return walks, node_to_idx

def create_node_features(G, walks, nodes, dim=32):
    """Create node features from walks"""
    print("Creating node features...")
    n_nodes = len(nodes)
    features = np.zeros((n_nodes, dim))
    
    # Add degree feature
    for i, node in enumerate(nodes):
        features[i, 0] = G.degree(node)
    
    # Add walk-based features
    for walk in tqdm(walks, desc="Processing walks"):
        for i in range(len(walk)-1):
            src_idx = walk[i]
            dst_idx = walk[i+1]
            features[src_idx, 1:] += 1
            features[dst_idx, 1:] += 1
    
    # Normalize features
    row_sums = features.sum(axis=1)
    row_sums[row_sums == 0] = 1
    features = features / row_sums[:, np.newaxis]
    
    return features

def node2vec_scores(adj_matrix_sparse, assoc_gene_vector, nodes):
    """Node2Vec-based prediction with improved node handling"""
    try:
        print("Converting to NetworkX graph...")
        G = nx.from_scipy_sparse_array(adj_matrix_sparse)
        nx.relabel_nodes(G, {i: str(nodes[i]) for i in range(len(nodes))}, copy=False)
        
        # Generate walks with proper node mapping
        walks, node_mapping = simplified_random_walks(G, nodes)
        
        # Create node features
        features = create_node_features(G, walks, nodes)
        
        # Train classifier
        print("Training classifier...")
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(features, assoc_gene_vector)
        
        return clf.predict_proba(features)[:, 1]
    
    except Exception as e:
        print(f"Error in Node2Vec implementation: {str(e)}")
        print("Debug info:")
        print(f"Number of nodes: {len(nodes)}")
        print(f"Matrix shape: {adj_matrix_sparse.shape}")
        print(f"Sample nodes: {list(nodes)[:5]}")
        raise