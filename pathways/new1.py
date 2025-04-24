import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import sys
import os
import gc
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# Add local paths
sys.path.append('pathways/prediction')
sys.path.append('pathways/characterization')

# Import prediction algorithms
from diamond import diamond_scores
from neighborhood import neighborhood_scores
from randomWalk import random_walk_scores
from calculate_community_scores import getCommunityScores
from node2vec_predict import node2vec_scores

def clear_memory():
    """Force garbage collection"""
    gc.collect()

def format_protein_id(protein_id):
    """Standardize protein ID format"""
    return str(protein_id).strip().lstrip('0')

def get_top_k(scores, nodes, k=10):
    """Get top K predictions with scores"""
    top_idx = scores.argsort()[::-1][:k]
    return [(nodes[i], scores[i]) for i in top_idx]

def batch_format_protein_ids(df, column, chunk_size=10000):
    """Process protein IDs in batches"""
    for i in range(0, len(df), chunk_size):
        df.loc[i:i+chunk_size-1, column] = df.loc[i:i+chunk_size-1, column].apply(format_protein_id)
    return df

def print_predictions(algo_name, predictions):
    """Print formatted predictions"""
    print(f"\n{algo_name} Predictions:")
    for node, score in predictions:
        print(f"{node}: {score:.4f}")

def calculate_metrics(predictions, true_indices, k_values=[10, 20, 50, 100]):
    """Calculate evaluation metrics"""
    metrics = {}
    n_true = len(true_indices)
    sorted_indices = predictions.argsort()[::-1]
    
    # Calculate Recall@k
    for k in k_values:
        top_k = set(sorted_indices[:k])
        recall = len(top_k.intersection(true_indices)) / n_true
        metrics[f'Recall@{k}'] = recall
    
    # Calculate MRR
    mrr = 0
    for true_idx in true_indices:
        rank = np.where(sorted_indices == true_idx)[0][0] + 1
        mrr += 1/rank
    metrics['MRR'] = mrr / n_true
    
    # Calculate AP
    binary_relevance = np.zeros_like(predictions)
    binary_relevance[list(true_indices)] = 1
    metrics['AP'] = average_precision_score(binary_relevance, predictions)
    
    return metrics

def main():
    try:
        # Load datasets
        print("Loading datasets...")
        ppi_df = pd.read_csv("./Dataset/bio-pathways-network (1)/bio-pathways-network.csv",
                            usecols=["Gene ID 1", "Gene ID 2"],
                            dtype=str)
        assoc_df = pd.read_csv("./Dataset/bio-pathways-associations (1)/bio-pathways-associations.csv")

        # Process network
        batch_format_protein_ids(ppi_df, "Gene ID 1")
        batch_format_protein_ids(ppi_df, "Gene ID 2")
        clear_memory()

        G = nx.from_pandas_edgelist(ppi_df, source="Gene ID 1", target="Gene ID 2")
        nodes = list(G.nodes)
        del ppi_df
        clear_memory()

        print(f"\nNetwork information:")
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(G.edges)}")

        # Process disease data
        disease_name = "Liver carcinoma"
        print(f"\nAnalyzing disease: {disease_name}")

        disease_data = assoc_df[assoc_df["Disease Name"] == disease_name]
        if len(disease_data) == 0:
            print("\nAvailable diseases:")
            print(assoc_df["Disease Name"].unique())
            raise ValueError(f"Disease '{disease_name}' not found")

        disease_proteins = disease_data["Associated Gene IDs"].iloc[0]
        if pd.isna(disease_proteins):
            raise ValueError(f"No proteins found for disease: {disease_name}")

        assoc_proteins_set = set(format_protein_id(p) for p in disease_proteins.split(",") if p.strip())
        print(f"Found {len(assoc_proteins_set)} associated proteins")

        # Prepare matrices
        adj_matrix_sparse = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float32)
        assoc_gene_vector = np.array([format_protein_id(node) in assoc_proteins_set 
                                    for node in nodes], dtype=np.float32)

        row_sums = np.array(adj_matrix_sparse.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1e-8
        norm_adj_matrix_sparse = adj_matrix_sparse.multiply(1 / row_sums[:, None])
        clear_memory()

        # Run predictions
        algorithms = [
            ("Random Walk", lambda m, v: random_walk_scores(norm_adj_matrix_sparse, v)),
            ("Neighborhood", lambda m, v: neighborhood_scores(adj_matrix_sparse, v)),
            ("Node2Vec", lambda m, v: node2vec_scores(m, v, nodes))
        ]

        results = {}
        for algo_name, algo_func in algorithms:
            print(f"\nStarting {algo_name} analysis...")
            try:
                results[algo_name] = algo_func(adj_matrix_sparse, assoc_gene_vector)
                print(f"{algo_name} complete")
            except Exception as e:
                print(f"Error in {algo_name}: {str(e)}")
            finally:
                clear_memory()

        # === Evaluation Metrics ===
        print("\n=== Evaluation Metrics ===")
        true_indices = set(i for i, node in enumerate(nodes) if format_protein_id(node) in assoc_proteins_set)
        metrics = {}
        for algo_name in results:
            metrics[algo_name] = calculate_metrics(results[algo_name], true_indices)
            print(f"\n{algo_name} Metrics:")
            for metric, value in metrics[algo_name].items():
                print(f"{metric}: {value:.4f}")

        # Print results
        print(f"\n=== Top 10 Predicted Proteins for '{disease_name}' ===")
        for algo_name in results:
            predictions = get_top_k(results[algo_name], nodes)
            print_predictions(algo_name, predictions)

        print("\n=== Community Structure Properties ===")
        for key, val in community_props.items():
            if isinstance(val, float):
                print(f"{key}: {val:.4f}")
            else:
                print(f"{key}: {val}")

        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = os.path.join(results_dir, f"{disease_name.replace(' ', '_')}_predictions.txt")
        with open(output_file, 'w') as f:
            f.write(f"Disease: {disease_name}\n")
            f.write(f"Network: {len(nodes)} nodes, {len(G.edges)} edges\n")
            f.write(f"Known disease proteins: {len(assoc_proteins_set)}\n\n")
            
            f.write("=== Evaluation Metrics ===\n")
            for algo_name, algo_metrics in metrics.items():
                f.write(f"\n{algo_name}:\n")
                for metric, value in algo_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\n=== Top 10 Predictions ===\n")
            for algo_name in results:
                f.write(f"\n{algo_name}:\n")
                predictions = get_top_k(results[algo_name], nodes)
                for node, score in predictions:
                    f.write(f"{node}: {score:.4f}\n")
            
            f.write("\n=== Community Structure Properties ===\n")
            for key, val in community_props.items():
                if isinstance(val, float):
                    f.write(f"{key}: {val:.4f}\n")
                else:
                    f.write(f"{key}: {val}\n")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("\nDebug Information:")
        print(f"Network: {len(nodes)} nodes, {len(G.edges)} edges")
        print(f"Matrix shape: {adj_matrix_sparse.shape}")
        print(f"Non-zero elements: {adj_matrix_sparse.nnz}")
        raise

if __name__ == "__main__":
    main()