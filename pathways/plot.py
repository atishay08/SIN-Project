import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_metrics(metrics, save_path=None):
    """Plot evaluation metrics for all algorithms"""
    algorithms = list(metrics.keys())
    metric_types = list(metrics[algorithms[0]].keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16)
    
    # Plot Recall@k values
    recall_metrics = [m for m in metric_types if m.startswith('Recall')]
    k_values = [int(m.split('@')[1]) for m in recall_metrics]
    recall_values = {algo: [metrics[algo][m] for m in recall_metrics] 
                    for algo in algorithms}
    
    axes[0,0].plot(k_values, [recall_values[algo] for algo in algorithms], marker='o')
    axes[0,0].set_xlabel('k')
    axes[0,0].set_ylabel('Recall@k')
    axes[0,0].legend(algorithms)
    axes[0,0].grid(True)
    
    # Plot MRR comparison
    mrr_values = [metrics[algo]['MRR'] for algo in algorithms]
    axes[0,1].bar(algorithms, mrr_values)
    axes[0,1].set_ylabel('Mean Reciprocal Rank')
    axes[0,1].set_title('MRR Comparison')
    for i, v in enumerate(mrr_values):
        axes[0,1].text(i, v, f'{v:.3f}', ha='center')
    
    # Plot AP comparison
    ap_values = [metrics[algo]['AP'] for algo in algorithms]
    axes[1,0].bar(algorithms, ap_values)
    axes[1,0].set_ylabel('Average Precision')
    axes[1,0].set_title('AP Comparison')
    for i, v in enumerate(ap_values):
        axes[1,0].text(i, v, f'{v:.3f}', ha='center')
    
    # Keep the last subplot empty for potential future metrics
    axes[1,1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_community_structure(community_props, save_path=None):
    """Plot community structure properties"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract numeric properties
    numeric_props = {k: v for k, v in community_props.items() 
                    if isinstance(v, (int, float))}
    
    # Create bar plot
    properties = list(numeric_props.keys())
    values = list(numeric_props.values())
    
    ax.bar(properties, values)
    ax.set_title('Community Structure Properties')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()