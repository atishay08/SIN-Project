import matplotlib.pyplot as plt
import numpy as np

# Data
algorithms = ['Neighborhood', 'DIAMOnD', 'Random Walk', 'Node2Vec', 'GCN']
metrics = {
    'MRR': [0.0081, 0.0274, 0.0275, 0.0002, 0.0198],
    'AP': [0.0725, 0.9954, 1.0000, 0.0118, 0.8976]
}

# Create figure
plt.figure(figsize=(10, 6))
colors = ['#2196F3', '#4CAF50', '#FFC107', '#E91E63', '#9C27B0']
markers = ['o', 's', '^', 'D', 'v']

# Plot MRR and AP
metric_names = ['MRR', 'AP']
x = np.arange(len(metric_names))
for i, algo in enumerate(algorithms):
    values = [metrics['MRR'][i], metrics['AP'][i]]
    plt.plot(x, values, '-',
             color=colors[i],
             marker=markers[i], 
             label=algo, 
             linewidth=2, 
             markersize=8)

plt.xticks(x, metric_names)
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.title('MRR and AP Performance Comparison', fontsize=14, pad=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add value labels
for line in plt.gca().get_lines():
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    for x, y in zip(x_data, y_data):
        plt.annotate(f'{y:.4f}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8)

plt.tight_layout()
plt.show()