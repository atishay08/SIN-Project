import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super().__init__()
        # First GCN layer
        self.conv1 = GCNConv(num_features, hidden_channels)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        # Output layer
        self.conv3 = GCNConv(hidden_channels//2, 1)
        # Dropout rate
        self.dropout = 0.5

    def forward(self, x, edge_index):
        # First layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def prepare_gcn_data(adj_matrix_sparse, assoc_gene_vector):
    """Prepare data for GCN input"""
    # Convert adjacency matrix to edge index
    edges = np.array(adj_matrix_sparse.nonzero())
    edge_index = torch.from_numpy(edges).long()
    
    # Create node features (using association vector as initial feature)
    x = torch.from_numpy(assoc_gene_vector).float().unsqueeze(1)
    
    # Add degree as additional feature
    degrees = np.array(adj_matrix_sparse.sum(axis=1)).flatten()
    degree_feature = torch.from_numpy(degrees).float().unsqueeze(1)
    x = torch.cat([x, degree_feature], dim=1)
    
    return x, edge_index

def train_gcn(model, x, edge_index, y, optimizer, num_epochs=200):
    """Train GCN model"""
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        out = model(x, edge_index)
        # Calculate loss
        loss = F.binary_cross_entropy(out.squeeze(), y)
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}')

def gcn_scores(adj_matrix_sparse, assoc_gene_vector, hidden_channels=64, num_epochs=200, learning_rate=0.01):
    """Main function to get GCN predictions"""
    try:
        # Prepare data
        x, edge_index = prepare_gcn_data(adj_matrix_sparse, assoc_gene_vector)
        y = torch.from_numpy(assoc_gene_vector).float()
        
        # Initialize model
        model = GCN(num_features=x.size(1), hidden_channels=hidden_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        print("Training GCN model...")
        train_gcn(model, x, edge_index, y, optimizer, num_epochs)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            scores = model(x, edge_index).squeeze().numpy()
        
        return scores
    
    except Exception as e:
        print(f"Error in GCN prediction: {str(e)}")
        raise

def test_gcn():
    """Test function for GCN implementation"""
    # Create small test network
    G = nx.erdos_renyi_graph(100, 0.1)
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
    assoc_vector = np.zeros(100)
    assoc_vector[:10] = 1  # Mark first 10 nodes as associated
    
    # Run GCN
    scores = gcn_scores(adj_matrix, assoc_vector)
    print("Test successful!")
    return scores

if __name__ == "__main__":
    test_gcn()