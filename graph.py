import numpy as np



def adjacency_to_incidence(adj_matrix):
    """
    Convert an adjacency matrix of a simple undirected graph to an incidence matrix.
    
    Parameters:
    - adj_matrix (2D array-like): An n x n adjacency matrix representing the graph.
    
    Returns:
    - incidence (numpy.ndarray): An n x m incidence matrix, where m is the number of edges.
    
    The function assumes that the graph is undirected and does not have multiple edges.
    It scans the upper triangle of the adjacency matrix (i < j) to extract each unique edge.
    """
    # Convert input to a NumPy array
    A = np.array(adj_matrix)
    n = A.shape[0]
    
    # List to hold edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] != 0:
                edges.append((i, j))

    # Incidence matrix
    incidence = np.zeros((n, len(edges)), dtype=int)
    for k, (i, j) in enumerate(edges):
        incidence[i, k] = 1
        incidence[j, k] = 1
        
    return incidence


def diagonal_edge_weights(adj_matrix):
    """
    Convert a weighted adjacency matrix of a simple undirected graph to a diagonal matrix of edge weights.
    
    Parameters:
    - adj_matrix (2D array-like): An n x n weighted adjacency matrix representing the graph.
      It is assumed that the graph is undirected (so the matrix is symmetric) and that edge weights are stored in the nonzero entries.
    
    Returns:
    - weight_diag (numpy.ndarray): An m x m diagonal matrix, where m is the number of edges.
      Each diagonal entry corresponds to the weight of an edge extracted from the upper triangle of the adjacency matrix.
    """
    # Convert input to a NumPy array
    A = np.array(adj_matrix)
    n = A.shape[0]
    
    # List to hold the weights of edges (from the upper triangle to avoid duplicates)
    edge_weights = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                edge_weights.append(A[i, j])
    
    # Create a diagonal matrix from the list of edge weights
    weight_diag = np.diag(edge_weights)
    return weight_diag


def weighted_laplacian_matrix(incidence_matrix, edge_weights):
    """
    Compute the Laplacian matrix of a weighted undirected graph.
    
    Parameters:
    - adj_matrix (2D array-like): An n x n adjacency matrix representing the graph.
    
    Returns:
    - laplacian (numpy.ndarray): An n x n Laplacian matrix.
    """
    return incidence_matrix @ edge_weights @ incidence_matrix.T



if __name__ == '__main__':
    # Define an adjacency matrix for a simple undirected graph
    adj_matrix = [
        [0, 1, 0, 4],
        [1, 0, 2, 3],
        [0, 2, 0, 0],
        [4, 3, 0, 0]
    ]
    
    incidence_matrix = adjacency_to_incidence(adj_matrix)
    print("Incidence Matrix:")
    print(incidence_matrix)

    weight_diag = diagonal_edge_weights(adj_matrix)
    print("\nWeight Diagonal Matrix:")
    print(weight_diag)

    laplacian = weighted_laplacian_matrix(incidence_matrix, weight_diag)
    print("\nWeighted Laplacian Matrix:")
    print(laplacian)
