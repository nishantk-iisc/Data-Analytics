#!/usr/bin/env python
# coding: utf-8
import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Facebook Data Questions
def import_facebook_data(file_path):
    # Create an empty set to store unique edges (i, j)
    unique_edges = set()
    
    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into two nodes (i and j)
            i, j = map(int, line.strip().split())
            
            # Ensure that the edge (i, j) is unique by using a set
            unique_edges.add((i, j))
    
    # Convert the set of unique edges to a numpy array
    nodes_connectivity_list_fb = np.array(list(unique_edges))

    return nodes_connectivity_list_fb

def spectralDecomp_OneIter(edges):
    # Now creating Graph 
    G = nx.Graph()
    G.add_edges_from(edges)

    # Compute the Laplacian matrix of above graph
    laplacian_matrix = nx.laplacian_matrix(G)

    # Perform spectral decomposition to calculate the eigenvalue and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix.toarray())

    # Sort eigenvalues and eigenvectors 
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Extract the Fiedler vector (for second smallest eigenvector)
    fielder_vec = eigenvectors[:, 1]

    # Determine graph partition using Fiedler vector
    graph_partition = []
    for node_id, value in enumerate(fielder_vec):
        community_id = int(value >= 0)  # Assign nodes to communities based on Fiedler vector
        graph_partition.append([node_id, community_id])

    # Create adjacency matrix of the graph G the is computer by using facebook dataset
    adj_mat = nx.to_numpy_array(G)

    return fielder_vec, adj_mat, np.array(graph_partition)

def spectralDecomposition(nodes_connectivity_list_fb):
    # Initialize the graph partition with all nodes in one community
    graph_partition = np.column_stack((np.arange(len(nodes_connectivity_list)), np.zeros(len(nodes_connectivity_list), dtype=int)))

    # Perform spectral decomposition iteratively for a certain number of iterations
    num_iterations = 7  # Adjust the number of iterations as needed
    for _ in range(num_iterations):
        # Call spectralDecomp_OneIter to update the graph_partition
        _, _, graph_partition = spectralDecomp_OneIter(nodes_connectivity_list)

    return graph_partition

def createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb):
    # Create a graph from the edges
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list)

    # Sort nodes by community (based on graph_partition)
    num_communities = len(np.unique(graph_partition[:, 1]))
    sorted_nodes = [node for node, _ in sorted(graph_partition, key=lambda x: x[1])]

    # Create a sorted adjacency matrix
    adj_mat = nx.to_numpy_array(G, nodelist=sorted_nodes)

    return adj_mat

def louvain_one_iter(nodes_connectivity_list_fb):
    # Create a graph from the edges
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_fb)

    # Initialize each node as its own community
    communities = list(G.nodes())

    # Create a dictionary to store the degrees of nodes for faster modularity computation
    degrees = dict(G.degree())

    # Flag to check if any community has been merged
    community_merged = True

    while community_merged:
        community_merged = False

        # Randomly shuffle nodes for fairness in merging order
        shuffled_nodes = list(G.nodes())
        np.random.shuffle(shuffled_nodes)

        for node in shuffled_nodes:
            current_community = communities[node]

            # Compute the modularity gain by moving the node to its neighbors' communities
            max_modularity_gain = 0
            best_community = current_community

            for neighbor in G.neighbors(node):
                neighbor_community = communities[neighbor]

                # Calculate the modularity gain
                delta_modularity = (
                    2 * G.get_edge_data(node, neighbor).get("weight", 1) - degrees[node] * degrees[neighbor] / (2 * len(nodes_connectivity_list_fb))
                )

                if delta_modularity > max_modularity_gain:
                    max_modularity_gain = delta_modularity
                    best_community = neighbor_community

            # Move the node to the community with the highest modularity gain
            if best_community != current_community:
                communities[node] = best_community
                community_merged = True

    # Create a dictionary to map nodes to their communities
    community_mapping = {node: comm for node, comm in enumerate(communities)}

    # Create the graph partition vector as a numpy array
    graph_partition = np.array([[node, community_mapping[node]] for node in G.nodes()])

    return graph_partition

# Bitcoin Data Questions
def import_bitcoin_data(file_path_btc):

    pass

def spectralDecomp_OneIter(nodes_connectivity_list_btc):

    pass

def spectralDecomposition(nodes_connectivity_list_btc):

    pass

def createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc):

    pass

def louvain_one_iter(nodes_connectivity_list_btc):

    pass
if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    # fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)
    # Plot the sorted Fiedler vector
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(fielder_vec_fb)
    plt.title('Fielder Vector')

    # Plot the adjacency matrix
    plt.subplot(122)
    plt.imshow(adj_mat_fb, cmap='viridis', origin='upper', interpolation='none')
    plt.title('Adjacency Matrix')

    # Plot the graph partition
    plt.figure(2)
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_fb)
    community_id = graph_partition_fb[:, 1]
    pos = nx.spring_layout(G)  # You can use different layouts
    nx.draw(G, pos, node_color=community_id, cmap=plt.cm.Paired, with_labels=True)
    plt.title("Graph based on Adjacency Matrix")
    plt.show()

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)
    # Visualize the sorted adjacency matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(clustered_adj_mat_fb, cmap='viridis', origin='upper', interpolation='none')
    plt.title("Sorted Adjacency Matrix")
    plt.colorbar()
    plt.show()
    
    # Create a graph from the sorted adjacency matrix
    G_clustered = nx.Graph(clustered_adj_mat_fb)
    # Visualize the graph
    pos = nx.spring_layout(G_clustered)
    colors = [graph_partition_fb[node_id][1] for node_id in range(len(graph_partition_fb))]
    nx.draw(G_clustered, pos, node_color=colors, cmap=plt.get_cmap('coolwarm'), with_labels=True)
    plt.title("Graph based on Sorted Adjacency Matrix")
    plt.show()

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    # Create a graph with the resulting communities
    G_louvain = nx.Graph()
    G_louvain.add_edges_from(nodes_connectivity_list_fb)

    # Visualize the graph with colored communities
    pos = nx.spring_layout(G_louvain)
    colors = [graph_partition_louvain_fb[node_id][1] for node_id in range(len(graph_partition_louvain_fb))]
    nx.draw(G_louvain, pos, node_color=colors, cmap=plt.get_cmap('coolwarm'), with_labels=True)
    plt.title("Louvain Community Detection (One Iteration)")
    plt.show()


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)

