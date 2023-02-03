import numpy as np
import heapq
from typing import Union


class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = np.zeros(self.adj_mat.shape)  # start with an unconnected graph

        current_node = 0
        all_nodes = set(range(self.adj_mat.shape[0]))
        visited_nodes = {current_node}
        outgoing_edges_weights = list(self.adj_mat[current_node, :])
        heapq.heapify(outgoing_edges_weights)
        while visited_nodes != all_nodes:
            while outgoing_edges_weights[0] == 0:
                heapq.heappop(outgoing_edges_weights)

            # find nodes of edges that have the lowest weight
            lowest_edge_indices_start, lowest_edge_indices_end = \
                np.where(self.adj_mat == outgoing_edges_weights[0])

            # find an edge (2 nodes) of the lowest weight that branches out the existing tree
            # from an existing node to a new node
            for i, potential_start_node in enumerate(lowest_edge_indices_start):
                start_node = potential_start_node
                end_node = lowest_edge_indices_end[i]
                if start_node in visited_nodes and end_node not in visited_nodes:
                    break
                start_node = -1  # edges of the lowest weight are already fully visited
            heapq.heappop(outgoing_edges_weights)
            if start_node == -1:
                continue
            self.mst[start_node, end_node] = self.adj_mat[start_node, end_node]
            self.mst[end_node, start_node] = self.adj_mat[end_node, start_node]
            visited_nodes.add(end_node)

            # add weights of edges from the recently-added node to the heap of possible next edge
            # weights
            for node, weight in enumerate(self.adj_mat[end_node, :]):
                if node not in visited_nodes:
                    heapq.heappush(outgoing_edges_weights, weight)
