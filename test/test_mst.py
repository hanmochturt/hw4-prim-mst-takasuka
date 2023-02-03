import pytest
import numpy as np

import random
import sys
import pathlib
from typing import Union

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: Union[int, float],
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    """A minimum spanning tree with n nodes that are connected should have n-1 edges. Each edge is 
    expressed twice, so there should be 2(n-1) nonzero entries in the matrix"""
    assert np.count_nonzero(mst) == 2 * (np.count_nonzero(np.count_nonzero(mst, axis=0)) - 1)

    assert np.array_equal(mst.shape, adj_mat.shape)

    # adjacency matrix and MST should be symmetric
    assert np.allclose(mst, mst.transpose())
    assert np.allclose(adj_mat, adj_mat.transpose())


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    Create a unique minimum spanning tree with a known weight, and compare this to the created
    algorithm
    """
    num_nodes = 9
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    minimum_path_weights = list(range(1, num_nodes))
    random.shuffle(minimum_path_weights)
    visited = {0}

    # create minimum spanning tree
    for weight in minimum_path_weights:
        not_visited = set(range(num_nodes)) - visited
        node_1 = random.choice(list(visited))
        node_2 = random.choice(list(not_visited))
        visited.add(node_2)
        adjacency_matrix[node_1, node_2] = weight
        adjacency_matrix[node_2, node_1] = weight

    # add other higher weight nodes that will not constitute the MST
    max_mst_weight = max(minimum_path_weights)
    weight = max_mst_weight + 1
    for _ in range(random.randint(1, sum(range(num_nodes-1))-num_nodes+1)):
        potential_start_nodes, potential_end_nodes = \
            np.where(adjacency_matrix == 0)
        index_of_pair = random.choice(range(potential_start_nodes.size))
        node_1 = potential_start_nodes[index_of_pair]
        node_2 = potential_end_nodes[index_of_pair]
        while node_1 == node_2:
            index_of_pair = random.choice(range(potential_start_nodes.size))
            node_1 = potential_start_nodes[index_of_pair]
            node_2 = potential_end_nodes[index_of_pair]
        adjacency_matrix[node_1, node_2] = weight
        adjacency_matrix[node_2, node_1] = weight

    g = Graph(adjacency_matrix)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, sum(minimum_path_weights))

