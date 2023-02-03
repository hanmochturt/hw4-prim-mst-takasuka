import pytest
import numpy as np

import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from mst import Graph
from sklearn.metrics import pairwise_distances

h = Graph(r"C:\Users\hanmo\Documents\GitHub\hw4-prim-mst-takasuka\data\small.csv")
h.construct_mst()
print(h.mst)

file_path = './data/slingshot_example.txt'
coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
g = Graph(dist_mat)
g.construct_mst()
#check_mst(g.adj_mat, g.mst, 57.263561605571695)

