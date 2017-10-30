"""
This script checks the computation of cluster vectors for three HCP-based
structures.
"""

from icetdev import clusterspace
from icetdev.clusterspace import create_clusterspace
from icetdev.structure import structure_from_atoms
from ase.build import bulk, make_supercell
import numpy as np


print(__doc__)

cutoffs = [8.0, 7.0]
subelements = ['Re', 'Ti']

prototype = bulk('Re')

print(prototype)

clusterspace = create_clusterspace(prototype, cutoffs, subelements)

conf = structure_from_atoms(prototype.copy())
print('Structure no. 1 (nat= {}):'.format(len(conf)))
cv = clusterspace.get_clustervector(conf)
cv_target = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                      -1.0, -1.0, -1.0])
print(cv)
assert np.all(np.abs(cv_target - cv) < 1e-6)

conf = make_supercell(prototype, [[2, 0, 1],
                                  [0, 1, 0],
                                  [0, 1, 2]])

conf[0].symbol = 'Ti'
conf[1].symbol = 'Ti'
print('Structure no. 2 (nat= {}):'.format(len(conf)))
cv = fj.get_cluster_vector(conf)
cv_target = np.array([1.0, -0.5, 0.3333333333333333,
                      0.3333333333333333, 0.0, 0.0, 0.3333333333333333,
                      0.0, 0.0,
                      0.3333333333333333, 0.3333333333333333,
                      0.3333333333333333, 0.0,
                      0.3333333333333333, 0.6666666666666666,
                      0.3333333333333333, 0.0, 0.0,
                      0.3333333333333333, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      0.16666666666666666, 0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, 0.16666666666666666,
                      -0.16666666666666666, -0.5, -0.16666666666666666,
                      -0.16666666666666666, -0.5, -0.16666666666666666,
                      0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, 0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      0.5, 0.16666666666666666,
                      0.16666666666666666, 0.16666666666666666,
                      -0.16666666666666666, 0.16666666666666666,
                      0.16666666666666666, 0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      0.16666666666666666, -0.16666666666666666, 0.5,
                      -0.16666666666666666, -0.5, -0.16666666666666666,
                      -0.16666666666666666, 0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666, 0.5,
                      0.5, 0.16666666666666666, -0.16666666666666666,
                      0.16666666666666666, -0.16666666666666666,
                      -0.16666666666666666, -0.5, -0.16666666666666666,
                      -0.16666666666666666, -0.16666666666666666])
print(cv)
assert np.all(np.abs(cv_target - cv) < 1e-6)

conf = make_supercell(prototype, [[1,  0, 1],
                                  [0,  1, 1],
                                  [0, -1, 3]])

conf[0].symbol = 'Ti'
conf[1].symbol = 'Ti'
conf[2].symbol = 'Ti'
print('Structure no. 3 (nat= {}):'.format(len(conf)))
cv = fj.get_cluster_vector(conf)
cv_target = np.array([1.0, -0.25, -0.5, 0.5, -0.5, 0.5, -0.5,
                      0.6666666666666666, 0.6666666666666666,
                      0.6666666666666666, -0.5, 0.6666666666666666, -0.5,
                      -0.5, 0.5, -0.5, 0.6666666666666666, -0.5,
                      -0.5, 0.25, 0.25, 0.08333333333333333, 0.25, 0.25,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.25,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333, -0.75,
                      -0.5833333333333334, -0.5833333333333334, 0.25, 0.25,
                      -0.5833333333333334, 0.25, 0.25, 0.25,
                      -0.4166666666666667, -0.4166666666666667,
                      -0.4166666666666667, 0.25, 0.25, 0.25,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333, 0.25,
                      -0.4166666666666667, 0.25, 0.25, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333,
                      -0.25, -0.4166666666666667,
                      0.08333333333333333, -0.4166666666666667,
                      -0.25, -0.4166666666666667,
                      0.08333333333333333, -0.25, 0.08333333333333333,
                      0.08333333333333333, 0.08333333333333333])
print(cv)
assert np.all(np.abs(cv_target - cv) < 1e-6)
