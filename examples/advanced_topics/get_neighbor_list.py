"""
This example demonstrates how to obtain the list of neighbors for a structure.
"""

# TODO: Why do we need this example?

# Import modules
from ase.build import bulk
from icet import Structure
from icet.core.neighbor_list import get_neighbor_lists

# Generate an icet structure from a 2x2x2 Al fcc supercell.
structure = bulk('Al', 'fcc', a=2).repeat(2)
structure.pbc = [True, True, True]
icet_structure = Structure.from_atoms(structure)

# Construct a list of all neighbors within the cutoff (1.5 A).
neighbor_cutoff = [1.5]
position_tolerance = 1e-5
nl = get_neighbor_lists(icet_structure, neighbor_cutoff, position_tolerance)[0]

# Loop over all atomic indices and print all of the neighbors.
for index in range(len(structure)):
    neighbors = nl.get_neighbors(index)
    print('Neighbors of atom with index {}'.format(index))
    for neighbor in neighbors:
        neighbor_index = neighbor.index
        neighbor_offset = neighbor.unitcell_offset
        distance_to_neighbor = icet_structure.get_distance(
            index, neighbor_index, [0, 0, 0], neighbor_offset)
        print('{0} {1} {2:1.5f}'.format(neighbor_index,
                                        neighbor_offset, distance_to_neighbor))
    print('')
print('fcc has {} nearest neighbors'.format(len(neighbors)))
