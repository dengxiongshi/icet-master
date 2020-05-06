import spglib as spg
from ase.neighborlist import NeighborList as ASENeighborList
from ase.db import connect
from icet.core.neighbor_list import get_neighbor_lists
"""
This script compares the neighbor list computed for structures in a database using
both icet.NeighborList and ase.neighborlist modules.
"""

db = connect('structures_for_testing.db')

neighbor_cutoff = 1.6
for row in db.select('natoms>1'):

    structure_row = row.toatoms()
    # ASE neighbor_list
    ase_nl = ASENeighborList(len(structure_row) * [neighbor_cutoff / 2], skin=1e-8,
                             bothways=True, self_interaction=False)
    ase_nl.update(structure_row)
    ase_indices, ase_offsets = ase_nl.get_neighbors(1)

    # icet neighbor_list
    neighbor_list = get_neighbor_lists(structure_row, [neighbor_cutoff])[0]
    neighbors = neighbor_list[1]
    indices = []
    offsets = []
    for ngb in neighbors:
        indices.append(ngb.index)
        offsets.append(ngb.unitcell_offset)

    msg = ['Testing size of neighbor_list indices']
    msg += ['failed for {}'.format(row.tag)]
    msg += ['{} != {} '.format(len(indices), len(ase_indices))]
    msg = ' '.join(msg)
    assert len(indices) == len(ase_indices), msg

    msg = ['Testing size of neighbor list offsets']
    msg += ['failed for {}'.format(row.tag)]
    msg += ['{} != {} '.format(len(offsets), len(ase_offsets))]
    msg = ' '.join(msg)
    assert len(offsets) == len(ase_offsets), msg

    for i, offset in zip(indices, offsets):
        msg = 'Testing offsets in neighbor list failed for {}'.format(row.tag)
        assert offset in ase_offsets, msg

        equiv_indices = \
            [x for x, ase_offset in enumerate(ase_offsets)
             if (ase_indices[x] == i and (ase_offset == offset).all())]
        if len(equiv_indices) > 1:
            print(i, offset, equiv_indices)
        assert len(equiv_indices) == 1, \
            'Testing duplicates offset failed for {}'.format(row.tag)
        assert i == ase_indices[equiv_indices[0]], \
            'Testing indices for offsets failed for {}'.format(row.tag)

    count_neighbors = {}
    inequiv_index = {}
    dataset = spg.get_symmetry_dataset(structure_row, symprec=1e-5,
                                       angle_tolerance=-1.0, hall_number=0)
    for index, equiv_index in enumerate(dataset['equivalent_atoms']):
        neighbors = neighbor_list[index]
        if equiv_index in count_neighbors:
            msg = ['Testing number of neighbors']
            msg += ['failed for {}'.format(row.tag)]
            msg += ['{}({}) at atom {}({}) '.format(
                len(neighbors),
                count_neighbors[equiv_index],
                index, inequiv_index[equiv_index])]
            msg = ' '.join(msg)
            assert count_neighbors[equiv_index] == len(neighbors), msg
        else:
            count_neighbors[equiv_index] = len(neighbors)
            inequiv_index[equiv_index] = index
