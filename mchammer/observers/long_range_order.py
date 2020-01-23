import numpy as np
from itertools import combinations_with_replacement
from mchammer.observers.base_observer import BaseObserver


class StructureFactorObserver(BaseObserver):

    def __init__(self, atoms, q_points, symbol_pairs, interval=None):
        super().__init__(interval=interval, return_type=dict, tag='StructureFactorObserver')

        self.q_points = q_points
        self.pairs = symbol_pairs
        self.unique_symbols = set(s for p in symbol_pairs for s in p)
        self._Sq_lookup = self._get_Sq_lookup(atoms, q_points)

    def _get_Sq_lookup(self, atoms, q_points):
        """ Get SQ lookup data for a given supercell and q-points"""
        n_atoms = len(atoms)
        dist_vectors = atoms.get_all_distances(mic=True, vector=True)
        Sq_lookup = np.zeros((n_atoms, n_atoms, len(self.q_points)))
        for i in range(n_atoms):
            for j in range(n_atoms):
                Rij = dist_vectors[i][j]
                Sq = np.exp(-1j * np.dot(q_points, Rij))
                Sq_lookup[i, j, :] = Sq.real
        return Sq_lookup

    def _get_indices(self, atoms):
        """ Returns indices for all unique symbols """
        indices = dict()
        symbols = atoms.get_chemical_symbols()
        for symbol in self.unique_symbols:
            indices[symbol] = np.where(np.array(symbols) == symbol)[0]
        return indices

    def _compute_structure_factor(self, atoms):

        indices = self._get_indices(atoms)

        # compute Sq
        Sq_dict = dict()
        for sym1, sym2 in self.pairs:
            inds1 = indices[sym1]
            inds2 = indices[sym2]
            norm = 1 / np.sqrt(len(inds1) * len(inds2))
            Sq = np.zeros(len(self.q_points), dtype=np.complex128)
            for i in inds1:
                Sq += self._Sq_lookup[i, inds2].sum(axis=0) * norm
            assert np.max(np.abs(Sq.imag)) < 1e-6
            Sq_dict[(sym1, sym2)] = Sq

        return Sq_dict

    def get_observable(self, atoms):
        Sq_dict = self._compute_structure_factor(atoms)
        return_dict = dict()
        for pair, Sq in Sq_dict.items():
            for i in range(len(Sq)):
                tag = 'sfo_{}_{}_q{}'.format(*pair, i)
                return_dict[tag] = Sq[i]
        return return_dict


def compute_structure_factor_naive(atoms, q_points):

    # generate unique pairs
    symbols = atoms.get_chemical_symbols()
    symbols_unique = sorted(set(symbols))
    pairs = list(combinations_with_replacement(symbols_unique, r=2))

    # find indices for each species
    indices = dict()
    for symbol in symbols_unique:
        indices[symbol] = np.where(np.array(symbols) == symbol)[0]

    # vector R_ij lookup
    vectors = atoms.get_all_distances(mic=True, vector=True)

    # maybe slow
    Sq_dict = dict()
    for pair in pairs:
        sym1, sym2 = pair

        inds1 = indices[sym1]
        inds2 = indices[sym2]
        norm = 1 / np.sqrt(len(inds1) * len(inds2))

        Sq = np.zeros(len(q_points), dtype=np.complex128)
        for i in inds1:
            for j in inds2:
                Rij = vectors[i][j]
                Sq_tmp = np.exp(-1j * np.dot(q_points, Rij))
                Sq += Sq_tmp * norm
        assert np.max(np.abs(Sq.imag)) < 1e-6, Sq
        Sq_dict[(sym1, sym2)] = Sq

    return Sq_dict


if __name__ == '__main__':

    import numpy as np
    from icet import ClusterSpace, ClusterExpansion
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.ensembles import CanonicalEnsemble
    from mchammer.observers import StructureFactorObserver
    from ase.build import bulk

    # parameters
    size = 4
    a0 = 4.0
    symbols = ['Al', 'Si']

    # setup
    prim = bulk('Al', a=a0)
    cs = ClusterSpace(prim, [5.0], symbols)
    ce = ClusterExpansion(cs, np.random.random(len(cs)))

    # make supercell
    supercell = prim.repeat(4)
    n2 = int(len(supercell) / 2)
    supercell.set_chemical_symbols(['Al'] * n2 + ['Si'] * n2)

    # q-points
    q_point = 2 * np.pi / a0 * np.array([1, 0, 0])
    q_points = np.array([q_point * i for i in np.linspace(0, 1, size+1)[1:]])
    q_norms = np.linalg.norm(q_points, axis=1) / (2 * np.pi / a0)

    sfo = StructureFactorObserver(supercell, q_points, [symbols])

    calc = ClusterExpansionCalculator(supercell, ce)
    mc = CanonicalEnsemble(supercell, calc, 300)
    mc.attach_observer(sfo)
    mc.run(5000)

    dc = mc.data_container
    print(dc.data.columns)
