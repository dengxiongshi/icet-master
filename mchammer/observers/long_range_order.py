import numpy as np
from itertools import combinations_with_replacement


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
        print(pair)
        symbol1, symbol2 = pair
        Sq = np.zeros(len(q_points), dtype=np.complex128)
        for i in indices[symbol1]:
            for j in indices[symbol2]:
                Rij = vectors[i][j]
                Sq_tmp = np.exp(-1j * np.dot(q_points, Rij))
                Sq += Sq_tmp
        assert np.max(np.abs(Sq.imag)) < 1e-6
        Sq_dict[(symbol1, symbol2)] = Sq

    return Sq_dict


def get_Sq_lookup_data(atoms, q_points):
    # pre-compute lookuo
    dist_vectors = atoms.get_all_distances(mic=True, vector=True)
    Sq_lookup = np.zeros((n_atoms, n_atoms, len(q_points)))
    for i in range(n_atoms):
        for j in range(n_atoms):
            Rij = dist_vectors[i][j]
            Sq = np.exp(-1j * np.dot(q_points, Rij))
            Sq_lookup[i, j, :] = Sq.real
    return Sq_lookup


def compute_structure_factor_lookup(atoms, q_points, Sq_lookup):

    # generate unique pairs
    symbols = atoms.get_chemical_symbols()
    symbols_unique = sorted(set(symbols))
    pairs = list(combinations_with_replacement(symbols_unique, r=2))

    # find indices for each species
    indices = dict()
    for symbol in symbols_unique:
        indices[symbol] = np.where(np.array(symbols) == symbol)[0]

    # compute Sq
    Sq_dict = dict()
    for sym1, sym2 in pairs:
        inds1 = indices[sym1]
        inds2 = indices[sym2]
        Sq = np.zeros(len(q_points), dtype=np.complex128)
        for i in inds1:
            Sq += Sq_lookup[i, inds2].sum(axis=0)
        assert np.max(np.abs(Sq.imag)) < 1e-6
        Sq_dict[(sym1, sym2)] = Sq

    return Sq_dict


if __name__ == '__main__':

    import time
    from ase.build import bulk
    import matplotlib.pyplot as plt

    np.random.seed(42)

    # setup
    a0 = 5.6
    size = 6
    atoms = bulk('NaCl', 'rocksalt', a=a0, cubic=True).repeat((size, size, size))
    n_atoms = len(atoms)
    dist_vectors = atoms.get_all_distances(mic=True, vector=True)

    # kpts
    q_point = 2 * np.pi / a0 * np.array([1, 0, 0])
    q_points = np.array([q_point * i for i in np.linspace(0, 1, size+1)[1:]])
    q_norms = np.linalg.norm(q_points, axis=1)

    # lookup data
    Sq_lookup = get_Sq_lookup_data(atoms, q_points)

    # Sanity tests
    # ------------------

    # compute Sq ordered
    t1 = time.time()
    Sq_order = compute_structure_factor_naive(atoms, q_points)
    print('naive', time.time()-t1)

    t1 = time.time()
    Sq_order2 = compute_structure_factor_lookup(atoms, q_points, Sq_lookup=Sq_lookup)
    print('lookup', time.time()-t1)

    pairs = sorted(Sq_order.keys())
    for p in pairs:
        assert np.allclose(Sq_order[p], Sq_order2[p])

    # compute Sq disordered
    n_rnd = 3
    for i in range(n_rnd):
        print(i)
        atoms_rnd = atoms.copy()
        np.random.shuffle(atoms_rnd.numbers)

        Sq1 = compute_structure_factor_naive(atoms_rnd, q_points)
        Sq2 = compute_structure_factor_lookup(atoms_rnd, q_points, Sq_lookup=Sq_lookup)

        pairs = sorted(Sq_order.keys())
        for p in pairs:
            assert np.allclose(Sq1[p], Sq2[p])

    # Sample NaCl S(q)
    # ------------------

    # compute Sq disordered
    n_rnd = 500
    Sq_rnd_ave = {p: np.zeros(len(q_points), dtype=np.complex128) for p in pairs}
    for i in range(n_rnd):
        print(i)
        atoms_rnd = atoms.copy()
        np.random.shuffle(atoms_rnd.numbers)
        Sq = compute_structure_factor_lookup(atoms_rnd, q_points, Sq_lookup=Sq_lookup)
        for key in Sq.keys():
            Sq_rnd_ave[key] += Sq[key]

    for key in Sq_rnd_ave.keys():
        Sq_rnd_ave[key] = np.array(Sq_rnd_ave[key])
        Sq_rnd_ave[key] /= n_rnd

    # plot
    norm = 4 / n_atoms

    fig = plt.figure(figsize=(9, 2.8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    for ax, pair in zip([ax1, ax2, ax3], Sq_order.keys()):
        ax.plot(q_norms, Sq_order[pair]*norm, '-o', label='ordered')
        ax.plot(q_norms, Sq_rnd_ave[pair]*norm, '-o', label='random')
        ax.set_xlabel('q-vector')
        ax.legend()
        ax.set_title(pair)
    ax1.set_ylabel('S(q)')
    fig.tight_layout()
    fig.savefig('test_long_range.png')
    plt.show()
