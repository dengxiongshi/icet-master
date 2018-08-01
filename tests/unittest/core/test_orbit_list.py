import unittest
from ase.build import bulk
from icet.core.lattice_site import LatticeSite
from icet.core.cluster import Cluster
from icet.core.orbit import Orbit
from icet.core.orbit_list import OrbitList, create_orbit_list
from icet.core.orbit_list import (
    __get_lattice_site_permutation_matrix as
    get_lattice_site_permutation_matrix)
from icet.core.neighbor_list import get_neighbor_lists
from icet.core.permutation_map import permutation_matrix_from_atoms
from icet.core.structure import Structure


class TestOrbitList(unittest.TestCase):
    """Container for test of the module functionality."""
    def __init__(self, *args, **kwargs):
        super(TestOrbitList, self).__init__(*args, **kwargs)
        self.cutoffs = [4.2]
        self.atoms = bulk('Ag', 'sc', a=4.1)

        # representative clusters for testing (singlet and pair)
        self.structure = Structure.from_atoms(self.atoms)

        self.cluster_singlet = Cluster(
            self.structure, [LatticeSite(0, [0, 0, 0])])

        lattice_sites = [LatticeSite(0, [i, 0, 0]) for i in range(2)]
        self.cluster_pair = Cluster(
            self.structure, [lattice_sites[0], lattice_sites[1]], True)

    def setUp(self):
        """Instantiate class before each test."""
        # permutation map
        permutation_matrix, self.prim_structure, _ = \
            permutation_matrix_from_atoms(self.atoms, self.cutoffs[0])
        self.pm_lattice_sites = \
            get_lattice_site_permutation_matrix(self.prim_structure,
                                                permutation_matrix)
        # neighbor-lists
        self.neighbor_lists = get_neighbor_lists(
            self.prim_structure, self.cutoffs)

        self.orbit_list = OrbitList(self.prim_structure,
                                    self.pm_lattice_sites,
                                    self.neighbor_lists)

    def test_init(self):
        """Test the different initializers."""
        # empty
        orbit_list = OrbitList()
        self.assertIsInstance(orbit_list, OrbitList)
        # with mnbl and structure
        orbit_list = OrbitList(self.neighbor_lists, self.prim_structure)
        self.assertIsInstance(orbit_list, OrbitList)
        # with mnbl, structure and permutation matrix
        orbit_list = OrbitList(
            self.prim_structure, self.pm_lattice_sites, self.neighbor_lists)
        self.assertIsInstance(self.orbit_list, OrbitList)

    def test_add_orbit(self):
        """Test add orbit funcionality."""
        orbit = Orbit(self.cluster_pair)
        self.orbit_list.add_orbit(orbit)
        self.assertEqual(len(self.orbit_list), 3)

    def test_get_number_of_NClusters(self):
        """Test counting orbits by number of bodies."""
        n_singlets = self.orbit_list.get_number_of_NClusters(1)
        self.assertEqual(n_singlets, 1)
        n_pairs = self.orbit_list.get_number_of_NClusters(2)
        self.assertEqual(n_pairs, 1)

    def test_get_orbit(self):
        """Test a copy of orbit is returned for a given index."""
        # singlet
        self.assertEqual(self.orbit_list.get_orbit(0).order, 1)
        # pair
        self.assertEqual(self.orbit_list.get_orbit(1).order, 2)
        # check there is not more listed orbits
        with self.assertRaises(IndexError):
            self.orbit_list.get_orbit(3)

    def test_clear(self):
        """Test orbit list is empty after calling this function."""
        self.orbit_list.clear()
        with self.assertRaises(IndexError):
            self.orbit_list.get_orbit(0)

    def test_sort(self):
        """Test sort functionality."""
        self.orbit_list.sort()
        for i in range(len(self.orbit_list) - 1):
            self.assertLess(
                self.orbit_list.get_orbit(i), self.orbit_list.get_orbit(i + 1))

    def test_orbits(self):
        """Test orbits property corresponds to the expected list of orbits."""
        # clusters for testing
        repr_clusters = [self.cluster_singlet, self.cluster_pair]

        for k, orbit in enumerate(self.orbit_list.orbits):
            with self.subTest(orbit=orbit):
                self.assertEqual(orbit.get_representative_cluster(),
                                 repr_clusters[k])

    def test_get_primitive_structure(self):
        """
        Test get primitive structure functionality.
        @todo This tests fails when comparing the structures intead of their
              positions.
        """
        prim_structure = self.orbit_list.get_primitive_structure()
        self.assertEqual(
            prim_structure.positions.tolist(),
            self.prim_structure.positions.tolist())

    def test_len(self):
        """Test len of orbit list."""
        self.assertEqual(len(self.orbit_list), 2)

    def test_create_orbit_list(self):
        """
        Test  orbit list is built from structure and cutoffs by calling
        this function.
        """
        orbit_list = create_orbit_list(self.atoms, self.cutoffs)
        for i in range(len(self.orbit_list)):
            orbit = self.orbit_list.get_orbit(i)
            orbit_ = orbit_list.get_orbit(i)
            # check all orbits in both lists are equal
            self.assertEqual(orbit, orbit_)

    def test_find_orbit(self):
        """Test orbit index is retuned from the given repr. cluster."""
        self.assertEqual(
            self.orbit_list._find_orbit(self.cluster_singlet), 0)
        self.assertEqual(
            self.orbit_list._find_orbit(self.cluster_pair), 1)
        non_repr = Cluster(
            self.structure, [LatticeSite(0, [0, 0, 0]),
                             LatticeSite(0, [1, 1, 1])])
        self.assertEqual(
            self.orbit_list._find_orbit(non_repr), -1)

    def test_rows_taken(self):
        """Test functionality."""
        taken_rows = set()
        row_indices = tuple([0, 1, 2])
        self.assertFalse(
            self.orbit_list._is_row_taken(taken_rows, row_indices))

        taken_rows = set([row_indices])
        self.assertTrue(
            self.orbit_list._is_row_taken(taken_rows, row_indices))

    def test_equivalent_sites_size(self):
        """Test that all the equivalent sites have the same radius."""
        for orbit in self.orbit_list.orbits:
            size = orbit.radius
            for eq_sites in orbit.equivalent_sites:
                cluster = Cluster(self.structure, eq_sites, True)
                self.assertAlmostEqual(
                    cluster.radius, size, places=5)

    def test_translate_to_unitcell(self):
        """Test the get all translated sites functionality."""
        # no offset site should get itself as translated
        sites = [LatticeSite(0, [0, 0, 0])]
        target = [[LatticeSite(0, [0, 0, 0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unit_cell(sites, False),
            sorted(target))

        # test a singlet site with offset
        sites = [LatticeSite(3, [0, 0, -1])]
        target = [[LatticeSite(3, [0, 0, 0])],
                  [LatticeSite(3, [0, 0, -1])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unit_cell(sites, False),
            sorted(target))

        # does it break when the offset is floats?
        sites = [LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unit_cell(sites, False),
            sorted(target))

        # float test continued
        sites = [LatticeSite(0, [1.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [1.0, 0.0, 0.0])],
                  [LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unit_cell(sites, False),
            sorted(target))

        # test two sites with floats and sort the output
        sites = [LatticeSite(0, [1.0, 0.0, 0.0]),
                 LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [1.0, 0.0, 0.0]),
                   LatticeSite(0, [0.0, 0.0, 0.0])],
                  [LatticeSite(0, [-1.0, 0.0, 0.0]),
                   LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unit_cell(sites, True),
            sorted(target))

        # test sites where none is inside unit cell
        sites = [LatticeSite(0, [1.0, 2.0, -1.0]),
                 LatticeSite(2, [2.0, 0.0, 0.0])]

        target = [sites,
                  [LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(2, [1.0, -2.0, 1.0])],
                  [LatticeSite(0, [-1.0, 2.0, -1.0]),
                   LatticeSite(2, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list._get_sites_translated_to_unit_cell(sites, False),
            sorted(target))

    def test_allowed_permutations(self):
        """Test allowed permutations of each orbit."""
        for orbit in self.orbit_list.orbits:
            for perm in orbit.allowed_permutations:
                self.assertEqual(len(set(perm)), len(perm))

    def test_get_colum1_from_pm(self):
        """Test the first column of the permutation matrix."""
        target = [LatticeSite(0, [0., 0., 0.]),
                  LatticeSite(0, [-1., 0., 0.]),
                  LatticeSite(0, [0., -1., 0.]),
                  LatticeSite(0, [0., 0., -1.]),
                  LatticeSite(0, [0., 0., 1.]),
                  LatticeSite(0, [0., 1., 0.]),
                  LatticeSite(0, [1., 0., 0.])]
        retval = \
            self.orbit_list._get_column1_from_pm(self.pm_lattice_sites, False)
        self.assertEqual(retval, target)

        retval = \
            self.orbit_list._get_column1_from_pm(self.pm_lattice_sites, True)
        self.assertEqual(retval, sorted(target))

    def test_find_rows_from_col1(self):
        """Test find_rows_from_col1 functionality."""
        column1 = \
            self.orbit_list._get_column1_from_pm(self.pm_lattice_sites, False)
        sites = [LatticeSite(0, [1.0, 0.0, 0.0]),
                 LatticeSite(0, [0.0, 0.0, 0.0])]
        rows = self.orbit_list._find_rows_from_col1(column1, sites, False)
        self.assertEqual(rows, [6, 0])
        # sort the rows
        rows = self.orbit_list._find_rows_from_col1(column1, sites, True)
        self.assertEqual(rows, [0, 6])

    def test_get_matches_in_pm(self):
        """Test get_matches_in_pm functionality."""
        sites = [LatticeSite(0, [1.0, 0.0, 0.0]),
                 LatticeSite(0, [0.0, 0.0, 0.0])]
        translated_sites = \
            self.orbit_list._get_sites_translated_to_unit_cell(sites, False)
        matches = self.orbit_list._get_matches_in_pm(translated_sites)
        sites, rows = zip(*matches)
        self.assertEqual(list(sites), sorted(translated_sites))
        self.assertEqual(list(rows), [[0, 1], [0, 6]])

    def test_orbit_list_non_pbc(self):
        """
        Test that singlets in orbit list retrieves the right number of unique
        sites of the structure with different periodic boundary conditions

        Todo
        ----
        Returned results are incorrect for simple-cubic structures.
        """
        atoms = bulk('Al', 'sc', a=4.0).repeat(4)
        structure = Structure.from_atoms(atoms)
        # [True, True, False]
        structure.set_pbc([True, True, False])
        orbit_list = create_orbit_list(structure, [0.])
        self.assertEqual(len(orbit_list), 4)
        # [True, False, False]
        structure.set_pbc([True, False, False])
        orbit_list = create_orbit_list(structure, [0.])
        self.assertEqual(len(orbit_list), 7)
        # [False]
        structure.set_pbc([False, False, False])
        orbit_list = create_orbit_list(structure, [0.])
        self.assertEqual(len(orbit_list), 20)

    def test_orbit_list_fcc(self):
        """
        Test orbit list has the right number of singlet and pairs for
        a fcc structure.
        """
        atoms = bulk('Al', 'fcc', a=3.0)
        cutoffs = [2.5]
        structure = Structure.from_atoms(atoms)
        orbit_list = create_orbit_list(structure, cutoffs)
        # only a singlet and a pair are expected
        self.assertEqual(len(orbit_list), 2)
        # singlet
        singlet = orbit_list.get_orbit(0)
        self.assertEqual(len(singlet), 1)
        # pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(1)
        self.assertEqual(len(pairs), 6)
        # not more orbits listed
        with self.assertRaises(IndexError):
            orbit_list.get_orbit(2)

    def test_orbit_list_bcc(self):
        """
        Test orbit list has the right number  of singlet and pairs for
        a bcc structure
        """
        atoms = bulk('Al', 'bcc', a=3.0)
        cutoffs = [3.0]
        structure = Structure.from_atoms(atoms)
        orbit_list = create_orbit_list(structure, cutoffs)
        # one singlet and two pairs expected
        self.assertEqual(len(orbit_list), 3)
        # singlet
        singlet = orbit_list.get_orbit(0)
        self.assertEqual(len(singlet), 1)
        # first pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(1)
        self.assertEqual(len(pairs), 4)
        # first pair has multiplicity equal to 3
        pairs = orbit_list.get_orbit(2)
        self.assertEqual(len(pairs), 3)
        # not more orbits listed
        with self.assertRaises(IndexError):
            orbit_list.get_orbit(3)

    def test_orbit_list_hcp(self):
        """
        Test orbit list has the right number of singlet and pairs for
        a hcp structure
        """
        atoms = bulk('Ni', 'hcp', a=3.0)
        cutoffs = [3.1]
        structure = Structure.from_atoms(atoms)
        orbit_list = create_orbit_list(structure, cutoffs)
        # only one singlet and one pair expected
        self.assertEqual(len(orbit_list), 3)
        # singlet
        singlet = orbit_list.get_orbit(0)
        self.assertEqual(len(singlet), 2)
        # pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(1)
        self.assertEqual(len(pairs), 6)
        # pair has multiplicity equal to 4
        pairs = orbit_list.get_orbit(2)
        self.assertEqual(len(pairs), 6)
        # not more orbits listed
        with self.assertRaises(IndexError):
            orbit_list.get_orbit(3)


if __name__ == '__main__':
    unittest.main()
