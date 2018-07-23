#!/usr/bin/env Python3
import unittest

from icet.core_py.orbit_list import OrbitList
from icet.core_py.lattice_site import LatticeSite
from icet.core_py.permutation_matrix import PermutationMatrix

# from icet.core.orbit_list import create_orbit_list
from ase.build import bulk
# from icet import Structure
# from icet import ClusterSpace
from ase.build import fcc111


class TestOrbitList(unittest.TestCase):
    """
    Container for tests of the class functionality
    """

    def __init__(self, *args, **kwargs):
        super(TestOrbitList, self).__init__(*args, **kwargs)
        self.cutoffs = [4.2]
        self.atoms = bulk('Ag', a=4.09)

    def setUp(self):
        """
        Instantiate class before each test.
        """
        self.orbit_list = OrbitList(self.atoms, self.cutoffs)
        # self._structure = Structure.from_atoms(self.atoms)

    def test_init(self):
        """
        Just testing that the setup
        (initialization) of tested class work
        """
        # initialize from ASE Atoms
        orbit_list = OrbitList(self.atoms, self.cutoffs)
        self.assertIsInstance(orbit_list, OrbitList)

    def test_len(self):
        """
        Test that length is equal to number of orbits.
        """
        self.assertEqual(len(self.orbit_list), len(self.orbit_list.orbits))
        self.assertEqual(len(self.orbit_list), 3)

    def test_sort(self):
        '''
        Testing sort functionality
        '''
        self.orbit_list.sort()
        for i in range(len(self.orbit_list) - 1):
            self.assertLess(
                self.orbit_list.orbits[i], self.orbit_list.orbits[i + 1])

    def test_property_primitive_structure(self):
        """ Tests the primitive_structure property. """
        self.orbit_list.primitive_structure
        self.assertEqual(
            self.orbit_list.primitive_structure,
            self.orbit_list.permutation_matrix.primitive_structure)

    def test_property_orbit(self):
        """
        Test orbit property.
        """
        self.orbit_list.orbits
        self.assertEqual(len(self.orbit_list), len(self.orbit_list.orbits))

    def test_is_new_orbit(self):
        """
        Test is new orbit method
        Test this method by going through
        all sites in the orbits and testing
        that there are no new orbits
        that could be made from these sites,
        i.e. they are already taken.
        """
        for orbit in self.orbit_list.orbits:
            for sites in orbit.equivalent_sites:
                self.assertFalse(self.orbit_list.is_new_orbit(sites))

    def test_make_orbit(self):
        """
        Test make a new orbit.
        Test that creating a new orbit from
        the orbits representative site
        gives back the same orbit.
        """
        # Need to reset taken rows for making a new orbit

        self.orbit_list.taken_rows = set()
        for orbit in self.orbit_list.orbits:
            eq_orbit = self.orbit_list.make_orbit(orbit.representative_sites)
            self.assertEqual(len(orbit), len(eq_orbit))

    def test_get_rows(self):
        """
        Test the get row method.
        """
        indices = [0, 1]
        sites = []
        for index in indices:
            sites.append(
                self.orbit_list.permutation_matrix.pm_lattice_sites[index][0])

        rows = self.orbit_list.get_rows(sites)
        self.assertEqual(
            rows[0], self.orbit_list.permutation_matrix.pm_lattice_sites[0])
        self.assertEqual(
            rows[1], self.orbit_list.permutation_matrix.pm_lattice_sites[1])

    def test_get_row_indices(self):
        """
        Test the get indices method
        """
        indices_target = (0, 1)
        sites = []
        for index in indices_target:
            sites.append(
                self.orbit_list.permutation_matrix.pm_lattice_sites[index][0])

        retval = self.orbit_list.get_row_indices(sites)
        self.assertEqual(indices_target, retval)

    def test_get_all_translated_sites(self):
        """
        Test the get all translated sites functionality.
        """

        # no offset site shoud get itself as translated
        sites = [LatticeSite(0, [0, 0, 0])]
        target = [[LatticeSite(0, [0, 0, 0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), sorted(target))

        # test a singlet site with offset
        sites = [LatticeSite(3, [0, 0, -1])]
        target = [[LatticeSite(3, [0, 0, 0])],
                  [LatticeSite(3, [0, 0, -1])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), sorted(target))

        # Does it break when the offset is floats?
        sites = [LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), sorted(target))
        # Float test continued
        sites = [LatticeSite(0, [1.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [1.0, 0.0, 0.0])], [
            LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), sorted(target))
        # Test two sites with floats
        sites = [LatticeSite(0, [1.0, 0.0, 0.0]),
                 LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [1.0, 0.0, 0.0]),
                   LatticeSite(0, [0.0, 0.0, 0.0])],
                  [LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(0, [-1, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), sorted(target))

        # Test sites where none is inside unit cell
        sites = [LatticeSite(0, [1.0, 2.0, -1.0]),
                 LatticeSite(2, [2.0, 0.0, 0.0])]

        target = [sites,
                  [LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(2, [1.0, -2.0, 1.0])],
                  [LatticeSite(0, [-1.0, 2.0, -1.0]),
                   LatticeSite(2, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), sorted(target))

    def test_no_duplicates_in_orbit(self):
        """
        Test that there are no duplicate sites in orbits
        equivalent sites.
        """
        for orbit in self.orbit_list.orbits:
            for i, site_i in enumerate(orbit.equivalent_sites):
                for j, site_j in enumerate(orbit.equivalent_sites):
                    if j <= i:
                        continue
                    self.assertNotEqual(sorted(site_i), sorted(site_j))

    def test_property_permutation_matrix(self):
        """ Tests the permutation_matrix property. """
        self.assertIsInstance(
            self.orbit_list.permutation_matrix, PermutationMatrix)
        self.assertEqual(
            self.orbit_list.permutation_matrix.cutoff, max(self.cutoffs))

    def test_str(self):
        """
        Test str functionality.

        Just print the orbitlist and see that it works.
        """
        print(self.orbit_list)

    def test_singlets_particle(self):
        """
        Test that a particle gets correct number of singlets.
        """
        # Below is an explanation on how the particle looks like
        # and what its unique sites are:
        # ------------------------------
        #
        #  One corner site,X,
        #  one next to the corner along the side, O,
        #  one site on the surface, K,
        #  and one of the center 2x2x2 sites J
        #
        #
        #    from the side:
        #    X O O X
        #    H K K H
        #    H K K H
        #    X O O X
        #
        #  below the K sites are another J site in the 2x2x2 positions
        #
        atoms = bulk("Al", 'sc', a=1).repeat(4)
        atoms.pbc = False

        orbit_list = OrbitList(atoms, [0])
        self.assertEqual(len(orbit_list), 4)

        # Size 3 also has 4 unique sites
        atoms = bulk("Al", 'sc', a=1).repeat(3)
        atoms.pbc = False

        orbit_list = OrbitList(atoms, [0])
        self.assertEqual(len(orbit_list), 4)

        # Test partial pbc:
        # Now we only distinguish sites by layers from
        # The surface. We have 4 layers but with mirror
        # symmetry we get 2 unique singlet orbits

        atoms = bulk("Al", 'sc', a=1).repeat(4)
        atoms.pbc = [True, True, False]

        orbit_list = OrbitList(atoms, [0])
        self.assertEqual(len(orbit_list), 2)

        # Pbc in only one direction, so we effectively have
        # a rod-particle the unique sites are only at the surface
        # so 1 corner, one side and one bulk
        atoms = bulk("Al", 'sc', a=1).repeat(4)
        atoms.pbc = [True, False, False]

        orbit_list = OrbitList(atoms, [0])
        self.assertEqual(len(orbit_list), 3)

        # Increasing size to 5 will give
        #  one corner
        # two on the side
        # the center will be 3x3 so 3 sites from that
        atoms = bulk("Al", 'sc', a=1).repeat(5)
        atoms.pbc = [True, False, False]

        orbit_list = OrbitList(atoms, [0])
        self.assertEqual(len(orbit_list), 6)

        # Test pbc = True
        atoms = bulk("Al", 'sc', a=1).repeat(3)
        atoms.pbc = True

        orbit_list = OrbitList(atoms, [0])
        self.assertEqual(len(orbit_list), 1)

    def test_pairs_fcc(self):
        """
        Test orbitlist with only pairs and singlets
        for a fcc system.
        """
        # Should get one singlet and one pair
        atoms = bulk("Al", 'fcc', a=3.0)
        cutoff = [2.5]

        orbit_list = OrbitList(atoms, cutoff)
        self.assertEqual(len(orbit_list), 2)

        # Multiplicity of pair should be 6
        orbit_pair = orbit_list.orbits[1]
        self.assertEqual(len(orbit_pair), 6)

        cutoff = [3.1]

        # should get another pair at cutoff 3
        # so three orbits with cutoff 3.1
        orbit_list = OrbitList(atoms, cutoff)
        self.assertEqual(len(orbit_list), 3)

        # Multiplicity of second pair should be 3
        orbit_pair = orbit_list.orbits[2]
        self.assertEqual(len(orbit_pair), 3)

    def test_pairs_bcc(self):
        """
        Test orbitlist with only pairs and singlets
        for a bcc system.
        """
        # Should get one singlet and one pair
        atoms = bulk("Al", 'bcc', a=3.0)
        cutoff = [2.7]
        orbit_list = OrbitList(atoms, cutoff)
        self.assertEqual(len(orbit_list), 2)

        # Multiplicity of pair should be 4
        # i.e. 8 neighboring cells each
        # with an atom in the center and
        # divided by 2 since it is per atom
        orbit_pair = orbit_list.orbits[1]
        self.assertEqual(len(orbit_pair), 4)

        #  should get another pair at cutoff 3
        # so three orbits with cutoff 3.1
        cutoff = [3.1]
        orbit_list = OrbitList(atoms, cutoff)
        self.assertEqual(len(orbit_list), 3)

        # Multiplicity of second pair should be 3
        orbit_pair1 = orbit_list.orbits[1]
        orbit_pair2 = orbit_list.orbits[2]
        self.assertEqual(len(orbit_pair1), 4)
        self.assertEqual(len(orbit_pair2), 3)

    def test_pairs_hcp(self):
        """
        Test orbitlist with only pairs and singlets
        for a bcc system.
        """

        atoms = bulk("Al", 'hcp', a=3.0, c=2.5)
        cutoff = [0]
        orbit_list = OrbitList(atoms, cutoff)
        # hcp has one singlet
        self.assertEqual(len(orbit_list), 1)

        # Gets two pairs
        #  TODO think if this is the correct result for hcp
        atoms = bulk("Al", 'hcp', a=3.0)
        cutoff = [3.1]
        orbit_list = OrbitList(atoms, cutoff)
        self.assertEqual(len(orbit_list), 3)

    def test_rows_taken(self):
        """
        This will test both method is_taken_rows
        and take_row
        """
        # clear taken rows
        self.orbit_list.taken_rows = set()
        row_indices = tuple([0, 1, 2])
        self.assertFalse(self.orbit_list.is_rows_taken(row_indices))

        # take rows
        self.orbit_list.take_row(row_indices)
        self.assertTrue(self.orbit_list.is_rows_taken(row_indices))

    def test_higher_order_orbits(self):
        """
        Try to construct higher order orbit list

        These tests will simply create orbit list
        and see that the orbit list gets created
        without any errors thrown.

        """
        atoms = bulk("Al", 'hcp', a=3.01)
        cutoff = [5] * 3
        orbit_list = OrbitList(atoms, cutoff)

        atoms = bulk("Al", 'fcc', a=3.01)
        cutoff = [5] * 3
        orbit_list = OrbitList(atoms, cutoff)

        atoms = bulk("Al", 'bcc', a=3.01)
        cutoff = [5] * 3
        orbit_list = OrbitList(atoms, cutoff)

        cutoff = [4.0] * 3
        atoms = bulk('Ag', a=4.09)
        orbit_list = OrbitList(atoms, cutoff)

        cutoff = [4.0] * 1
        atoms = fcc111('Ag', a=4.09, vacuum=5.0, size=[1, 1, 3])
        atoms.pbc = [True, True, False]
        orbit_list = OrbitList(atoms, cutoff)  # noqa

    def test_get_matches_in_pm(self):
        """
        Test function get_matches_in_pm
        """

        # Find indices in column1 for list of lattice sites
        # permutation matrix
        indices = [0, 5, 2]
        # manually get sites
        sites = [[self.orbit_list.column1[i] for i in indices]]

        match = self.orbit_list.get_matches_in_pm(sites)
        # match[0][1] is the indices we found
        self.assertEqual(match[0][1], tuple(indices))

    def test_property_permutations_to_representative(self):
        """
        Test permutations to representative.
        """
        for orbit in self.orbit_list.orbits:
            # Test that the permutations to representative is
            # the same length as the number of equivalent_sites
            # i.e. len(orbit)
            self.assertEqual(
                len(orbit.permutations_to_representative), len(orbit))
            for perm in orbit.permutations_to_representative:
                # Assert that the permutation doesn't have any
                # duplicates
                self.assertEqual(len(set(perm)), len(perm))

    def test_property_allowed_permutations(self):
        """
        Test property allowed permutations
        """
        for orbit in self.orbit_list.orbits:
            for perm in orbit.allowed_permutations:
                # Assert that the permutation doesn't have any
                # duplicates
                self.assertEqual(len(set(perm)), len(perm))


if __name__ == '__main__':
    unittest.main()
