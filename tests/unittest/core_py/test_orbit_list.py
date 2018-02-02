#!/usr/bin/env Python3


import unittest

from icet.core_py.orbit_list import OrbitList
from icet.core_py.lattice_site import LatticeSite
from icet.core_py.permutation_matrix import PermutationMatrix
# from icet.core.orbit_list import create_orbit_list

# from icet import Structure
from icet import ClusterSpace
class TestOrbitList(unittest.TestCase):
    '''
    Container for tests of the class functionality
    '''

    def __init__(self, *args, **kwargs):
        from ase.build import bulk
        super(TestOrbitList, self).__init__(*args, **kwargs)
        self.cutoffs = [0, 5]
        self.atoms = bulk('Ag', a=4.09)

    def setUp(self):
        '''
        Instantiate class before each test.
        '''
        self.orbit_list = OrbitList(self.atoms, self.cutoffs)
        # self._structure = Structure.from_atoms(self.atoms)

        self.cluster_space_cpp = ClusterSpace(self.atoms, self.cutoffs,["Al","H"])

    def test_init(self):
        '''
        Just testing that the setup
        (initialization) of tested class work
        '''
        # initialize from ASE Atoms
        orbit_list = OrbitList(self.atoms, self.cutoffs)
        self.assertIsInstance(orbit_list, OrbitList)

    def test_sort(self):
        '''
        Testing len functionality
        '''
        self.orbit_list.sort()
        for i in range(len(self.orbit_list) - 1):
            self.assertLess(
                self.orbit_list.orbits[i], self.orbit_list.orbits[i + 1])

    def test_property_primitive_structure(self):
        '''
        Testing get_orbit_list_info functionality
        '''
        self.orbit_list.primitive_structure
        self.assertEqual(self.orbit_list.primitive_structure,
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
        """
        pass

    def test_make_orbit(self):
        """
        Test make a new orbit.
        """
        pass

    def test_get_rows(self):
        """
        Test the get row method.
        """
        pass

    def test_get_indices(self):
        """
        Test the get indices method
        """
        pass

    def test_get_all_translated_sites(self):
        """
        Test teh get all translated sites functionality.
        """
        sites = [LatticeSite(0, [0, 0, 0])]
        target = [[LatticeSite(0, [0, 0, 0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), target)

        # Does it break when the offset is floats?
        sites = [LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), target)

        sites = [LatticeSite(0, [1.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), target)

        sites = [LatticeSite(0, [1.0, 0.0, 0.0]),
                 LatticeSite(0, [0.0, 0.0, 0.0])]
        target = [[LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(0, [-1, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), target)

        sites = [LatticeSite(0, [1.0, 2.0, -1.0]),
                 LatticeSite(2, [2.0, 0.0, 0.0])]

        target = [[LatticeSite(0, [0.0, 0.0, 0.0]),
                   LatticeSite(2, [1.0, -2.0, 1.0])],
                  [LatticeSite(0, [-1.0, 2.0, -1.0]),
                   LatticeSite(2, [0.0, 0.0, 0.0])]]
        self.assertListEqual(
            self.orbit_list.get_all_translated_sites(sites), target)

    def test_property_permutation_matrix(self):
        '''
        Test the permutation matrix property.
        '''
        self.assertIsInstance(
            self.orbit_list.permutation_matrix, PermutationMatrix)
        self.assertEqual(
            self.orbit_list.permutation_matrix.cutoff, max(self.cutoffs))

    def test_str(self):
        print(self.orbit_list)
        print(self.cluster_space_cpp)
        for orbit in self.orbit_list.orbits:
            for sites in orbit.equivalent_sites:
                for site in sites:
                    print(site, end= ' ')
                print()                    
            print("----")
        print("C++ version")
        for orbit in self.cluster_space_cpp.get_orbit_list().get_orbit_list():
            for sites in orbit.equivalent_sites:
                for site in sites:
                    print(site, end= ' ')
                print()                    
            print("----")
            
if __name__ == '__main__':
    unittest.main()
