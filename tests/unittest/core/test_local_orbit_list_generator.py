import unittest
from ase.build import bulk

from icet.core.structure import Structure
from icet.core.orbit_list import create_orbit_list
from icet.core.local_orbit_list_generator import LocalOrbitListGenerator

class TestLocalOrbitListGenerator(unittest.TestCase):
    """
    Container for test of class functionality.
    """

    def __init__(self, *args, **kwargs):
        super(TestLocalOrbitListGenerator, self).__init__(*args, **kwargs)
        atoms = bulk('Al')
        cutoffs = [3.2, 3.2]
        # Build an orbit list and supercell
        self.orbit_list = create_orbit_list(atoms, cutoffs)
        self.structure = Structure.from_atoms(atoms.repeat(2))

    def setUp(self):
        """
        Instantiate class for each test case.
        """
        self.log = LocalOrbitListGenerator(self.orbit_list, self.structure)

    def test_generate_local_orbit_list(self):
        """
        Testing this function generates an orbit list by passing
        an index that corresponds to a specific offset of the supercell.
        """
        index = 0
        local_orbit_list = self.log.generate_local_orbit_list(index)
        for orbit in local_orbit_list.get_orbit_list():
            print(orbit.get_representative_sites().__str__())

    def test_unique_offsets(self):
        """
        Testing unique offsets obtained by mapping supercell sites to primitive cell.
        """
        unique_offsets =  self.log.get_unique_primcell_offsets()
        print(unique_offsets)


if __name__ == '__main__':
    unittest.main()
