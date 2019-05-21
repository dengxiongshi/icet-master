import unittest

from ase.build import bulk
from icet import ClusterSpace
from mchammer.observers import PairCountObserver


class TestPairCountObserver(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestPairCountObserver,
              self).__init__(*args, **kwargs)

        self.atoms = bulk('Au').repeat([2, 2, 1])
        self.atoms[1].symbol = 'Pd'
        self.atoms = self.atoms.repeat(2)

        self.rmax = 3
        self.rmin = 0
        self.tag = 'AuPd'
        self.species = ['Au', 'Pd']
        subelements = [['Au', 'Pd']]*len(self.atoms)
        cutoffs = [6]
        self.cs = ClusterSpace(self.atoms, cutoffs, subelements)
        self.interval = 10

    def setUp(self):
        self.observer = PairCountObserver(self.cs, self.atoms, interval=self.interval,
                                          rmax=self.rmax, rmin=self.rmin, tag=self.tag,
                                          species=self.species)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def test_count_values(self):
        """Tests the sro values for a few systems where it is easily
        done by hand.
        """

        # 1 Pd in pure Au sro
        self.atoms.set_chemical_symbols(['Au'] * len(self.atoms))
        self.atoms[0].symbol = 'Pd'
        count = self.observer.get_observable(self.atoms)
        # In total there will be 12 Pd neighboring an Au atom

        self.assertEqual(count, 12)

        # 1 Au in Pure Pd sro
        self.atoms.set_chemical_symbols(['Pd'] * len(self.atoms))
        self.atoms[0].symbol = 'Au'
        count = self.observer.get_observable(self.atoms)

        # That one Au will have 12 Pd in its shell
        self.assertEqual(count, 12)

    def test_checkerboard_sro(self):
        """Tests the SRO parameter values for a BCC structure with
        checkerboard decoration.
        """

        atoms = bulk('Al', 'bcc', cubic=True, a=4)
        atoms[1].symbol = 'Ge'
        atoms = atoms.repeat(6)
        cutoffs = [5]
        subelements = [['Al', 'Ge']]*len(atoms)
        cluster_space = ClusterSpace(atoms, cutoffs, subelements)
        species = ['Al', 'Ge']
        observer = PairCountObserver(cluster_space, atoms, interval=self.interval,
                                     rmax=5, rmin=0, tag=self.tag, species=species)

        count = observer.get_observable(atoms)
        self.assertEqual(count, 1728)

    def test_property_tag(self):
        """Tests property tag."""
        self.assertEqual(self.observer.tag, self.tag)

    def test_property_interval(self):
        """Tests property interval."""
        self.assertEqual(self.observer.interval, self.interval)


if __name__ == '__main__':
    unittest.main()
