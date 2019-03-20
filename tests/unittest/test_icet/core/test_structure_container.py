#!/usr/bin/env python3

"""
This file contains unit tests and other tests. It can be executed by
simply executing this file from a shell prompt:

    $ ./test_structure_container.py

In which case it will use the system's default python version. If a specific
python version should be used, run that python version with this file as input,
e.g.:

    python3 test_structure_container.py

For a description of the python unit testing framework, see this link:
https://docs.python.org/3/library/unittest.html

When executing this file doc testing is also performed on all doc tests in
the structure_container.py file

"""

import sys
import tempfile
import unittest
import numpy as np

from io import StringIO
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from icet import ClusterSpace, StructureContainer
from icet.core.structure_container import FitStructure


def strip_surrounding_spaces(input_string):
    """
    Helper function that removes both leading and trailing spaces from a
    multi-line string.

    Returns
    -------
    str
        original string minus surrounding spaces and empty lines
    """
    s = []
    with StringIO(input_string) as data:
        for line in data:
            if len(line.strip()) == 0:
                continue
            s += [line.strip()]
    return '\n'.join(s)


class TestStructureContainer(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestStructureContainer, self).__init__(*args, **kwargs)
        prim = bulk('Ag', a=4.09)
        chemical_symbols = ['Ag', 'Au']
        self.cs = ClusterSpace(atoms=prim,
                               cutoffs=[4.0, 4.0, 4.0],
                               chemical_symbols=chemical_symbols)
        self.structure_list = []
        self.user_tags = []
        for k in range(4):
            atoms = prim.repeat(2)
            symbols = [chemical_symbols[0]] * len(atoms)
            symbols[:k] = [chemical_symbols[1]] * k
            atoms.set_chemical_symbols(symbols)
            self.structure_list.append(atoms)
            self.user_tags.append('Structure {}'.format(k))

        self.properties_list = []
        self.add_properties_list = []
        for k, atoms in enumerate(self.structure_list):
            atoms.set_calculator(EMT())
            properties = {'energy': atoms.get_potential_energy(),
                          'volume': atoms.get_volume(),
                          'Au atoms': atoms.get_chemical_symbols().count('Au')}
            self.properties_list.append(properties)
            add_properties = {'total_energy': atoms.get_total_energy()}
            self.add_properties_list.append(add_properties)

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.sc = StructureContainer(self.cs)
        for atoms, tag, props in zip(self.structure_list, self.user_tags, self.properties_list):
            self.sc.add_structure(atoms, tag, props)

    def test_init(self):
        """Tests that initialization of tested class works."""
        # check empty initialization
        self.assertIsInstance(StructureContainer(self.cs), StructureContainer)

        # check whether method raises Exceptions
        with self.assertRaises(TypeError) as cm:
            StructureContainer('my_sc.sc')
        self.assertIn('cluster_space must be a ClusterSpace', str(cm.exception))

    def test_len(self):
        """Tests length functionality."""
        len_structure_container = self.sc.__len__()
        self.assertEqual(len_structure_container, len(self.structure_list))

    def test_getitem(self):
        """Tests getitem functionality."""
        structure = self.sc.__getitem__(1)
        self.assertIsNotNone(structure)

    def test_get_structure_indices(self):
        """Tests get_structure_indices functionality."""
        list_index = [x for x in range(len(self.structure_list))]
        self.assertEqual(self.sc.get_structure_indices(), list_index)

    def test_add_structure(self):
        """Tests add_structure functionality."""
        # add atoms with tag and property
        atoms = self.structure_list[0]
        properties = self.properties_list[0]
        tag = 'Structure 4'
        self.sc.add_structure(atoms, tag, properties)
        self.assertEqual(len(self.sc), len(self.structure_list) + 1)

        # add atom and read property from calculator
        self.sc.add_structure(atoms)
        self.assertEqual(len(self.sc), len(self.structure_list) + 2)
        self.assertEqual(self.sc[5].properties['energy'],
                         self.properties_list[0]['energy'] / len(atoms))

        # add atom and don't read property from calculator
        atoms_cpy = atoms.copy()
        atoms_cpy.set_calculator(EMT())
        self.sc.add_structure(atoms_cpy)
        self.assertEqual(len(self.sc), len(self.structure_list) + 3)
        self.assertNotIn('energy', self.sc[6].properties)

        # check that duplicate structure is not added.
        with self.assertRaises(ValueError) as cm:
            self.sc.add_structure(atoms, allow_duplicate=False)
        msg = 'Input atoms have identical cluster vector with' \
              ' Structure 0 at index 0'
        self.assertEqual(msg, str(cm.exception))
        self.assertEqual(len(self.sc), len(self.structure_list) + 3)

        symbols = ['Au' for i in range(len(atoms))]
        atoms.set_chemical_symbols(symbols)
        self.sc.add_structure(atoms, 'Structure 5', allow_duplicate=False)
        self.assertEqual(len(self.sc), len(self.structure_list) + 4)

    def test_get_condition_number(self):
        """Tests get_condition_number functionality."""
        target = np.linalg.cond(self.sc.get_fit_data()[0])
        retval = self.sc.get_condition_number()

        self.assertEqual(target, retval)

    def test_get_fit_data(self):
        """Tests get_fit_data functionality."""
        import numpy as np
        cluster_vectors, properties = self.sc.get_fit_data()
        # testing outputs have ndarray type
        self.assertIsInstance(cluster_vectors, np.ndarray)
        self.assertIsInstance(properties, np.ndarray)
        # testing values of cluster_vectors and properties
        for atoms, cv in zip(self.structure_list, cluster_vectors):
            retval = list(cv)
            target = list(self.cs.get_cluster_vector(atoms))
            self.assertAlmostEqual(retval, target, places=9)
        for target, retval in zip(self.properties_list, properties):
            self.assertEqual(retval, target['energy'])
        # passing a list of indexes
        cluster_vectors, properties = self.sc.get_fit_data([0])
        retval = list(cluster_vectors[0])
        atoms = self.structure_list[0]
        target = list(self.cs.get_cluster_vector(atoms))
        self.assertAlmostEqual(retval, target, places=9)
        retval2 = properties[0]
        target2 = self.properties_list[0]
        self.assertEqual(retval2, target2['energy'])

    def test_repr(self):
        """Tests repr functionality."""
        retval = self.sc.__repr__()
        target = """
================================== Structure Container ===================================
Total number of structures: 4
------------------------------------------------------------------------------------------
index |       user_tag        | natoms | chemical formula | Au atoms |  energy  |  volume
------------------------------------------------------------------------------------------
   0  | Structure 0           |     8  | Ag8              |        0 |    0.013 |  136.836
   1  | Structure 1           |     8  | Ag7Au            |        1 |   -0.007 |  136.836
   2  | Structure 2           |     8  | Ag6Au2           |        2 |   -0.026 |  136.836
   3  | Structure 3           |     8  | Ag5Au3           |        3 |   -0.038 |  136.836
==========================================================================================
""" # noqa
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

        # test representation of an empty structure container
        sc = StructureContainer(self.cs)
        self.assertEqual(sc.__repr__(), 'Empty StructureContainer')

    def test_get_string_representation(self):
        """Tests _get_string_representation functionality."""
        retval = self.sc._get_string_representation(print_threshold=2,
                                                    print_minimum=1)
        target = """
================================== Structure Container ===================================
Total number of structures: 4
------------------------------------------------------------------------------------------
index |       user_tag        | natoms | chemical formula | Au atoms |  energy  |  volume
------------------------------------------------------------------------------------------
   0  | Structure 0           |     8  | Ag8              |        0 |    0.013 |  136.836
 ...
   3  | Structure 3           |     8  | Ag5Au3           |        3 |   -0.038 |  136.836
==========================================================================================
""" # noqa
        self.assertEqual(strip_surrounding_spaces(target),
                         strip_surrounding_spaces(retval))

    def test_print_overview(self):
        """Tests print_overview functionality."""
        with StringIO() as capturedOutput:
            sys.stdout = capturedOutput  # redirect stdout
            self.sc.print_overview()
            sys.stdout = sys.__stdout__  # reset redirect
            self.assertTrue('Structure Container' in capturedOutput.getvalue())

    def test_cluster_space(self):
        """Tests cluster space functionality."""
        cs_onlyread = self.sc.cluster_space
        self.assertEqual(cs_onlyread, self.cs)

    def test_available_properties(self):
        """Tests available_properties property."""
        available_properties = sorted(self.properties_list[0])
        self.sc.add_structure(self.structure_list[0], properties=self.properties_list[0])

        self.assertSequenceEqual(available_properties, self.sc.available_properties)

    def test_read_write(self):
        """Tests the read and write functionality."""
        temp_file = tempfile.NamedTemporaryFile()

        # check before with a non-tar file
        with self.assertRaises(TypeError) as context:
            self.sc.read(temp_file)
        self.assertTrue('{} is not a tar file'.format(str(temp_file.name))
                        in str(context.exception))

        # save and read an empty structure container
        sc = StructureContainer(self.cs)
        sc.write(temp_file.name)
        sc_read = StructureContainer.read(temp_file.name)
        self.assertEqual(sc_read.__str__(), 'Empty StructureContainer')

        # save to file
        self.sc.write(temp_file.name)

        # read from file object
        sc_read = self.sc.read(temp_file.name)

        # check data
        self.assertEqual(len(self.sc), len(sc_read))
        self.assertEqual(self.sc.__str__(), sc_read.__str__())

        for fs, fs_read in zip(self.sc._structure_list, sc_read._structure_list):
            self.assertEqual(list(fs.cluster_vector),
                             list(fs_read.cluster_vector))
            self.assertEqual(fs.atoms, fs_read.atoms)
            self.assertEqual(fs.user_tag, fs_read.user_tag)
            self.assertEqual(fs.properties, fs_read.properties)
        temp_file.close()


class TestFitStructure(unittest.TestCase):
    """Container for tests of the class functionality."""

    def __init__(self, *args, **kwargs):
        super(TestFitStructure, self).__init__(*args, **kwargs)
        self.prim = bulk('Ag', a=4.09)
        self.cs = ClusterSpace(atoms=self.prim, cutoffs=[4.0, 4.0, 4.0],
                               chemical_symbols=['Ag', 'Au'])

        self.atoms = self.prim.repeat(2)
        self.prop = {'energy': 0.0126746}
        self.cv = self.cs.get_cluster_vector(self.atoms)
        self.tag = 'struct1'

    def shortDescription(self):
        """Silences unittest from printing the docstrings in test cases."""
        return None

    def setUp(self):
        """Instantiates class before each test."""
        self.fit_structure = FitStructure(self.atoms, self.tag, self.cv, self.prop)

    def test_init(self):
        """Tests that initialization of tested class works."""
        atoms = self.prim.repeat(2)
        tag = 'struct1'
        self.fit_structure = FitStructure(atoms, tag, [1, 2, 3, 4])

    def test_cluster_vector(self):
        """Tests cluster vector attribute."""
        atoms = self.prim.repeat(2)
        cv_from_cluster_space = list(self.cs.get_cluster_vector(atoms))
        cv = list(self.fit_structure.cluster_vector)
        self.assertEqual(cv, cv_from_cluster_space)

    def test_atoms(self):
        """Tests atoms attribute."""
        atoms = self.fit_structure.atoms
        self.assertTrue(isinstance(atoms, Atoms))

    def test_user_tag(self):
        """Tests user_tag attribute."""
        user_tag = self.fit_structure.user_tag
        self.assertTrue(isinstance(user_tag, str))

    def test_properties(self):
        """Tests properties attribute."""
        properties = self.fit_structure.properties
        self.assertTrue(isinstance(properties, dict))

    def test_getattr(self):
        """Tests custom getattr function."""
        properties = dict(energy=2.123, nvac=48, c=[0.5, 0.5], fname='asd.xml')
        fs = FitStructure(self.atoms, self.tag, self.cv, properties)

        # test the function call
        for key, val in properties.items():
            self.assertEqual(fs.__getattr__(key), val)

        # test the attributes
        self.assertEqual(fs.energy, properties['energy'])
        self.assertEqual(fs.nvac, properties['nvac'])
        self.assertEqual(fs.c, properties['c'])
        self.assertEqual(fs.fname, properties['fname'])

        # test regular attribute call
        fs.properties
        fs.atoms
        with self.assertRaises(AttributeError):
            fs.hello_world


if __name__ == '__main__':
    unittest.main()
