from mchammer.calculators.base_calculator import BaseCalculator
from typing import List
from icet import Structure
from icet import ClusterSpace
import numpy as np

class ClusterExpansionCalculator(BaseCalculator):
    """
    Cluster expansion calculator.

    Class for efficiently calculating the
    cluster expansion property
    for a specific structure

    Parameters
    ----------
    atoms : ASE Atoms object
        the structure that the calculator can use
        to optimize the calculate functions.

    cluster_expansion : icet ClusterExpansion object
    name : str
        human readable identifier for this calculator

    Todo
    ----
    * add the real occupation constraints when
      that is setup in the cluster space.

    """

    def __init__(self, atoms, cluster_expansion,
                 name='Cluster Expansion Calculator'):
        super().__init__(atoms=atoms, name=name)

        self._cluster_expansion = cluster_expansion
        self._local_cluster_space = ClusterSpace(self.cluster_expansion.cluster_space._input_atoms.copy(),
                    self.cluster_expansion.cluster_space._cutoffs,
                    self.cluster_expansion.cluster_space._chemical_symbols,
                    self.cluster_expansion.cluster_space._mi,
                    bothways=True)

    @property
    def cluster_expansion(self):
        """
        icet ClusterExpansion object.
        """
        return self._cluster_expansion

    def calculate_total(self, occupations: List[int]):
        """
        Calculator the total property of the current configuration.

        Return
        -------
            total_property : float
        """
        self.atoms.set_atomic_numbers(occupations)
        return self.cluster_expansion.predict(self.atoms)

    def calculate_local_contribution(self, local_indices: List[int] = None,
                                     occupations: List[int] = None):
        """
        Return the sum of the contributions from the indices in the input list.

        Parameters
        ----------
        local_indices : list of ints
            the lattice indices to get the local energy of
        occupations : list of ints
            the occupation vector

        Return
        ------
        float : sum of contributions
        """
        if local_indices is None:
            raise TypeError("Local indices need to be set")
        if occupations is None:
            raise TypeError("occupations need to be set")

        self.atoms.set_atomic_numbers(occupations)
        local_contribution = 0
        for index in local_indices:
            local_contribution += self._calculate_local_contribution(index)

        return local_contribution

    def _calculate_local_contribution(self, index):
        """
        Internal method to calculate the local contribution for one
        index.

        Parameters
        ----------
        index : int
            lattice index

        """
        structure = Structure.from_atoms(self.atoms)
        local_cv = self._local_cluster_space.get_local_cluster_vector(structure, index)
        return np.dot(local_cv, self.cluster_expansion.parameters)

        # return self.calculate_total(self.atoms.numbers)

    @property
    def occupation_constraints(self):
        """A map from site to allowed species."""
        elements = list(
            self.cluster_expansion.cluster_space.element_map.keys())
        return [elements] * len(self.atoms)
