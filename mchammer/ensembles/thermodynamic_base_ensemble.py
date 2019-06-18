from abc import abstractproperty
from typing import Dict, List

import numpy as np

from ase import Atoms
from ase.units import kB
from ase.data import chemical_symbols

from .. import DataContainer
from ..calculators.base_calculator import BaseCalculator
from .base_ensemble import BaseEnsemble


class ThermodynamicBaseEnsemble(BaseEnsemble):
    """
    Thermodynamic base ensemble class.

    Parameters
    ----------
    atoms : :class:`Atoms <ase.Atoms>`
        atomic configuration to be used in the Monte Carlo simulation;
        also defines the initial occupation vector
    calculator : :class:`BaseCalculator <mchammer.calculators.ClusterExpansionCalculator>`
        calculator to be used for calculating the potential changes
        that enter the evaluation of the Metropolis criterion
    boltzmann_constant : float
        Boltzmann constant :math:`k_B` in appropriate
        units, i.e. units that are consistent
        with the underlying cluster expansion
        and the temperature units [default: eV/K]
    user_tag : str
        human-readable tag for ensemble [default: None]
    data_container : str
        name of file the data container associated with the ensemble
        will be written to; if the file exists it will be read, the
        data container will be appended, and the file will be
        updated/overwritten
    random_seed : int
        seed for the random number generator used in the Monte Carlo
        simulation
    ensemble_data_write_interval : int
        interval at which data is written to the data container; this
        includes for example the current value of the calculator
        (i.e. usually the energy) as well as ensembles specific fields
        such as temperature or the number of atoms of different species
    data_container_write_period : float
        period in units of seconds at which the data container is
        written to file; writing periodically to file provides both
        a way to examine the progress of the simulation and to back up
        the data [default: np.inf]
    trajectory_write_interval : int
        interval at which the current occupation vector of the atomic
        configuration is written to the data container.
    """

    def __init__(self, atoms: Atoms, calculator: BaseCalculator,
                 user_tag: str = None,
                 boltzmann_constant: float = kB,
                 data_container: DataContainer = None, random_seed: int = None,
                 data_container_write_period: float = np.inf,
                 ensemble_data_write_interval: int = None,
                 trajectory_write_interval: int = None) -> None:

        self._boltzmann_constant = boltzmann_constant

        super().__init__(
            atoms=atoms, calculator=calculator, user_tag=user_tag,
            data_container=data_container,
            random_seed=random_seed,
            data_container_write_period=data_container_write_period,
            ensemble_data_write_interval=ensemble_data_write_interval,
            trajectory_write_interval=trajectory_write_interval)

    @abstractproperty
    @property
    def temperature(self) -> float:
        pass

    @property
    def boltzmann_constant(self) -> float:
        """ Boltzmann constant :math:`k_B` (see parameters section above) """
        return self._boltzmann_constant

    def _acceptance_condition(self, potential_diff: float) -> bool:
        """
        Evaluates Metropolis acceptance criterion.

        Parameters
        ----------
        potential_diff
            change in the thermodynamic potential associated
            with the trial step
        """
        if potential_diff < 0:
            return True
        elif abs(self.temperature) < 1e-6:  # temperature is numerically zero
            return False
        else:
            p = np.exp(-potential_diff / (self.boltzmann_constant * self.temperature))
            return p > self._next_random_number()

    def do_canonical_swap(self, sublattice_index: int, allowed_species: List[int] = None) -> None:
        """ Carries out one Monte Carlo trial step.

        Parameters
        ---------
        sublattice_index
            the sublattice the swap will act on
        allowed_species
            list of atomic numbers for allowed species
         """
        self._total_trials += 1
        sites, species = self.configuration.get_swapped_state(sublattice_index, allowed_species)
        potential_diff = self._get_property_change(sites, species)

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations(sites, species)

    def do_sgc_flip(self, chemical_potentials: Dict[int, float], sublattice_index: int,
                    allowed_species: List[int] = None) -> None:
        """ Carries out one Monte Carlo trial step.

        chemical_potentials
            chemical potentials used to calculate the potential
             difference
        sublattice_index
            the sublattice the flip will act on
        allowed_species
            list of atomic numbers for allowed species
        """
        self._total_trials += 1
        index, species = self.configuration.get_flip_state(sublattice_index, allowed_species)
        potential_diff = self._get_property_change([index], [species])

        # change in chemical potential
        old_species = self.configuration.occupations[index]
        chemical_potential_diff = chemical_potentials[old_species] - chemical_potentials[species]
        potential_diff += chemical_potential_diff

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations([index], [species])

    def do_vcsgc_flip(self, phis: Dict[int, float], kappa: float, sublattice_index: int,
                      allowed_species: List[int] = None) -> None:
        """Carries out one Monte Carlo trial step.

        Parameters
        ----------
        phis
            average constraint parameters
        kappa
            parameter that constrains the variance of the concentration
        sublattice_index
            the sublattice the flip will act on
        allowed_species
            list of atomic numbers for allowed species
        """
        self._total_trials += 1
        index, new_species = self.configuration.get_flip_state(
            sublattice_index, allowed_species)
        old_species = self.configuration.occupations[index]

        # Calculate difference in VCSGC thermodynamic potential.
        # Note that this assumes that only one atom was flipped.
        sl_occupations = self.configuration.get_occupations_on_sublattice(sublattice_index)
        N = len(sl_occupations)
        potential_diff = 1.0  # dN
        for species in phis:
            if species == old_species:
                factor = -1
            elif species == new_species:
                factor = 1
            else:
                continue
            potential_diff += factor * (N * phis[species] + 2 * sl_occupations.count(species))
        potential_diff *= kappa * self.boltzmann_constant * self.temperature / N
        potential_diff += self._get_property_change([index], [new_species])

        if self._acceptance_condition(potential_diff):
            self._accepted_trials += 1
            self.update_occupations([index], [new_species])

    def _get_swap_sublattice_probabilities(self) -> List[float]:
        """ Returns sublattice probabilities suitable for swaps."""
        sublattice_probabilities = []
        for i, sl in enumerate(self.sublattices):
            if self.configuration.is_swap_possible(i):
                sublattice_probabilities.append(len(sl.indices))
            else:
                sublattice_probabilities.append(0)
        norm = sum(sublattice_probabilities)
        if norm == 0:
            raise ValueError('No canonical swaps are possible on any of the active sublattices.')
        sublattice_probabilities = [p / norm for p in sublattice_probabilities]
        return sublattice_probabilities

    def _get_flip_sublattice_probabilities(self) -> List[float]:
        """Returns the default sublattice probability which is based on
        the sizes of a sublattice.
        """
        probability_distribution = []
        for i, sl in enumerate(self.sublattices):
            if len(sl.chemical_symbols) > 1:
                probability_distribution.append(len(sl.indices))
            else:
                probability_distribution.append(0)

        norm = sum(probability_distribution)
        probability_distribution = [p / norm for p in probability_distribution]
        return probability_distribution

    def _get_vcsgc_free_energy_derivatives(self, phis: Dict[int, str], kappa: float,
                                           sublattice_index: int = None) -> Dict:
        """
        Returns a dict with the free energy derivatives.

        Parameters
        ----------
        phis
            average constraint parameters
        kappa
            parameter that constrains the variance of the concentration
        sublattice_index
            sublattice index
        """
        data = {}

        for atnum in phis:
            for i, sublattice in enumerate(self.sublattices):
                if sublattice_index is not None and i != sublattice_index:
                    continue
                if len(sublattice.chemical_symbols) > 0 and atnum in sublattice.atomic_numbers:
                    N = len(sublattice.indices)
                    sl_occupations = self.configuration.get_occupations_on_sublattice(i)
                    concentration = sl_occupations.count(atnum) / N
                    data['free_energy_derivative_{}'.format(chemical_symbols[atnum])] \
                        = kappa * self.boltzmann_constant * self.temperature * \
                        (- 2 * concentration - phis[atnum])

        return data

    def _get_species_counts(self) -> Dict:
        """
        Returns a dict with the species counts.
        """
        data = {}
        atoms = self.configuration.atoms
        unique, counts = np.unique(atoms.numbers, return_counts=True)
        for sl in self.sublattices:
            for symbol in sl.chemical_symbols:
                data['{}_count'.format(symbol)] = 0
        for atnum, count in zip(unique, counts):
            data['{}_count'.format(chemical_symbols[atnum])] = count

        return data