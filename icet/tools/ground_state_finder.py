from math import inf
import numpy as np
from typing import List, Dict

from ase import Atoms
from ase.data import chemical_symbols as periodic_table
from .. import ClusterExpansion
from ..core.local_orbit_list_generator import LocalOrbitListGenerator
from ..core.structure import Structure
from .variable_transformation import transform_ECIs
from ..input_output.logging_tools import logger
from pkg_resources import VersionConflict

try:
    import mip
    from mip.constants import BINARY
    from distutils.version import LooseVersion

    if LooseVersion(mip.constants.VERSION) < '1.6.3':
        raise VersionConflict('Python-MIP version 1.6.3 or later is required in order to use the '
                              'ground state finder.')
except ImportError:
    raise ImportError('Python-MIP (https://python-mip.readthedocs.io/en/latest/) is required in '
                      'order to use the ground state finder.')


class GroundStateFinder:
    """
    This class provides functionality for determining the ground states
    using a binary cluster expansion. This is efficiently achieved through the
    use of mixed integer programming (MIP) as shown by Larsen *et al.* in
    `Phys. Rev. Lett. 120, 256101 (2018)
    <https://doi.org/10.1103/PhysRevLett.120.256101>`_.

    This class relies on the `Python-MIP package
    <https://python-mip.readthedocs.io>`_. Python-MIP can be used together
    with `Gurobi <https://www.gurobi.com/>`_, which is not open source
    but issues academic licenses free of charge. Pleaase note that
    Gurobi needs to be installed separately. The `GroundStateFinder` works
    also without Gurobi, but if performance is critical, Gurobi is highly
    recommended.

    Warning
    -------
    In order to be able to use Gurobi with python-mip one must ensure that
    `GUROBI_HOME` should point to the installation directory
    (``<installdir>``)::

        export GUROBI_HOME=<installdir>

    Note
    ----
    The current implementation only works for binary systems.


    Parameters
    ----------
    cluster_expansion : ClusterExpansion
        cluster expansion for which to find ground states
    structure : Atoms
        atomic configuration
    solver_name : str, optional
        'gurobi', alternatively 'grb', or 'cbc', searches for available
        solvers if not informed
    verbose : bool, optional
        whether to display solver messages on the screen
        (default: True)


    Example
    -------
    The following snippet illustrates how to determine the ground state for a
    Au-Ag alloy. Here, the parameters of the cluster
    expansion are set to emulate a simple Ising model in order to obtain an
    example that can be run without modification. In practice, one should of
    course use a proper cluster expansion::

        from ase.build import bulk
        from icet import ClusterExpansion, ClusterSpace
        from icet.tools.ground_state_finder import GroundStateFinder

        # prepare cluster expansion
        # the setup emulates a second nearest-neighbor (NN) Ising model
        # (zerolet and singlet ECIs are zero; only first and second neighbor
        # pairs are included)
        prim = bulk('Au')
        chemical_symbols = ['Ag', 'Au']
        cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=chemical_symbols)
        ce = ClusterExpansion(cs, [0, 0, 0.1, -0.02])

        # prepare initial configuration
        structure = prim.repeat(3)

        # set up the ground state finder and calculate the ground state energy
        gsf = GroundStateFinder(ce, structure)
        ground_state = gsf.get_ground_state({'Ag': 5})
        print('Ground state energy:', ce.predict(ground_state))
    """

    def __init__(self,
                 cluster_expansion: ClusterExpansion,
                 structure: Atoms,
                 solver_name: str = None,
                 verbose: bool = True) -> None:
        # Check that there is only one active sublattice
        self._cluster_expansion = cluster_expansion
        self._fractional_position_tolerance = cluster_expansion.fractional_position_tolerance
        self.structure = structure
        cluster_space = self._cluster_expansion.get_cluster_space_copy()
        primitive_structure = cluster_space.primitive_structure
        sublattices = cluster_space.get_sublattices(structure)
        self._active_sublattices = sublattices.active_sublattices

        # Check that there are no more than two allowed species
        active_species = [subl.chemical_symbols for subl in self._active_sublattices]
        if any(len(species) > 2 for species in active_species):
            raise NotImplementedError('Currently, systems with more than two allowed species on '
                                      'any sublattice are not supported.')
        self._active_species = active_species
        self._active_indices = [subl.indices for subl in self._active_sublattices]
        self._all_active_species = [symbol for species in active_species for symbol in species]

        # Define cluster functions for elements
        self._id_maps = []
        self._reverse_id_maps = []
        for species in active_species:
            for species_map in cluster_space.species_maps:
                symbols = [periodic_table[n] for n in species_map.keys()]
                if set(symbols) == set(species):
                    id_map = {periodic_table[n]: 1 - species_map[n] for n in species_map.keys()}
                    reverse_id_map = {value: key for key, value in id_map.items()}
                    self._id_maps.append(id_map)
                    self._reverse_id_maps.append(reverse_id_map)
                    break
        self._count_symbols = [reverse_id_map[1] for reverse_id_map in self._reverse_id_maps]

        # Generate full orbit list
        self._orbit_list = cluster_space.orbit_list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(primitive_structure),
                                       self._fractional_position_tolerance)
        self._full_orbit_list = lolg.generate_full_orbit_list()

        # Transform the ECIs
        binary_ecis = transform_ECIs(primitive_structure,
                                     self._full_orbit_list,
                                     self._cluster_expansion.parameters)
        self._transformed_parameters = binary_ecis

        # Build model
        if solver_name is None:
            solver_name = ''
        self._model = self._build_model(structure, solver_name, verbose)

        # Properties that are defined when searching for a ground state
        self._optimization_status = None

    def _build_model(self,
                     structure: Atoms,
                     solver_name: str,
                     verbose: bool) -> mip.Model:
        """
        Build a Python-MIP model based on the provided structure

        Parameters
        ----------
        structure
            atomic configuration
        solver_name
            'gurobi', alternatively 'grb', or 'cbc', searches for
            available solvers if not informed
        verbose
            whether to display solver messages on the screen
        """

        # Create cluster maps
        self._create_cluster_maps(structure)

        # Initiate MIP model
        model = mip.Model('CE', solver_name=solver_name)
        model.solver.set_mip_gap(0)   # avoid stopping prematurely
        model.solver.set_emphasis(2)  # focus on finding optimal solution
        model.preprocess = 2          # maximum preprocessing

        # Set verbosity
        model.verbose = int(verbose)

        # Spin variables (remapped) for all atoms in the structure
        xs = []
        site_to_active_index_map = {}
        symbol_to_active_indices = {count_symbol: [] for count_symbol in self._count_symbols}
        active_index_to_sublattice_map = {}
        for i in range(len(structure)):
            for j, indices in enumerate(self._active_indices):
                if i in indices:
                    site_to_active_index_map[i] = len(xs)
                    symbol_to_active_indices[self._count_symbols[j]].append(len(xs))
                    active_index_to_sublattice_map[i] = j
                    xs.append(model.add_var(name='atom_{}'.format(i), var_type=BINARY))
                    break
        self.xs = xs
        self._active_index_to_sublattice_map = active_index_to_sublattice_map

        ys = []
        for i in range(len(self._cluster_to_orbit_map)):
            ys.append(model.add_var(name='cluster_{}'.format(i), var_type=BINARY))

        # The objective function is added to 'model' first
        model.objective = mip.minimize(mip.xsum(self._get_total_energy(ys)))

        # The five constraints are entered
        # TODO: don't create cluster constraints for singlets
        constraint_count = 0
        for i, cluster in enumerate(self._cluster_to_sites_map):
            orbit = self._cluster_to_orbit_map[i]
            ECI = self._transformed_parameters[orbit + 1]
            assert ECI != 0

            if len(cluster) < 2 or ECI < 0:  # no "downwards" pressure
                for atom in cluster:
                    model.add_constr(ys[i] <= xs[site_to_active_index_map[atom]],
                                     'Decoration -> cluster {}'.format(constraint_count))
                    constraint_count += 1

            if len(cluster) < 2 or ECI > 0:  # no "upwards" pressure
                model.add_constr(ys[i] >= 1 - len(cluster) +
                                 mip.xsum(xs[site_to_active_index_map[atom]]
                                          for atom in cluster),
                                 'Decoration -> cluster {}'.format(constraint_count))
                constraint_count += 1

        # Set species constraint
        for sym in self._count_symbols:
            xs_symbol = [xs[i] for i in symbol_to_active_indices[sym]]
            model.add_constr(mip.xsum(xs_symbol) == -1, '{} count'.format(sym))

        # Update the model so that variables and constraints can be queried
        if model.solver_name.upper() in ['GRB', 'GUROBI']:
            model.solver.update()
        return model

    def _create_cluster_maps(self, structure: Atoms) -> None:
        """
        Create maps that include information regarding which sites and orbits
        are associated with each cluster as well as the number of clusters per
        orbit

        Parameters
        ----------
        structure
            atomic configuration
        """
        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(structure),
                                       self._fractional_position_tolerance)
        full_orbit_list = lolg.generate_full_orbit_list()

        # Create maps of site indices and orbits for all clusters
        cluster_to_sites_map = []
        cluster_to_orbit_map = []
        for orb_index in range(len(full_orbit_list)):

            equivalent_clusters = full_orbit_list.get_orbit(orb_index).get_equivalent_sites()

            # Determine the sites and the orbit associated with each cluster
            for cluster in equivalent_clusters:

                # Do not include clusters for which the ECI is 0
                ECI = self._transformed_parameters[orb_index + 1]
                if ECI == 0:
                    continue

                # Add the the list of sites and the orbit to the respective cluster maps
                cluster_sites = [site.index for site in cluster]
                cluster_to_sites_map.append(cluster_sites)
                cluster_to_orbit_map.append(orb_index)

        # calculate the number of clusters per orbit
        nclusters_per_orbit = [cluster_to_orbit_map.count(i) for i in
                               range(cluster_to_orbit_map[-1] + 1)]
        nclusters_per_orbit = [1] + nclusters_per_orbit

        self._cluster_to_sites_map = cluster_to_sites_map
        self._cluster_to_orbit_map = cluster_to_orbit_map
        self._nclusters_per_orbit = nclusters_per_orbit

    def _get_total_energy(self, cluster_instance_activities: List[int]) -> List[float]:
        """
        Calculates the total energy using the expression based on binary
        variables

        .. math::

            H({\\boldsymbol x}, {\\boldsymbol E})=E_0+
            \\sum\\limits_j\\sum\\limits_{{\\boldsymbol c}
            \\in{\\boldsymbol C}_j}E_jy_{{\\boldsymbol c}},

        where (:math:`y_{{\\boldsymbol c}}=
        \\prod\\limits_{i\\in{\\boldsymbol c}}x_i`).

        Parameters
        ----------
        cluster_instance_activities
            list of cluster instance activities, (:math:`y_{{\\boldsymbol c}}`)
        """

        E = [0.0 for _ in self._transformed_parameters]
        for i in range(len(cluster_instance_activities)):
            orbit = self._cluster_to_orbit_map[i]
            E[orbit + 1] += cluster_instance_activities[i]
        E[0] = 1

        E = [0.0 if np.isclose(self._transformed_parameters[orbit], 0.0) else
             E[orbit] * self._transformed_parameters[orbit] / self._nclusters_per_orbit[orbit]
             for orbit in range(len(self._transformed_parameters))]
        return E

    def get_ground_state(self,
                         species_count: Dict[str, int],
                         max_seconds: float = inf,
                         threads: int = 0) -> Atoms:
        """
        Finds the ground state for a given structure and species count, which
        refers to the `count_species`, if provided when initializing the
        instance of this class, or the first species in the list of chemical
        symbols for the active sublattice.

        Parameters
        ----------
        species_count
            dictionary with count for one of the species on each active
            sublattice
        max_seconds
            maximum runtime in seconds (default: inf)
        threads
            number of threads to be used when solving the problem, given that a
            positive integer has been provided. If set to 0 the solver default
            configuration is used while -1 corresponds to all available
            processing cores.
        """
        # Check that the species_count is consistent with the cluster space
        for symbol in species_count.keys():
            if symbol not in self._all_active_species:
                raise ValueError('The species {} is not present on any of the active sublattices'
                                 ' ({})'.format(symbol, self._active_species))
        species_to_count = []
        for i, species in enumerate(self._active_species):
            symbols_to_add = [sym for sym in species_count if sym in species]
            if len(symbols_to_add) != 1:
                raise ValueError('Provide counts for one of the species on each active sublattice '
                                 '({}), not {}!'.format(self._active_species,
                                                        list(species_count.keys())))
            species_to_count += symbols_to_add
        for i, species in enumerate(species_to_count):
            count = species_count[species]
            max_count = len(self._active_sublattices[i].indices)
            if count < 0 or count > max_count:
                raise ValueError('The count for species {} ({}) must be a positive integer and '
                                 'cannot exceed the number of sites on the active sublattice '
                                 '({})'.format(species, count, max_count))

        # Determine the maximum species count for each sublattice
        species_constr_xcount = {}
        for j, id_map in enumerate(self._id_maps):
            if id_map[species_to_count[j]] == 1:
                xcount = species_count[species_to_count[j]]
            else:
                active_count = len([i for i in range(len(self.structure)) if i in
                                    self._active_indices[j]])
                xcount = active_count - species_count[species_to_count[j]]
            species_constr = '{} count'.format(self._count_symbols[j])
            species_constr_xcount[species_constr] = xcount

        # The model is solved using python-MIPs choice of solver, which is
        # Gurobi, if available, and COIN-OR Branch-and-Cut, otherwise.
        model = self._model

        # Update the species counts
        for species_constr, xcount in species_constr_xcount.items():
            model.constr_by_name(species_constr).rhs = xcount

        # Set the number of threads
        model.threads = threads

        # Optimize the model
        self._optimization_status = model.optimize(max_seconds=max_seconds)

        # The status of the solution is printed to the screen
        if str(self._optimization_status) != 'OptimizationStatus.OPTIMAL':
            if str(self._optimization_status) == 'OptimizationStatus.FEASIBLE':
                logger.warning('Solution optimality not proven.')
            else:
                raise Exception('No solution found.')

        # Each of the variables is printed with it's resolved optimum value
        gs = self.structure.copy()

        for v in model.vars:
            if 'atom' in v.name:
                index = int(v.name.split('_')[-1])
                gs[index].symbol = self._reverse_id_maps[
                    self._active_index_to_sublattice_map[index]][int(v.x)]

        # Assert that the solution agrees with the prediction
        prediction = self._cluster_expansion.predict(gs)
        assert abs(model.objective_value - prediction) < 1e-6

        return gs

    @property
    def optimization_status(self) -> mip.OptimizationStatus:
        """Optimization status"""
        return self._optimization_status

    @property
    def model(self) -> mip.Model:
        """Python-MIP model"""
        return self._model.copy()
