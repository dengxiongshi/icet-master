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


class MIPCluster:

    def __init__(self, orbit_index, cluster_sites, active):
        self.orbit_index = orbit_index
        self.cluster_sites = cluster_sites
        self.active = active
        self.variable = None


class MIPSite:

    def __init__(self, sublattice_index, variable, allowed_symbols, active):
        self.sublattice_index = sublattice_index
        self.variable = variable
        self.allowed_symbols = allowed_symbols
        self.active = active


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

        >>> from ase.build import bulk
        >>> from icet import ClusterExpansion, ClusterSpace

        >>> # prepare cluster expansion
        >>> # the setup emulates a second nearest-neighbor (NN) Ising model
        >>> # (zerolet and singlet ECIs are zero; only first and second neighbor
        >>> # pairs are included)
        >>> prim = bulk('Au')
        >>> chemical_symbols = ['Ag', 'Au']
        >>> cs = ClusterSpace(prim, cutoffs=[4.3], chemical_symbols=chemical_symbols)
        >>> ce = ClusterExpansion(cs, [0, 0, 0.1, -0.02])

        >>> # prepare initial configuration
        >>> structure = prim.repeat(3)

        >>> # set up the ground state finder and calculate the ground state energy
        >>> gsf = GroundStateFinder(ce, structure)
        >>> ground_state = gsf.get_ground_state({'Ag': 5})
        >>> print('Ground state energy:', ce.predict(ground_state))
    """

    def __init__(self,
                 cluster_expansion: ClusterExpansion,
                 structure: Atoms,
                 active_indices: List[int] = None,
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

        if active_indices is None:
            self._active_indices = [atom.index for atom in structure]
        else:
            self._active_indices = active_indices

        # Check that there are no more than two allowed species
        active_species = [
            subl.chemical_symbols for subl in self._active_sublattices]
        if any(len(species) > 2 for species in active_species):
            raise NotImplementedError('Currently, systems with more than two allowed species on '
                                      'any sublattice are not supported.')
        self._active_species = active_species

        # Define cluster functions for elements
        self._symbol_to_variable = []
        self._variable_to_symbol = []
        for species in active_species:
            for species_map in cluster_space.species_maps:
                symbols = [periodic_table[n] for n in species_map.keys()]
                if set(symbols) == set(species):
                    sym_to_var = {periodic_table[
                        n]: 1 - species_map[n] for n in species_map.keys()}
                    self._symbol_to_variable.append(sym_to_var)
                    self._variable_to_symbol.append(
                        {value: key for key, value in sym_to_var.items()})
                    break

        # Generate full orbit list
        self._orbit_list = cluster_space.orbit_list
        lolg = LocalOrbitListGenerator(
            orbit_list=self._orbit_list,
            structure=Structure.from_atoms(primitive_structure),
            fractional_position_tolerance=self._fractional_position_tolerance)
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

        # Initiate MIP model
        model = mip.Model('CE', solver_name=solver_name)
        model.solver.set_mip_gap(0)   # avoid stopping prematurely
        model.solver.set_emphasis(2)  # focus on finding optimal solution
        model.preprocess = 2          # maximum preprocessing

        # Set verbosity
        model.verbose = int(verbose)

        # Create sites
        self._create_mip_sites(structure, model)

        # Create clusters
        self._create_mip_clusters(structure, model)

        # The objective function is added to 'model' first
        model.objective = mip.minimize(mip.xsum(self._get_total_energy()))

        # The five constraints are entered
        # TODO: don't create cluster constraints for singlets
        constraint_count = 0
        for cluster in self._clusters:
            if not cluster.active:
                continue

            ECI = self._transformed_parameters[cluster.orbit_index + 1]
            assert ECI != 0

            if len(cluster.cluster_sites) < 2 or ECI < 0:  # no "downwards" pressure
                for site in cluster.cluster_sites:
                    if site in self._active_indices:
                        model.add_constr(cluster.variable <= model.var_by_name('atom_{}'.format(site)),
                                     'Decoration -> cluster {}'.format(constraint_count))
                        constraint_count += 1

            if len(cluster.cluster_sites) < 2 or ECI > 0:  # no "upwards" pressure
                site_variables = []
                inactive_count = 0
                for site in cluster.cluster_sites:
                    if site in self._active_indices:
                        site_variables.append(model.var_by_name('atom_{}'.format(site)))
                    else:
                        mip_site = self._sites[site]
                        inactive_count += mip_site.variable
                        print('hej', inactive_count)
                model.add_constr(cluster.variable >= 1 - len(cluster.cluster_sites) +
                                 + inactive_count + mip.xsum(site_variables),
                                 'Decoration -> cluster {}'.format(constraint_count))
                constraint_count += 1

        # Update the model so that variables and constraints can be queried
        if model.solver_name.upper() in ['GRB', 'GUROBI']:
            model.solver.update()
        return model

    def _create_mip_clusters(self, structure: Atoms, model) -> None:
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
        lolg = LocalOrbitListGenerator(
            orbit_list=self._orbit_list,
            structure=Structure.from_atoms(structure),
            fractional_position_tolerance=self._fractional_position_tolerance)
        full_orbit_list = lolg.generate_full_orbit_list()

        # Create clusters
        nclusters_per_orbit = [1.0]  # zerolet
        clusters = []
        for orb_index in range(len(full_orbit_list)):

            equivalent_clusters = full_orbit_list.get_orbit(
                orb_index).get_equivalent_sites()
            nclusters_per_orbit.append(0)

            # Determine the sites and the orbit associated with each cluster
            for cluster in equivalent_clusters:

                # Do not include clusters for which the ECI is 0
                ECI = self._transformed_parameters[orb_index + 1]
                if ECI == 0:
                    continue

                # Add the the list of sites and the orbit to the respective
                # cluster maps
                cluster_sites = [site.index for site in cluster]
                active = False
                for site_index in cluster_sites:
                    if site_index not in self._active_indices:
                        site = self._sites[site_index]
                        var = self._symbol_to_variable[site.sublattice_index][structure[site_index].symbol]
                        if var == 0:
                            active = False
                            break
                    else:
                        active = True

                cluster = MIPCluster(orbit_index=orb_index,
                                     cluster_sites=cluster_sites,
                                     active=active)
                if active:
                    var = model.add_var(
                        name='cluster_{}'.format(len(clusters)), var_type=BINARY)
                else:
                    var = 1
                    for site_index in cluster_sites:
                        site = self._sites[site_index]
                        var *= self._symbol_to_variable[site.sublattice_index][structure[site_index].symbol]
                cluster.variable = var
                clusters.append(cluster)
                nclusters_per_orbit[-1] += 1

        self._clusters = clusters
        self._nclusters_per_orbit = nclusters_per_orbit

    def _create_mip_sites(self, structure, model):

        # Spin variables (remapped) for all atoms in the structure
        sites = {}
        for j, sublattice in enumerate(self._active_sublattices):
            for i in sublattice.indices:
                if i in self._active_indices:
                    active = True
                    x = model.add_var(name='atom_{}'.format(i), var_type=BINARY)
                else:
                    active = False
                    x = self._symbol_to_variable[j][structure[i].symbol]
                site = MIPSite(sublattice_index=j,
                               variable=x, allowed_symbols=self._active_species[j],
                               active=active)
                sites[i] = site
        self._sites = sites

    def _get_total_energy(self) -> List[float]:
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
        for cluster in self._clusters:
            E[cluster.orbit_index + 1] += cluster.variable
        E[0] = 1

        E = [0.0 if np.isclose(self._transformed_parameters[orbit], 0.0) else
             E[orbit] * self._transformed_parameters[orbit] /
             self._nclusters_per_orbit[orbit]
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
        
        all_active_species = [
            symbol for species in self._active_species for symbol in species]
        for symbol in species_count.keys():
            if symbol not in all_active_species:
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

        # The model is solved using python-MIPs choice of solver, which is
        # Gurobi, if available, and COIN-OR Branch-and-Cut, otherwise.
        model = self._model

        # Set species constraint
        for variables in self._symbol_to_variable:
            for sym, variable in variables.items():
                if sym in species_to_count:
                    xs_symbol = [site.variable for site in self._sites.values()
                        if sym in site.allowed_symbols]
                    if variable == 1:
                        xcount = species_count[symbol]
                    else:
                        xcount = len(xs_symbol) - species_count[symbol]
                    model.add_constr(mip.xsum(xs_symbol) == xcount, '{} count'.format(sym))

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

        # Translate solution to Atoms object
        gs = self.structure.copy()
        for site_index, site in self._sites.items():
            if site.active:
                gs[site_index].symbol = self._variable_to_symbol[
                    site.sublattice_index][site.variable.x]

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
