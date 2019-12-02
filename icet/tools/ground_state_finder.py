from math import inf
from typing import List, Dict

from ase import Atoms
from ase.data import chemical_symbols as periodic_table
from .. import ClusterExpansion
from ..core.orbit_list import OrbitList
from ..core.local_orbit_list_generator import LocalOrbitListGenerator
from ..core.structure import Structure
from .variable_transformation import transform_ECIs
from ..io.logging import logger

try:
    import mip
    from mip.constants import BINARY
except ImportError:
    raise ImportError('Python-MIP '
                      '(https://python-mip.readthedocs.io/en/latest/) is '
                      'required in order to use the ground state finder.')


class GSCluster:

    def __init__(self, sites, orbit_index, active=True):
        self.sites = sites
        self.orbit_index = orbit_index
        self.active = active


class GSClusters:

    def __init__(self):
        self.clusters = []
        self.nclusters_per_orbit = {}

    def add_cluster(self, sites, orbit_index, active):
        self.add_to_multiplicity(orbit_index)
        self.clusters.append(GSCluster(sites, orbit_index, active))

    def add_to_multiplicity(self, orbit_index):
        if orbit_index not in self.nclusters_per_orbit:
            self.nclusters_per_orbit[orbit_index] = 1
        else:
            self.nclusters_per_orbit[orbit_index] += 1

    def generate_active_clusters(self):
        for cluster in self.clusters:
            if cluster.active:
                yield cluster


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
                 active_atom_indices=None,
                 solver_name: str = None,
                 verbose: bool = True) -> None:
        # Check that there is only one active sublattice
        self._cluster_expansion = cluster_expansion
        self.structure = structure
        cluster_space = self._cluster_expansion.get_cluster_space_copy()
        primitive_structure = cluster_space.primitive_structure
        sublattices = cluster_space.get_sublattices(primitive_structure)
        if len(sublattices.active_sublattices) > 1:
            raise NotImplementedError('Only binaries are implemented '
                                      'as of yet.')

        # Check that there are no more than two allowed species
        species = list(sublattices.active_sublattices[0].chemical_symbols)
        if len(species) > 2:
            raise NotImplementedError('Only binaries are implemented '
                                      'as of yet.')
        self._species = species

        if active_atom_indices is None:
            self.active_atom_indices = [atom.index for atom in structure]
        else:
            self.active_atom_indices = active_atom_indices

        # Define cluster functions for elements
        species_map = cluster_space.species_maps[0]
        self._id_map = {periodic_table[n]: 1 - species_map[n]
                        for n in species_map.keys()}
        self._reverse_id_map = {}
        for key, value in self._id_map.items():
            self._reverse_id_map[value] = key

        # Generate orbit list
        primitive_structure.set_chemical_symbols(
            [els[0] for els in cluster_space.chemical_symbols])
        cutoffs = cluster_space.cutoffs
        self._orbit_list = OrbitList(primitive_structure, cutoffs)

        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(primitive_structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Determine the number of active orbits
        self.active_orbit_indices = self._get_active_orbit_indices(primitive_structure)

        # Transform the ECIs
        binary_ecis = transform_ECIs(primitive_structure,
                                     full_orbit_list,
                                     self.active_orbit_indices,
                                     self._cluster_expansion.parameters)
        self._transformed_parameters = binary_ecis

        # Build model
        if solver_name is None:
            solver_name = ''
        self._model = self._build_model(structure, solver_name, verbose)

        # Properties that are defined when searching for a ground state
        self._optimization_status = None

    def _build_model(self, structure: Atoms, solver_name: str,
                     verbose: bool, xcount: int = -1) -> mip.Model:
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
        xcount
            constraint for the species count
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
        for atom in structure:
            if atom.symbol in self._species and atom.index in self.active_atom_indices:
                site_to_active_index_map[atom.index] = len(xs)
                xs.append(model.add_var(name='atom_{}'.format(atom.index),
                                        var_type=BINARY))
        self.xs = xs

        for i, cluster in enumerate(self.gs_clusters.generate_active_clusters()):
            cluster.model_var = model.add_var(name='cluster_{}'.format(i),
                                              var_type=BINARY)

        # The objective function is added to 'model' first
        model.objective = mip.minimize(mip.xsum(self._get_total_energy()))

        # The five constraints are entered
        # TODO: don't create cluster constraints for singlets
        constraint_count = 0
        for i, cluster in enumerate(self.gs_clusters.generate_active_clusters()):
            # Test whether constraint can be binding
            orbit_index = cluster.orbit_index
            ECI = self._transformed_parameters[orbit_index + 1]

            if len(cluster.sites) < 2 or ECI < 0:  # no "downwards" pressure
                for atom_index in cluster.sites:
                    if atom_index in self.active_atom_indices:
                        model.add_constr(cluster.model_var <= xs[site_to_active_index_map[atom_index]],
                                         'Decoration -> cluster {}'.format(constraint_count))
                        constraint_count += 1

            if len(cluster.sites) < 2 or ECI > 0:  # no "upwards" pressure
                constr = 1 - len(cluster.sites)
                to_xsum = []
                for site in cluster.sites:
                    if site in self.active_atom_indices:
                        to_xsum.append(xs[site_to_active_index_map[site]])
                    else:
                        constr += 1
                # model.add_constr(cluster.model_var >= 1 - len(cluster.sites) +
                        # mip.xsum(xs[site_to_active_index_map[site]]
                        #         for site in cluster.sites),
                        #'Decoration -> cluster {}'.format(constraint_count))
                model.add_constr(cluster.model_var >= constr + mip.xsum(to_xsum),
                                 'Decoration -> cluster {}'.format(constraint_count))
                constraint_count += 1

        # Set species constraint
        model.add_constr(mip.xsum(xs) == xcount, 'Species count')

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
                                       Structure.from_atoms(structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Create maps of site indices and orbits for all clusters
        self.gs_clusters = GSClusters()

        orbit_counter = 0
        for i, old_orbit_index in enumerate(self.active_orbit_indices):

            if abs(self._transformed_parameters[i + 1]) < 1e-9:
                continue

            equivalent_clusters = full_orbit_list.get_orbit(
                old_orbit_index).get_equivalent_sites()

            # Determine the sites and the orbit associated with each cluster
            for cluster in equivalent_clusters:

                # Go through all sites in the cluster
                cluster_sites = []
                active = True
                for site in cluster:

                    # Ensure that all sites in the cluster are occupied by allowed elements
                    if structure[site.index].symbol not in self._species:
                        break

                    if site.index not in self.active_atom_indices:
                        if self._id_map[structure[site.index].symbol] == 0:
                            self.gs_clusters.add_to_multiplicity(i)
                            break
                        else:
                            active = True

                    # Add the site to the list of sites for this cluster
                    cluster_sites.append(site.index)

                else:
                    # Add the the list of sites and the orbit to the respective cluster maps
                    self.gs_clusters.add_cluster(cluster_sites, i, active=active)

    def _get_active_orbit_indices(self,  structure: Atoms) -> List[int]:
        """
        Generate a list with the indices of all active orbits

        Parameters
        ----------
        structure
            atomic configuration
        """
        # Generate full orbit list
        lolg = LocalOrbitListGenerator(self._orbit_list,
                                       Structure.from_atoms(structure))
        full_orbit_list = lolg.generate_full_orbit_list()

        # Determine the active orbits
        active_orbit_indices = []
        for i in range(len(full_orbit_list)):
            equivalent_clusters = full_orbit_list.get_orbit(
                i).get_equivalent_sites()
            if all(structure[site.index].symbol in self._species
                   for cluster in equivalent_clusters for site in cluster):
                active_orbit_indices.append(i)

        return active_orbit_indices

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

        activity_sums = [0 for _ in self._transformed_parameters[1:]]
        for cluster in self.gs_clusters.clusters:
            orbit_index = cluster.orbit_index
            if cluster.active:
                activity_sums[orbit_index] += cluster.model_var
            else:
                activity_sums[orbit_index] += 1
        # for i, cluster_instance_activity in enumerate(cluster_instance_activities):
        #    orbit_index = self.gs_clusters.clusters[i].orbit_index
        #    activity_sums[orbit_index + 1] += cluster_instance_activity

        E = [self._transformed_parameters[0]]
        for i in range(len(self.active_orbit_indices)):
            if i not in self.gs_clusters.nclusters_per_orbit:
                continue
            E.append(activity_sums[i] * self._transformed_parameters[i + 1] /
                     self.gs_clusters.nclusters_per_orbit[i])
        return E

    def get_ground_state(self,
                         species_count: Dict[str, float],
                         max_seconds: float=inf) -> Atoms:
        """
        Finds the ground state for a given structure and species count, which
        refers to the `count_species`, if provided when initializing the
        instance of this class, or the first species in the list of chemical
        symbols for the active sublattice.

        Parameters
        ----------
        species_count
            dictionary with count for one of the species on the active
            sublattice
        max_seconds
            maximum runtime in seconds (default: inf)
        """
        # Check that the species_count is consistent with the cluster space
        if len(species_count) != 1:
            raise ValueError('Provide counts for one of the species on the '
                             'active sublattice ({}), '
                             'not {}!'.format(self._species,
                                              list(species_count.keys())))
        species_to_count = list(species_count.keys())[0]
        if species_to_count not in self._species:
            raise ValueError('The species {} is not present on the active '
                             'sublattice'
                             ' ({})'.format(species_to_count, self._species))
        if self._id_map[species_to_count] == 1:
            xcount = species_count[species_to_count]
        else:
            active_count = len([atom.symbol for atom in self.structure
                                if atom.symbol in self._species
                                and atom.index in self.active_atom_indices])
            xcount = active_count - species_count[species_to_count]

        # The model is solved using python-MIPs choice of solver, which is
        # Gurobi, if available, and COIN-OR Branch-and-Cut, otherwise.
        model = self._model

        # Update the species count
        # temporary hack until python-mip supports setting RHS directly:
        if model.solver_name.upper() in ['GUROBI', 'GRB']:
            # remove the old constraint and add a new one
            idx = model.solver.constr_get_index('Species count')
            model.solver.remove_constrs([idx])
            model.add_constr(mip.xsum(self.xs) == xcount, 'Species count')
        else:
            # rebuild the whole model
            self._model = model = self._build_model(self.structure,
                                                    model.solver_name,
                                                    bool(model.verbose),
                                                    xcount=xcount)

        # Optimize the model
        self._optimization_status = model.optimize(max_seconds=max_seconds)

        # The status of the solution is printed to the screen
        if str(self._optimization_status) != 'OptimizationStatus.OPTIMAL':
            logger.warning('No optimal solution found.')

        # Each of the variables is printed with it's resolved optimum value
        gs = self.structure.copy()

        for v in model.vars:
            if 'atom' in v.name:
                index = int(v.name.split('_')[-1])
                gs[index].symbol = self._reverse_id_map[int(v.x)]

        # Assert that the solution agrees with the prediction
        prediction = self._cluster_expansion.predict(gs)

        from ase.io import write
        write('out.xyz', gs)
        print(model.objective_value, prediction)
        print(gs)
        if model.solver_name.upper() in ['GUROBI', 'GRB']:
            assert abs(model.objective_value - prediction) < 1e-6
        elif model.solver_name.upper() == 'CBC':
            assert abs(model.objective_const +
                       model.objective_value - prediction) < 1e-6
        return gs

    @property
    def optimization_status(self) -> mip.OptimizationStatus:
        """Optimization status"""
        return self._optimization_status

    @property
    def model(self) -> mip.Model:
        """Python-MIP model"""
        return self._model.copy()
