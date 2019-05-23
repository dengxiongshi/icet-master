from typing import Dict, List


from ase import Atoms
from icet import ClusterSpace
from mchammer.observers import ClusterCountObserver
from mchammer.observers.base_observer import BaseObserver


class PairCountObserver(BaseObserver):
    """
    This class represents a short range order (SRO) observer for a
    binary system.


    Parameters
    ----------
    cluster_space : icet.ClusterSpace
        cluster space used for initialization
    structure : ase.Atoms
        defines the lattice which the observer will work on
    interval : int
        observation interval during the Monte Carlo simulation
    rmin : float
        the minimum radius for the neigbhor shells considered
    rmax : float
        the maximum radius for the neigbhor shells considered
    tag : str
        human readable observer name
    interval : int
        observation interval

    """

    def __init__(self, cluster_space, structure: Atoms,
                 interval: int, rmin: float, rmax: float, tag: str, species: List[str]) -> None:
        super().__init__(interval=interval, return_type=int,
                         tag=tag)

        self._structure = structure.copy()
        self._structure.wrap()
        self._species = sorted(species)

        self._cluster_space = ClusterSpace(
            atoms=cluster_space.primitive_structure,
            cutoffs=[rmax], chemical_symbols=cluster_space.chemical_symbols)
        del_orbits = []
        for i, orbit in enumerate(self._cluster_space.orbit_list.orbits):
            radius = orbit.get_representative_cluster().radius
            if rmin > 2*radius > rmax:
                del_orbits.append(i+1)  # plus 1 to account for the zerolet
        self._cluster_space._prune_orbit_list(del_orbits)
        self._cluster_count_observer = ClusterCountObserver(
            cluster_space=self._cluster_space, atoms=structure,
            interval=interval)

    def get_observable(self, atoms: Atoms) -> Dict[str, float]:
        """Returns the value of the property from a cluster expansion
        model for a given atomic configurations.

        Parameters
        ----------
        atoms
            input atomic structure
        """
        atoms.wrap()

        self._cluster_count_observer._generate_counts(atoms)
        df = self._cluster_count_observer.count_frame

        pair_orbit_indices = set(df.loc[df['order'] == 2]['orbit_index'].tolist())
        pair_count = 0
        for k, orbit_index in enumerate(sorted(pair_orbit_indices)):
            orbit_df = df.loc[df['orbit_index'] == orbit_index]
            for i, row in orbit_df.iterrows():
                if self._species == sorted(row.decoration):
                    pair_count += row.cluster_count

        return pair_count
