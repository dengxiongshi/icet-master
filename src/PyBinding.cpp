#include "Structure.hpp"
#include "Neighborlist.hpp"
#include "ManybodyNeighborlist.hpp"
#include "Cluster.hpp"
#include "PermutationMap.hpp"
#include "LatticeNeighbor.hpp"
#include "ClusterCounts.hpp"
#include "LocalOrbitlistGenerator.hpp"
#include "ClusterSpace.hpp"
#include <pybind11/pybind11.h>
#include "Symmetry.hpp"
#include "Orbit.hpp"
#include "OrbitList.hpp"
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <pybind11/operators.h>

PYBIND11_PLUGIN(_icetdev)
{
    py::module m("_icetdev", "pybind11 _icetdev plugin");

        // Disable the automatically generated signatures that prepend the
        // docstrings by default.
        py::options options;
        options.disable_function_signatures();

        py::class_<Structure>(m, "Structure",
            "This class stores the cell metric, positions, chemical symbols,"
            " and periodic boundary conditions that describe a structure. It"
            " also holds information pertaining to the components that are"
            " allowed on each site and provides functionality for computing"
            " distances between sites.")
        .def(py::init<>())
        .def(py::init<const Eigen::Matrix<double, Dynamic, 3, Eigen::RowMajor> &,
                      const std::vector<std::string> &,
                      const Eigen::Matrix3d &,
                      const std::vector<bool> &,
                      double>(),
             "Initialize a structure.\n\n"

             ,
             py::arg("positions"),
             py::arg("elements"),
             py::arg("cell"),
             py::arg("pbc"),
             py::arg("precision") = 1e-5)
        .def("get_positions",
             &Structure::getPositions,
             "Returns the positions in Cartesian coordinates.\n"
             "\nReturns\n-------\n"
             "list of NumPy arrays")
        .def("set_positions",
             &Structure::setPositions,
             py::arg("positions"),
             "Set the positions in Cartesian coordinates.\n"
             "\nParameters\n----------\n"
             "positions : list of NumPy arrays\n"
             "    new positions in Cartesian coordinates")
        .def_property("positions",
                      &Structure::getPositions,
                      &Structure::setPositions,
                      "list of lists : atomic positions in Cartesian coordinates")
        .def("get_atomic_numbers",
             &Structure::getAtomicNumbers,
             "Returns a list of the species occupying each site by atomic number.")
        .def("set_atomic_numbers",
             &Structure::setAtomicNumbers,
             py::arg("atomic_numbers"),
             "Sets the species occupying each site by atomic number.\n"
             "\nParameters\n----------\n"
             "atomic_numbers : list of ints\n"
             "    new species by atomic number")
        .def_property("atomic_numbers",
                      &Structure::getAtomicNumbers,
                      &Structure::setAtomicNumbers,
                      "list of ints : atomic numbers of species on each site")
        .def("get_chemical_symbols",
             &Structure::getChemicalSymbols,
             "Returns a list of the species occupying each site by chemical symbol.")
        .def("set_chemical_symbols",
             &Structure::setChemicalSymbols,
             py::arg("atomic_numbers"),
             "Sets the species occupying each site by chemical symbol.\n"
             "\nParameters\n----------\n"
             "chemical_symbols : list of strings\n"
             "    new species by chemical symbol")
        .def_property("chemical_symbols",
                      &Structure::getChemicalSymbols,
                      &Structure::setChemicalSymbols,
                      "list of strings : chemical symbols of species on each site")
        .def("set_unique_sites",
             &Structure::setUniqueSites,
             py::arg("unique_sites"),
             "Set unique sites.\n"
             "This allows one to specify for each site in the structure the"
             " unique site it is related to.\n"
             "\nParameters\n----------\n"
             "unique_sites : list of ints\n"
             "    site of interest")
        .def("get_unique_sites",
             &Structure::getUniqueSites,
             "Returns the unique sites.\n"
             "\nReturns\n-------\n"
             "list of ints")
        .def_property("unique_sites",
                      &Structure::getUniqueSites,
                      &Structure::setUniqueSites,
                      "list of ints : unique sites")
        .def("get_unique_site",
             &Structure::getUniqueSite,
             py::arg("index"),
             "Returns unique site.\n"
             "\nParameters\n----------\n"
             "index : int\n"
             "    index of site of interest\n"
             "\nReturns\n-------\n"
             "int\n"
             "    index of unique site")
        .def("get_position",
             &Structure::getPosition,
             py::arg("site"),
             "Returns position of specified site\n"
             "\nParameters\n----------\n"
             "site : LatticeNeighbor object\n"
             "    site of interest\n"
             "\nReturns\n-------\n"
             "vector\n"
             "    position in Cartesian coordinates")
        .def("get_distance",
             &Structure::getDistance,
             py::arg("index1"),
             py::arg("index2"),
             py::arg("offset1") = Vector3d(0, 0, 0),
             py::arg("offset2") = Vector3d(0, 0, 0),
             "Returns distance between two sites\n"
             "\nParameters\n----------\n"
             "index1 : int\n"
             "    index of the first site\n",
             "index2 : int\n"
             "    index of the second site\n",
             "offset1 : vector\n"
             "    offset to be applied to the first site\n",
             "offset2 : vector\n"
             "    offset to be applied to the second site\n",
             "\nReturns\n-------\n"
             "float\n"
             "    distance in length units")
        .def("find_site_by_position",
             &Structure::findSiteByPosition,
             py::arg("position"),
             "Returns the index of the site that matches the position.\n"
             "\nParameters\n----------\n"
             "position : list/NumPy array\n"
             "    position in Cartesian coordinates\n",
             "\nReturns\n-------\n"
             "int\n"
             "    site index")
        .def("find_lattice_neighbor_by_position",
             &Structure::findLatticeNeighborByPosition,
             py::arg("position"),
             "Returns the lattice neighbor that matches the position.\n"
             "\nParameters\n----------\n"
             "position : list/NumPy array\n"
             "    position in Cartesian coordinates\n",
             "\nReturns\n-------\n"
             "LatticeNeighbor object\n"
             "    lattice neighbor")
        .def("find_lattice_neighbors_by_positions",
             &Structure::findLatticeNeighborsByPositions,
             py::arg("positions"),
             "Returns the lattice neighbors that match the positions.\n"
             "\nParameters\n----------\n"
             "positions : list of lists/NumPy arrays\n"
             "    list of positions in Cartesian coordinates\n",
             "\nReturns\n-------\n"
             "list of LatticeNeighbor object\n"
             "    list of lattice neighbors")
        .def("get_pbc", &Structure::getPBC,
             "Returns the periodic boundary conditions.")
        .def("set_pbc", &Structure::setPBC,
             "Sets the periodic boundary conditions.")
        .def_property("pbc",
                      &Structure::getPBC,
                      &Structure::setPBC,
                      "3-dimensional vector : periodic boundary conditions")
        .def("get_cell", &Structure::getCell,
             "Returns the cell metric.")
        .def("set_cell", &Structure::setCell,
             "Sets the cell metric.")
        .def_property("cell",
                      &Structure::getCell,
                      &Structure::setCell,
                      "3x3 array : cell metric")
        .def("__len__", &Structure::size);

    py::class_<Neighborlist>(m, "Neighborlist")
        .def(py::init<const double>())
        .def("build", &Neighborlist::build)
        .def("is_neighbor", &Neighborlist::isNeighbor)
        .def("get_neighbors", &Neighborlist::getNeighbors)
        .def("__len__", &Neighborlist::size)

        ;

    py::class_<ManybodyNeighborlist>(m, "ManybodyNeighborlist")
        .def(py::init<>())
        .def("calc_intersection", &ManybodyNeighborlist::getIntersection)
        .def("build", &ManybodyNeighborlist::build);

    py::class_<Cluster>(m, "Cluster")
        // .def(py::init<std::vector<int> &, std::vector<double> &, const bool, const int>(), pybind11::arg("sites"),
        //      pybind11::arg("distances"), pybind11::arg("sortedCluster") = true, pybind11::arg("clusterTag") = 0)
        .def(py::init<const Structure &, const std::vector<LatticeNeighbor> &, const bool, const int>(), pybind11::arg("structure"),
             pybind11::arg("latticeNeighbors"), pybind11::arg("sortedCluster") = true, pybind11::arg("clusterTag") = 0)
        .def("count", &Cluster::count)
        .def("get_count", &Cluster::getCount)
        .def("get_sites", &Cluster::getSites)
        .def("get_distances", &Cluster::getDistances)
        .def("print", &Cluster::print)
        .def("is_sorted", &Cluster::isSorted)
        .def("get_clustertag", &Cluster::getClusterTag)
        .def("get_geometrical_size", &Cluster::getGeometricalSize)
        .def("get_number_of_bodies",&Cluster::getNumberOfBodies)
        .def("__hash__", [](const Cluster &cluster) { return std::hash<Cluster>{}(cluster); })
        .def(py::self < py::self)
        .def(py::self == py::self)
        // .def(hash(py::self))
        ;

    py::class_<PermutationMap>(m, "PermutationMap")
        .def(py::init<const std::vector<Vector3d> &,
                      const std::vector<Matrix3d> &>())
        .def("build", &PermutationMap::build)
        .def("get_permutated_positions", &PermutationMap::getPermutatedPositions)
        .def("get_indiced_positions", &PermutationMap::getIndicedPermutatedPositions)

        ;

    py::class_<LatticeNeighbor>(m, "LatticeNeighbor")
        .def(py::init<const int, const Vector3d &>())
        .def("print", &LatticeNeighbor::print)
        .def_property("index", &LatticeNeighbor::index, &LatticeNeighbor::setIndex)
        .def_property("unitcellOffset", &LatticeNeighbor::unitcellOffset, &LatticeNeighbor::setUnitcellOffset)
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self + Eigen::Vector3d())
        .def("__hash__", [](const LatticeNeighbor &latticeNeighbor) { return std::hash<LatticeNeighbor>{}(latticeNeighbor); })

        ;

    py::class_<ClusterCounts>(m, "ClusterCounts")
        .def(py::init<>())
        .def("count_lattice_neighbors", &ClusterCounts::countLatticeNeighbors)
        .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<LatticeNeighbor> &)) & ClusterCounts::count)
        .def("count", (void (ClusterCounts::*)(const Structure &, const std::vector<std::vector<LatticeNeighbor>> &, const Cluster &)) & ClusterCounts::count)
        .def("count_orbitlist",&ClusterCounts::countOrbitlist)
        .def("__len__", &ClusterCounts::size)
        .def("reset", &ClusterCounts::reset)
        .def("get_cluster_counts", [](const ClusterCounts &clusterCounts) {
            //&ClusterCounts::getClusterCounts
            py::dict clusterCountDict;
            for (const auto &mapPair : clusterCounts.getClusterCounts())
            {
                py::dict d;
                for (const auto &vecInt_intPair : mapPair.second)
                {
                    d[py::tuple(py::cast(vecInt_intPair.first))] = vecInt_intPair.second;
                }
                clusterCountDict[py::cast(mapPair.first)] = d;
            }
            return clusterCountDict;
        })
        .def("print", &ClusterCounts::print);

    py::class_<Orbit>(m, "Orbit")
        .def(py::init<const Cluster &>())
        .def("add_equivalent_sites", (void (Orbit::*)(const std::vector<LatticeNeighbor> &, bool)) & Orbit::addEquivalentSites, py::arg("lattice_neighbors"), py::arg("sort")=false)
        .def("add_equivalent_sites", (void (Orbit::*)(const std::vector<std::vector<LatticeNeighbor>> &, bool)) & Orbit::addEquivalentSites, py::arg("lattice_neighbors"), py::arg("sort")=false)
        .def("get_representative_cluster", &Orbit::getRepresentativeCluster)
        .def("get_equivalent_sites", &Orbit::getEquivalentSites)
        .def("get_representative_sites", &Orbit::getRepresentativeSites)
        .def("get_equivalent_sites_permutations", &Orbit::getEquivalentSitesPermutations)
        .def("get_sites_with_permutation", &Orbit::getSitesWithPermutation)
        .def("__len__", &Orbit::size)
        .def("get_number_of_duplicates", &Orbit::getNumberOfDuplicates, py::arg("verbosity") = 0)
        .def("get_MC_vectors",&Orbit::getMCVectors)
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::self + Eigen::Vector3d());

    py::class_<OrbitList>(m, "OrbitList")
        .def(py::init<>())
        .def(py::init<const std::vector<Neighborlist> &, const Structure &>())
        .def(py::init<const Structure &, const std::vector<std::vector<LatticeNeighbor>> &, const std::vector<Neighborlist> &>())
        .def("add_orbit", &OrbitList::addOrbit)
        .def("get_number_of_NClusters", &OrbitList::getNumberOfNClusters)
        .def("get_orbit", &OrbitList::getOrbit)
        .def("clear", &OrbitList::clear)
        .def("sort", &OrbitList::sort)
        .def("get_orbitList", &OrbitList::getOrbitList)
        .def("get_primitive_structure",&OrbitList::getPrimitiveStructure)
        .def("__len__", &OrbitList::size)
        .def("print", &OrbitList::print, py::arg("verbosity") = 0)
        .def("get_supercell_orbitlist", &OrbitList::getSupercellOrbitlist);

    py::class_<LocalOrbitlistGenerator>(m, "LocalOrbitlistGenerator")
        .def(py::init<const OrbitList &, const Structure &>())
        .def("generate_local_orbitlist", (OrbitList(LocalOrbitlistGenerator::*)(const unsigned int)) & LocalOrbitlistGenerator::generateLocalOrbitlist)
        .def("generate_local_orbitlist", (OrbitList(LocalOrbitlistGenerator::*)(const Vector3d &)) & LocalOrbitlistGenerator::generateLocalOrbitlist)
        .def("clear", &LocalOrbitlistGenerator::clear)
        .def("get_unique_offsets_count", &LocalOrbitlistGenerator::getUniqueOffsetsCount)
        .def("get_prim_to_supercell_map", &LocalOrbitlistGenerator::getPrimToSupercellMap)
        .def("get_unique_primcell_offsets", &LocalOrbitlistGenerator::getUniquePrimcellOffsets);

        py::class_<ClusterSpace>(m, "ClusterSpace",py::dynamic_attr())
        .def(py::init<std::vector<int>, std::vector<std::string>, const OrbitList &>())
        .def("_get_clustervector",&ClusterSpace::generateClustervector)
        .def("get_orbit", &ClusterSpace::getOrbit)
        .def("get_cluster_product", &ClusterSpace::getClusterProduct)
        .def("get_clusterspace_info", &ClusterSpace::getClusterSpaceInfo)
        .def("get_clusterspace_size", &ClusterSpace::getClusterSpaceSize)
        .def("get_elements", &ClusterSpace::getAtomicNumbers)
        .def("get_cutoffs",&ClusterSpace::getCutoffs)
        .def("get_primitive_structure",&ClusterSpace::getPrimitiveStructure)
        .def("get_native_clusters",&ClusterSpace::getNativeClusters)
        ;


    return m.ptr();
}
