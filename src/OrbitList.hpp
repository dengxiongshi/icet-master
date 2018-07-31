#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <vector>
#include "Orbit.hpp"
#include "ManyBodyNeighborList.hpp"
#include "Structure.hpp"
#include "Cluster.hpp"
#include "NeighborList.hpp"
#include <unordered_map>
#include <unordered_set>
#include "LatticeSite.hpp"
#include "VectorHash.hpp"
#include "Vector3dCompare.hpp"
#include "Symmetry.hpp"
#include "Geometry.hpp"
/**
@brief This class contains a set of sorted vectors or orbits
*/

class OrbitList
{
  public:

    /// Constructors.
    OrbitList();
    OrbitList(const std::vector<NeighborList> &neighbor_lists, const Structure &);
    OrbitList(const Structure &, const std::vector<std::vector<LatticeSite>> &, const std::vector<NeighborList> &);

    /// Returns a supercell orbit.
    Orbit getSuperCellOrbit(const Structure &, const Vector3d &, const unsigned int, std::unordered_map<LatticeSite, LatticeSite> &) const;

    /// Returns a local orbit list.
    OrbitList getLocalOrbitList(const Structure &, const Vector3d &, std::unordered_map<LatticeSite, LatticeSite> &) const;
    
    /// Adds a set of sites that are equivalent to the ones in this orbit.
    void addOrbit(const Orbit &orbit)
    {
        _orbitList.push_back(orbit);
    }

    /// Returns number of orbits.
    size_t size() const
    {
        return _orbitList.size();
    }

    /// Returns the number of orbits which are made up of N bodies.
    unsigned int getNumberOfNClusters(unsigned int N) const
    {
        unsigned int count = 0;
        for (const auto &orbit : _orbitList)
        {
            if (orbit.getRepresentativeCluster().order() == N)
            {
                count++;
            }
        }
        return count;
    }

    /// Returns a copy of the orbit located at given position in the internal orbit list.
    Orbit getOrbit(unsigned int i) const
    {
        if (i >= size())
        {
            throw std::out_of_range("Error: Tried accessing orbit at out of bound index. Orbit OrbitList::getOrbit");
        }
        return _orbitList[i];
    }

    /// Clears the orbit list.
    void clear()
    {
        _orbitList.clear();
    }

    /// Sorts the orbit list.
    void sort()
    {
        std::sort(_orbitList.begin(), _orbitList.end());
    }

    /// Returns all orbits in the orbit list.
    std::vector<Orbit> getOrbitList() const
    {
        return _orbitList;
    }

    /// If exists, returns the location of the orbit with the given representative cluster.
    int findOrbit(const Cluster &) const;

    /// Prints information about the orbit list
    void print(int verbosity = 0) const
    {
        int orbitCount = 0;
        for (const auto &orbit : _orbitList)
        {
            std::cout << "Multiplicities: " << orbit.size() << std::endl;
            if (verbosity > 1)
            {
                std::cout << "Duplicates: " << orbit.getNumberOfDuplicates() << std::endl;
            }
            std::cout << std::endl;
        }
    }

    /// Adds a cluster to orbit list.
    void addClusterToOrbitList(const Cluster &cluster, const std::vector<LatticeSite> &, std::unordered_map<Cluster, int> &);

    /// Adds lattice sites from column1 to neighbor list in orbit list.
    void addPermutationMatrixColumns(std::vector<std::vector<std::vector<LatticeSite>>> &lattice_neighbors, std::unordered_set<std::vector<int>, VectorHash> &taken_rows, const std::vector<LatticeSite> &lat_nbrs, const std::vector<int> &pm_rows,
                                     const std::vector<std::vector<LatticeSite>> &permutation_matrix, const std::vector<LatticeSite> &col1, bool) const;
    
    /// Returns the first column of the permutation matrix.
    std::vector<LatticeSite> getColumn1FromPM(const std::vector<std::vector<LatticeSite>> &, bool sortIt) const;
    
    /// Searches for lattice neighbors in the first column of the permutation matrix and returns the row index. 
    std::vector<int> findRowsFromCol1(const std::vector<LatticeSite> &col1, const std::vector<LatticeSite> &latNbrs, bool sortit = false) const;

    /// Checks that at least one of the lattice sites is inside the unit cell.
    bool validatedCluster(const std::vector<LatticeSite> &) const;

    /// Iterates over the neighbor list, that also includes permuted sites, to add orbit to orbit list.
    /// @todo Think about to rename this function.
    void addOrbitsFromPM(const Structure &, const std::vector<std::vector<std::vector<LatticeSite>>> &);
    
    /// Creates and adds orbit along with its representative cluster to the orbit list.
    void addOrbitFromPM(const Structure &, const std::vector<std::vector<LatticeSite>> &);
    
    /// Checks that equivalent sites generates same cluster as the representative sites. 
    void checkEquivalentClusters() const;

    /// Translates all the lattice sites by substracting the unitcell offset of a given site.
    std::vector<LatticeSite> translateSites(const std::vector<LatticeSite> &, const unsigned int) const;
    
    /// Iterates over all the lattice sites and creates for each iteration a set of translated sites.
    std::vector<std::vector<LatticeSite>> getSitesTranslatedToUnitcell(const std::vector<LatticeSite> &, bool sortit) const;

    /// Returns a set of translated sites that exists in the first column of permutation matrix.
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> getMatchesInPM(const std::vector<std::vector<LatticeSite>> &) const;

    /// Maps a given lattice site to supercell structure.
    void transformSiteToSupercell(LatticeSite &site, const Structure &superCell, std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const;
    
    /// Sets a primitive structure.
    void setPrimitiveStructure(const Structure &primitive)
    {
        _primitiveStructure = primitive;
    }

    /// Inserts a list of indexes corresponding to the rows of the permutation matrix to be accounted for.
    /// @todo Check if this description is correct.
    void takeRows(std::unordered_set<std::vector<int>, VectorHash> &taken_rows, std::vector<int> rows) const;

    /// Returns the primitive structure.
    Structure getPrimitiveStructure() const
    {
        return _primitiveStructure;
    }

    /// Appends an orbit list. If current orbit list is not empty, both orbit list must have the same
    /// number of orbits.
    OrbitList &operator+=(const OrbitList &rhs_ol)
    {
        if (size() == 0)
        {
            _orbitList = rhs_ol.getOrbitList();
            return *this;
        }

        if (size() != rhs_ol.size())
        {
            std::string errorMsg = "lhs.size() and rhs.size() are not equal in  OrbitList& operator+= " + std::to_string(size()) + " != " + std::to_string(rhs_ol.size());
            throw std::runtime_error(errorMsg);
        }

        for (size_t i = 0; i < rhs_ol.size(); i++)
        {
            _orbitList[i] += rhs_ol.getOrbit(i); // .addEquivalentSites(.getEquivalentSites());
        }
        return *this;
    }

    /// @todo Check this unused function and its dependencies.
    // OrbitList getSupercellOrbitList(const Structure &superCell) const;

    /// Adds the permutation information to the orbits.
    void addPermutationInformationToOrbits(const std::vector<LatticeSite> &, const std::vector<std::vector<LatticeSite>> &);

    /// Returns all columns from the given rows in permutation matrix.
    std::vector<std::vector<LatticeSite>> getAllColumnsFromRow(const std::vector<int> &, const std::vector<std::vector<LatticeSite>> &, bool, bool sortIt=true ) const;

    /// Checks if rows of the permutation matrix have been taken.
    bool isRowsTaken(const std::unordered_set<std::vector<int>, VectorHash> &taken_rows, std::vector<int> rows) const;

    /// Finds the sites in the first column of permutation matrix, extracts and returns all columns along
    /// with their unit cell translated indistinguishable sites
    std::vector<std::vector<LatticeSite>> getAllColumnsFromSites(const std::vector<LatticeSite> &,
        const std::vector<LatticeSite> &,
        const std::vector<std::vector<LatticeSite>> & ) const;

    /// Checks that the lattice neighbors do not have any unitcell offsets in a non-pbc direction
    bool isSitesPBCCorrect(const std::vector<LatticeSite> &sites) const;

  private:

    /// Checks that an orbit with the given representative cluster exists. If so, returns its location in the orbit list.
    int findOrbit(const Cluster &, const std::unordered_map<Cluster, int> &) const;

    /// Primitive structure used to set up the orbits
    Structure _primitiveStructure;

    /// Internal list of orbits
    std::vector<Orbit> _orbitList;

    /// Permutation matrix
    std::vector<std::vector<LatticeSite>> _permutation_matrix;
    
    /// First column of the permutation matrix
    std::vector<LatticeSite> _column1;
};
