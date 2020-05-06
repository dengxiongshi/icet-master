#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <set>

#include "Cluster.hpp"
#include "LatticeSite.hpp"
#include "Symmetry.hpp"
#include "VectorOperations.hpp"

using namespace Eigen;

/**
This class handles an orbit.

An orbit is a set of clusters that are equivalent under the symmetry operations
of the underlying lattice. Each cluster is represented by a set of lattice
sites. An orbit is characterized by a representative cluster.

*/

class Orbit
{
public:
    /// Constructor.
    Orbit(const Cluster &cluster) : _representativeCluster(cluster) {}

    /// Adds one cluster to the orbit.
    void addEquivalentCluster(const std::vector<LatticeSite> &, bool = false);

    /// Adds several several clusters to the orbit.
    void addEquivalentClusters(const std::vector<std::vector<LatticeSite>> &, bool = false);

    /// Returns the number of equivalent clusters in this orbit.
    size_t size() const { return _equivalentClusters.size(); }

    /// Returns the radius of the representative cluster in this orbit.
    double radius() const { return _representativeCluster.radius(); }

    /// Returns the sorted, representative cluster for this orbit.
    Cluster getRepresentativeCluster() const { return _representativeCluster; }

    /// Returns the sites that define the representative cluster of this orbit.
    std::vector<LatticeSite> getSitesOfRepresentativeCluster() const { return _equivalentClusters[0]; }

    /// Returns the equivalent cluster.
    std::vector<LatticeSite> getClusterByIndex(unsigned int) const;

    /// Returns the permuted equivalent cluster.
    std::vector<LatticeSite> getPermutedClusterByIndex(unsigned int) const;

    /// Returns all equivalent clusters.
    std::vector<std::vector<LatticeSite>> getEquivalentClusters() const { return _equivalentClusters; }

    /// Returns all permuted equivalent clusters.
    std::vector<std::vector<LatticeSite>> getPermutedEquivalentClusters() const;

    /// Sets the equivalent clusters.
    void setEquivalentClusters(const std::vector<std::vector<LatticeSite>> &equivalentClusters) { _equivalentClusters = equivalentClusters; }

    /// Sorts equivalent clusters.
    void sort() { std::sort(_equivalentClusters.begin(), _equivalentClusters.end()); }

    /// Returns the number of bodies of the cluster that represent this orbit.
    unsigned int getClusterSize() const { return _representativeCluster.order(); }

    /// Returns permutations of equivalent clusters.
    std::vector<std::vector<int>> getPermutationsOfEquivalentClusters() const { return _equivalentClusterPermutations; }

    /// Assigns the permutations of the equivalent clusters.
    void setPermutationsOfEquivalentClusters(std::vector<std::vector<int>> &permutations) { _equivalentClusterPermutations = permutations; }

    /// Assigns the allowed permutations.
    void setAllowedClusterPermutations(std::set<std::vector<int>> &permutations) { _allowedClusterPermutations = permutations; }

    /// Gets the allowed permutations of clusters.
    std::set<std::vector<int>> getAllowedClusterPermutations() const { return _allowedClusterPermutations; }

    /// Returns the relevant multicomponent vectors of this orbit given the number of allowed components.
    std::vector<std::vector<int>> getMultiComponentVectors(const std::vector<int> &Mi_local) const;

    std::vector<std::vector<int>> getAllPossibleMultiComponentVectorPermutations(const std::vector<int> &Mi_local) const;

    /// Returns true if the input sites exists in _equivalentClusters, order does not matter if sorted=false.
    bool contains(const std::vector<LatticeSite>, bool) const;

    /// Removes all clusters from the list of equivalent clusters that contain the site with the given index.
    void removeClustersWithSiteIndex(const size_t, bool);

    /// Removes all clusters from the list of equivalent clusters that do not contain the site with the given index.
    void removeClustersWithoutIndex(const size_t index, bool);

    /// Remove a specific cluster (defined by a list of lattice sites) from the list of equivalent clusters.
    void removeCluster(std::vector<LatticeSite>);

    /// Comparison operator for automatic sorting in containers.
    friend bool operator==(const Orbit &orbit1, const Orbit &orbit2)
    {
        throw std::logic_error("Reached equal operator in Orbit");
    }

    /// Comparison operator for automatic sorting in containers.
    friend bool operator<(const Orbit &orbit1, const Orbit &orbit2)
    {
        throw std::logic_error("Reached < operator in Orbit");
    }

    /**
    Creates a copy of this orbit and translates all LatticeSite offsets in equivalent sites.
    This will also transfer any existing permutations directly, which should be fine since an offset does not change permutations to the prototype sites.
    */
    friend Orbit operator+(const Orbit &orbit, const Eigen::Vector3d &offset)
    {
        Orbit orbitOffset = orbit;
        for (auto &latticeSites : orbitOffset._equivalentClusters)
        {
            for (auto &latticeSite : latticeSites)
            {
                latticeSite = latticeSite + offset;
            }
        }
        return orbitOffset;
    }

    /// Appends an orbit to this orbit.
    Orbit &operator+=(const Orbit &orbit_rhs)
    {
        // This orbit does not have any eq. sites permutations: check that orbit_rhs also doesn't have them
        if (_equivalentClusterPermutations.size() == 0)
        {
            if (orbit_rhs.getPermutationsOfEquivalentClusters().size() != 0)
            {
                throw std::runtime_error("One orbit has equivalent site permutations and one does not (Orbit &operator+=)");
            }
        }
        else // This orbit has some eq. sites permutations: check that orbit_rhs also has them
        {
            if (orbit_rhs.getPermutationsOfEquivalentClusters().size() == 0)
            {
                throw std::runtime_error("One orbit has equivalent site permutations and one does not (Orbit &operator+=)");
            }
        }

        // Get representative sites
        auto rep_sites_rhs = orbit_rhs.getSitesOfRepresentativeCluster();
        auto rep_sites_this = getSitesOfRepresentativeCluster();

        if (rep_sites_this.size() != rep_sites_rhs.size())
        {
            throw std::runtime_error("Orbit order is not equal (Orbit &operator+=)");
        }

        const auto rhsEquivalentClusters = orbit_rhs.getEquivalentClusters();
        const auto rhsEquivalentClusterPermutations = orbit_rhs.getPermutationsOfEquivalentClusters();

        // Insert rhs eq sites and corresponding permutations
        _equivalentClusters.insert(_equivalentClusters.end(), rhsEquivalentClusters.begin(), rhsEquivalentClusters.end());
        _equivalentClusterPermutations.insert(_equivalentClusterPermutations.end(), rhsEquivalentClusterPermutations.begin(), rhsEquivalentClusterPermutations.end());
        return *this;
    }

public:
    /// Container of equivalent sites for this orbit
    std::vector<std::vector<LatticeSite>> _equivalentClusters;

    /// Representative sorted cluster for this orbit
    Cluster _representativeCluster;

private:
    /// Contains the permutations of the equivalent sites which takes it to the order of the reference cluster
    std::vector<std::vector<int>> _equivalentClusterPermutations;

    /// Contains the allowed sites permutations. i.e. if 0,2,1 is in this set then 0,1,0 is the same MC vector as 0,0,1
    std::set<std::vector<int>> _allowedClusterPermutations;
};


namespace std
{
    /// Stream operator.
    ostream& operator<<(ostream&, const Orbit&);
}
