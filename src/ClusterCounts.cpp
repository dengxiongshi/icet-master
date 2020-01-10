#include "ClusterCounts.hpp"

/**
@details Will count the vectors in latticeSites and assuming these sets of sites are represented by the cluster 'cluster'.
@param structure the structure that will have its clusters counted
@param latticeSites A group of sites, represented by 'cluster', that will be counted
@param cluster A cluster used as identification on what sites the clusters belong to
@param orderIntact if true the order of the sites will stay the same otherwise the vector of species being counted will be sorted
*/
void ClusterCounts::count(const Structure &structure, const std::vector<std::vector<LatticeSite>> &latticeSites,
                          const Cluster &cluster, bool orderIntact)
{
    std::vector<int> elements(latticeSites[0].size());
    for (const auto &sites : latticeSites)
    {
        for (size_t i = 0; i < sites.size(); i++)
        {
            elements[i] = structure._atomicNumbers[sites[i].index()];
        }
        countCluster(cluster, elements, orderIntact);
    }
}

/// Counts cluster only through this function.
void ClusterCounts::countCluster(const Cluster &cluster, const std::vector<int> &elements, bool orderIntact)
{
    if (orderIntact)
    {
        _clusterCounts[cluster][elements] += 1;
    }
    else
    {
        std::vector<int> sortedElements = elements;
        std::sort(sortedElements.begin(), sortedElements.end());
        _clusterCounts[cluster][sortedElements] += 1;
    }
}

/**
 @brief Counts the clusters in the input structure.
 @param structure input configuration
 @param orbitList orbit list
 @param orderIntact if true do not reorder clusters before comparison (i.e., ABC != ACB)
 @param permuteSites if true the sites will be permuted according to the correspondin permutations in the orbit
*/
void ClusterCounts::countOrbitList(const Structure &structure, const OrbitList &orbitList, bool orderIntact, bool permuteSites)
{
    for (size_t i = 0; i < orbitList.size(); i++)
    {
        Cluster representativeCluster = orbitList._orbits[i].getRepresentativeCluster();
        representativeCluster.setTag(i);
        if (permuteSites && orderIntact && representativeCluster.order() != 1)
        {
            count(structure, orbitList.getOrbit(i).getPermutedEquivalentSites(), representativeCluster, orderIntact);
        }
        else if (!permuteSites && orderIntact && representativeCluster.order() != 1)
        {
            count(structure, orbitList._orbits[i]._equivalentSites, representativeCluster, orderIntact);
        }
        else
        {
            count(structure, orbitList._orbits[i]._equivalentSites, representativeCluster, orderIntact);
        }
    }
}
