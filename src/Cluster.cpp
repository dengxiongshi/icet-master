#include "Cluster.hpp"


Cluster::Cluster(const Structure &structure,
                 const std::vector<LatticeNeighbor> &latticeNeighbors,
                 const bool sortedCluster,
                 const int clusterTag )
{
    _symprec = 1e-6;
    size_t clusterSize = latticeNeighbors.size();
    std::vector<int> sites(clusterSize);
    std::vector<double> distances;
    distances.reserve((clusterSize * (clusterSize - 1) / 2));
    for (size_t i = 0; i < latticeNeighbors.size(); i++)
    {
        sites[i] = structure.getUniqueSite(latticeNeighbors[i].index());
        for (size_t j = i + 1; j < latticeNeighbors.size(); j++)
        {
            double distance = roundDouble(structure.getDistance(latticeNeighbors[i].index(),
                                                                latticeNeighbors[j].index(),
                                                                latticeNeighbors[i].unitcellOffset(),
                                                                latticeNeighbors[j].unitcellOffset()));

            distances.push_back(distance);
        }
    }


    _sites = sites;
    _distances = distances;
    _sortedCluster = sortedCluster;
    _clusterTag = clusterTag;
    _geometricalSize = icet::getGeometricalRadius(latticeNeighbors, structure);
    if (_sortedCluster)
    {
        sortCluster();
    }
}



