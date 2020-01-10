#include "Cluster.hpp"

/**
@details Create an instance of a cluster.
@param structure icet structure object
@param latticeSites list of lattice sites that form the cluster
@param tag integer identifier
*/
Cluster::Cluster(const Structure &structure,
                 const std::vector<LatticeSite> &latticeSites,
                 const int tag)
{
    size_t clusterSize = latticeSites.size();
    std::vector<int> sites(clusterSize);
    std::vector<double> distances;
    distances.reserve((clusterSize * (clusterSize - 1) / 2));
    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        sites[i] = structure.getUniqueSite(latticeSites[i].index());
        for (size_t j = i + 1; j < latticeSites.size(); j++)
        {
            double distance = structure.getDistance(latticeSites[i].index(),
                                                    latticeSites[j].index(),
                                                    latticeSites[i].unitcellOffset(),
                                                    latticeSites[j].unitcellOffset());
            distances.push_back(distance);
        }
    }
    _sites = sites;
    _distances = distances;
    _tag = tag;
    _radius = icet::getGeometricalRadius(latticeSites, structure);
}

namespace std
{
    /// Stream operator.
    ostream& operator<<(ostream& os, const Cluster& cluster)
    {
        for (const auto d : cluster.distances())
        {
            os << d << " ";
        }
        os << cluster.radius();
        return os;
    }
}
