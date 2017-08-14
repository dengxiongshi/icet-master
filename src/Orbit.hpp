#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <vector>
#include <string>
#include "LatticeNeighbor.hpp"
#include "Cluster.hpp"
using namespace Eigen;

/**
Class Orbit

contains equivalent vector<LatticeNeighbors>
contains a sorted Cluster for representation

Can be compared to other orbits

*/

class Orbit
{
  public:
    Orbit(const Cluster &);

    ///Add a group sites that are equivalent to the ones in this orbit
    void addEquivalentSites(const std::vector<LatticeNeighbor> &latNbrs)
    {
        _equivalentSites.push_back(latNbrs);
    }

    friend bool operator<(const Orbit &orbit1, const Orbit &orbit2);

    ///Returns amount of equivalent sites in this orbit
    size_t size() const
    {
        return _equivalentSites.size();
    }

    ///Return the sorted, reprasentative cluster for this orbit
    Cluster getRepresentativeCluster() const
    {
        return _sortedCluster;
    }

    ///Returns equivalent sites
    std::vector<std::vector<LatticeNeighbor>>  getEquivalentSites() const
    {
        return _equivalentSites;
    }

  private:
    ///Reprasentative sorted cluster for this orbit
    Cluster _sortedCluster;

    ///Container of equivalent sites for this orbit
    std::vector<std::vector<LatticeNeighbor>> _equivalentSites;
};