#pragma once

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>

#include "Cluster.hpp"
#include "LatticeSite.hpp"
//#include "ManyBodyNeighborList.hpp"
#include "OrbitList.hpp"
//#include "PeriodicTable.hpp"
#include "Structure.hpp"

using namespace Eigen;

namespace py = pybind11;

class ClusterCounts
{
public:
    ClusterCounts() {  }
    void count(const Structure &, const std::vector<std::vector<LatticeSite>> &, const Cluster &, bool);
    void countCluster(const Cluster &, const std::vector<int> &, bool);
    void countOrbitList(const Structure &, const OrbitList &, bool orderIntact, bool permuteSites = false);

    std::unordered_map<Cluster, std::map<std::vector<int>, int>> getClusterCounts() const
    {
      return _clusterCounts;
    }

    /// Returns the cluster counts size i.e. the total number of clusters
    size_t size() const
    {
      return _clusterCounts.size();
    }
    /// Reset cluster counts
    void reset()
    {
      _clusterCounts.clear();
    }

    std::unordered_map<Cluster, std::map<std::vector<int>, int>> _clusterCounts;
};
