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
    Orbit(const Cluster &, const double geometricSize);

    ///Add a group sites that are equivalent to the ones in this orbit
    void addEquivalentSites(const std::vector<LatticeNeighbor> &latNbrs)
    {
        _equivalentSites.push_back(latNbrs);
    }
    ///add many lattice neigbhors
    void addEquivalentSites(const std::vector<std::vector<LatticeNeighbor>> &LatticeNeighbors)
    {
        _equivalentSites.insert(_equivalentSites.end(), LatticeNeighbors.begin(), LatticeNeighbors.end());
    }

    ///Returns amount of equivalent sites in this orbit
    size_t size() const
    {
        return _equivalentSites.size();
    }
    ///Returns the geometric size of the orbit defines as the mean distance to the center of the 
    double geometricSize() const
    {
        
    }
    ///Return the sorted, reprasentative cluster for this orbit
    Cluster getRepresentativeCluster() const
    {
        return _representativeCluster;
    }

    ///Returns equivalent sites
    std::vector<std::vector<LatticeNeighbor>> getEquivalentSites() const
    {
        return _equivalentSites;
    }

    std::vector<LatticeNeighbor> GetSitesOfIndex(unsigned int index) const
    {
        if (index >= _equivalentSites.size())
        {
            throw std::out_of_range("Index out of range in function Orbit::GetSitesOfIndex");
        }
        return _equivalentSites[index];
    }
    void setEquivalentSites(const std::vector<std::vector<LatticeNeighbor>> &equivalentSites)
    {
        _equivalentSites = equivalentSites;
    }
    ///Return the number of bodies of the cluster that represent this orbit
    unsigned int getClusterSize() const
    {
        return _representativeCluster.getNumberOfBodies();
    }
    ///Compare operator for automatic sorting in containers
    friend bool operator<(const Orbit &orbit1, const Orbit &orbit2)
    {
        if (orbit1.getRepresentativeCluster() < orbit2.getRepresentativeCluster())
        {
            return true;
        }
        //note the order is changed here "o2 < o1"
        if (orbit2.getRepresentativeCluster() < orbit1.getRepresentativeCluster())
        {
            return false;
        }
        //representative cluster is equal
        //Try comparing length of equivalent sites
        if (orbit1.size() < orbit2.size())
        {
            return true;
        }
        if (orbit1.size() > orbit2.size())
        {
            return false;
        }
        //Both representative cluster and size of equivalent sites are equal.
        //throw error to see if this ever happens

        bool debug = true;

        if (debug)
        {
            std::cout << "Clusters:" << std::endl;
            orbit1.getRepresentativeCluster().print();
            orbit2.getRepresentativeCluster().print();
            std::cout << "Length of eq sites: (orbit1.size(), orbit2.size()) " << orbit1.size() << " , " << orbit2.size() << std::endl;

            int maxCols = 7;
            std::cout << "First " << maxCols << " equivalent sites in bort orbits" << std::endl;

            int count = 0;
            for (auto sites : orbit1.getEquivalentSites())
            {
                std::cout << "site " << count << std::endl;
                for (auto site : sites)
                {
                    site.print();
                }

                if (count++ == maxCols)
                {
                    break;
                }
            }
            count = 0;
            std::cout << std::endl;
            for (auto sites : orbit2.getEquivalentSites())
            {
                std::cout << "site " << count << std::endl;
                for (auto site : sites)
                {
                    site.print();
                }

                if (count++ == maxCols)
                {
                    break;
                }
            }
        }

        throw std::runtime_error("Both representative cluster and size of equivalent sites are equal in orbit < comparison");
    }

    int getNumberOfDuplicates(int verbosity = 0) const;

    friend Orbit operator+(const Orbit &orbit, const Eigen::Vector3d &offset)
    {
        Orbit orbitOffset = orbit;
        for (auto &latNbrs : orbitOffset._equivalentSites)
        {
            for (auto &latNbr : latNbrs)
            {
                latNbr = latNbr + offset;
            }
        }
        return orbitOffset;
    }

  private:
    ///Reprasentative sorted cluster for this orbit
    Cluster _representativeCluster;

    ///Container of equivalent sites for this orbit
    std::vector<std::vector<LatticeNeighbor>> _equivalentSites;
};