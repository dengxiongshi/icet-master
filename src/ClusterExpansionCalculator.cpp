#include "ClusterExpansionCalculator.hpp"

ClusterExpansionCalculator::ClusterExpansionCalculator(const ClusterSpace &clusterSpace,
                                                       const Structure &structure,
                                                       const double fractionalPositionTolerance)
{
    _clusterSpace = clusterSpace;
    _supercell = structure;
    LocalOrbitListGenerator LOLG = LocalOrbitListGenerator(clusterSpace.getOrbitList(), _supercell, fractionalPositionTolerance);
    size_t uniqueOffsets = LOLG.getNumberOfUniqueOffsets();
    int numberOfOrbits = _clusterSpace._orbitList.size();
    std::vector<Orbit> orbitVector;
    _fullOrbitList = LOLG.getFullOrbitList();
    for (const auto orbit : clusterSpace._orbitList._orbits)
    {
        orbitVector.push_back(Orbit(orbit.getRepresentativeCluster()));
        
    }

    // Permutations for the sites in the orbits
    std::vector<std::vector<std::vector<int>>> permutations(numberOfOrbits);

    /* Strategy for constructing the "full" primitive orbitlists.

    First we fill up a std::vector<Orbit> orbitVector,
    where vector<orbit> is essentially an orbit list.

    The existing method for constructing the _full_ orbit list proceeds
    by looping over all local orbit lists with LocalOrbitListGenerator and
    adding the sites to the local orbit list.

    Now we do something similar by looping over each local orbit list
    (by looping over offsetIndex)
    The local orbitlist is retrieved here:
        `LOLG.getLocalOrbitList(offsetIndex).getOrbits()`

    Then for each orbit `orbitIndex` in `LOLG.getLocalOrbitList(offsetIndex).getOrbits()`
    each group of lattice sites in orbit.equivalentSites() is added to
    orbitVector[orbitIndex] if the lattice sites have a site with offset [0, 0, 0].

    When the full primitive orbitlist is used to create a local orbit list for
    site `index` in the supercell it should thus contain all lattice sites that
    contain `index`.
    */

    for (size_t offsetIndex = 0; offsetIndex < uniqueOffsets; offsetIndex++)
    {
        int orbitIndex = -1;
        // This orbit is a local orbit related to the supercell
        for (const auto orbit : LOLG.getLocalOrbitList(offsetIndex).getOrbits())
        {
            orbitIndex++;

            auto orbitPermutations = orbit.getPermutationsOfEquivalentSites();

            int eqSiteIndex = -1;

            for (const auto latticeSites : orbit.getEquivalentSites())
            {
                eqSiteIndex++;

                std::vector<LatticeSite> primitiveEquivalentSites;
                for (const auto site : latticeSites)
                {
                    Vector3d sitePosition = _supercell.getPosition(site);
                    auto primitiveSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(sitePosition, fractionalPositionTolerance);
                    primitiveEquivalentSites.push_back(primitiveSite);
                }
                std::vector<std::vector<LatticeSite>> latticeSitesTranslated = _clusterSpace._orbitList.getSitesTranslatedToUnitcell(primitiveEquivalentSites, false);

                for (auto latticesitesPrimTrans : latticeSitesTranslated)
                {
                    if (std::any_of(latticesitesPrimTrans.begin(), latticesitesPrimTrans.end(), [=](LatticeSite ls) { return (ls.unitcellOffset()).norm() < 1e-4; }))
                    {
                        // false or true here seems to not matter
                        if (!orbitVector[orbitIndex].contains(latticesitesPrimTrans, true))
                        {
                            orbitVector[orbitIndex].addEquivalentSites(latticesitesPrimTrans);
                            permutations[orbitIndex].push_back(orbitPermutations[eqSiteIndex]);
                        }
                    }
                }
            }
        }
    }

    // Now create the full primitive orbit list using the vector<orbit>
    _fullPrimitiveOrbitList.setPrimitiveStructure(_clusterSpace.getPrimitiveStructure());
    int orbitIndex = -1;
    for (auto orbit : orbitVector)
    {
        orbitIndex++;
        _fullPrimitiveOrbitList.addOrbit(orbit);
    }

    // Calculate the permutation for each orbit in this orbit list.
    // This is normally done in the constructor but since we made one manually
    // we have to do this ourself.
    _fullPrimitiveOrbitList.addPermutationInformationToOrbits(_clusterSpace.getOrbitList().getFirstColumnOfMatrixOfEquivalentSites(),
                                                              _clusterSpace.getOrbitList().getMatrixOfEquivalentSites());

    _primToSupercellMap.clear();
    _indexToOffset.clear();

    // Precompute all possible local orbitlists for this supercell and map it to the offset
    for (size_t i = 0; i < structure.size(); i++)
    {
        Vector3d localPosition = structure.getPositions().row(i);
        LatticeSite localSite = _clusterSpace.getPrimitiveStructure().findLatticeSiteByPosition(localPosition, fractionalPositionTolerance);
        Vector3d offsetVector = localSite.unitcellOffset();
        _indexToOffset[i] = offsetVector;

        if (_localOrbitlists.find(offsetVector) == _localOrbitlists.end())
        {
            _localOrbitlists[offsetVector] = _fullPrimitiveOrbitList.getLocalOrbitList(structure, offsetVector, _primToSupercellMap, fractionalPositionTolerance);

            // Set eq sites equal to the permuted sites so no permutation is required in the orbit list counting.
            /// @todo If one replaces the reference to the internal _orbits member of OrbitList with getOrbits(),
            /// multiple tests fails (for ternaries). This needs to be fixed.
            for (auto &orbit : _localOrbitlists[offsetVector]._orbits)
            {
                auto permutedSites = orbit.getPermutedEquivalentSites();
                orbit._equivalentSites = permutedSites;
            }
        }
    }
}

/**
@details This constructs a cluster vector that only considers clusters that contain the input index.
@param occupations the occupation vector for the supercell
@param index the local index of the supercell
@param ignoredIndices a vector of indices which have already had their local energy calculated. This is required to input so that no double counting occurs.
*/
std::vector<double> ClusterExpansionCalculator::getLocalClusterVector(const std::vector<int> &occupations,
                                                                      int index,
                                                                      std::vector<size_t> ignoredIndices)
{
    _supercell.setAtomicNumbers(occupations);
    _clusterCounts.reset();
    if (occupations.size() != _supercell.size())
    {
        throw std::runtime_error("Input occupations and internal supercell structure mismatch in size (ClusterExpansionCalculator::getLocalClusterVector)");
    }

    for (auto ignoreIndex : ignoredIndices)
    {
        if (ignoreIndex >= _supercell.size())
        {
            throw std::runtime_error("Index larger than input structure size (ClusterExpansionCalculator::getLocalClusterVector)");
        }
    }

    // do not sort the clusters
    bool keepOrder = true;

    // count the clusters in the order they lie in equivalent sites
    // since these sites are already in the permuted order
    bool permuteSites = false;
    

    // Get one of the translated orbitlists
    _translatedOrbitList = _localOrbitlists[_indexToOffset[index]];

    // Remove sites not containing the local index
    if (_clusterSpace._primitiveStructure.size() > 1)
    {
        // true meaning we only look at zero offset sites
        _translatedOrbitList.removeSitesNotContainingIndex(index, true);
    }

    // Purge the orbitlist of all sites containing the ignored indices
    for (auto ignoredIndex : ignoredIndices)
    {   
        // False meaning we consider all offsets 
        _translatedOrbitList.removeSitesContainingIndex(ignoredIndex, false);
    }

    // Count clusters and get cluster count map
    _clusterCounts.countOrbitList(_supercell, _translatedOrbitList, keepOrder, permuteSites);
    

    // Finally begin occupying the cluster vector
    std::vector<double> clusterVector;
    clusterVector.push_back(1.0 / _supercell.size());
    for (size_t i = 0; i < _fullPrimitiveOrbitList.size(); i++)
    {
        Cluster representativeCluster = _fullPrimitiveOrbitList._orbits[i]._representativeCluster;
        auto representativeSites = _clusterSpace._orbitList._orbits[i].getRepresentativeSites();
        std::vector<int> allowedOccupations;

        if (i >= _clusterSpace._orbitList.size())
        {
            std::cout << _fullPrimitiveOrbitList.size() << " >= " << _clusterSpace._orbitList.size() << std::endl;
            throw std::runtime_error("Index i larger than cs.orbit_list.size() (ClusterExpansionCalculator::getLocalClusterVector)");
        }
        try
        {
            allowedOccupations = _clusterSpace.getNumberOfAllowedSpeciesBySite(_clusterSpace._primitiveStructure, representativeSites);
        }
        catch (const std::exception &e)
        {
            std::cout << e.what() << std::endl;
            throw std::runtime_error("Failed getting allowed occupations (ClusterExpansionCalculator::getLocalClusterVector)");
        }

        // Skip the rest if any of the sites are inactive (i.e. allowed occupation < 2)
        if (std::any_of(allowedOccupations.begin(), allowedOccupations.end(), [](int allowedOccupation) { return allowedOccupation < 2; }))
        {
            continue;
        }
        
        std::vector<int> representativeSitesIndices;
        for (const auto site : representativeSites)
        {
            representativeSitesIndices.push_back(site.index());
        }

        const auto &mcVectors = _clusterSpace._multiComponentVectors[i];
        representativeCluster.setTag(i);

        /// Loop over all multi component vectors for this orbit
        for (size_t currentMCVectorIndex = 0; currentMCVectorIndex < _clusterSpace._multiComponentVectors[i].size(); currentMCVectorIndex++)
        {
            double clusterVectorElement = 0;

            auto clusterFind = _clusterCounts._clusterCounts.find(representativeCluster);

            /// Push back zero if nothing was counted for this orbit
            if (clusterFind == _clusterCounts._clusterCounts.end())
            {
                clusterVector.push_back(0);
                continue;
            }

            std::vector<int> permutedMCVector;
            std::vector<int> permutedAllowedOccupations;
            std::vector<int> permutedRepresentativeIndices;
            /// Loop over all the counts for this orbit
            for (const auto &elementsCountPair : _clusterCounts._clusterCounts.at(representativeCluster))
            {
                /// Loop over all equivalent permutations for this orbit and mc vector
                for (const auto &perm : _clusterSpace._sitePermutations[i][currentMCVectorIndex])
                {
                    /// Permute the mc vector and the allowed occupations
                    permutedMCVector = icet::getPermutedVector(mcVectors[currentMCVectorIndex], perm);
                    permutedAllowedOccupations = icet::getPermutedVector(allowedOccupations, perm);
                    permutedRepresentativeIndices = icet::getPermutedVector(representativeSitesIndices, perm);

                    clusterVectorElement += _clusterSpace.evaluateClusterProduct(permutedMCVector, permutedAllowedOccupations, elementsCountPair.first, permutedRepresentativeIndices) * elementsCountPair.second;
                }
            }

            /// This is the multiplicity one would have gotten during a full cluster vector calculation and is needed as normalizing factor
            double realMultiplicity = (double)_clusterSpace._sitePermutations[i][currentMCVectorIndex].size() * (double)_clusterSpace._orbitList._orbits[i]._equivalentSites.size() / (double)_clusterSpace._primitiveStructure.size();
            clusterVectorElement /= ((double)realMultiplicity * (double)_supercell.size());
            clusterVector.push_back(clusterVectorElement);
        }
    }
    return clusterVector;
}



std::vector<double> ClusterExpansionCalculator::getClusterVector(const std::vector<int> &occupations)
{
    _supercell.setAtomicNumbers(occupations);
    _clusterCounts.reset();
    if (occupations.size() != _supercell.size())
    {
        throw std::runtime_error("Input occupations and internal supercell structure mismatch in size (ClusterExpansionCalculator::getLocalClusterVector)");
    }


    // do not sort the clusters
    bool keepOrder = true;

    /// Permute the sites
    bool permuteSites = true;

    // Count clusters and get cluster count map
    _clusterCounts.countOrbitList(_supercell, _fullOrbitList, keepOrder, permuteSites);


    // Finally begin occupying the cluster vector
    std::vector<double> clusterVector;
    clusterVector.push_back(1.0);
    for (size_t i = 0; i < _fullOrbitList.size(); i++)
    {
        Cluster representativeCluster = _fullOrbitList._orbits[i]._representativeCluster;
        auto representativeSites = _clusterSpace._orbitList._orbits[i].getRepresentativeSites();
        std::vector<int> allowedOccupations;

        if (i >= _clusterSpace._orbitList.size())
        {
            std::cout << _fullOrbitList.size() << " >= " << _clusterSpace._orbitList.size() << std::endl;
            throw std::runtime_error("Index i larger than cs.orbit_list.size() (ClusterExpansionCalculator::getLocalClusterVector)");
        }
        try
        {
            allowedOccupations = _clusterSpace.getNumberOfAllowedSpeciesBySite(_clusterSpace._primitiveStructure, representativeSites);
        }
        catch (const std::exception &e)
        {
            std::cout << e.what() << std::endl;
            throw std::runtime_error("Failed getting allowed occupations (ClusterExpansionCalculator::getLocalClusterVector)");
        }

        // Skip the rest if any of the sites are inactive (i.e. allowed occupation < 2)
        if (std::any_of(allowedOccupations.begin(), allowedOccupations.end(), [](int allowedOccupation) { return allowedOccupation < 2; }))
        {
            continue;
        }
        
        std::vector<int> representativeSitesIndices;
        for (const auto site : representativeSites)
        {
            representativeSitesIndices.push_back(site.index());
        }

        const auto &mcVectors = _clusterSpace._multiComponentVectors[i];
        representativeCluster.setTag(i);

        /// Loop over all multi component vectors for this orbit
        for (size_t currentMCVectorIndex = 0; currentMCVectorIndex < _clusterSpace._multiComponentVectors[i].size(); currentMCVectorIndex++)
        {
            double clusterVectorElement = 0;
            int multiplicity = 0;

            auto clusterFind = _clusterCounts._clusterCounts.find(representativeCluster);

            /// Push back zero if nothing was counted for this orbit
            if (clusterFind == _clusterCounts._clusterCounts.end())
            {
                clusterVector.push_back(0);
                continue;
            }

            std::vector<int> permutedMCVector;
            std::vector<int> permutedAllowedOccupations;
            std::vector<int> permutedRepresentativeIndices;
            /// Loop over all the counts for this orbit
            for (const auto &elementsCountPair : _clusterCounts._clusterCounts.at(representativeCluster))
            {
                /// Loop over all equivalent permutations for this orbit and mc vector
                for (const auto &perm : _clusterSpace._sitePermutations[i][currentMCVectorIndex])
                {
                    /// Permute the mc vector and the allowed occupations
                    permutedMCVector = icet::getPermutedVector(mcVectors[currentMCVectorIndex], perm);
                    permutedAllowedOccupations = icet::getPermutedVector(allowedOccupations, perm);
                    permutedRepresentativeIndices = icet::getPermutedVector(representativeSitesIndices, perm);

                    clusterVectorElement += _clusterSpace.evaluateClusterProduct(permutedMCVector, permutedAllowedOccupations, elementsCountPair.first, permutedRepresentativeIndices) * elementsCountPair.second;
                    multiplicity += elementsCountPair.second;

                }
            }
            clusterVectorElement /= (double)multiplicity;
            clusterVector.push_back(clusterVectorElement);
        }
    }
    return clusterVector;
}
