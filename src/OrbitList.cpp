#include "OrbitList.hpp"

/**
@details This constructor generates an orbit list for the given (supercell) structure from a set of neighbor lists and a matrix of (symmetry) equivalent sites.
@param structure (supercell) structure for which to generate orbit list
@param matrixOfEquivalentSites matrix of symmetry equivalent sites
@param neighborLists neighbor lists for each (cluster) order (0=pairs, 1=triplets etc)
@param positionTolerance tolerance applied when comparing positions in Cartesian coordinates
**/
OrbitList::OrbitList(const Structure &structure,
                     const std::vector<std::vector<LatticeSite>> &matrixOfEquivalentSites,
                     const std::vector<std::vector<std::vector<LatticeSite>>> &neighborLists,
                     const double positionTolerance)
{
    _primitiveStructure = structure;
    _matrixOfEquivalentSites = matrixOfEquivalentSites;
    _referenceLatticeSites = getReferenceLatticeSites(false);

    /**
    The following list is used to compile "raw data" for the orbit list.
    The first index (vector) runs over the orbits, the second index (vector) over the
    equivalent cluster in a given orbit, and the final vector runs over the lattice sites
    that represent a particular cluster. **/
    std::vector<std::vector<std::vector<LatticeSite>>> listOfEquivalentClusters;

    // rows that have already been accounted for
    std::unordered_set<std::vector<int>, VectorHash> rowsTaken;

    ManyBodyNeighborList mbnl = ManyBodyNeighborList();

    // check that there are no duplicates in the first column of the matrix of equivalent sites
    std::set<LatticeSite> uniqueReferenceLatticeSites(_referenceLatticeSites.begin(), _referenceLatticeSites.end());
    if (_referenceLatticeSites.size() != uniqueReferenceLatticeSites.size())
    {
        std::ostringstream msg;
        msg << "Found duplicates in the list of reference lattice sites (= first column of matrix of equivalent sites): ";
        msg << std::to_string(_referenceLatticeSites.size()) << " != " << std::to_string(uniqueReferenceLatticeSites.size());
        msg << " (OrbitList::OrbitList)";
        throw std::runtime_error(msg.str());
    }

    for (size_t index = 0; index < neighborLists[0].size(); index++)
    {
        std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> mbnlLatticeSites = mbnl.build(neighborLists, index, false);
        for (const auto &mbnlPair : mbnlLatticeSites)
        {
            for (const auto &latticeSite : mbnlPair.second)
            {
                // complete cluster by combining the first and the second part of the MBNL pair
                std::vector<LatticeSite> cluster = mbnlPair.first;
                cluster.push_back(latticeSite);

                // check that original sites are sorted
                auto copyOfCluster = cluster;
                std::sort(copyOfCluster.begin(), copyOfCluster.end());
                if (copyOfCluster != cluster)
                {
                    throw std::runtime_error("Original sites are not sorted (OrbitList::OrbitList)");
                }

                // get all translational variants of cluster
                std::vector<std::vector<LatticeSite>> clusterWithTranslations = getSitesTranslatedToUnitcell(cluster);

                // get all sites from the matrix of equivalent sites
                auto pairsOfSiteAndIndex = getMatchesInMatrixOfEquivalenSites(clusterWithTranslations);
                if (!isRowsTaken(rowsTaken, pairsOfSiteAndIndex[0].second))
                {
                    // Found new stuff
                    addColumnsFromMatrixOfEquivalentSites(listOfEquivalentClusters, rowsTaken, pairsOfSiteAndIndex[0].second, true);
                }
            }

            // special singlet case
            if (mbnlPair.second.size() == 0)
            {
                std::vector<LatticeSite> cluster = mbnlPair.first;
                auto indices = getIndicesOfEquivalentLatticeSites(cluster);
                auto find = rowsTaken.find(indices);
                if (find == rowsTaken.end())
                {
                    // Found new stuff
                    addColumnsFromMatrixOfEquivalentSites(listOfEquivalentClusters, rowsTaken, indices, true);
                }
            }
        }
    }

    // Sort list of equivalent clusters
    for (size_t i = 0; i < listOfEquivalentClusters.size(); i++)
    {
        std::sort(listOfEquivalentClusters[i].begin(), listOfEquivalentClusters[i].end());
    }

    // Add orbits from list of equivalent clusters to this orbit list
    for (const auto &equivalentClusters : listOfEquivalentClusters)
    {
        Cluster representativeCluster = Cluster(structure, equivalentClusters[0]);
        Orbit newOrbit = Orbit(representativeCluster);
        _orbits.push_back(newOrbit);

        for (const auto &cluster : equivalentClusters)
        {
            _orbits.back().addEquivalentCluster(cluster);
        }
        _orbits.back().sort();

    }

    addPermutationInformationToOrbits();

    // Sort the orbit list.
    sort(positionTolerance);

}

/**
@details This function sorts the orbit list by order and radius. This is done to obtain a reproducable (stable) order of the orbit list.
@param positionTolerance tolerance applied when comparing positions in Cartesian coordinates
*/
void OrbitList::sort(const double positionTolerance)
{
    std::sort(_orbits.begin(), _orbits.end(),
              [positionTolerance](const Orbit& lhs, const Orbit& rhs)
              {
                  // Test against number of bodies in cluster.
                  if (lhs.getRepresentativeCluster().order() != rhs.getRepresentativeCluster().order())
                  {
                      return lhs.getRepresentativeCluster().order() < rhs.getRepresentativeCluster().order();
                  }
                  // Compare by radius.
                  if (fabs(lhs.radius() - rhs.radius()) > positionTolerance)
                  {
                      return lhs.radius() < rhs.radius();
                  }

                  // Check size of vector of equivalent sites.
                  if (lhs.size() < rhs.size())
                  {
                      return true;
                  }
                  if (lhs.size() > rhs.size())
                  {
                      return false;
                  }

                  // Check the individual equivalent sites.
                  return lhs.getEquivalentClusters() < rhs.getEquivalentClusters();
              });
}

/**
@param orbit orbit to add to orbit list
**/
void OrbitList::addOrbit(const Orbit &orbit) {
    _orbits.push_back(orbit);
}

/**
@param nbody number of bodies for which to return the number of clusters
**/
unsigned int OrbitList::getNumberOfNBodyClusters(unsigned int nbody) const
{
    unsigned int count = 0;
    for (const auto &orbit : _orbits)
    {
        if (orbit.getRepresentativeCluster().order() == nbody)
        {
            count++;
        }
    }
    return count;
}

/**
@details Returns the index of the orbit for which the given cluster is representative.
@param cluster cluster to search for
@param clusterIndexMap map of cluster indices for fast lookup
@returns orbit index; -1 if nothing is found
**/
int OrbitList::findOrbitIndex(const Cluster &cluster,
                              const std::unordered_map<Cluster, int> &clusterIndexMap) const
{
    auto search = clusterIndexMap.find(cluster);
    if (search != clusterIndexMap.end())
    {
        return search->second;
    }
    else
    {
        return -1;
    }
}

/**
@details Returns a copy of the orbit at the given index.
@param index index of orbit
@returns copy of orbit
**/
Orbit OrbitList::getOrbit(unsigned int index) const
{
    if (index >= size())
    {
        throw std::out_of_range("Tried accessing orbit at out of bound index (Orbit OrbitList::getOrbit)");
    }
    return _orbits[index];
}

/**
@details
This function adds permutation related information to the orbits.

Algorithm
---------

For each orbit:

1. Take representative cluster
2. Find the rows, which match the sites that belong to this cluster, and ...
3. Get all columns for these rows, i.e the sites that are directly equivalent, call these equivalentClustersWithTranslations.
4. Construct all possible permutations for the representative cluster, call these representativeClusterWithTranslationsAndPermutations.
5. Construct the intersection of p_equal and p_all, call this consistentEquivalentClustersWithTranslations.
6. Get the index version of consistentEquivalentClustersWithTranslations and these are then the allowed permutations for this orbit.
7. Take the clusters in the orbit:
    if site exists in p_all:
        those cluster are then related to representative_cluster via the permutation
    else:
        loop over permutations of the clusters:
            if the permutation exists in p_all:
                that permutation is then related to repr_cluster through that permutation
            else:
                continue

**/
void OrbitList::addPermutationInformationToOrbits()
{
    for (size_t i = 0; i < size(); i++)
    {

        bool sortRows = false;

        // step one: Get representative cluster
        std::vector<LatticeSite> sitesOfRepresentativeCluster = _orbits[i].getSitesOfRepresentativeCluster();
        auto representativeClusterWithTranslations = getSitesTranslatedToUnitcell(sitesOfRepresentativeCluster, sortRows);

        // step two: Find the rows these sites belong to and
        // step three: Get all columns for these rows
        std::vector<std::vector<LatticeSite>> equivalentClustersWithTranslations;
        for (auto reprCluster : representativeClusterWithTranslations)
        {
            auto equivClusters = getAllColumnsFromCluster(reprCluster);
            equivalentClustersWithTranslations.insert(equivalentClustersWithTranslations.end(), equivClusters.begin(), equivClusters.end());
        }
        std::sort(equivalentClustersWithTranslations.begin(), equivalentClustersWithTranslations.end());

        // Step four: Construct all possible permutations of the representative cluster
        std::vector<std::vector<LatticeSite>> representativeClusterWithTranslationsAndPermutations;
        for (auto reprCluster : representativeClusterWithTranslations)
        {
            std::vector<std::vector<LatticeSite>> permClusters = icet::getAllPermutations<LatticeSite>(reprCluster);
            representativeClusterWithTranslationsAndPermutations.insert(representativeClusterWithTranslationsAndPermutations.end(), permClusters.begin(), permClusters.end());
        }
        std::sort(representativeClusterWithTranslationsAndPermutations.begin(), representativeClusterWithTranslationsAndPermutations.end());

        // Step five: Construct intersection of equivalentClustersWithTranslations and
        // representativeClusterWithTranslationsAndPermutations. This will
        // generate the list of equivalent clusters that is consistent with the
        // permutations of the representative cluster. This is relevant for
        // systems with more than two components, for which one must deal with
        // multi-component vectors.
        std::vector<std::vector<LatticeSite>> consistentEquivalentClustersWithTranslations;
        std::set_intersection(equivalentClustersWithTranslations.begin(), equivalentClustersWithTranslations.end(),
                              representativeClusterWithTranslationsAndPermutations.begin(), representativeClusterWithTranslationsAndPermutations.end(),
                              std::back_inserter(consistentEquivalentClustersWithTranslations));

        // Step six: Get the index version of consistentEquivalentClustersWithTranslations
        std::set<std::vector<int>> allowedPermutations;
        for (const auto &equivCluster : consistentEquivalentClustersWithTranslations)
        {
            size_t failedLoops = 0;
            for (auto reprCluster : representativeClusterWithTranslations)
            {
                try
                {
                    std::vector<int> allowedPermutation = icet::getPermutation<LatticeSite>(reprCluster, equivCluster);
                    allowedPermutations.insert(allowedPermutation);
                }
                catch (const std::runtime_error &e)
                {
                    {
                        failedLoops++;
                        if (failedLoops == representativeClusterWithTranslations.size())
                        {
                            throw std::runtime_error("Did not find integer permutation from allowed permutation to any translated representative site (OrbitList::addPermutationInformationToOrbits)");
                        }
                        continue;
                    }
                }
            }
        }

        // Step seven: Relate equivalent clusters to the representative cluster, i.e. what is the consistent ordering of the cluster
        const auto orbitSites = _orbits[i].getEquivalentClusters();
        std::unordered_set<std::vector<LatticeSite>> p_equal_set;
        p_equal_set.insert(equivalentClustersWithTranslations.begin(), equivalentClustersWithTranslations.end());

        std::vector<std::vector<int>> sitePermutations;
        sitePermutations.reserve(orbitSites.size());

        for (const auto &eqOrbitSites : orbitSites)
        {
            if (p_equal_set.find(eqOrbitSites) == p_equal_set.end())
            {
                // Did not find the orbit.eq_sites in p_equal meaning that this eq site does not have an allowed permutation.
                auto equivalently_translated_eqOrbitsites = getSitesTranslatedToUnitcell(eqOrbitSites, sortRows);
                std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> translatedPermutationsOfSites;
                for (const auto eq_trans_eqOrbitsites : equivalently_translated_eqOrbitsites)
                {
                    const auto allPermutationsOfSites_i = icet::getAllPermutations<LatticeSite>(eq_trans_eqOrbitsites);
                    for (const auto perm : allPermutationsOfSites_i)
                    {
                        translatedPermutationsOfSites.push_back(std::make_pair(perm, eq_trans_eqOrbitsites));
                    }
                }
                for (const auto &onePermPair : translatedPermutationsOfSites)
                {
                    const auto findOnePerm = p_equal_set.find(onePermPair.first);
                    if (findOnePerm != p_equal_set.end()) // one perm is one of the equivalent sites. This means that eqOrbitSites is associated to p_equal
                    {
                        std::vector<int> permutationToEquivalentSites = icet::getPermutation<LatticeSite>(onePermPair.first, onePermPair.second);
                        sitePermutations.push_back(permutationToEquivalentSites);
                        break;
                    }
                    if (onePermPair == translatedPermutationsOfSites.back())
                    {
                        throw std::runtime_error("Did not find a permutation of the orbit sites to the permutations of the representative sites (OrbitList::addPermutationInformationToOrbits)");
                    }
                }
            }
            else
            {
                std::vector<int> permutationToEquivalentSites = icet::getPermutation<LatticeSite>(eqOrbitSites, eqOrbitSites); //the identical permutation
                sitePermutations.push_back(permutationToEquivalentSites);
            }
        }

        if (sitePermutations.size() != _orbits[i].getEquivalentClusters().size() || sitePermutations.size() == 0)
        {
            std::ostringstream msg;
            msg << "Not each set of site got a permutation (OrbitList::addPermutationInformationToOrbits) " << std::endl;
            msg << sitePermutations.size() << " != " << _orbits[i].getEquivalentClusters().size();
            throw std::runtime_error(msg.str());
        }

        _orbits[i].setPermutationsOfEquivalentClusters(sitePermutations);
        _orbits[i].setAllowedClusterPermutations(allowedPermutations);
    }
}

/**
@details Finds the sites in referenceLatticeSites, extract all columns along with their unit cell translated indistinguishable sites.
@param sites sites that correspond to the columns that will be returned
@returns columns along with their unit cell translated indistinguishable sites
**/
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromCluster(const std::vector<LatticeSite> &sites) const
{
    bool sortRows = false;
    std::vector<int> rowsFromReferenceLatticeSites = getIndicesOfEquivalentLatticeSites(sites, sortRows);
    std::vector<std::vector<LatticeSite>> p_equal = getAllColumnsFromRow(rowsFromReferenceLatticeSites, true, sortRows);
    return p_equal;
}

/// Returns true if rows_sort exists in rowsTaken.
bool OrbitList::isRowsTaken(const std::unordered_set<std::vector<int>, VectorHash> &rowsTaken,
                            std::vector<int> rows) const
{
    const auto find = rowsTaken.find(rows);
    if (find == rowsTaken.end())
    {
        return false;
    }
    else
    {
        return true;
    }
}

/**
@brief Returns all columns from the given rows in matrix of symmetry equivalent sites
@param rows indices of rows to return
@param includeTranslatedSites If true it will also include the equivalent sites found from the rows by moving each site into the unitcell.
@param sort if true (default) the first column will be sorted
**/
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromRow(const std::vector<int> &rows,
                                                                      bool includeTranslatedSites,
                                                                      bool sort) const
{
    std::vector<std::vector<LatticeSite>> allColumns;

    for (size_t column = 0; column < _matrixOfEquivalentSites[0].size(); column++)
    {
        std::vector<LatticeSite> nondistinctLatticeSites;

        for (const int &row : rows)
        {
            nondistinctLatticeSites.push_back(_matrixOfEquivalentSites[row][column]);
        }

        if (includeTranslatedSites)
        {
            auto translatedEquivalentSites = getSitesTranslatedToUnitcell(nondistinctLatticeSites, sort);
            allColumns.insert(allColumns.end(), translatedEquivalentSites.begin(), translatedEquivalentSites.end());
        }
        else
        {
            allColumns.push_back(nondistinctLatticeSites);
        }
    }
    return allColumns;
}

/**
@details
This function creates all possible translations of the input list of lattice sites, for which at
least one of the lattice sites is inside the (original) unit cell.
For example, given a pair with unit cell offsets
  [0, 0, 1], [-3, 0, 3]
one gets
  [0, 0, 0], [-3, 0, 2]
  [3, 0, -2], [0, 0, 0]

This translation gives rise to equivalent clusters that sometimes
are not found by using the set of crystal symmetries given by spglib.

@param latticeSites list of lattice sites
@param sort if true sort the translated sites
*/
std::vector<std::vector<LatticeSite>> OrbitList::getSitesTranslatedToUnitcell(const std::vector<LatticeSite> &latticeSites,
                                                                              bool sort) const
{

    std::vector<std::vector<LatticeSite>> listOfTranslatedLatticeSites;
    listOfTranslatedLatticeSites.push_back(latticeSites);
    Vector3d zeroVector = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < latticeSites.size(); i++)
    {
        if ((latticeSites[i].unitcellOffset() - zeroVector).norm() > 0.5) // only translate those outside unitcell
        {
            auto translatedSites = translateSites(latticeSites, i);
            if (sort)
            {
                std::sort(translatedSites.begin(), translatedSites.end());
            }
            listOfTranslatedLatticeSites.push_back(translatedSites);
        }
    }

    // sort this so that the lowest vec<latticeSite> will be chosen and therefore the sorting of orbits should be consistent.
    std::sort(listOfTranslatedLatticeSites.begin(), listOfTranslatedLatticeSites.end());

    return listOfTranslatedLatticeSites;
}

/**
@detail Takes all lattice sites in vector latticeSites and subtracts the unitcell offset of site latticeSites[index].
@param latticeSites list of lattice sites, typically a cluster
@param index index of site relative to which to shift
*/
std::vector<LatticeSite> OrbitList::translateSites(const std::vector<LatticeSite> &latticeSites,
                                                   const unsigned int index) const
{
    Vector3d offset = latticeSites[index].unitcellOffset();
    auto translatedSites = latticeSites;
    for (auto &latticeSite : translatedSites)
    {
        latticeSite.addUnitcellOffset(-offset);
    }
    return translatedSites;
}

/**
@details Adds columns of the matrix of equivalent sites to the orbit list.
@param listOfEquivalentClusters list of lattice sites to which to add
The first index (vector) runs over the orbits,
the second index (vector) over the equivalent cluster in a given orbit, and
the final vector runs over the lattice sites that represent a particular cluster.

@param rowsTaken
@param pm_rows indices of rows in matrix of symmetry equivalent sites
@param add
@todo fix the description of this function, including its name
**/
void OrbitList::addColumnsFromMatrixOfEquivalentSites(std::vector<std::vector<std::vector<LatticeSite>>> &listOfEquivalentClusters,
                                                      std::unordered_set<std::vector<int>, VectorHash> &rowsTaken,
                                                      const std::vector<int> &pm_rows,
                                                      bool add) const
{

    std::vector<std::vector<LatticeSite>> columnLatticeSites;
    columnLatticeSites.reserve(_matrixOfEquivalentSites[0].size());
    for (size_t column = 0; column < _matrixOfEquivalentSites[0].size(); column++)
    {
        std::vector<LatticeSite> nondistinctLatticeSites;

        for (const int &row : pm_rows)
        {
            nondistinctLatticeSites.push_back(_matrixOfEquivalentSites[row][column]);
        }
        auto translatedEquivalentSites = getSitesTranslatedToUnitcell(nondistinctLatticeSites);

        auto pairsOfSiteAndIndex = getMatchesInMatrixOfEquivalenSites(translatedEquivalentSites);

        auto find = rowsTaken.find(pairsOfSiteAndIndex[0].second);
        bool findOnlyOne = true;
        if (find == rowsTaken.end())
        {
            for (size_t i = 0; i < pairsOfSiteAndIndex.size(); i++)
            {
                find = rowsTaken.find(pairsOfSiteAndIndex[i].second);
                if (find == rowsTaken.end())
                {
                    if (add && findOnlyOne && validCluster(pairsOfSiteAndIndex[i].first))
                    {
                        columnLatticeSites.push_back(pairsOfSiteAndIndex[0].first);
                        findOnlyOne = false;
                    }
                    rowsTaken.insert(pairsOfSiteAndIndex[i].second);
                }
            }
        }
    }
    if (columnLatticeSites.size() > 0)
    {
        listOfEquivalentClusters.push_back(columnLatticeSites);
    }
}

/**
@details Returns the first set of translated sites that exist in referenceLatticeSites.
*/
std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> OrbitList::getMatchesInMatrixOfEquivalenSites(
    const std::vector<std::vector<LatticeSite>> &translatedSites) const
{
    std::vector<int> perm_matrix_rows;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> matchedSites;
    for (const auto &sites : translatedSites)
    {
        try
        {
            perm_matrix_rows = getIndicesOfEquivalentLatticeSites(sites);
        }
        catch (const std::runtime_error)
        {
            continue;
        }
        // No error here indicating that we found matching rows in reference lattice sites
        matchedSites.push_back(std::make_pair(sites, perm_matrix_rows));
    }
    if (matchedSites.size() > 0)
    {
        return matchedSites;
    }
    else
    {
        // No matching rows in matrix of equivalent sites, this should not happen so we throw an error.
        throw std::runtime_error("Did not find any of the translated sites in reference lattice sites in the matrix of equivalent sites (OrbitList::addColumnsFromMatrixOfEquivalentSites)");
    }
}

/**
@details This function returns true if the cluster includes at least on site from the unit cell at the origin, i.e. its unitcell offset is zero.
@param latticeSites list of sites to check
*/
bool OrbitList::validCluster(const std::vector<LatticeSite> &latticeSites) const
{
    Vector3d zeroVector = {0., 0., 0.};
    for (const auto &latticeSite : latticeSites)
    {
        if (latticeSite.unitcellOffset() == zeroVector)
        {
            return true;
        }
    }
    return false;
}

/**
@details Returns a list of indices of entries in latticeSites that are equivalent to the sites in reference lattice sites.
@param sort if true the first column will be sorted
@param latticeSites list of sites to search in
@return indices of entries in latticeSites that are equivalent to sites in the reference lattice sites
**/
std::vector<int> OrbitList::getIndicesOfEquivalentLatticeSites(const std::vector<LatticeSite> &latticeSites,
							                                   bool sort) const
{
    std::vector<int> rows;
    for (const auto &latticeSite : latticeSites)
    {
        const auto find = std::find(_referenceLatticeSites.begin(), _referenceLatticeSites.end(), latticeSite);
        if (find == _referenceLatticeSites.end())
        {
            throw std::runtime_error("Did not find lattice site in the reference lattice sites in the matrix of equivalent sites (OrbitList::getIndicesOfEquivalentLatticeSites)");
        }
        else
        {
            int row_in_referenceLatticeSites = std::distance(_referenceLatticeSites.begin(), find);
            rows.push_back(row_in_referenceLatticeSites);
        }
    }
    if (sort)
    {
        std::sort(rows.begin(), rows.end());
    }
    return rows;
}

/**
@details Returns reference lattice sites, which is equivalent to returning the first column of the matrix of equivalent sites.
@todo Expand description.
@param sort if true (default) the first column will be sorted
**/
std::vector<LatticeSite> OrbitList::getReferenceLatticeSites(bool sort) const
{
    std::vector<LatticeSite> referenceLatticeSites;
    referenceLatticeSites.reserve(_matrixOfEquivalentSites[0].size());
    for (const auto &row : _matrixOfEquivalentSites)
    {
        referenceLatticeSites.push_back(row[0]);
    }
    if (sort)
    {
        std::sort(referenceLatticeSites.begin(), referenceLatticeSites.end());
    }
    return referenceLatticeSites;
}

/**
@details This function returns the orbit for a supercell that is associated with a given orbit in the primitive structure.
@param supercell input structure
@param cellOffset offset by which to translate the orbit
@param orbitIndex index of orbit in list of orbits
@param primToSuperMap map from sites in the primitive cell to sites in the supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
Orbit OrbitList::getSuperCellOrbit(const Structure &supercell,
                                   const Vector3d &cellOffset,
                                   const unsigned int orbitIndex,
                                   std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap,
                                   const double fractionalPositionTolerance) const
{
    if (orbitIndex >= _orbits.size())
    {
        std::ostringstream msg;
        msg << "Orbit index out of range (OrbitList::getSuperCellOrbit).";
        msg << orbitIndex << " >= " << _orbits.size();
        throw std::out_of_range(msg.str());
    }

    Orbit supercellOrbit = _orbits[orbitIndex] + cellOffset;

    auto equivalentSites = supercellOrbit.getEquivalentClusters();

    for (auto &sites : equivalentSites)
    {
        for (auto &site : sites)
        {
            // Technically we should use the fractional position tolerance
            // corresponding to the cell metric of the supercell structure.
            // This is, however, not uniquely defined. Moreover, the difference
            // would only matter for very large supercells. We (@angqvist,
            // @erikfransson, @erhart) therefore decide to defer this issue
            // until someone encounters the problem in a practical situation.
            // In principle, one should not handle coordinates (floats) at this
            // level anymore. Rather one should transform any (supercell)
            // structure into an effective representation in terms of lattice
            // sites before any further operations.
            transformSiteToSupercell(site, supercell, primToSuperMap, fractionalPositionTolerance);
        }
    }

    supercellOrbit.setEquivalentClusters(equivalentSites);
    return supercellOrbit;
}

/**
@details Transforms a site from the primitive structure to a given supercell.
This involves finding a map from the site in the primitive cell to the supercell.
If no map is found mapping is attempted based on the position of the site in the supercell.
@param site lattice site to transform
@param supercell supercell structure
@param primToSuperMap map from primitive to supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
void OrbitList::transformSiteToSupercell(LatticeSite &site,
                                         const Structure &supercell,
                                         std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap,
                                         const double fractionalPositionTolerance) const
{
    auto find = primToSuperMap.find(site);
    LatticeSite supercellSite;
    if (find == primToSuperMap.end())
    {
        Vector3d sitePosition = _primitiveStructure.getPosition(site);
        supercellSite = supercell.findLatticeSiteByPosition(sitePosition, fractionalPositionTolerance);
        primToSuperMap[site] = supercellSite;
    }
    else
    {
        supercellSite = primToSuperMap[site];
    }

    // overwrite site to match supercell index offset
    site.setIndex(supercellSite.index());
    site.setUnitcellOffset(supercellSite.unitcellOffset());
}

/**
@details Returns a "local" orbitList by offsetting each site in the primitive cell by an offset.
@param supercell supercell structure
@param cellOffset offset to be applied to sites
@param primToSuperMap map from primitive to supercell
@param fractionalPositionTolerance tolerance applied when comparing positions in fractional coordinates
**/
OrbitList OrbitList::getLocalOrbitList(const Structure &supercell,
                                       const Vector3d &cellOffset,
                                       std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap,
                                       const double fractionalPositionTolerance) const
{
    OrbitList localOrbitList = OrbitList();
    localOrbitList.setPrimitiveStructure(_primitiveStructure);

    for (size_t orbitIndex = 0; orbitIndex < _orbits.size(); orbitIndex++)
    {
        localOrbitList.addOrbit(getSuperCellOrbit(supercell, cellOffset, orbitIndex, primToSuperMap, fractionalPositionTolerance));
    }
    return localOrbitList;
}

/**
@details This function will loop over all orbits in the list and remove from each orbit the clusters that match the given index.
@param index the index for which to check
@param onlyConsiderZeroOffset if true only clusters with zero offset will be removed
**/
void OrbitList::removeClustersContainingIndex(const int index,
                                              bool onlyConsiderZeroOffset)
{
    for (auto &orbit : _orbits)
    {
        orbit.removeClustersWithSiteIndex(index, onlyConsiderZeroOffset);
    }
}

/**
@details This function will loop over all orbits in the list and remove from each orbit the clusters that _do _not_ match the given index.
@param index the index for which to check
@param onlyConsiderZeroOffset if true only clusters with zero offset will be removed
**/
void OrbitList::removeClustersWithoutIndex(const int index,
                                           bool onlyConsiderZeroOffset)
{
    for (auto &orbit : _orbits)
    {
        orbit.removeClustersWithoutIndex(index, onlyConsiderZeroOffset);
    }
}

/**
@details Removes from the current orbit list all clusters that are in the input orbit list.
@param orbitList orbit list with clusters that are to be removed.
 **/
void OrbitList::subtractClustersFromOrbitList(const OrbitList &orbitList)
{
    if (orbitList.size() != size())
    {
        throw std::runtime_error("Orbit lists differ in size (OrbitList::subtractClustersFromOrbitList)");
    }
    for (size_t i = 0; i < size(); i++)
    {
        for (const auto sites : orbitList.getOrbit(i)._equivalentClusters)
        {
            if (_orbits[i].contains(sites, true))
            {
                _orbits[i].removeCluster(sites);
            }
        }
    }
}

/**
@details This function removes an orbit identified by index from the orbit list.
@param index index of the orbit in question
**/
void OrbitList::removeOrbit(const size_t index)
{
    if (index >= size())
    {
        std::ostringstream msg;
        msg << "Index " << index << " was out of bounds (OrbitList::removeOrbit)." << std::endl;
        msg << "OrbitList size: " << size();
        throw std::out_of_range(msg.str());
    }
    _orbits.erase(_orbits.begin() + index);
}

/**
@details Removes all orbits that have inactive sites.
@param structure the structure containining the number of allowed species on each lattice site
**/
void OrbitList::removeInactiveOrbits(const Structure &structure)
{
    for (int i = _orbits.size() - 1; i >= 0; i--)
    {
        auto numberOfAllowedSpecies = structure.getNumberOfAllowedSpeciesBySites(_orbits[i].getSitesOfRepresentativeCluster());
        if (std::any_of(numberOfAllowedSpecies.begin(), numberOfAllowedSpecies.end(), [](int n) { return n < 2; }))
        {
            removeOrbit(i);
        }
    }
}

/**
@details Provides the "+=" operator for adding orbit lists.
First assert that they have the same number of orbits or that this is empty and
then add equivalent sites of orbit i of rhs to orbit i to ->this
**/
OrbitList &OrbitList::operator+=(const OrbitList &rhs_ol)
{
    if (size() == 0)
    {
        _orbits = rhs_ol.getOrbits();
        return *this;
    }

    if (size() != rhs_ol.size())
    {
	    std::ostringstream msg;
        msg << "Left (" << size() << ") and right hand side (" << rhs_ol.size();
        msg << ") differ in size (OrbitList& operator+=).";
        throw std::runtime_error(msg.str());
    }

    for (size_t i = 0; i < rhs_ol.size(); i++)
    {
        _orbits[i] += rhs_ol.getOrbit(i);
    }
    return *this;
}
