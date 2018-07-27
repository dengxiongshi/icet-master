#include "OrbitList.hpp"

/**
@todo Think about adding a string tag here to keep track of different orbit lists.
@todo Commented out lines of code may rot even if they are used for debugging. Those lines should be cleaned up. 
*/
OrbitList::OrbitList()
{
    // Empty constructor
}

/**
@details Constructs an OrbitList object from a many body neighbor-list and a primitive structure.
@param neighbor_lists list of NeighborList objects.
@param structure primitive atomic structure.
*/
OrbitList::OrbitList(const std::vector<NeighborList> &neighbor_lists, const Structure &structure)
{
    _primitiveStructure = structure;
    std::unordered_map<Cluster, int> clusterIndexMap;
    ManyBodyNeighborList mbnl = ManyBodyNeighborList();

    for (int index = 0; index < structure.size(); index++)
    {
        mbnl.build(neighbor_lists, index, false); //bothways=false
        for (size_t i = 0; i < mbnl.getNumberOfSites(); i++)
        {
            // Special case for singlet
            if (mbnl.getNumberOfSites(i) == 0)
            {
                std::vector<LatticeSite> sites = mbnl.getSites(i, 0);
                Cluster cluster = Cluster(structure, sites);
                addClusterToOrbitList(cluster, sites, clusterIndexMap);
            }

            for (size_t j = 0; j < mbnl.getNumberOfSites(i); j++)
            {
                std::vector<LatticeSite> sites = mbnl.getSites(i, j);
                Cluster cluster = Cluster(structure, sites);
                addClusterToOrbitList(cluster, sites, clusterIndexMap);
            }
        }
    }
    bool debug = true;

    for (auto &orbit : _orbitList)
    {
        orbit.sortOrbit();
    }

    if (debug)
    {
        checkEquivalentClusters();
    }
}

/**
@details Adds a cluster to orbit list. If cluster exists, then add sites. 
Otherwise a new orbit is created from the cluster and appended to orbit list.
@param cluster a Cluster object.
@param sites equivalent sites.
@param clusterIndexMap
@todo Complete the description for the parameters.
*/
void OrbitList::addClusterToOrbitList(const Cluster &cluster, const std::vector<LatticeSite> &sites, std::unordered_map<Cluster, int> &clusterIndexMap)
{
    int orbitNumber = findOrbit(cluster, clusterIndexMap);
    if (orbitNumber == -1)
    {
        Orbit newOrbit = Orbit(cluster);
        addOrbit(newOrbit);
        _orbitList.back().addEquivalentSites(sites);
        clusterIndexMap[cluster] = _orbitList.size() - 1;
        _orbitList.back().sortOrbit();
    }
    else
    {
        _orbitList[orbitNumber].addEquivalentSites(sites, true);
    }
}

/**
@details Returns the index in the orbit list of the orbit for which the
input cluster is the representative cluster. If orbit is not found, returns -1.
@param cluster a Cluster object.
*/
int OrbitList::findOrbit(const Cluster &cluster) const
{
    for (size_t i = 0; i < _orbitList.size(); i++)
    {
        if (_orbitList[i].getRepresentativeCluster() == cluster)
        {
            return i;
        }
    }
    return -1;
}

/**
@details Returns the index in the orbit list of the orbit for which the
input cluster is listed in the cluster index map. If orbit is not found, returns -1.
@param cluster an icet Cluster object
@param clusterIndexMap
*/
int OrbitList::findOrbit(const Cluster &cluster, const std::unordered_map<Cluster, int> &clusterIndexMap) const
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
@details This constructor creates an OrbitList object from a primitive.
structure, a permutation matrix, and a many body neighbor-list.
@param structure primitive atomic structure.
@param permutation_matrix permutation matrix.
@param neighbor_lists list of NeighborList objects.
@todo This constructor needs a more detailed step-by-step description.
*/
OrbitList::OrbitList(const Structure &structure, const std::vector<std::vector<LatticeSite>> &permutation_matrix, const std::vector<NeighborList> &neighbor_lists)
{
    _primitiveStructure = structure;
    _permutation_matrix = permutation_matrix;
    std::vector<std::vector<std::vector<LatticeSite>>> lattice_neighbors;
    
    /// @todo This seems to be unnecessary.
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> many_bodyNeighborIndices;
    
    // Set up a many body neighbor-list.
    bool saveBothWays = false;
    ManyBodyNeighborList mbnl = ManyBodyNeighborList();

    // If [0,1,2] exists in taken_rows then these three rows (with columns) have been accounted for and should not be looked at
    /// @todo Clean up this description of taken rows.
    std::unordered_set<std::vector<int>, VectorHash> taken_rows;
    
    // Get the first column of the permutation matrix which contains all the original (non-permuted) lattice sites.
    std::vector<LatticeSite> col1 = getColumn1FromPM(permutation_matrix, false);
    _column1 = col1;

    // Check all lattice sites in column1 are uniques.
    /// @todo Uniqueness of these elements must be verified during the implementation of permutation matrix.
    std::set<LatticeSite> col1_uniques(col1.begin(), col1.end());
    if (col1.size() != col1_uniques.size())
    {
        std::string errMSG = "Found duplicates in column1 of permutation matrix " + std::to_string(col1.size()) + " != " + std::to_string(col1_uniques.size());
        throw std::runtime_error(errMSG);
    }
    // Iterates through the atoms in the structure used to build the neighbor-list.
    for (size_t index = 0; index < neighbor_lists[0].size(); index++)
    {
        // Build a many body neighbor-list for each atom. 
        std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> mbnl_latnbrs = mbnl.build(neighbor_lists, index, saveBothWays);
        for (const auto &mbnl_pair : mbnl_latnbrs)
        {
            for (const auto &latnbr : mbnl_pair.second)
            {
                // Create a vector of lattice neighbors.
                std::vector<LatticeSite> lat_nbrs = mbnl_pair.first;
                lat_nbrs.push_back(latnbr);

                // Check that lattice neighbors are sorted in the vector.
                auto lat_nbrs_copy = lat_nbrs;
                std::sort(lat_nbrs_copy.begin(), lat_nbrs_copy.end());
                if (lat_nbrs_copy != lat_nbrs)
                {
                    throw std::runtime_error("lattice neighbors sites are not sorted");
                }

                // Translated lattice neighbors sites to unitcell.
                std::vector<std::vector<LatticeSite>> translatedSites = getSitesTranslatedToUnitcell(lat_nbrs, false);
                
                /// @todo This seems to be unnecessary.
                int missedSites = 0;

                // Add permuted sites to the input lattice neighbors.
                auto sites_index_pair = getMatchesInPM(translatedSites);
                if (!isRowsTaken(taken_rows, sites_index_pair[0].second))
                {
                    addPermutationMatrixColumns(lattice_neighbors, taken_rows, sites_index_pair[0].first, sites_index_pair[0].second, permutation_matrix, col1, true);
                }
            }

            // Add permuted sites for special singlet cases
            if (mbnl_pair.second.size() == 0)
            {
                std::vector<LatticeSite> lat_nbrs = mbnl_pair.first;
                auto pm_rows = findRowsFromCol1(col1, lat_nbrs);
                // auto find = taken_rows.find(pm_rows);
                // if (find == taken_rows.end())
                if (!isRowsTaken(taken_rows, pm_rows))
                {
                    addPermutationMatrixColumns(lattice_neighbors, taken_rows, lat_nbrs, pm_rows, permutation_matrix, col1, true);
                }
            }
        }
    }

    for (int i = 0; i < lattice_neighbors.size(); i++)
    {
        std::sort(lattice_neighbors[i].begin(), lattice_neighbors[i].end());
    }

    // Add orbit once permuted sites have been added to lattice neighbors
    addOrbitsFromPM(structure, lattice_neighbors);

    // 
    /// @todo Rename this function
    addPermutationInformationToOrbits(col1, permutation_matrix);
    bool debug = true;

    if (debug)
    {
        checkEquivalentClusters();
        // std::cout << "Done checking equivalent structures" << std::endl;
    }
}

// /// Check if the indices, when sorted, exits in taken_rows
// bool OrbitList::isRowsTaken(const std::vector<int> indices, const std::unordered_set<std::vector<int>, VectorHash> &taken_rows) const
// {
//     std::sort(indices.begin(),indices.end());
//     auto find = taken_rows.find(indices);
//     return find != taken_rows.end();

// }

/**
    Add permutation stuff to orbits

    steps:

    For each orbit:

    1. Take representative sites
    2. Find the rows these sites belong to (also find the unit cell offsets equivalent sites??)
    3. Get all columns for these rows, i.e the sites that are directly equivalent, call these p_equal.
    4. Construct all possible permutations for the representative sites, call these p_all
    5. Construct the intersect of p_equal and p_all, call this p_allowed_permutations.
    6. Get the indice version of p_allowed_permutations and these are then the allowed permutations for this orbit.
    7. take the sites in the orbit:
        site exist in p_all?:
            those sites are then related to representative_sites through the permutation
        else:
           loop over permutations of the sites:
              does the permutation exist in p_all?:
                 that permutation is then related to rep_sites through that permutation
              else:
                 continue



*/
void OrbitList::addPermutationInformationToOrbits(const std::vector<LatticeSite> &col1, const std::vector<std::vector<LatticeSite>> &permutation_matrix)
{
    for (size_t i = 0; i < size(); i++)
    {

        bool sortRows = false;

        // step one: Take representative sites
        std::vector<LatticeSite> representativeSites_i = _orbitList[i].getRepresentativeSites();
        auto translatedRepresentativeSites = getSitesTranslatedToUnitcell(representativeSites_i, sortRows);

        // step two: Find the rows these sites belong to and,

        // step three: Get all columns for these rows
        std::vector<std::vector<LatticeSite>> all_translated_p_equal;

        for (auto translated_rep_sites : translatedRepresentativeSites)
        {
            auto p_equal_i = getAllColumnsFromSites(translated_rep_sites, col1, permutation_matrix);
            all_translated_p_equal.insert(all_translated_p_equal.end(), p_equal_i.begin(), p_equal_i.end());
        }

        std::sort(all_translated_p_equal.begin(), all_translated_p_equal.end());

        // Step four: Construct all possible permutations for the representative sites
        std::vector<std::vector<LatticeSite>> p_all_with_translated_equivalent;
        for (auto translated_rep_sites : translatedRepresentativeSites)
        {
            std::vector<std::vector<LatticeSite>> p_all_i = icet::getAllPermutations<LatticeSite>(translated_rep_sites);
            p_all_with_translated_equivalent.insert(p_all_with_translated_equivalent.end(), p_all_i.begin(), p_all_i.end());
        }
        std::sort(p_all_with_translated_equivalent.begin(), p_all_with_translated_equivalent.end());

        // Step five:  Construct the intersect of p_equal and p_all
        std::vector<std::vector<LatticeSite>> p_allowed_permutations;
        std::set_intersection(all_translated_p_equal.begin(), all_translated_p_equal.end(),
                              p_all_with_translated_equivalent.begin(), p_all_with_translated_equivalent.end(),
                              std::back_inserter(p_allowed_permutations));

        // Step six: Get the indice version of p_allowed_permutations
        std::unordered_set<std::vector<int>, VectorHash> allowedPermutations;
        for (const auto &p_lattNbr : p_allowed_permutations)
        {
            int failedLoops = 0;
            for (auto translated_rep_sites : translatedRepresentativeSites)
            {
                try
                {
                    std::vector<int> allowedPermutation = icet::getPermutation<LatticeSite>(translated_rep_sites, p_lattNbr);
                    allowedPermutations.insert(allowedPermutation);
                }
                catch (const std::runtime_error &e)
                {
                    {
                        failedLoops++;
                        if (failedLoops == translatedRepresentativeSites.size())
                        {
                            throw std::runtime_error("Error: did not find any integer permutation from allowed permutation to any translated representative site ");
                        }
                        continue;
                    }
                }
            }
        }

        // std::cout << i << "/" << size() << " | " << representativeSites_i.size() << " " << std::endl;
        // Step 7
        const auto orbitSites = _orbitList[i].getEquivalentSites();
        std::unordered_set<std::vector<LatticeSite>> p_equal_set;
        p_equal_set.insert(all_translated_p_equal.begin(), all_translated_p_equal.end());

        std::vector<std::vector<int>> sitePermutations;
        sitePermutations.reserve(orbitSites.size());

        for (const auto &eqOrbitSites : orbitSites)
        {
            if (p_equal_set.find(eqOrbitSites) == p_equal_set.end())
            {
                // for (auto latNbr : eqOrbitSites)
                // {
                //     latNbr.print();
                // }
                // std::cout << "====" << std::endl;
                //Did not find the orbit.eq_sites in p_equal meaning that this eq site does not have an allowed permutation
                auto equivalently_translated_eqOrbitsites = getSitesTranslatedToUnitcell(eqOrbitSites, sortRows);
                std::vector<std::pair<std::vector<LatticeSite>, std::vector<LatticeSite>>> translatedPermutationsOfSites;
                for (const auto eq_trans_eqOrbitsites : equivalently_translated_eqOrbitsites)
                {
                    const auto allPermutationsOfSites_i = icet::getAllPermutations<LatticeSite>(eq_trans_eqOrbitsites);
                    for (const auto perm : allPermutationsOfSites_i)
                    {
                        translatedPermutationsOfSites.push_back(std::make_pair(perm, eq_trans_eqOrbitsites));
                    }
                    // translatedPermutationsOfSites.insert(translatedPermutationsOfSites.end(),allPermutationsOfSites_i.begin(), allPermutationsOfSites_i.end());
                }
                for (const auto &onePermPair : translatedPermutationsOfSites)
                {
                    // for (auto latNbr : onePermPair.first)
                    // {
                    //     std::cout << "\t";
                    //     latNbr.print();
                    // }
                    // std::cout << "----" << std::endl;

                    const auto findOnePerm = p_equal_set.find(onePermPair.first);
                    if (findOnePerm != p_equal_set.end()) // one perm is one of the equivalent sites. This means that eqOrbitSites is associated to p_equal
                    {
                        std::vector<int> permutationToEquivalentSites = icet::getPermutation<LatticeSite>(onePermPair.first, onePermPair.second);
                        sitePermutations.push_back(permutationToEquivalentSites);
                        break;
                    }
                    if (onePermPair == translatedPermutationsOfSites.back())
                    {

                        // std::cout << "Target sites " << std::endl;
                        // for (auto latNbrs : p_equal_set)
                        // {
                        //     for (auto latNbr : latNbrs)
                        //     {
                        //         latNbr.print();
                        //     }
                        //     std::cout << "-=-=-=-=-=-=-=" << std::endl;
                        // }
                        std::string errMSG = "Error: did not find a permutation of the orbit sites to the permutations of the representative sites";
                        throw std::runtime_error(errMSG);
                    }
                }
            }
            else
            {
                std::vector<int> permutationToEquivalentSites = icet::getPermutation<LatticeSite>(eqOrbitSites, eqOrbitSites); //the identical permutation
                sitePermutations.push_back(permutationToEquivalentSites);
            }
        }

        if (sitePermutations.size() != _orbitList[i].getEquivalentSites().size() || sitePermutations.size() == 0)
        {
            std::string errMSG = "Each set of site did not get a permutation " + std::to_string(sitePermutations.size()) + " != " + std::to_string(_orbitList[i].getEquivalentSites().size());
            throw std::runtime_error(errMSG);
        }

        _orbitList[i].setEquivalentSitesPermutations(sitePermutations);
        _orbitList[i].setAllowedSitesPermutations(allowedPermutations);
        ///debug prints

        // for (auto perm : allowedPermutations)
        // {
        //     for (auto i : perm)
        //     {
        //         std::cout << i << " ";
        //     }
        //     std::cout << " | ";
        // }
        // std::cout << std::endl;
        //    std::cout<<representativeSites.size()<< " "<<p_all.size()<< " "<< p_equal.size()<< " " << p_allowed_permutations.size()<<std::endl;
    }
}

/// Finds the sites in the first column of permutation matrix, extracts and returns all columns along with their unit cell translated indistinguishable sites
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromSites(const std::vector<LatticeSite> &sites,
                                                                        const std::vector<LatticeSite> &col1,
                                                                        const std::vector<std::vector<LatticeSite>> &permutation_matrix) const
{
    bool sortRows = false;
    std::vector<int> rowsFromCol1 = findRowsFromCol1(col1, sites, sortRows);
    std::vector<std::vector<LatticeSite>> p_equal = getAllColumnsFromRow(rowsFromCol1, permutation_matrix, true, sortRows);

    return p_equal;
}

/// Returns true if row indexes has already be taken into account.
bool OrbitList::isRowsTaken(const std::unordered_set<std::vector<int>, VectorHash> &taken_rows, std::vector<int> rows) const
{
    std::sort(rows.begin(), rows.end());

    const auto find = taken_rows.find(rows);
    if (find == taken_rows.end())
    {
        return false;
    }
    else
    {
        return true;
    }
}

/// Inserts a list of indexes corresponding to the rows of the permutation matrix to be accounted for.
void OrbitList::takeRows(std::unordered_set<std::vector<int>, VectorHash> &taken_rows, std::vector<int> rows) const
{
    std::sort(rows.begin(), rows.end());
    taken_rows.insert(rows);
}

/**
@details Returns all columns from the given rows in permutation matrix.

@param rows set of row indices .
@param permutation_matrix permutation matrix
@param includeTranslatedSites if true, it will also include the equivalent sites found
from the rows by moving each site into the unitcell
*/
std::vector<std::vector<LatticeSite>> OrbitList::getAllColumnsFromRow(const std::vector<int> &rows, const std::vector<std::vector<LatticeSite>> &permutation_matrix, bool includeTranslatedSites, bool sortIt) const
{

    std::vector<std::vector<LatticeSite>> allColumns;

    for (size_t column = 0; column < permutation_matrix[0].size(); column++)
    {

        std::vector<LatticeSite> indistinctLatNbrs;

        for (const int &row : rows)
        {
            indistinctLatNbrs.push_back(permutation_matrix[row][column]);
        }

        if (includeTranslatedSites)
        {
            auto translatedEquivalentSites = getSitesTranslatedToUnitcell(indistinctLatNbrs, sortIt);
            allColumns.insert(allColumns.end(), translatedEquivalentSites.begin(), translatedEquivalentSites.end());
        }
        else
        {
            allColumns.push_back(indistinctLatNbrs);
        }
    }
    return allColumns;
}

/**
@details This function will take the lattice neighbors, and for each site outside the unitcell 
will translate it inside the unitcell and translate the other sites with the same translation.

This translation will give rise to equivalent sites that sometimes are not found by using the
set of crystal symmetries given by spglib.

An added requirement is that if the primitive structure has non-pbc in a given direction
then this function should not give rise to any sites in that direction.

@param latticeNeighbors vector of lattice sites. 
@param sortIt if true, sort the sites in the translated sites.  
*/
std::vector<std::vector<LatticeSite>> OrbitList::getSitesTranslatedToUnitcell(const std::vector<LatticeSite> &latticeNeighbors, bool sortIt) const
{

    // Check that pbc is currently respected
    if (!isSitesPBCCorrect(latticeNeighbors))
    {
        throw std::runtime_error("A lattice neighbor with a repeated site in the unitcell direction with non-pbc has been passed to getSitesTranslatedToUnitCell function");
    }

    std::vector<std::vector<LatticeSite>> translatedLatticeSites;
    translatedLatticeSites.push_back(latticeNeighbors);
    Vector3d zeroVector = {0.0, 0.0, 0.0};
    for (int i = 0; i < latticeNeighbors.size(); i++)
    {
        if ((latticeNeighbors[i].unitcellOffset() - zeroVector).norm() > 0.1)
        {
            auto translatedSites = translateSites(latticeNeighbors, i);
            if (sortIt)
            {
                std::sort(translatedSites.begin(), translatedSites.end());
            }

            if (!isSitesPBCCorrect(translatedSites))
            {
                throw std::runtime_error("A repeated site in the unitcell direction with non-pbc was catched after call getSitesTranslatedToUnitCell function");
            }

            translatedLatticeSites.push_back(translatedSites);
        }
    }

    // Sort this so that the lowest vector of lattice sites will be chosen and
    // therefore the sorting of orbits should be consistent.
    std::sort(translatedLatticeSites.begin(), translatedLatticeSites.end());

    return translatedLatticeSites;
}

/// Check that the lattice neighbors do not have any unitcell offsets in a non-pbc direction
bool OrbitList::isSitesPBCCorrect(const std::vector<LatticeSite> &sites) const
{
    for (const auto &latNbr : sites)
    {
        for (int i = 0; i < 3; i++)
        {
            if (!_primitiveStructure.hasPBC(i) && latNbr.unitcellOffset()[i] != 0)
            {
                return false;
            }
        }
    }
    return true;
}

/// Take all lattice neighbors in vector latticeNeighbors and subtract the unitcelloffset of site latticeNeighbors[index]
std::vector<LatticeSite> OrbitList::translateSites(const std::vector<LatticeSite> &latticeNeighbors, const unsigned int index) const
{
    Vector3d offset = latticeNeighbors[index].unitcellOffset();
    auto translatedNeighbors = latticeNeighbors;
    for (auto &latNbr : translatedNeighbors)
    {
        latNbr.addUnitcellOffset(-offset);
    }
    return translatedNeighbors;
}

/// Debug function to check that all equivalent sites in every orbit give same sorted cluster
void OrbitList::checkEquivalentClusters() const
{

    for (const auto &orbit : _orbitList)
    {
        Cluster representative_cluster = orbit.getRepresentativeCluster();
        for (const auto &sites : orbit.getEquivalentSites())
        {
            Cluster equivalentCluster = Cluster(_primitiveStructure, sites);
            if (representative_cluster != equivalentCluster)
            {
                std::cout << " found an 'equivalent' cluster that was not equal representative cluster" << std::endl;
                std::cout << "representative_cluster:" << std::endl;
                representative_cluster.print();

                std::cout << "equivalentCluster:" << std::endl;
                equivalentCluster.print();

                throw std::runtime_error("found a \"equivalent\" cluster that were not equal representative cluster");
            }
            if (fabs(equivalentCluster.radius() - representative_cluster.radius()) > 1e-3)
            {
                std::cout << " found an 'equivalent' cluster that does not equal the representative cluster" << std::endl;
                std::cout << "representative_cluster:" << std::endl;
                representative_cluster.print();

                std::cout << "equivalentCluster:" << std::endl;
                equivalentCluster.print();
                std::cout << " test geometric size: " << icet::getGeometricalRadius(sites, _primitiveStructure) << " " << std::endl;
                throw std::runtime_error("found an 'equivalent' cluster that does not equal the representative cluster");
            }
        }
    }
}

/**
This function adds the lattice_neighbors container found in the constructor to the orbits.

Each outer vector is an orbit and the inner vectors are identical sites
*/

void OrbitList::addOrbitsFromPM(const Structure &structure, const std::vector<std::vector<std::vector<LatticeSite>>> &lattice_neighbors)
{

    for (const auto &equivalent_sites : lattice_neighbors)
    {
        addOrbitFromPM(structure, equivalent_sites);
    }
}

/// Adds equivalent sites as an orbit to orbit list.
void OrbitList::addOrbitFromPM(const Structure &structure, const std::vector<std::vector<LatticeSite>> &equivalent_sites)
{

    Cluster representativeCluster = Cluster(structure, equivalent_sites[0]);
    Orbit newOrbit = Orbit(representativeCluster);
    _orbitList.push_back(newOrbit);

    for (const auto &sites : equivalent_sites)
    {
        _orbitList.back().addEquivalentSites(sites);
    }
    _orbitList.back().sortOrbit();
}
/**
@details This function appends at most one lattice site per column to the lattice
neighbors from all columns of permutation matrix. The appended site corresponds
to a found match in colum1 of permuted sites. Even if a matched lattice site 
is not added, a record of its row index is saved in a set called taken_rows.

@param lattice_neighbor vector of NeighborList objects
@param taken_rows set of integers indicating row indices of the permutation matrix
@param lat_nbrs unused vector of lattice sites (to be removed)
@param pm_rows set of index of the permutation matrix rows to be considered
@param permutation_matrix permutation matrix
@param col1 first column of the permutation matrix
@param add if true, add lattice sites from permutation matrix (default true)

@todo lat_nbrs argument seems to be not used here.
@todo Boolean parameter *add* is only used for debugging.
*/
void OrbitList::addPermutationMatrixColumns(
    std::vector<std::vector<std::vector<LatticeSite>>> &lattice_neighbors, std::unordered_set<std::vector<int>, VectorHash> &taken_rows, const std::vector<LatticeSite> &lat_nbrs, const std::vector<int> &pm_rows,
    const std::vector<std::vector<LatticeSite>> &permutation_matrix, const std::vector<LatticeSite> &col1, bool add) const
{

    std::vector<std::vector<LatticeSite>> columnLatticeSites;
    columnLatticeSites.reserve(permutation_matrix[0].size());
    for (size_t column = 0; column < permutation_matrix[0].size(); column++)
    {
        std::vector<LatticeSite> indistinctLatNbrs;

        for (const int &row : pm_rows)
        {
            indistinctLatNbrs.push_back(permutation_matrix[row][column]);
        }

        auto translatedEquivalentSites = getSitesTranslatedToUnitcell(indistinctLatNbrs, false);

        auto sites_index_pair = getMatchesInPM(translatedEquivalentSites);
        // for (int i = 1; i < sites_index_pair.size(); i++)
        // {
        //     auto find = taken_rows.find(sites_index_pair[i].second);
        //     if( find == taken_rows.end())
        //     {

        //     }
        // }
        // auto find_first_validCluster = std::find_if(sites_index_pair.begin(), sites_index_pair.end(),[](const std::pair<std::vector<LatticeSite>,std::vector<int>> &site_index_pair){return validatedCluster(site_index_pair.second);});
        // auto find = taken_rows.find(sites_index_pair[0].second);
        bool findOnlyOne = true;
        // if (find == taken_rows.end())
        if (!isRowsTaken(taken_rows, sites_index_pair[0].second))
        {
            for (int i = 0; i < sites_index_pair.size(); i++)
            {
                // find = taken_rows.find(sites_index_pair[i].second);
                // if (find == taken_rows.end())
                if (!isRowsTaken(taken_rows, sites_index_pair[i].second))
                {
                    if (add && findOnlyOne && validatedCluster(sites_index_pair[i].first))
                    {
                        columnLatticeSites.push_back(sites_index_pair[0].first);
                        findOnlyOne = false;
                    }
                    // taken_rows.insert(sites_index_pair[i].second);
                    takeRows(taken_rows, sites_index_pair[i].second);
                }
            }
            // taken_rows.insert(sites_index_pair[0].second);
            // if (add && validatedCluster(sites_index_pair[0].first))
            // {
            //     columnLatticeSites.push_back(sites_index_pair[0].first);
            // }
            // for (int i = 1; i < sites_index_pair.size(); i++)
            // {
            //     find = taken_rows.find(sites_index_pair[i].second);
            //     if (find == taken_rows.end())
            //     {
            //         taken_rows.insert(sites_index_pair[i].second);
            //     }
            // }
        }
    }
    if (columnLatticeSites.size() > 0)
    {
        lattice_neighbors.push_back(columnLatticeSites);
    }
}

/**
@details This function looks for a set of elements in the column1 that matches the input translated sites.
When a match is found, returns a pair with the matched sites and their row indexes.

@param translatedSites vector of LatticeSite objects

@todo Current name seems not obvious when one is actually matching againts column1.
*/
std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> OrbitList::getMatchesInPM(const std::vector<std::vector<LatticeSite>> &translatedSites) const
{
    std::vector<int> perm_matrix_rows;
    std::vector<std::pair<std::vector<LatticeSite>, std::vector<int>>> matchedSites;
    for (const auto &sites : translatedSites)
    {
        try
        {
            perm_matrix_rows = findRowsFromCol1(_column1, sites, false);
        }
        catch (const std::runtime_error)
        {
            continue;
        }
        //no error here indicating we found matching rows in col1
        matchedSites.push_back(std::make_pair(sites, perm_matrix_rows));
    }
    if (matchedSites.size() > 0)
    {
        return matchedSites;
    }
    else
    {
        //we found no matching rows in permutation matrix, this should not happen so we throw an error

        //first print some debug info
        std::cout << "number of translated sites: " << translatedSites.size() << std::endl;
        std::cout << "sites: " << std::endl;
        for (auto latnbrs : translatedSites)
        {
            for (auto latnbr : latnbrs)
            {
                latnbr.print();
            }
            std::cout << " ========= " << std::endl;
        }
        std::cout << "col1:" << std::endl;
        for (auto row : _column1)
        {
            row.print();
        }
        throw std::runtime_error("Did not find any of the translated sites in col1 of permutation matrix in function getFirstMatchInPM in orbit list");
    }
}

/**
Checks that at least one lattice neigbhor originates in the unit cell, i.e. has one cell offset = [0,0,0]
*/
bool OrbitList::validatedCluster(const std::vector<LatticeSite> &latticeNeighbors) const
{
    Vector3d zeroVector = {0., 0., 0.};
    for (const auto &latNbr : latticeNeighbors)
    {
        if (latNbr.unitcellOffset() == zeroVector)
        {
            return true;
        }
    }
    return false;
}

/**
Searches for lattice neighbors in column1 of permutation matrix and find the corresponding rows.
*/
std::vector<int> OrbitList::findRowsFromCol1(const std::vector<LatticeSite> &col1, const std::vector<LatticeSite> &latNbrs, bool sortIt) const
{
    std::vector<int> rows;
    for (const auto &latNbr : latNbrs)
    {
        const auto find = std::find(col1.begin(), col1.end(), latNbr);
        if (find == col1.end())
        {
            //for (const auto &latNbrp : latNbrs)
            //{
                //latNbrp.print();
            //}
            //  latNbr.print();
            throw std::runtime_error("lattice neigbhor is not found in column1 in function findRowsFromCol1");
        }
        else
        {
            int row_in_col1 = std::distance(col1.begin(), find);
            rows.push_back(row_in_col1);
        }
    }
    if (sortIt)
    {
        std::sort(rows.begin(), rows.end());
    }
    return rows;
}

/**
@details This function returns the first column of the permutation matrix.
@param permutation_matrix permutation matrix
@param sortIt if true it will sort column1 (default true)
*/
std::vector<LatticeSite> OrbitList::getColumn1FromPM(const std::vector<std::vector<LatticeSite>> &permutation_matrix, bool sortIt) const
{
    std::vector<LatticeSite> col1;
    col1.reserve(permutation_matrix[0].size());
    for (const auto &row : permutation_matrix)
    {
        col1.push_back(row[0]);
    }
    if (sortIt)
    {
        std::sort(col1.begin(), col1.end());
    }
    return col1;
}

/**
@details Returns a supercell orbit by translating and mapping an orbit to the supercell structure.

@param superCell a supercell atomic structure.
@param cellOffset the offset you translate the orbit with.
@param orbitIndex index of the orbit.
@param primToSuperMap maps primitive lattice neighbors to lattice neighbors in the supercell.
*/
Orbit OrbitList::getSuperCellOrbit(const Structure &superCell, const Vector3d &cellOffset, const unsigned int orbitIndex, std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const
{
    if (orbitIndex >= _orbitList.size())
    {
        std::string errorMsg = "Error: orbitIndex out of range in OrbitList::getSuperCellOrbit " + std::to_string(orbitIndex) + " >= " + std::to_string(_orbitList.size());
        throw std::out_of_range(errorMsg);
    }

    Orbit superCellOrbit = _orbitList[orbitIndex] + cellOffset;

    auto equivalentSites = superCellOrbit.getEquivalentSites();

    for (auto &sites : equivalentSites)
    {
        for (auto &site : sites)
        {
            transformSiteToSupercell(site, superCell, primToSuperMap);
        }
    }

    superCellOrbit.setEquivalentSites(equivalentSites);
    return superCellOrbit;
}

/**
Takes the site and tries to find it in the map to supercell

if it does not find it it gets the xyz position and then find the lattice neighbor in the supercell corresponding to that position and adds it to the map

in the end site is modified to correspond to the index, offset of the supercell
*/
void OrbitList::transformSiteToSupercell(LatticeSite &site, const Structure &superCell, std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const
{
    auto find = primToSuperMap.find(site);
    LatticeSite supercellSite;
    if (find == primToSuperMap.end())
    {
        Vector3d sitePosition = _primitiveStructure.getPosition(site);
        supercellSite = superCell.findLatticeSiteByPosition(sitePosition);
        primToSuperMap[site] = supercellSite;
    }
    else
    {
        supercellSite = primToSuperMap[site];
    }

    //write over site to match supercell index offset
    site.setIndex(supercellSite.index());
    site.setUnitcellOffset(supercellSite.unitcellOffset());
}

///Create and return a "local" orbitList by offsetting each site in the primitve by cellOffset
OrbitList OrbitList::getLocalOrbitList(const Structure &superCell, const Vector3d &cellOffset, std::unordered_map<LatticeSite, LatticeSite> &primToSuperMap) const
{
    OrbitList localOrbitList = OrbitList();
    localOrbitList.setPrimitiveStructure(_primitiveStructure);

    for (size_t orbitIndex = 0; orbitIndex < _orbitList.size(); orbitIndex++)
    {
        localOrbitList.addOrbit(getSuperCellOrbit(superCell, cellOffset, orbitIndex, primToSuperMap));
    }
    return localOrbitList;
}
