#include "ManybodyNeighborlist.hpp"

/**
    This will use the neighborlist to combine all possible neighbors up to the given order.
    The output will be: std::vector<std::pair<originalNeighbors, manyNeigbhors>>
    the many body neigbhors can be retrieved by doing:
    for (const auto nbr : manybodyNeighborIndices)
    {
        std::vector<std::pair<int,Vector3d>> neighbors = nbr.first; // this are the first orignal neighbors
        for(const auto manynbr : nbr.second)
        {
            manybody_neigbhor = neighbors;
            manybody_neigbhor.append(manynbr);
        }
    }

    this means that if originalNeigbhors.size() == 2 then for each lattice site in manyNeigbhors 
    you can combine it with originalNeigbhors to get all triplets that have these first two original neighbors (lattice indices)


    saveBothWays : bool
    if true then both i,j,k and j,i,k etc.. will be saved
    otherwise only i,j,k will be saved if i < j < k
*/

std::vector<std::pair<std::vector<std::pair<int, Vector3d>>, std::vector<std::pair<int, Vector3d>>>> ManybodyNeighborlist::build(const Neighborlist &nl, int index, int maxOrder, bool saveBothWays)
{

    std::vector<std::pair<std::vector<std::pair<int, Vector3d>>, std::vector<std::pair<int, Vector3d>>>> manybodyNeighborIndices;
    auto Ni = nl.getNeighbors(index);
    int numberOfSites = nl.size();

    int c = 2;
    Vector3d zeroVector = {0.0, 0.0, 0.0};
    std::vector<std::pair<int, Vector3d>> currentOriginalNeighbors;
    currentOriginalNeighbors.push_back(std::make_pair(index, zeroVector)); // index is always first index

    combineToHigherOrder(nl, manybodyNeighborIndices, Ni, currentOriginalNeighbors, c, saveBothWays, maxOrder);
    // }

    return manybodyNeighborIndices;
}



/*
    for each j in Ni construct the intersect of N_j and N_i = N_ij.
    all k in N_ij are then neighbors with i,j
    what is saved is then i,j and N_ij up to the desired order "maxorder"
*/
void ManybodyNeighborlist::combineToHigherOrder(const Neighborlist &nl,
                                                std::vector<std::pair<std::vector<std::pair<int, Vector3d>>, std::vector<std::pair<int, Vector3d>>>> &manybodyNeighborIndices,
                                                const std::vector<std::pair<int, Vector3d>> &Ni, std::vector<std::pair<int, Vector3d>> &currentOriginalNeighbors, int order, bool saveBothWays, const int maxOrder)
{
        NeighborPairCompare comp;
    for (const auto &j : Ni)
    {
        //if j is smaller than last added site then continue
        // if bothways = True then don't compare to first
        if ((!saveBothWays && currentOriginalNeighbors.size() == 1) && comp(j, currentOriginalNeighbors.back()))
        {
            continue;
        }

        auto originalNeighborCopy = currentOriginalNeighbors;
        originalNeighborCopy.push_back(j); // put j in originalNeigbhors

        auto Nj = nl.getNeighbors(j.first);
        translateAllNi(Nj, j.second);

        //exclude smaller neighbors
        const auto N_j_filtered = getFilteredNj(Nj, j);        

        const auto intersection_N_ij = getIntersection(Ni, N_j_filtered);


        if (originalNeighborCopy.size() + 1 < maxOrder)
        {
            combineToHigherOrder(nl, manybodyNeighborIndices, intersection_N_ij, originalNeighborCopy, order++, saveBothWays, maxOrder);
        }

        if (intersection_N_ij.size() > 0)
        {
            manybodyNeighborIndices.push_back(std::make_pair(originalNeighborCopy, intersection_N_ij));
        }
    }
}

/*
Since N_j is always sorted then simply search for first k in N_j that have k>= j
and then filtered are from indexof(k) to end()

*/
std::vector<std::pair<int, Vector3d>> ManybodyNeighborlist::getFilteredNj(const std::vector<std::pair<int, Vector3d>> &N_j, const std::pair<int, Vector3d> &j) const
{
    NeighborPairCompare comp;
    auto first = std::upper_bound(N_j.begin(), N_j.end(), j,comp);

    std::vector<std::pair<int, Vector3d>> ret(first, N_j.end());
    return ret;
}

/**
    Offsets all indice, offsets pairs in Ni with the input offset, e.g:
    For all j in Ni:
     offset j.offset with "unitCellOffset"

*/
void ManybodyNeighborlist::translateAllNi(std::vector<std::pair<int, Vector3d>> &Ni, const Vector3d &unitCellOffset) const
{
    for (auto &latNbr : Ni)
    {
        latNbr.second += unitCellOffset;
    }
    
}
