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

std::vector<std::pair<std::vector<LatticeNeighbor>, std::vector<LatticeNeighbor>>> ManybodyNeighborlist::build(const std::vector<Neighborlist> &neighborlists, int index, bool saveBothWays)
{

    std::vector<std::pair<std::vector<LatticeNeighbor>, std::vector<LatticeNeighbor>>> manybodyNeighborIndices;
    for(size_t c = 2; c < neighborlists.size()+2; c++)
    {
        auto Ni = neighborlists[c-2].getNeighbors(index);
        int numberOfSites = neighborlists[c-2].size();
        Vector3d zeroVector = {0.0, 0.0, 0.0};
        std::vector<LatticeNeighbor> currentOriginalNeighbors;
        currentOriginalNeighbors.push_back(LatticeNeighbor(index, zeroVector)); // index is always first index

        combineToHigherOrder(neighborlists[c-2], manybodyNeighborIndices, Ni, currentOriginalNeighbors, saveBothWays, c);   
    }
    return manybodyNeighborIndices;
}




/*
    for each j in Ni construct the intersect of N_j and N_i = N_ij.
    all k in N_ij are then neighbors with i,j
    what is saved is then i,j and N_ij up to the desired order "maxorder"
*/
void ManybodyNeighborlist::combineToHigherOrder(const Neighborlist &nl,
                                                std::vector<std::pair<std::vector<LatticeNeighbor>, std::vector<LatticeNeighbor>>> &manybodyNeighborIndices,
                                                const std::vector<LatticeNeighbor> &Ni, std::vector<LatticeNeighbor> &currentOriginalNeighbors, bool saveBothWays, const int maxOrder)
{

    for (const auto &j : Ni)
    {
        //if j is smaller than last added site then continue
        // if bothways = True then don't compare to first
        bool cont = false;

        if( saveBothWays )
        {
            if(currentOriginalNeighbors.size() > 1)
            {
                if(j < currentOriginalNeighbors.back())
                {
                    cont = true;
                }
            }
        }
        else
        {
            if(j < currentOriginalNeighbors.back()) 
            {
                cont = true;
            }
        }
        if(cont)
        {
            continue;
        }
        // if ((!saveBothWays && currentOriginalNeighbors.size() == 1) && j < currentOriginalNeighbors.back())
        // {
        //     continue;
        // }

        auto originalNeighborCopy = currentOriginalNeighbors;
        originalNeighborCopy.push_back(j); // put j in originalNeigbhors

        auto Nj = nl.getNeighbors(j.index);

        //translate the neighbors
        translateAllNi(Nj, j.unitcellOffset);

        //exclude smaller neighbors
        if(!saveBothWays)
        {
            Nj = getFilteredNj(Nj, j);
        }
   
        

        //construct the intersection
        const auto intersection_N_ij = getIntersection(Ni, Nj);

        if (originalNeighborCopy.size() + 1 < maxOrder)
        {
            combineToHigherOrder(nl, manybodyNeighborIndices, intersection_N_ij, originalNeighborCopy, saveBothWays, maxOrder);
        }

        if (intersection_N_ij.size() > 0 && originalNeighborCopy.size() == (maxOrder - 1) )
        {
            manybodyNeighborIndices.push_back(std::make_pair(originalNeighborCopy, intersection_N_ij));
        }
    }
}

/*
Since N_j is always sorted then simply search for first k in N_j that have k>= j
and then filtered are from indexof(k) to end()

*/
std::vector<LatticeNeighbor> ManybodyNeighborlist::getFilteredNj(const std::vector<LatticeNeighbor> &N_j, const LatticeNeighbor &j) const
{
    auto first = std::upper_bound(N_j.begin(), N_j.end(), j);

    std::vector<LatticeNeighbor> ret(first, N_j.end());
    return ret;
}

/**
    Offsets all indice, offsets pairs in Ni with the input offset, e.g:
    For all j in Ni:
     offset j.offset with "unitCellOffset"

*/
void ManybodyNeighborlist::translateAllNi(std::vector<LatticeNeighbor> &Ni, const Vector3d &unitCellOffset) const
{
    for (auto &latNbr : Ni)
    {
        latNbr.unitcellOffset += unitCellOffset;
    }
}
