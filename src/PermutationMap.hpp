#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <pybind11/stl.h>
#include <map>
#include <unordered_map>
#include <algorithm>

using namespace Eigen;

namespace py = pybind11;

///@TODO: Think about renaming this to permutation matrix since it doesnt really map anything

class PermutationMap
{
  public:
    PermutationMap(const std::vector<Vector3d> &translations,
                   const std::vector<Matrix3d> &rotations)
    {
        _translations = translations;
        _rotations = rotations;
        symprec = 1e-7;
    }

    ///Build the permutationmap using the input positions
    void build(const Eigen::Matrix<double, Dynamic, 3, RowMajor> &positions);

    std::vector<std::vector<Vector3d>> getPermutedPositions() const
    {
        return _permutedPositions;
    }

    /**
     Returns indices for unique positions as well as the representative positions,
     same indices share the same position
    */
    std::pair<std::vector<std::vector<int>>, std::vector<Vector3d>> getIndexedPermutedPositions()
    {
        std::vector<Vector3d> uniquePositions;
        std::vector<std::vector<int>> indicePositions(_permutedPositions.size(), std::vector<int>(_permutedPositions[0].size()));
        for (size_t row = 0; row < _permutedPositions.size(); row++)
        {
            for (size_t col = 0; col < _permutedPositions[0].size(); col++)
            {

                Vector3d pos = _permutedPositions[row][col];
                const auto find = std::find(uniquePositions.begin(), uniquePositions.end(), pos);
                if (find == uniquePositions.end())
                {
                    uniquePositions.push_back(pos);
                    indicePositions[row][col] = uniquePositions.size() - 1;
                }
                else
                {
                    indicePositions[row][col] = std::distance(uniquePositions.begin(), find);
                }
            }
        }

        return std::make_pair(indicePositions, uniquePositions);
    }

  private:
    std::vector<Vector3d> _translations;
    std::vector<Matrix3d> _rotations;
    std::vector<std::vector<Vector3d>> _permutedPositions;

    /** 
    Help function to round a Vector3d for easier comparing the transmutated positions

    @TODO: move this outside the class into some type of support/tools collection
    */
    void roundVector3d(Vector3d &vec3d)
    {
        vec3d[0] = round(vec3d[0] * 1.0 / symprec) / (1.0 / symprec);
        vec3d[1] = round(vec3d[1] * 1.0 / symprec) / (1.0 / symprec);
        vec3d[2] = round(vec3d[2] * 1.0 / symprec) / (1.0 / symprec);
    }
    double symprec;
};
