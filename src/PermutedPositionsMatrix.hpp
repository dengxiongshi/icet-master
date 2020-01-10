#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>

class PermutedPositionsMatrix
{
public:
    PermutedPositionsMatrix(const std::vector<Eigen::Vector3d> &translations,
                            const std::vector<Eigen::Matrix3d> &rotations)
    {
        _translations = translations;
        _rotations = rotations;
    }

    /// Build the permutation map using the input positions
    void build(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> &positions);
    std::vector<std::vector<Eigen::Vector3d>> getPermutedPositions() const { return _permutedPositions;
     }

private:
    std::vector<Eigen::Vector3d> _translations;

    std::vector<Eigen::Matrix3d> _rotations;

    std::vector<std::vector<Eigen::Vector3d>> _permutedPositions;
};
