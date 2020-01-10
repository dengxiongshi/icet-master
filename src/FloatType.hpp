#pragma once

#include <Eigen/Dense>

namespace icet {

    /// A small epsilon value for floats.
    #define FLOATTYPE_EPSILON 1e-9

    /// Returns number rounded to specified tolerance.
    double roundDouble(const double &, const double);

    /// Rounds a vector.
    void roundVector3d(Eigen::Vector3d &, const double);

}
