#include "FloatType.hpp"

namespace icet {

    /**
    @brief Returns number rounded to specified tolerance.
    @details Computes tolerance * round(value / tolerance).
    @param value number to operate on
    @param tolerance
    */
    double roundDouble(const double &value, const double tolerance)
    {
        return round(value / tolerance) * tolerance;
    }

    /// Rounds a vector.
    void roundVector3d(Eigen::Vector3d &vector, const double tolerance)
    {
        for (int i = 0; i < 3; i++)
        {
            vector[i] = roundDouble(vector[i], tolerance);
        }
    }

}
