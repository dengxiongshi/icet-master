#pragma once

#include <Eigen/Dense>

#include "LatticeSite.hpp"

using namespace Eigen;

/**
  @brief Class for storing a structure.
  @details This class stores the cell metric, positions, chemical symbols, and
  periodic boundary conditions that describe a structure. It also holds
  information pertaining to the components that are allowed on each site and
  provides functionality for computing distances between sites.
*/
class Structure
{
  public:

    /// Default constructor.
    Structure() {};

    /// Overloaded constructor.
    Structure(const Matrix<double, Dynamic, 3, RowMajor> &,
              const std::vector<std::string> &,
              const Matrix3d &,
              const std::vector<bool> &);

    /// Returns distance vector between two sites.
    double getDistance(const size_t, const size_t, const Vector3d, const Vector3d) const;

    /// Return the position of a site in Cartesian coordinates.
    Vector3d getPosition(const LatticeSite &) const;

    /// Return the position of a site in Cartesian coordinates.
    Vector3d getPositionByIndex(const size_t &index) const;

    /// Returns atomic number of site.
    int getAtomicNumber(const size_t) const;

    /// Returns the list of unique sites.
    std::vector<size_t> getUniqueSites() const { return _uniqueSites; }

    /// Set list of unique sites.
    /// @todo add example for how the unique sites are supposed to work.
    void setUniqueSites(const std::vector<size_t> &);

    /// Returns a unique site.
    size_t getUniqueSite(const size_t) const;

    /// Return LatticeSite object that matches the given position.
    LatticeSite findLatticeSiteByPosition(const Vector3d &, const double) const;

    /// Return list of LatticeSite objects that matche a given list of positions.
    std::vector<LatticeSite> findLatticeSitesByPositions(const std::vector<Vector3d> &, const double) const;

  public:

    /// Returns the size of the structure, i.e., the number of sites.
    size_t size() const { return (_atomicNumbers.size()); }

    /// Set the atomic positions.
    void setPositions(const Matrix<double, Dynamic, 3> &positions) { _positions = positions; }

    /// Returns positions.
    Matrix<double, Dynamic, 3, RowMajor> getPositions() const { return _positions; }

    /// Set atomic numbers.
    void setAtomicNumbers(const std::vector<int> &atomicNumbers) { _atomicNumbers = atomicNumbers; }

    /// Returns atomic numbers.
    std::vector<int> getAtomicNumbers() const { return _atomicNumbers; }

    /// Set atomic numbers via chemical symbols.
    void setChemicalSymbols(const std::vector<std::string> &chemicalSymbols) { setAtomicNumbers(convertChemicalSymbolsToAtomicNumbers(chemicalSymbols)); }

    /// Returns chemical symbols.
    std::vector<std::string> getChemicalSymbols() const { return convertAtomicNumbersToChemicalSymbols(_atomicNumbers); }

    /// Returns periodic boundary condition along direction k.
    bool hasPBC(const int k) const { return _pbc[k]; }

    /// Returns periodic boundary conditions.
    std::vector<bool> getPBC() const { return _pbc; }

    /// Set periodic boundary conditions.
    void setPBC(const std::vector<bool> pbc) { _pbc = pbc; }

    /// Set the cell metric.
    void setCell(const Matrix<double, 3, 3> &cell) { _cell = cell; }

    /// Returns the cell metric.
    Matrix<double, 3, 3> getCell() const { return _cell; }

    /// Set allowed components for each site by vector.
    void setNumberOfAllowedSpecies(const std::vector<int> &);

    /// Set allowed components for all sites to the same value.
    void setNumberOfAllowedSpecies(const int);

    /// Returns number of allowed components on each site.
    int getNumberOfAllowedSpeciesBySite(const size_t) const;

    /// Returns number of allowed components on each site.
    std::vector<int> getNumberOfAllowedSpeciesBySites(const std::vector<LatticeSite> &) const;

    /// List of atomic numbers.
    std::vector<int> _atomicNumbers;

  private:

    /// Convert chemical symbols to atomic numbers.
    std::vector<int> convertChemicalSymbolsToAtomicNumbers(const std::vector<std::string> &) const;

    /// Convert chemical symbols to atomic numbers.
    std::vector<std::string> convertAtomicNumbersToChemicalSymbols(const std::vector<int> &) const;

  private:

    /// Positions of sites in Cartesian coordinates.
    Matrix<double, Dynamic, 3, RowMajor> _positions;

    /// Cell metric.
    Matrix3d _cell;

    /// Periodic boundary conditions.
    std::vector<bool> _pbc;

    /// List of unique sites.
    std::vector<size_t> _uniqueSites;

    /// List of the number of allowed components on each site.
    std::vector<int> _numbersOfAllowedSpecies;

};
