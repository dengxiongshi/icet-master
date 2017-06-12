#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "PeriodicTable.hpp"
using namespace Eigen;

namespace py = pybind11;

class Structure
{
  public:
    Structure(const Eigen::Matrix<double, Dynamic, 3, RowMajor> &,
              const std::vector<std::string> &,
              const Eigen::Matrix3d &,
              const std::vector<bool> &);

    double getDistance(const int, const int) const;

    /**
        Returns the distance for index1 with unitcell offset offset 1 to index2 with unit-cell offset offset2
    */
    double getDistance2(const int index1, const Vector3d offset1,
                        const int index2, const Vector3d offset2) const
    {
        if (index1 >= _positions.rows() or index2 >= _positions.rows())
        {
            throw std::out_of_range("Error: Tried accessing position at out of bound index. Structure::getDistance2");
        }

        Vector3d pos1 = _positions.row(index1) + offset1.transpose() * _cell;
        Vector3d pos2 = _positions.row(index2) + offset2.transpose() * _cell;

        return (pos1 - pos2).norm();
    }
   
    // Getters - Setters
    void setPositions(const Eigen::Matrix<double, Dynamic, 3> &positions)
    {
        _positions = positions;
    }

    Eigen::Matrix<double, Dynamic, 3, RowMajor> getPositions() const
    {
        return _positions;
    }

    void setStrElements(const std::vector<std::string> &elements)
    {
        _strelements = elements;
        setElements(convertStrElements(_strelements));
    }

    std::vector<std::string> getStrElements() const
    {
        return _strelements;
    }

    void setElements(const std::vector<int> &elements)
    {
        _elements = elements;
    }

    std::vector<int> getElements() const
    {
        return _elements;
    }

    int getElement(const size_t i) const
    {
        if ( i >= _elements.size())
        {
            std::string errorMessage = "Error: out of range in function get element:index : elements.size()  _elements.size() ";
            errorMessage += std::to_string(i) + " : ";
            errorMessage += std::to_string(_elements.size());
            throw std::out_of_range(errorMessage);
        }        

        return _elements[i];
    }


    void setUniqueSites(const std::vector<int> &sites)
    {
        _uniqueSites = sites;
    }

    std::vector<int> getUniqueSites() const
    {
        return _uniqueSites;
    }
    
    int getSite(const size_t i) const
    {
       if ( i >= _uniqueSites.size())
        {
            std::string errorMessage = "Error: out of range in function getSite : index :  _uniqueSites.size() ";
            errorMessage += std::to_string(i) + " : ";
            errorMessage += std::to_string(_uniqueSites.size());

            throw std::out_of_range(errorMessage);
        }        
        return _uniqueSites[i];

    }

    bool has_pbc(const int k) const
    {
        return _pbc[k];
    }
    std::vector<bool> get_pbc() const
    {
        return _pbc;
    }
    void set_pbc(const std::vector<bool> pbc)
    {
        _pbc = pbc;
    }

    void set_cell(const Eigen::Matrix<double, 3, 3> &cell)
    {
        _cell = cell;
    }

    Eigen::Matrix<double, 3, 3> get_cell() const
    {
        return _cell;
    }

    size_t size() const
    {
        if (_elements.size() != _positions.rows())
        {
            throw std::out_of_range("Error: Positions and elements do not match in size");
        }        
        return( _elements.size());
    }    

  private:
    Eigen::Matrix<double, Dynamic, 3, RowMajor> _positions;
    Eigen::Matrix3d _cell;
    std::vector<int> _elements;
    std::vector<std::string> _strelements;
    std::vector<bool> _pbc;
    std::vector<int> _uniqueSites;

std::vector<int> convertStrElements( const std::vector<std::string> &elements) 
{
    std::vector<int> intElements(elements.size());
    for (int i = 0; i < elements.size(); i++)
    {
        intElements[i] = PeriodicTable::strInt[elements[i]];
    }
    return intElements;
}

std::vector<std::string> convertIntElements( const std::vector<int> &elements) 
{
    std::vector<std::string> strElements(elements.size());
    for (int i = 0; i < elements.size(); i++)
    {
        strElements[i] = PeriodicTable::intStr[elements[i]];
    }
    return strElements;
}




};
