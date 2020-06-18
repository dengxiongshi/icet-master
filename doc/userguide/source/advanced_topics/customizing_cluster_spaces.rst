.. highlight:: python
.. index::
   single: Advanced topics; Customizing cluster spaces
   single: Advanced topics; Merging orbits
   single: Advanced topics; Local symmetries

Customizing cluster spaces
==========================

When decomposing a structure :program:`icet` uses all available crystal
symmetries to determine which clusters are symmetry equivalent and hence
belong to the same orbit. This is a rigorous and well defined procedure. For
systems of lower symmetry, it can, however, lead to a large number of orbits
and hence :term:`ECIs`. At the same time, one can often still find "local"
symmetries in these systems and physical (or chemical) intuition tells us that
the parameters associated with such clusters should be very similar (even
if they are not strictly identical).

Motivating example
------------------

For illustration consider the construction of a model for a (111) :term:`FCC`
surface. Using :program:`ase` we can readily construct a 10-layer slab
representing such a system and construct a corresponding cluster space.

.. testsetup::

    from ase.build import fcc111
    from icet import ClusterSpace

.. doctest::

    >>> structure = fcc111('Au', size=(1, 1, 10), a=4.1, vacuum=10, periodic=True)
    >>> cs = ClusterSpace(structure=structure, cutoffs=[4.2], chemical_symbols=['Au', 'Ag'])
    >>> print(cs)
    ====================================== Cluster Space =======================================
     space group                            : P-3m1 (164)
     chemical species                       : ['Ag', 'Au'] (sublattice A)
     cutoffs                                : 4.2000
     total number of parameters             : 21
     number of parameters by order          : 0= 1  1= 5  2= 15
     fractional_position_tolerance          : 1e-06
     position_tolerance                     : 1e-05
     symprec                                : 1e-05
    --------------------------------------------------------------------------------------------
    index | order |  radius  | multiplicity | orbit_index | multi_component_vector | sublattices
    --------------------------------------------------------------------------------------------
       0  |   0   |   0.0000 |        1     |      -1     |           .            |      .
       1  |   1   |   0.0000 |        2     |       0     |          [0]           |      A
       2  |   1   |   0.0000 |        2     |       1     |          [0]           |      A
       3  |   1   |   0.0000 |        2     |       2     |          [0]           |      A
       4  |   1   |   0.0000 |        2     |       3     |          [0]           |      A
       5  |   1   |   0.0000 |        2     |       4     |          [0]           |      A
       6  |   2   |   1.4496 |        3     |       5     |         [0, 0]         |     A-A
       7  |   2   |   1.4496 |        6     |       6     |         [0, 0]         |     A-A
       8  |   2   |   1.4496 |        6     |       7     |         [0, 0]         |     A-A
       9  |   2   |   1.4496 |        6     |       8     |         [0, 0]         |     A-A
      10  |   2   |   1.4496 |        6     |       9     |         [0, 0]         |     A-A
      11  |   2   |   1.4496 |        6     |      10     |         [0, 0]         |     A-A
      12  |   2   |   1.4496 |        6     |      11     |         [0, 0]         |     A-A
      13  |   2   |   1.4496 |        6     |      12     |         [0, 0]         |     A-A
      14  |   2   |   1.4496 |        6     |      13     |         [0, 0]         |     A-A
      15  |   2   |   1.4496 |        6     |      14     |         [0, 0]         |     A-A
      16  |   2   |   2.0500 |        3     |      15     |         [0, 0]         |     A-A
      17  |   2   |   2.0500 |        6     |      16     |         [0, 0]         |     A-A
      18  |   2   |   2.0500 |        6     |      17     |         [0, 0]         |     A-A
      19  |   2   |   2.0500 |        6     |      18     |         [0, 0]         |     A-A
      20  |   2   |   2.0500 |        6     |      19     |         [0, 0]         |     A-A
    ============================================================================================

This cluster space comprises 5 singlets and 15 pairs. For a binary system
one thus obtains 21 :term:`ECIs` (including the zerolet). In practice, one
would use longer cutoffs and higher orders typically leading to an even larger
number of parameters.

Inspect orbits
--------------

We can use the :func:`get_coordinates_of_representative_cluster()
<icet.ClusterSpace.get_coordinates_of_representative_cluster>` method to look
up the representative cluster for each orbit and thus sort out which singlet
corresponds to which layer.

.. note::

    The argument of :func:`get_coordinates_of_representative_cluster()
    <icet.ClusterSpace.get_coordinates_of_representative_cluster>` is the
    index of the orbit in the orbit list, i.e. the value from the
    ``orbit_index`` column that is shown when printing a :class:`ClusterSpace
    <icet.ClusterSpace>` object.

.. doctest::

    >>> for k in range(0, 5):
    >>>     pos = cs.get_coordinates_of_representative_cluster(k)[0]
    >>>     print(f'orbit: {k}  pos: {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}')
    orbit: 0  pos:   0.000   0.000  10.000
    orbit: 1  pos:   1.450   0.837  12.367
    orbit: 2  pos:  -0.000   1.674  14.734
    orbit: 3  pos:   0.000   0.000  17.101
    orbit: 4  pos:   1.450   0.837  19.469

As one can expect, there is one singlet for each symmetry *inequivalent* layer
with the first/second/third... singlet corresponding to the
first/second/third... layer. Based on physical intuition, we can expect
corresponding clusters
in the center of the slab to behave nearly identical, i.e. the :term:`ECIs`
associated with, say, the fourth and fifth singlets should be very close if
not identical, and similarly for pairs, triplets etc.

Merge orbits
------------

To handle such situations, :program:`icet` allows one to merge orbits via the
:func:`merge_orbits <icet.ClusterSpace.merge_orbits>`. Which orbits should be
merged is entirely up to the user. In the present example, one could for
example merge the singlets for the third, fourth, and fifth layers,
effectively treating them as bulk sites, while keeping the singlets for the
first two layers distinct.

.. doctest::

    >>> cs.merge_orbits({2: [3, 4]})
    >>> print(cs)
    ====================================== Cluster Space =======================================
     space group                            : P-3m1 (164)
     chemical species                       : ['Ag', 'Au'] (sublattice A)
     cutoffs                                : 4.2000
     total number of parameters             : 19
     number of parameters by order          : 0= 1  1= 3  2= 15
     fractional_position_tolerance          : 1e-06
     position_tolerance                     : 1e-05
     symprec                                : 1e-05
    --------------------------------------------------------------------------------------------
    index | order |  radius  | multiplicity | orbit_index | multi_component_vector | sublattices
    --------------------------------------------------------------------------------------------
       0  |   0   |   0.0000 |        1     |      -1     |           .            |      .
       1  |   1   |   0.0000 |        2     |       0     |          [0]           |      A
       2  |   1   |   0.0000 |        2     |       1     |          [0]           |      A
       3  |   1   |   0.0000 |        6     |       2     |          [0]           |      A
       4  |   2   |   1.4496 |        3     |       3     |         [0, 0]         |     A-A
       5  |   2   |   1.4496 |        6     |       4     |         [0, 0]         |     A-A
       6  |   2   |   1.4496 |        6     |       5     |         [0, 0]         |     A-A
       7  |   2   |   1.4496 |        6     |       6     |         [0, 0]         |     A-A
       8  |   2   |   1.4496 |        6     |       7     |         [0, 0]         |     A-A
       9  |   2   |   1.4496 |        6     |       8     |         [0, 0]         |     A-A
      10  |   2   |   1.4496 |        6     |       9     |         [0, 0]         |     A-A
      11  |   2   |   1.4496 |        6     |      10     |         [0, 0]         |     A-A
      12  |   2   |   1.4496 |        6     |      11     |         [0, 0]         |     A-A
      13  |   2   |   1.4496 |        6     |      12     |         [0, 0]         |     A-A
      14  |   2   |   2.0500 |        3     |      13     |         [0, 0]         |     A-A
      15  |   2   |   2.0500 |        6     |      14     |         [0, 0]         |     A-A
      16  |   2   |   2.0500 |        6     |      15     |         [0, 0]         |     A-A
      17  |   2   |   2.0500 |        6     |      16     |         [0, 0]         |     A-A
      18  |   2   |   2.0500 |        6     |      17     |         [0, 0]         |     A-A
    ============================================================================================

We are left with 10 first and 5 second-nearest neighbor orbits. Applying a
similar logic, we could now decide to merge the bulk interactions and keep
only interactions that involve the first surface layer separate. To this end,
we inspect the representative cluster of all pair orbits.

.. doctest::
    >>> for k in range(len(cs) - 1):
    >>>     coords = cs.get_coordinates_of_representative_cluster(k)
    >>>     if len(coords) != 2:
    >>>         continue
    >>>     print(f'cluster: {k}   order: {len(coords)}')
    >>>     for m, pos in enumerate(coords):
    >>>         print(f'  site: {m}   pos: {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}')
    cluster: 3   order: 2
      site: 0   pos:   0.000  -1.674  19.469
      site: 1   pos:  -1.450  -0.837  21.836
    cluster: 4   order: 2
      site: 0   pos:  -1.450  -2.511  10.000
      site: 1   pos:   0.000   0.000  10.000
    cluster: 5   order: 2
      site: 0   pos:   0.000   0.000  10.000
      site: 1   pos:   0.000  -1.674  12.367
    cluster: 6   order: 2
      site: 0   pos:  -2.899  -1.674  12.367
      site: 1   pos:   0.000  -1.674  12.367
    cluster: 7   order: 2
      site: 0   pos:   0.000  -1.674  12.367
      site: 1   pos:  -1.450  -0.837  14.734
    cluster: 8   order: 2
      site: 0   pos:  -1.450  -0.837  14.734
      site: 1   pos:   1.450  -0.837  14.734
    cluster: 9   order: 2
      site: 0   pos:  -1.450  -0.837  14.734
      site: 1   pos:   0.000   0.000  17.101
    cluster: 10   order: 2
      site: 0   pos:  -1.450  -2.511  17.101
      site: 1   pos:   0.000   0.000  17.101
    cluster: 11   order: 2
      site: 0   pos:   0.000   0.000  17.101
      site: 1   pos:   0.000  -1.674  19.469
    cluster: 12   order: 2
      site: 0   pos:  -2.899  -1.674  19.469
      site: 1   pos:   0.000  -1.674  19.469
    cluster: 13   order: 2
      site: 0   pos:   0.000  -1.674  19.469
      site: 1   pos:  -0.000   1.674  21.836
    cluster: 14   order: 2
      site: 0   pos:   1.450  -2.511  10.000
      site: 1   pos:   1.450   0.837  12.367
    cluster: 15   order: 2
      site: 0   pos:   0.000  -1.674  12.367
      site: 1   pos:  -0.000   1.674  14.734
    cluster: 16   order: 2
      site: 0   pos:  -0.000  -3.348  14.734
      site: 1   pos:   0.000   0.000  17.101
    cluster: 17   order: 2
      site: 0   pos:   1.450  -2.511  17.101
      site: 1   pos:   1.450   0.837  19.469

According to this analysis orbits 4, 5, and 14 correspond, respectively, to
nearest-neighbor in-plane, first nearest-neighbor out-of-plane, and
next-nearest-neighbor interactions. We can thus merge all remaining clusters.

.. doctest::

    >>> cs.merge_orbits({3: [k for k in range(6, 13)],
    >>>                  13: [k for k in range(15, 18)]})
    >>> print(cs)
    ====================================== Cluster Space =======================================
     space group                            : P-3m1 (164)
     chemical species                       : ['Ag', 'Au'] (sublattice A)
     cutoffs                                : 4.2000
     total number of parameters             : 9
     number of parameters by order          : 0= 1  1= 3  2= 5
     fractional_position_tolerance          : 1e-06
     position_tolerance                     : 1e-05
     symprec                                : 1e-05
    --------------------------------------------------------------------------------------------
    index | order |  radius  | multiplicity | orbit_index | multi_component_vector | sublattices
    --------------------------------------------------------------------------------------------
       0  |   0   |   0.0000 |        1     |      -1     |           .            |      .
       1  |   1   |   0.0000 |        2     |       0     |          [0]           |      A
       2  |   1   |   0.0000 |        2     |       1     |          [0]           |      A
       3  |   1   |   0.0000 |        6     |       2     |          [0]           |      A
       4  |   2   |   1.4496 |       45     |       3     |         [0, 0]         |     A-A
       5  |   2   |   1.4496 |        6     |       4     |         [0, 0]         |     A-A
       6  |   2   |   1.4496 |        6     |       5     |         [0, 0]         |     A-A
       7  |   2   |   2.0500 |       21     |       6     |         [0, 0]         |     A-A
       8  |   2   |   2.0500 |        6     |       7     |         [0, 0]         |     A-A
    ============================================================================================

By merging singlet and pair orbits we have cut the number of parameters by
more than half, from 21 to 9. As a final check we can print all the
representative clusters.

    >>> for k in range(len(cs) - 1):
    >>>     coords = cs.get_coordinates_of_representative_cluster(k)
    >>>     print(f'orbit_index: {k}   order: {len(coords)}')
    >>>     for m, pos in enumerate(coords):
    >>>         print(f'  site: {m}   pos: {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}')
    orbit_index: 0   order: 1
      site: 0   pos:   0.000   0.000  10.000
    orbit_index: 1   order: 1
      site: 0   pos:   1.450   0.837  12.367
    orbit_index: 2   order: 1
      site: 0   pos:  -0.000   1.674  14.734
    orbit_index: 3   order: 2
      site: 0   pos:   0.000  -1.674  19.469
      site: 1   pos:  -1.450  -0.837  21.836
    orbit_index: 4   order: 2
      site: 0   pos:  -1.450  -2.511  10.000
      site: 1   pos:   0.000   0.000  10.000
    orbit_index: 5   order: 2
      site: 0   pos:   0.000   0.000  10.000
      site: 1   pos:   0.000  -1.674  12.367
    orbit_index: 6   order: 2
      site: 0   pos:   0.000  -1.674  19.469
      site: 1   pos:  -0.000   1.674  21.836
    orbit_index: 7   order: 2
      site: 0   pos:   1.450  -2.511  10.000
      site: 1   pos:   1.450   0.837  12.367

The cluster space obtained in this fashion can be used for constructing and
sampling cluster expansions in exactly the same way as if no orbits had been
merged.

Further considerations
----------------------

The example above illustrates the basic principles that can be applied reduce
the dimensionality of cluster spaces. In applications it is can be impractical
to "manually" inspect all orbits. Instead one can define to a simple set of
criteria including, e.g., radii (and order) of the orbits as well as the sites
that are involved.
