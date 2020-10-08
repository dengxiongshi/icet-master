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

.. testcode::

    structure = fcc111('Au', size=(1, 1, 10), a=4.1, vacuum=10, periodic=True)
    cs = ClusterSpace(structure=structure, cutoffs=[4.2], chemical_symbols=['Au', 'Ag'])
    print(cs)

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ====================================== Cluster Space =======================================
     space group                            : P-3m1 (164)
     chemical species                       : ['Ag', 'Au'] (sublattice A)
     cutoffs                                : 4.2000
     total number of parameters             : 21
     number of parameters by order          : 0= 1  1= 5  2= 15
     fractional_position_tolerance          : 1e-06
     position_tolerance                     : 1e-05
     symprec                                : 1e-05
     ...
    ============================================================================================


This cluster space comprises 5 singlets and 15 pairs. For this binary system
one thus obtains 21 :term:`ECIs` (including the zerolet). In practice, one
would typically use longer cutoffs and higher orders, leading to an even
larger number of parameters.

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

.. testcode::

    for k in range(0, 5):
        pos = cs.get_coordinates_of_representative_cluster(k)[0]
        print(f'orbit_index: {k}  pos: {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}')

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    orbit_index: 0  pos:   0.000   0.000  10.000
    orbit_index: 1  pos:   1.450   0.837  12.367
    orbit_index: 2  pos:  -0.000   1.674  14.734
    orbit_index: 3  pos:   0.000   0.000  17.101
    orbit_index: 4  pos:   1.450   0.837  19.469

In this case, there is one singlet for each symmetry *inequivalent* layer with
the first/second/third... singlet corresponding to the first/second/third...
layer. Based on physical intuition, we can expect corresponding clusters in
the center of the slab to behave nearly identical, i.e. the :term:`ECIs`
associated with, say, the fourth and fifth singlets should be very close if
not identical, and similarly for pairs, triplets etc.

Merge orbits
------------

To handle such situations, :program:`icet` allows one to merge orbits
via the :func:`merge_orbits <icet.ClusterSpace.merge_orbits>`
method. Which orbits should be merged is entirely up to the user. In
the present example, one could for example merge the singlets for the
third, fourth, and fifth layers, effectively treating them as bulk
sites, while keeping the singlets for the first two layers
distinct. The following snippet achieves this by merging the orbits
with indices 3 and 4 into the one with index 2 (compare output above).

.. testcode::

    cs.merge_orbits({2: [3, 4]})
    print(cs)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    ====================================== Cluster Space =======================================
     space group                            : P-3m1 (164)
     chemical species                       : ['Ag', 'Au'] (sublattice A)
     cutoffs                                : 4.2000
     total number of parameters             : 19
     number of parameters by order          : 0= 1  1= 3  2= 15
     fractional_position_tolerance          : 1e-06
     position_tolerance                     : 1e-05
     symprec                                : 1e-05
     ...
    ============================================================================================

There are now only 3 singlets but we are still left with 10 first and
5 second-nearest neighbor orbits. Applying a similar logic as above,
we could now decide to merge the bulk orbits and keep only orbits that
involve the first surface layer distinct. To this end, we inspect the
representative clusters of all pair orbits.

.. testcode::

    for k in range(len(cs) - 1):
        coords = cs.get_coordinates_of_representative_cluster(k)
        if len(coords) != 2:
            continue
        print(f'orbit_index: {k}   order: {len(coords)}')
        for m, pos in enumerate(coords):
            print(f'  site: {m}   pos: {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}')

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    orbit_index: 3   order: 2
      site: 0   pos:   0.000  -1.674  19.469
      site: 1   pos:  -1.450  -0.837  21.836
    orbit_index: 4   order: 2
      site: 0   pos:  -1.450  -2.511  10.000
      site: 1   pos:   0.000   0.000  10.000
    orbit_index: 5   order: 2
      site: 0   pos:   0.000   0.000  10.000
      site: 1   pos:   0.000  -1.674  12.367
    ...
    orbit_index: 14   order: 2
      site: 0   pos:   1.450  -2.511  10.000
      site: 1   pos:   1.450   0.837  12.367
    orbit_index: 15   order: 2
      site: 0   pos:   0.000  -1.674  12.367
      site: 1   pos:  -0.000   1.674  14.734
    ...

According to this analysis, orbits 4, 5 and 14 are related to
interactions involving the surface layer. We can thus keep these
orbits separate and merge all remaining ones.

.. testcode::

    cs.merge_orbits({3: [k for k in range(6, 13)],
                     13: [k for k in range(15, 18)]})
    print(cs)

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

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

.. testcode::

    for k in range(len(cs) - 1):
        coords = cs.get_coordinates_of_representative_cluster(k)
        print(f'orbit_index: {k}   order: {len(coords)}')
        for m, pos in enumerate(coords):
            print(f'  site: {m}   pos: {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}')

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

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
