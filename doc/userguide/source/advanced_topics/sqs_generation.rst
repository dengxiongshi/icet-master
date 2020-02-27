.. _advanced_topics_sqs_structures:
.. highlight:: python
.. index::
   single: Advanced topics; Special quasirandom structures

Special quasirandom structures
==============================

Random alloys are often of special interest. This is true in particular for
systems that form random solid solutions below the melting point. It is,
however, not always easy to model such structures, because the system sizes
that lend themselves to, for example, DFT calculations, are often too small to
accomodate a structure that may be regarded as random; the periodicity imposed
by boundary conditions introduces correlations that make the modeled structure
deviate from the random alloy. This problem can sometimes be alleviated with
the use of so-called special quasirandom structures (SQS) [ZunWeiFer90]_. SQS
cells are the best possible approximations to random alloys in the sense that
their cluster vectors closely resemble the cluster vectors of truly random
alloys. This tutorial demonstrates how SQS cells can be generated in
:program:`icet` using a simulated annealing approach.

There is no unique way to measure the similarity between the cluster vector of
the SQS cell and the random alloy. The implementation in :program:`icet` uses
the measure proposed in [WalTiwJon13]_. Specifically, the objective function
:math:`Q` is calculated as

.. math::
    Q = - \omega L + \sum_{\alpha}
         \left| \Gamma_{\alpha} - \Gamma^{\text{target}}_{\alpha}
         \right|.

Here, :math:`\Gamma_{\alpha}` are components in the cluster vector and
:math:`\Gamma^\text{target}_{\alpha}` the corresponding target values. The
factor :math:`\omega` is the radius (in Ångström) of the largest pair cluster
such that all clusters with the same or smaller radii have
:math:`\Gamma_{\alpha} - \Gamma^\text{target}_{\alpha} = 0`. The parameter
:math:`L`, by default ``1.0``, can be specified by the user.

The functionality for generating SQS cells is just a special case of a more
general algorithm for generating a structure with a cluster vector similar to
*any* target cluster vector. The below example demonstrates both applications.


Import modules
--------------

The :func:`generate_sqs <icet.tools.structure_generation.generate_sqs>` and/or
:func:`generate_target_structure
<icet.tools.structure_generation.generate_target_structure>` functions need to
be imported together with some additional functions from `ASE
<https://wiki.fysik.dtu.dk/ase>`_ and :program:`icet`. It is advisable to turn
on logging, since the SQS cell generation may otherwise run quietly for a few
minutes.

.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Import modules
   :end-before: # Generate SQS for binary

Generate binary SQS cells
-------------------------

In the following example, a binary :term:`FCC` SQS cell with 8 atoms will be
generated. To this end, an :class:`icet.ClusterSpace` and target
concentrations need to be defined. The cutoffs in the cluster space are
important, since they determine how many elements are to be included when
cluster vectors are compared. It is usually sufficient to use cutoffs such
that the length of the cluster vector is on the order of 10. Target
concentrations are specified via a dictionary, which should contain all the
involved elements and their fractions of the total number of atoms.
Internally, the function carries out simulated annealing with Monte Carlo
trial swaps and can be expected to run for a minute or so.

.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Generate SQS for binary fcc
   :end-before: # Generate SQS for binary fcc with specified supercells

If for some reason a particular supercell is needed, there is another function
:func:`generate_sqs_from_supercells
<icet.tools.structure_generation.generate_sqs_from_supercells>`
that works similarly, but in which it is possible to explicitly provide
the accepted supercell. The code will then look for the optimal SQS among
the provided supercells.

.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Generate SQS for binary fcc with specified supercells
   :end-before: # Use enumeration

Generate SQS cells by enumeration
---------------------------------

In the above simple case, in which the target structure size is very small, it
is more efficient to generate the best SQS cell by exhaustive enumeration of
all binary :term:`FCC` structures having up to 8 atoms in the supercell:

.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Use enumeration
   :end-before: # Generate SQS for

Generation of SQS cells by enumeration is preferable over the Monte Carlo
approach if the size of the system permits, because with enumeration there is
no risk that the optimal SQS cell is missed.


Generate SQS cells for a system with sublattices
------------------------------------------------

It is possible to generate SQS cells also for systems with sublattices. In the
below example, an SQS cell is generated for a system with two sublattices; one
:term:`FCC` sublattice on which Au, Cu, and Pd are allowed, and another
:term:`FCC` sublattice on which H and vacancies (V) are allowed. Target
concentrations are specified per sublattice. The sublattices are defined by
the letters shown at the top of the printout of a `ClusterSpace`. 

.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Generate SQS for a system with two sublattices
   :end-before: # Target concentrations are specified per sublattice

This should result in something similar to this::

  ====================================== Cluster Space =======================================
   space group                            : Fm-3m (225)
   chemical species                       : ['Au', 'Cu', 'Pd'] (sublattice A), ['H', 'V'] (sublattice B)
   cutoffs                                : 7.0000
   total number of parameters             : 40
   number of parameters by order          : 0= 1  1= 3  2= 36
   fractional_position_tolerance          : 2e-06
   position_tolerance                     : 1e-05
   symprec                                : 1e-05
  --------------------------------------------------------------------------------------------
  index | order |  radius  | multiplicity | orbit_index | multi_component_vector | sublattices
  --------------------------------------------------------------------------------------------
     0  |   0   |   0.0000 |        1     |      -1     |           .            |      .     
     1  |   1   |   0.0000 |        1     |       0     |          [0]           |      A     
     2  |   1   |   0.0000 |        1     |       0     |          [1]           |      A     
     3  |   1   |   0.0000 |        1     |       1     |          [0]           |      B     
     4  |   2   |   1.0000 |        6     |       2     |         [0, 0]         |     A-B 
  ...

Here we see that the sublattice with Au, Cu and Pd is sublattice `A`, while H
and V are on sublattice B. These letters can now be used when the target
concentrations are specified.

In the below example, an SQS cell is generated for a supercell that is 16
times larger than the primitive cell, in total 32 atoms. The keyword
``include_smaller_cells=False`` guarantees that the generated structure has 32
atoms (otherwise the structure search would have been carried out among
structures having 32 atoms *or less*).

In this example, the number of trial steps is manually set to 50,000. This
number may be insufficient, but will most likely provide a reasonable SQS
cell, albeit perhaps not *the* best one. The default number of trial steps is
3,000 times the number of inequivalent supercell shapes. The latter quantity
increases quickly with the size of the supercell.


.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Target concentrations are specified per sublattice
   :end-before: # Generate structure with a specified cluster vector

Generate a structure matching an arbitrary cluster vector
---------------------------------------------------------

The SQS cell generation approach can be utilized to generate the structure
that most closely resembles *any* cluster vector. To do so, one can employ the
same procedure but the target cluster vector must be specified manually. Note
that there are no restrictions on what target vectors can be specified (except
their length, which must match the cluster space length), but the space of
cluster vectors that can be realized by structures is restricted in multiple
ways. The similarity between the target cluster vector and the cluster vector
of the generated structure may thus appear poor.

.. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
   :start-after: # Generate structure with a specified cluster vector

Source code
-----------

.. container:: toggle

    .. container:: header

       The complete source code is available in
       ``examples/sqs_generation.py``

    .. literalinclude:: ../../../../examples/advanced_topics/sqs_generation.py
