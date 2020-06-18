.. _faq:
.. index:: FAQ

***
FAQ
***

Failing tests
-------------

When starting to use :program:`icet` it is a good idea to 
:ref:`run all the tests <run_tests>` to
check that everything works as expected. A few common reasons why tests may
fail include:

* Out of sync test and source-code. The tests have to match the version of the
  package.

* Old packages: If the numpy/sympy/spglib version that has been installed does
  not match the version requirements some tests can fail.

Does :program:`icet` run on Linux/Mac/Windows?
----------------------------------------------

:program:`icet` is currently well tested only on recent OS versions in the
Linux and Mac family, but :program:`icet` should in principle work fine on
Windows too. If you encounter any issues when running :program:`icet` on
Windows, please send an e-mail too icet@materialsmodeling.org and describe
the problem.

:program:`icet` crashes with the error `Failed to find site by position (findLatticeSiteByPosition)`
----------------------------------------------------------------------------------------------------

This error usually occurs when trying to calculate the cluster vector of a
structure that is *not* a supercell of the primitive structure that the
:class:`ClusterSpace <icet.ClusterSpace>` is based on. Sometimes, the reason
is too strict numerical tolerances, which is why the error message suggests
changing the tolerances when creating the cluster space. If changing the
tolerances does not help, we recommend the following:

* Ensure that the input structure is sensible, i.e., that it matches the
  primitive structure of the cluster space.

* Is the structure relaxed, such that the atoms are no longer at the ideal
  lattice sites? If yes, you need to map the structure back to the ideal cell.
  This can be accomplished with the function :meth:`map_structure_to_reference
  <icet.tools.map_structure_to_reference>`.

* Occasionally, `wrapping
  <https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.wrap>`_ your
  structure may do the trick.

Does :program:`icet` support 2D systems?
----------------------------------------

Structures in :program:`icet` should always have periodic boundary conditions
in all three directions. What you can still do, however, is to add sufficient
vacuum in one direction such that the cutoffs in your cluster space do not
extend from one periodic image to the next in that direction. For more
information, see :ref:`here <2Dsystems>`.

Does :program:`icet` support arbitrary point group symmetries?
--------------------------------------------------------------

No, :program:`icet` currently only supports crystallographic point group
symmetries. The symmetries of, for example, an icosahedral nanoparticle
will thus *not* be properly handled in :program:`icet`.

The SQS generator gives me a weird supercell
--------------------------------------------

The function :meth:`generate_sqs
<icet.tools.structure_generation.generate_sqs>` searches for the optimal 
:term:`SQS (special quasirandom structure) <SQS>`
not only by decorating a supercell with a fixed shape, but also by trying
different supercells. Sometimes the optimal SQS has a cell shape that looks
quite different from the input primitive cell, but it is always a supercell of
the primitive structure, i.e., it has the same symmetry etc.

In some cases, it can be a good idea to generate SQS with a specific
supercell. To this end, there is a separate function,
:meth:`generate_sqs_from_supercell
<icet.tools.structure_generation.generate_sqs_from_supercell>`, to which the
user can specify among which supercell(s) the SQS should be sought.

The structure mapping function fails
------------------------------------

The structure mapping function, :meth:`map_structure_to_reference
<icet.tools.map_structure_to_reference>`, sometimes fails if the structure to
be mapped differs too much from the ideal cell. A particularly important
aspect is that the structure to be mapped cannot be significantly *rotated*
from the primitive cell. Any major rotation has to be reverted by the user
prior to mapping.

