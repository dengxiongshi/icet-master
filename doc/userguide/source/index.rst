.. raw:: html

  <p>
  <a href="https://badge.fury.io/py/icet"><img src="https://badge.fury.io/py/icet.svg" alt="PyPI version" height="18"></a>
  </p>

:program:`icet` — A Pythonic approach to cluster expansions
***********************************************************

:program:`icet` is a tool for the construction and sampling of alloy cluster
expansions. A detailed description of the functionality provided as well as an
extensive tutorial can be found in the `user guide
<https://icet.materialsmodeling.org/>`_.

:program:`icet` is written in Python, which enables easy integration with
many first-principles codes and analysis tools accessible from Python, and
allows for a simple and intuitive user interface. All computationally demanding
parts are, however, written in C++ ensuring performance while maintaining
portability.

The following snippet provides a minimal example for its usage:

.. testsetup::

   from ase.build import bulk
   from icet import ClusterExpansion, ClusterSpace, Optimizer, StructureContainer
   primitive_cell = bulk('Au', a=1.0, crystalstructure='fcc')
   cutoffs = [1.1]
   species = ['Au', 'Ag']

   training_structures, energies = [], []

   s = primitive_cell.repeat(1)
   training_structures.append(s)
   energies.append(-1.0)

   s = primitive_cell.repeat(2)
   s[0].symbol = 'Ag'
   training_structures.append(s)
   energies.append(-4.0)

   s = primitive_cell.repeat(2)
   s[0].symbol = 'Ag'
   s[1].symbol = 'Ag'
   training_structures.append(s)
   energies.append(-5.0)

.. doctest::

   >>> cs = ClusterSpace(primitive_cell, cutoffs, species)
   >>> sc = StructureContainer(cs)
   >>> for structure, energy in zip(training_structures, energies):
   ...     sc.add_structure(structure, properties={'energy': energy})
   >>> opt = Optimizer(sc.get_fit_data())
   >>> opt.train()
   >>> ce = ClusterExpansion(cs, opt.parameters)

:program:`icet` has been developed at the `Department of Physics
<https://www.chalmers.se/en/departments/physics/Pages/default.aspx>`_
of `Chalmers University of Technology <https://www.chalmers.se/>`_ in
Gothenburg, Sweden, and the `Data and Software Management Center
<https://europeanspallationsource.se/data-management-software#data-analysis-modelling>`_
at the European Spallation Source in Copenhagen, Denmark. Please
consult the :ref:`credits page <credits>` for information on how to
cite :program:`icet`.

:program:`icet` and its development are hosted on `gitlab
<https://gitlab.com/materials-modeling/icet>`_. Bugs and feature
requests are ideally submitted via the `gitlab issue tracker
<https://gitlab.com/materials-modeling/icet/issues>`_. The development
team can also be reached by email via icet@materialsmodeling.org.

.. toctree::
   :maxdepth: 2
   :caption: Main

   background/index
   installation
   tutorial/index
   advanced_topics/index
   faq
   credits

.. toctree::
   :maxdepth: 2
   :caption: Function reference

   moduleref_icet/index
   moduleref_mchammer/index

.. toctree::
   :maxdepth: 2
   :caption: Backmatter

   bibliography
   publications
   glossary
   genindex
