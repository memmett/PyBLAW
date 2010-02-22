PyBLAW Reference
================

There are a few things to keep in mind:

* each class has an *allocate* method that gets called by the solver
  during initialisation;

* each class has a *pre_run* method that gets called after the
  initial conditions are computed;

* each class has a *debug* dictionary and a *trace* variable to help
  you debug your solver; and

* some classes are linked together through various instance
  variables (eg, the flux class has a *system* variable that points
  to the system class).

Again, checking out some :doc:`examples <examples>` might be helpful.


Base
----

.. autoclass:: pyblaw.base.Base
   :members:


System
------

.. autoclass:: pyblaw.system.System
   :members:

.. autoclass:: pyblaw.system.SimpleSystem


Flux
----

.. autoclass:: pyblaw.flux.Flux
   :members:

.. autoclass:: pyblaw.flux.SimpleFlux

.. autoclass:: pyblaw.flux.LFFlux

Source
------

.. autoclass:: pyblaw.source.Source
   :members:

.. autoclass:: pyblaw.source.SimpleSource

.. :
  .. autoclass:: pyblaw.source.LinearQuad3Source


Reconstructor
-------------

.. autoclass:: pyblaw.reconstructor.Reconstructor
   :members:

.. autoclass:: pyblaw.wenoclaw.WENOCLAWReconstructor


Evolver
-------

.. autoclass:: pyblaw.evolver.Evolver
   :members:

.. :

  .. autoclass:: pyblaw.evolver.FE

  .. autoclass:: pyblaw.evolver.ERK2

.. autoclass:: pyblaw.evolver.SSPERK3


Dumper
------

.. autoclass:: pyblaw.dumper.Dumper
   :members:

.. autoclass:: pyblaw.dumper.MATDumper

.. autoclass:: pyblaw.h5dumper.H5PYDumper


Solver
------

.. autoclass:: pyblaw.solver.Solver
   :members:

.. autoclass:: pyblaw.wenoclaw.WENOCLAWLFSolver
