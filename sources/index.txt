PyBLAW
======

PyBLAW is a python framework for solving one-dimensional system of
hyperbolic balance laws.

A one-dimensional system of hyperbolic balance laws is a system of the
form

.. math:: q_t + \bigl( f(q) \bigr)_x = s(q).

Check out some :doc:`examples <examples>` to learn more about how to
use PyBLAW.


Using PyBLAW
------------

To solve a one-dimensional system of hyperbolic balance laws, we
integrate the system over a grid cell (finite volume) to obtain

.. math:: \frac{d}{dt} q_i = \frac{1}{\Delta x_i} \Bigl( f(q_{i-1/2}) - f(q_{i+1/2}) \Bigr) + \frac{1}{\Delta x_i} \int_{x_{i-1/2}}^{x_{i+1/2}} s(q) \;dx.

where :math:`q_i` is the cell average of :math:`q`.  As such, we need
to approximate

  1. the flux :math:`f(q)` at the cell boundaries, and
  2. the integral of the source :math:`s(q)` over the cells;

and evolve the sum these of to obtain the time evolution of the cell
averages.

The key to approximating the flux term is being able to approximate
the solution :math:`q` at the cell boundaries given the cell averages
of :math:`q`.  This is the *reconstruction* problem.

To solve a particular system of hyperbolic balance laws, we need to

  1. reconstruct the solution at various points (cell boundaries and source quadrature points),
  2. compute the flux and source terms, and
  3. evolve the system.

This is similar to Godunov's REA algorithm.

The PyBLAW framework defines base classes to handle/define

  * the system,
  * the flux,
  * the source,
  * a reconstructor,
  * an evolver,
  * a dumper,
  * and a solver.

Briefly, the base classes are

  * pyblaw.system.System - system parameters, initial conditions, etc;
  * pyblaw.flux.Flux - compute the flux (given a reconstruction);
  * pyblaw.source.Source - compute the source (given a reconstruction);
  * pyblaw.evolver.Evolver - evolve the system;
  * pyblaw.dumper.Dumper - dump the solution; and
  * pyblaw.solver.Solver - run the solver!


Once you have defined your system by inheriting and overriding the
PyBLAW base classes, you would call the Solver's *run* method to run
the solver.

There are a few things to keep in mind:

  * each class has an *allocate* method that gets called by the solver
    during initialisation;

  * each class has a *pre_run* method that gets called after the
    initial conditions are computed; and

  * some classes are linked together through various instance
    variables (eg, the Flux class has a *system* variable that points
    to the System class).


API
---

* :doc:`PyBLAW API <pyblaw>`


Obtaining PyBLAW
----------------

Download, build, and install from source.

The latest source distribution is available in either zip_ or tar_
format.  You can also obtain the source code on GitHub through the
`PyBLAW project page`_.  You can clone the project by running::

  $ git clone git://github.com/memmett/PyBLAW

PyBLAW uses the Python setuptools_ package for installation.


Contributing
------------

Contributions are welcome!  Please send comments, suggestions, and/or
patches to the primary author (Matthew Emmett).  You will be credited.

If you plan to extend or modify PyBLAW in a more substantial way,
please see the `PyBLAW project page`_.



.. toctree::
   :hidden:

   self
   pyblaw
   examples

.. _zip: http://github.com/memmett/PyBLAW/zipball/master
.. _tar: http://github.com/memmett/PyBLAW/tarball/master
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _`PyBLAW project page`: http://github.com/memmett/PyBLAW
