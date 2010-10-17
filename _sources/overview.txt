PyBLAW Overview
---------------

PyBLAW is based upon the *finite volume* approach to solving a system
of hyperbolic balance laws numerically.

We integrate the system over a grid cell (finite volume) from
:math:`x_{i-1/2}` to :math:`x_{i+1/2}` and re-arrange to obtain

.. math:: \frac{d}{dt} q_i = \frac{1}{\Delta x_i} \Bigl( f(q_{i-1/2}) - f(q_{i+1/2}) \Bigr) + \frac{1}{\Delta x_i} \int_{x_{i-1/2}}^{x_{i+1/2}} s(q) \;dx.

where :math:`q_i` is the cell average of :math:`q`.  As such, we need
to approximate

1. the flux :math:`f(q)` at the cell boundaries :math:`q_{i \pm 1/2}`, and
2. the integral of the source :math:`s(q)` over the cells;

and evolve the sum of these to obtain the time evolution of the cell
averages.  The key to approximating the flux and source terms is being
able to approximate the solution :math:`q` at the cell boundaries and
quadrature points given the cell averages of :math:`q`.  This is the
*reconstruction* problem.  Therefore, in order to solve a particular
system of hyperbolic balance laws we need to

1. reconstruct the solution at various points (cell boundaries and
   source quadrature points),
2. compute the flux and source terms, and
3. evolve the system.

As such, the PyBLAW framework consists of base classes that define

* the *system*,
* the *flux*,
* the *source*,
* a *reconstructor*,
* an *evolver*,
* a *dumper*,
* and a *solver*.

Again, these are only lightweight base classes (although PyBLAW has
some predefined classes to help).  It is up to you to extend these
base classes for your particular problem.  Briefly, the

* *system* class defines system parameters, initial conditions, etc; the
* *flux* class computes the net flux into each cell given a
  reconstruction; the
* *source* class computes the source in each cell given a
  reconstruction; the
* *reconstructor* class reconstructs the solution at the cell
  boundaries and quadrature points given the cell averages of the
  solution; the
* *evolver* class evolves the system from one time step to the next; the
* *dumper* class dumps the solution to a file; and the
* *solver* class glues everything together.

PyBLAW has some predefined classes to save you some time:

* a WENO reconstructor and WENO conservation law solver,
* a Runge-Kutta evolver,
* an HDF5 and MAT dumper, and a
* generic solver.

The `Python WENO`_ package `PyWENO`_ is a good resource for building
your reconstructor.

Once you have defined your system by inheriting and overriding the
PyBLAW base classes, you call the Solver's *run* method to run the
solver.

Check out some :doc:`examples <examples>` to learn more about how to
use PyBLAW.

.. _`Python WENO`: http://memmett.github.com/PyWENO/
.. _`PyWENO`: http://memmett.github.com/PyWENO/
