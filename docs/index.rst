PyBLAW
======

PyBLAW is a lightweight Python framework for solving one-dimensional
systems of hyperbolic balance laws of the form

.. math:: q_t + \bigl( f(q) \bigr)_x = s(q).


Using PyBLAW
------------

To solve a one-dimensional system of hyperbolic balance laws, we
integrate the system over a grid cell (finite volume) from
:math:`x_{i-1/2}` to :math:`x_{i+1/2}` and re-arrange to obtain

.. math:: \frac{d}{dt} q_i = \frac{1}{\Delta x_i} \Bigl( f(q_{i-1/2}) - f(q_{i+1/2}) \Bigr) + \frac{1}{\Delta x_i} \int_{x_{i-1/2}}^{x_{i+1/2}} s(q) \;dx.

where :math:`q_i` is the cell average of :math:`q`.  As such, we need
to approximate

  1. the flux :math:`f(q)` at the cell boundaries :math:`q_{i \pm 1/2}`, and
  2. the integral of the source :math:`s(q)` over the cells;

and evolve the sum these of to obtain the time evolution of the cell
averages.  The key to approximating the flux and source terms is being
able to approximate the solution :math:`q` at the cell boundaries and
quadrature points given the cell averages of :math:`q`.  This is the
*reconstruction* problem.  So, in order to solve a particular system
of hyperbolic balance laws we need to

  1. reconstruct the solution at various points (cell boundaries and source quadrature points),
  2. compute the flux and source terms, and
  3. evolve the system.

As such, the PyBLAW framework defines base classes to handle/define

  * the *system*,
  * the *flux*,
  * the *source*,
  * a *reconstructor*,
  * an *evolver*,
  * a *dumper*,
  * and a *solver*.

Again, these are only lightweight base classes (although PyBLAW has
some predefined classes to help, see below).  It is up to you to
extend these base classes for your particular problem.  Briefly, the

  * *system* class defines system parameters, initial conditions, etc; the
  * *flux* class computes the flux at the cell boundaries given a
    reconstruction; the
  * *source* class computes the source in each cell given a
    reconstruction; the
  * *reconstructor* class reconstructs the solution at the cell
    boundaries and quadrature points given the cell averages of the
    solution; the
  * *evolver* class evolves the system from one time step to the next;
  * *dumper* class dumps the solution to a file; and the
  * *solver* class glues everything together.

PyBLAW has some predefined classes to save you some time:

  * a WENO reconstructor,
  * a few Runge-Kutta evolvers,
  * an HDF5 dumper, and a
  * generic solver.

The `Python WENO`_ package `PyWENO`_ is a good resource for building
your reconstructor.

Once you have defined your system by inheriting and overriding the
PyBLAW base classes, you call the Solver's *run* method to run the
solver.

Check out some :doc:`examples <examples>` to learn more about how to
use PyBLAW.

API
---

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

Please see the :doc:`PyBLAW API <pyblaw>` page for the detailed API
documentation.  Again, checking out some :doc:`examples <examples>`
might be helpful.


Obtaining and installing PyBLAW
-------------------------------

To install PyBLAW, please download, build, and install from source
(there aren't any pre-built packages).

The latest source distribution is available in either zip_ or tar_
format.  You can also obtain the source code on GitHub through the
`PyBLAW project page`_, and you can clone the project by running::

  $ git clone git://github.com/memmett/PyBLAW

Once you have downloaded and unpacked the source package, you can
install PyBLAW by simply running::

  $ python setup.py install

For more installation options, please see the Python `Installing
Python Modules`_ document and the setuptools_ documentation.


Contributing
------------

Contributions are welcome!  Please send comments, suggestions, and/or
patches to the primary author (`Matthew Emmett`_).  You will be
credited.

If you plan to extend or modify PyBLAW in a more substantial way,
please see the `PyBLAW project page`_.



.. toctree::
   :hidden:

   self
   pyblaw
   examples

.. _zip: http://github.com/memmett/PyBLAW/zipball/master
.. _tar: http://github.com/memmett/PyBLAW/tarball/master
.. _`Installing Python Modules`: http://docs.python.org/install/index.html
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _`PyBLAW project page`: http://github.com/memmett/PyBLAW
.. _`Python WENO`: http://github.com/memmett/PyWENO
.. _`PyWENO`: http://github.com/memmett/PyWENO
.. _`Matthew Emmett`: http://www.math.ualberta.ca/~memmett/
