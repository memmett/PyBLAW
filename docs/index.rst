PyBLAW
======

PyBLAW is a lightweight Python framework for solving one-dimensional
systems of hyperbolic balance laws of the form

.. math:: q_t + \bigl( f(q) \bigr)_x = s(q).

**News**

* November 27 2010: `PyBLAW 0.5.9`_ released.
* October 17 2010: `PyBLAW 0.5.7`_ released (periodic WENOCLAW solver, fixed LF flux).
* May 3 2010: `PyBLAW 0.5.2`_ released (easy Cython support).
* March 15 2010: `PyBLAW 0.5.1`_ released (minor fixes to the WENOCLAW solver).
* February 22 2010: `PyBLAW 0.5.0`_ released.
* Fall 2009: PyBLAW is in early development, and is a bit rough around
  the edges.

Please check out the documentation (below) or the `PyBLAW project
page`_ for more infomation about using and contributing to PyBLAW.


Documentation
-------------

**Main parts of the documentation**

* :doc:`Overview <overview>` - overview and basic usage.
* :doc:`Examples <examples>` - more detailed examples.
* :doc:`Reference <reference>` - reference documentation.


Download
--------

Check out the :doc:`download page <download>` for instructions on
obtaining and installing PyBLAW.


Contributing
------------

Contributions are welcome!  Please send comments, suggestions, and/or
patches to the primary author (`Matthew Emmett`_).  You will be credited.


.. toctree::
   :hidden:

   self
   overview
   examples
   reference
   download

.. _`PyBLAW project page`: http://github.com/memmett/PyBLAW
.. _`Matthew Emmett`: http://www.math.ualberta.ca/~memmett/
.. _`PyBLAW 0.5.0`: http://github.com/memmett/PyBLAW/downloads
.. _`PyBLAW 0.5.1`: http://github.com/memmett/PyBLAW/downloads
.. _`PyBLAW 0.5.2`: http://github.com/memmett/PyBLAW/downloads
.. _`PyBLAW 0.5.7`: http://github.com/memmett/PyBLAW/downloads
.. _`PyBLAW 0.5.9`: http://github.com/memmett/PyBLAW/downloads
