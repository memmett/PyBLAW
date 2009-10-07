"""PyBLAW setup script."""

import setuptools


setuptools.setup(

    name = "PyBLAW",
    version = "0.1",
    packages = ['pyblaw'],
    zip_safe = True,

    install_requires = [ "numpy >= 1.0.3", "h5py >= 1.1.0", "pyweno >= 0.1" ],

    author = "Matthew Emmett",
    author_email = "matthew.emmett@ualberta.ca",
    description = "Hyperbolic PDE (balance law) solver",
    license = "BSD",
    url = "http://www.math.ualberta.ca/~memmett/pyblaw/",
    keywords = "balance law, conservation law, hyperbolic pde, pde"

    )
