"""PyBLAW setup script."""

import setuptools
import numpy as np

setuptools.setup(

    name = "PyBLAW",
    version = "0.1",
    packages = ['pyblaw'],
    zip_safe = True,

    install_requires = [ "numpy >= 1.0.3", "scipy >= 0.7.0", "h5py >= 1.1.0", "pyweno >= 0.1" ],

    ext_modules = [
        setuptools.Extension('pyblaw.clinearflux',
                             sources = ['src/clinearflux.c'],
                             include_dirs=[np.get_include()]
                             ),
        setuptools.Extension('pyblaw.clinearsource',
                             sources = ['src/clinearsource.c'],
                             include_dirs=[np.get_include()]
                             )],

    author = "Matthew Emmett",
    author_email = "matthew.emmett@ualberta.ca",
    description = "Hyperbolic PDE (balance law) solver",
    license = "BSD",
    url = "http://www.math.ualberta.ca/~memmett/pyblaw/",
    keywords = "balance law, conservation law, hyperbolic pde, pde"

    )
