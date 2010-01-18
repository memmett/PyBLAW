"""PyBLAW setup script."""

import os
import re

import setuptools
import numpy as np


######################################################################
# version
execfile('version.py')                  # this sets 'version'


######################################################################
# save git version to 'pyblaw/__git_version__.py'

try:
    git_head_file = os.path.join(os.path.dirname(__file__), '.git', 'HEAD')
    f = open(git_head_file)
    m = re.match(r'ref: (.+)', f.readline())
    ref = m.group(1)
    f.close()

    git_head_file = os.path.join(os.path.dirname(__file__), '.git', ref)
    f = open(git_head_file)
    git_version = f.readline().rstrip()
    f.close()

except:
    git_version = 'not_available'

git_version_file = os.path.join(os.path.dirname(__file__),
                                'pyblaw','__git_version__.py')
f = open(git_version_file, 'w')
f.write("version = '%s'\n" % (git_version))
f.close()


######################################################################
# save version to 'pyblaw/__version__.py'

version_file = os.path.join(os.path.dirname(__file__),
                            'pyblaw','__version__.py')
f = open(version_file, 'w')
f.write("version = '%s'\n" % (version))
f.close()


######################################################################
# setup!

setuptools.setup(

    name = "PyBLAW",
    version = version,
    packages = ['pyblaw'],
    zip_safe = True,

    install_requires = [ "numpy >= 1.0.3", "scipy >= 0.7.0", "pyweno >= 0.1" ],

    ext_modules = [
        setuptools.Extension('pyblaw.clfflux',
                             sources = ['src/clfflux.c'],
                             include_dirs=[np.get_include()]
                             )],

    package_data = {'': ['__version__.py', '__git_version__.py']},
    exclude_package_data = {'': ['.gitignore']},

    author = "Matthew Emmett",
    author_email = "matthew.emmett@ualberta.ca",
    description = "Hyperbolic PDE (balance law) solver",
    license = "BSD",
    url = "http://www.math.ualberta.ca/~memmett/pyblaw/",
    keywords = "balance law, conservation law, hyperbolic pde, pde"

    )
