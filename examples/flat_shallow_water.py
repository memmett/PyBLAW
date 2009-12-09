"""Shallow-water solver.

   This flat-bed shallow-water solver serves as short example of how
   to use the PyBLAW framework.

   It solves the depth-averaged shallow-water equations, which are

     * height:   h_t + (u h)_x = 0
     * momentum: (h u)_t + ( h u^2 + 1/2 h^2 )_x = 0

   where h is the depth of the fluid, and u is the velocity of the
   fluid.

   Throughout, q is:

     * q[i,0] - average depth in cell C_i
     * q[i,1] - average momentum in cell C_i

"""

import numpy as np
import pyblaw.wenoclaw
import pyblaw.system

# import cython flux and source functions
import pyximport; pyximport.install()
import cflatshallowwater

# initial conditions (dambreak into shallow tail water)
def h0(x, t):
    if x <= 0:
        return 1.0

    return 0.05

def q0(x, t):
    return np.array([h0(x,t), 0.0])

# the solver (XXX: try other k values)
solver = pyblaw.wenoclaw.WENOCLAWLFSolver(
    order=3,
    times=np.linspace(0.0, 15.0, 15*10*4+1),
    flux={ 'f': cflatshallowwater.f, 'alpha': 2.0 },
    system=pyblaw.system.SimpleSystem(q0),
    cache='flat_shallow_water_cache.mat',
    output='flat_shallow_water.mat'
    )

# build/load grid and cache
if not solver.load_cache():
    solver.build_cache(np.linspace(-25.0, 25.0, 50*10+1))

# giv'r!
solver.run()
