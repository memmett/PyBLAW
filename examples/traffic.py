"""Traffic flow solver."""

import cProfile as profile
import pstats

import numpy as np
import pyblaw.wenosolver
import pyblaw.system

# use a cython based flux function (see traffic_flux.pyx)
import pyximport; pyximport.install()
import traffic_flux

# initial condition
def q0(x, t):
    return np.array([0.0, 0.0, 0.0])

# the solver
solver = pyblaw.wenosolver.WENOCLAWLFSolver(
    f=traffic_flux.f,
    order=3,
    alpha=4.0,
    times=np.linspace(0.0, 10.0, 41),
    system=pyblaw.system.SimpleSystem(q0)
    )

# build/load grid and cache
if not solver.load_cache():
    solver.build_cache(np.linspace(-10.0, 10.0, 41))

# giv'r! (and profile)
profile.run("solver.run()", 'traffic.prof')
p = pstats.Stats('traffic.prof')
p.strip_dirs().sort_stats('time', 'cum').print_stats(10)
