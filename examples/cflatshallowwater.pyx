"""Flat-bed shallow-water flux."""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def f(np.ndarray[np.double_t, ndim=2] q, np.ndarray[np.double_t, ndim=2] flux):
    cdef int N = q.shape[0]
    cdef unsigned int i

    for i in range(N):
        if q[i,0] > 0.0:
            flux[i,0] = q[i,1]
            flux[i,1] = q[i,1]*q[i,1]/q[i,0] + 0.5 * q[i,0]*q[i,0]
        else:
            flux[i,0] = 0.0
            flux[i,1] = 0.0
