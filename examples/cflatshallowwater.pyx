"""Flat-bed shallow-water flux."""

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
def f(np.ndarray[np.double_t, ndim=2] q,
      np.ndarray[np.double_t, ndim=2] flux,
      **kwargs):
    cdef int N = q.shape[0]
    cdef unsigned int i

    for i in range(1, N-1):
        if q[i,0] > 0.0:
            flux[i,0] = q[i,1]
            flux[i,1] = q[i,1]*q[i,1]/q[i,0] + 0.5 * q[i,0]*q[i,0]
        else:
            flux[i,0] = 0.0
            flux[i,1] = 0.0

    flux[0,:] = flux[1,:]
    flux[N-1,:] = flux[N-2,:]

