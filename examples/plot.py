"""Plot the first component of the solutions contained in
   'output.hdf5' for all dump times."""

import h5py as h5
import matplotlib.pyplot as plt

hdf = h5.File('output.hdf5')
q   = hdf['data/q']
(M, N, p) = q.shape

for i in range(M):
    plt.plot(q[i,:,0], 'o-')

plt.show()
hdf.close()
