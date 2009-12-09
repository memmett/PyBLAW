import os
import scipy.io as sio

import matplotlib
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

import matplotlib.pyplot as plt

mat = sio.loadmat('shallow_water.mat')
q = mat['data.q']
x = mat['dims.xdim']

(M, N, p) = q.shape

plt.figure(figsize=(10,4))

for i in range(M):
    plt.clf()
    plt.plot(x, q[i,:,0]+q[0,:,2], '-k') # height
    plt.plot(x, q[i,:,1], '-r')          # momentum
    plt.plot(x, q[i,:,2], '-b')          # bed
    plt.xlim([0.0, 3.0])
    plt.ylim([-0.2, 1.4])

    fname = '_tmp%05d.png' % (i)
    plt.savefig(fname)


fps = 10
os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o shallow_water.mpg")
os.system("rm -f _tmp*.png")
