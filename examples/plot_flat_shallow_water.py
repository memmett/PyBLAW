import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('flat_shallow_water.mat')
q = mat['data.q']
x = mat['dims.xdim']

(M, N, p) = q.shape

plt.plot(x, q[0,:,0], '-.k')            # initial height

plt.plot(x, q[9,:,0], '-k')             # final height
plt.plot(x, q[9,:,1], '-r')             # final momentum

plt.title('Flat-bed shallow-water')
plt.xlabel('x')
plt.ylabel('height/momentum')
plt.legend(['initial height', 'height', 'momentum'])
plt.show()
