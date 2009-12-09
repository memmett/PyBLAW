import scipy.io as sio
import matplotlib.pyplot as plt

mat = sio.loadmat('shallow_water.mat')
q = mat['data.q']
x = mat['dims.xdim']

(M, N, p) = q.shape

plt.plot(x, q[0,:,0]+q[0,:,2], '-.k')     # initial height
plt.plot(x, q[0,:,2], '-b')               # bed

plt.plot(x, q[M-1,:,0]+q[M-1,:,2], '-k')  # final height
plt.plot(x, q[M-1,:,1], '-r')             # final momentum

plt.title('Shallow-water')
plt.xlabel('x')
plt.ylabel('height/momentum')
plt.legend(['initial height', 'bed', 'height', 'momentum'])
plt.show()
