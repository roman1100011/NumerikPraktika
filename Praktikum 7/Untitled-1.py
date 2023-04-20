# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
data = np.array([
    [0.1, 0.3, 0.7, 1.2, 1.6, 2.2, 2.7, 3.1, 3.5, 3.9],
    [0.558, 0.569, 0.176, -0.207, -0.133, 0.132, 0.055, -0.090, -0.069, 0.027]
]).T

# %%
plt.plot(*data.T,'o')

# %%
ti = data[:,0]
yi = data[:,1]

print(ti,yi)

# %%
def y(t,x):
    A, tau, omega, phi = x
    return A*np.exp(-tau*t)*np.sin(omega*t+phi)

def F(x):
    return y(ti,x)-yi

def dF(x):
    A, tau, omega, phi = x
    return np.array([np.exp(-tau*ti)*np.sin(omega*ti+phi),
                     -ti*A*np.exp(-tau*ti)*np.sin(omega*ti+phi),
                     ti*A*np.exp(-tau*ti)*np.cos(omega*ti+phi),
                     A*np.exp(-tau*ti)*np.cos(omega*ti+phi)]).T


# %%
x0 = np.array([1,1,3,1],dtype=float)
plt.plot(*data.T,'o')
plt.plot(ti,y(ti,x0))

# %%
from scipy.linalg import solve_triangular

# %%
maxIter = 100
tol = 1e-10
x = x0.copy()
for k in range(maxIter):
    A = dF(x)
    b = F(x)
    q,r = np.linalg.qr(A)
    s = solve_triangular(r,q.T@b)
    x -= s
    err = np.linalg.norm(dF(x).T@F(x))
    print(k, err)
    if err < tol:
        break


# %%



