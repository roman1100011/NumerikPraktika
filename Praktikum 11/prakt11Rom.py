import numpy as np
import matplotlib.pyplot as plt
def Hesse(time,x):
    t,u = map(np.array, (time,x))
    ks = 0.05
    k = 0.5 * 0.45*1.184*0.031
    m = 0.05
    v = 3.
    H = np.zeros([2,2])
    H[0, 0] = 0
    H[0, 1] = 1
    H[1, 0] = -ks/m
    H[1, 1] = -(2*k)/m*np.abs(u+v)
    return H


def rk4(func, X0, tim):
    """
    Runge Kutta 4 solver.
    """
    dt = tim[1] - tim[0]
    nt = len(tim)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], tim[i])
        k2 = func(X[i] + dt / 2. * k1, tim[i] + dt / 2.)
        k3 = func(X[i] + dt / 2. * k2, tim[i] + dt / 2.)
        k4 = func(X[i] + dt * k3, tim[i] + dt)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X


t = np.arange(0,100,0.1)

plt.plot(t,rk4(Hesse, [0, 0],t))


