import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from scipy.linalg import lu

g = 9.81
#k = 0.5 * 0.45 * 1.184 * 0.1**2 * np.pi
k = 0
m = 0.05
ks = 0.05

simulation_time = 50

def f(t,x,v0):
    ##v0 = np.sqrt(g*m/k)
    x0 = x[0]
    x1 = x[1]
    return np.array([x1,g-(k/m)*(x1+v0)*np.abs(x1+v0) - x0*(ks/m)],dtype=float)

def df(t,x,v0):
    x0 = x[0]
    x1 = x[1]
    return np.array([[0,1],[-(ks/m),-2*(k/m)*np.abs(x1+v0)]],dtype=float)

def Runge_Kutta(xend, h, y0, f, v0):
    x = np.arange(0, xend + h, h)
    y = np.zeros((len(x), len(y0)))
    y[0] = y0

    r = np.zeros((len(y0),4))
    a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
    c = np.array([0,0.5,0.5,1])
    b = np.array([1/6,1/3,1/3,1/6])



    for xpos in range(len(x) - 1):
        # Runge-Kutta Verfahren Schritt
        for i in range(4):
            r[:,i] = f(0, y[xpos] + h * np.sum(r@a[i],axis=0), v0)
        y[xpos+1] = y[xpos] + h * (r@b)

    return np.array(x), np.array(y)


def Implizite_Mittelpunkt(xend, h, y0, f, df, v0):
    x = np.arange(0, xend + h, h)
    y = np.zeros((len(x), len(y0)))
    y[0] = y0
    x[0] = 0

    r = np.zeros(len(y0))

    def G(r0, xk, yk):
        return r0 - f(xk+h/2,yk + (h/2) * r0,v0)
    
    def dG(r0, xk, yk):
        return np.eye(len(yk)) - df(xk+h/2,yk + (h/2) * r0,v0)*(h/2)
    
    def GaussNewton(xk,yk,F,dF):
        r_input = np.array([0,0],dtype=float)
        k = 0
        tol = 1e-10
        delta = 10*tol*np.ones(len(yk))
        while np.linalg.norm(delta,np.inf) > tol and k < 1000:
            Asolve = dF(r_input,xk,yk)
            bsolve = F(r_input,xk,yk)
            p,l,u = lu(Asolve)
            z = solve_triangular(l,p.T@bsolve,lower=False)
            delta = solve_triangular(u,z,lower=True)
            r_input -= delta
            k += 1
        return r_input

    for xpos in range(len(x) - 1):
        r = GaussNewton(x[xpos],y[xpos],G,dG)
        y[xpos+1] = y[xpos] + h*r

    return x, y

## Energieerhaltung testen

def Energie(pos):
    return 0.5 * ks * ((pos[:,0] - m*g/ks)**2) + 0.5 * m * (pos[:,1]**2)

xm, ym = Implizite_Mittelpunkt(simulation_time, 0.8, np.array([0,0],dtype=float), f, df,0.)
xr, yr = Runge_Kutta(simulation_time, 0.8, np.array([0,0],dtype=float), f, 0.)


plt.plot(xm,Energie(ym),label='Energie implizite Mittelpunktregel')
plt.plot(xr,Energie(yr),label='Energie RK4 Verfahren')
plt.title('$v_0 = 0, timestep = 0.1$, Energieerhaltung iM vs RK4')
plt.legend()
plt.show()