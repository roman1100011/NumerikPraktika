import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from scipy.linalg import lu


g = 9.81
k = 0.5 * 0.45 * 1.184 * 0.1**2 * np.pi
m = 0.05
ks = 0.05

simulation_time = 50
# Define the system of differential equations
def f(t,x):
    v0 = 1
    ##v0 = np.sqrt(g*m/k)
    x0 = x[0]
    x1 = x[1]
    return np.array([x1,g-(k/m)*(x1+v0)*np.abs(x1+v0) - x0*ks/m],dtype=float)

def df(t,x):
    v0 = 1
    x0 = x[0]
    x1 = x[1]
    return np.array([[0,1],[-(ks/m),-2*(k/m)*np.abs(x1+v0)]],dtype=float)

def impliziteulerVerfahren(T,phi0,h,f,df):
    ti = [0]
    phi = [phi0]
        
    # Verfahrensfunktion für implizit Euler
    def G(s, tk, phik):
        return s-f(tk+h,phik+h/2*s)

    # Partielle Ableitung nach s der Verfahrensfunktion
    def dG(s, tk, phik):
        return np.eye(len(phik))-df(tk+h,phik+h/2*s)*h/2

    def newton(s, tk, phik, tol=1e-12, maxIter=20):
        k = 0
        delta = 10*tol*np.ones(phik.shape[0])
        while np.linalg.norm(delta,np.inf) > tol and k < maxIter:
            b = G(s,tk,phik)
            A = dG(s,tk,phik)
            p,l,u = lu(A)
            z = solve_triangular(l,p.T@b,lower=False)
            delta = solve_triangular(u,z,lower=True)
            s -= delta
            k += 1
        return s
    
    s = f(ti[-1],phi[-1])
    while ti[-1] < T-h/2:
        r1 = newton(s, ti[-1], phi[-1])
        phi.append(phi[-1]+h*r1)
        ti.append(ti[-1]+h)
    return np.array(ti), np.array(phi)
y0 = np.array([0,0],dtype=float)
ti_i, phi_i = impliziteulerVerfahren(100,y0,0.01,f,df)

plt.plot(ti_i,phi_i[:,0])
plt.plot(ti_i,phi_i[:,1])
plt.show()