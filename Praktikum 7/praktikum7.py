## imports 
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.linalg import solve_triangular

## start
imported_data = np.loadtxt("Praktikum 7\dataP7.txt")

x_data = imported_data[:,0]
x_data = (x_data - x_data.min())/(x_data.max() - x_data.min())
y_data = imported_data[:,1]

print(x_data,y_data)

## erste Aufgabe
data = np.array([
    [0.1, 0.3, 0.7, 1.2, 1.6, 2.2, 2.7, 3.1, 3.5, 3.9],
    [0.558, 0.569, 0.176, -0.207, -0.133, 0.132, 0.055, -0.090, -0.069, 0.027]
]).T

ti = data[:,0]
yi = data[:,1]

def u(t,x):
    u0, tau, omega, phi = x
    return u0 * np.exp(- tau * t) * np.sin(omega * t + phi)

def P(x):
    return u(ti,x) - yi

def dP(x):
    u0, tau, omega, phi = x
    return np.array([np.exp(- tau * ti) * np.sin(omega * ti + phi), 
            - ti * u0 * np.exp(- tau * ti) * np.sin(omega * ti + phi),
            ti * u0 * np.exp(- tau * ti) * np.cos(omega * ti + phi),
            u0 * np.exp(- tau * ti) * np.cos(omega * ti + phi)],dtype=float).T

x0 = np.array([1,1,3,1],dtype=float)

maxIter = 100
tol = 1e-10
x = x0.copy()
for k in range(maxIter):
    A = dP(x)
    b = P(x)
    q,r = np.linalg.qr(A)
    s = solve_triangular(r,q.T@b)
    x -= s
    err = np.linalg.norm(dP(x).T@P(x))
    print(k, err)
    if err < tol:
        break

tplot = np.linspace(min(ti),max(ti),num=401)

first = plt.subplot(2,2,4)
first.plot(ti,yi,'.',tplot,u(tplot,x))
first.legend(["Daten","gefittet"])
first.set_title("Modellproblem Skript")


def f(p,x):
    a,b,c,d,h,s,x0 = p
    return a*x**3 + b*x**2 + c*x + d + (h/(1 + ((x - x0)/(s/2))**2))

def poly3(p,x):
    a,b,c,d = p
    return a*x**3 + b*x**2 + c*x + d

def F(p):
    return f(p,x_data) - y_data

def dF(p):
    a,b,c,d,h,s,x0 = p
    ret_array = np.zeros([7,len(x_data)],dtype=float)
    ret_array[0] = x_data**3
    ret_array[1] = x_data**2
    ret_array[2] = x_data**1
    ret_array[3] = 1
    ret_array[4] = 1/(((4*((x_data-x0)**2))/(s**2))+1)
    ret_array[5] = (8*h*s*((x_data-x0)**2))/((s**2 + 4*((x_data-x0)**2))**2)
    ret_array[6] = (8*h*(s**2)*(x_data-x0))/(((s**2)+4*((x_data-x0)**2))**2)
    return ret_array.T

iter = 20
delta_min = 1e-10

#Ungedämpft
#Durch einsetzen in den Code der linearen Ausgleichsrechnung und approximation von s=0.2 und x0 = 0.35: [ 0.79961601  8.15741679 -9.91712293  3.66662723  3.03707973]
x0 = np.array([0.79961601,8.15741679,-9.91712293,3.66662723,3.03707973,0.2,0.35],dtype=float)
param = x0.copy()

err_ungedaempft = []

for k in range(iter):
    A = dF(param)
    b = F(param)
    q,r = np.linalg.qr(A)
    s = solve_triangular(r,q.T@b)
    param -= s
    err_ungedaempft.append(np.linalg.norm(dF(param).T@F(param)))
    print("ungedaempft: ",k, err_ungedaempft[k])

xplot = np.linspace(0,1,num=len(x_data))
unplot = plt.subplot(2,2,1)
unplot.plot(x_data,y_data,'.',xplot,f(param,xplot),xplot,y_data-poly3(param[0:4],xplot),'.',xplot,poly3(param[0:4],xplot))
unplot.legend(["Messdaten","Fit","Nutzsignal","Baseline"])
unplot.set_title("Ungedaempft")


#Gedämpft 
param = x0.copy()

err_gedaempft = []

for k in range(iter):
    A = dF(param)
    b = F(param)
    q,r = np.linalg.qr(A)
    delta = 1
    s = solve_triangular(r,q.T@b)
    while (np.linalg.norm(F(param - delta * s)) > np.linalg.norm(F(param)) and delta > delta_min):
        delta /= 2
    param -= delta*s
    err_gedaempft.append(np.linalg.norm(dF(param).T@F(param)))
    print("gedaempft: ",k, err_gedaempft[k])

xplot = np.linspace(0,1,num=len(x_data))
geplot = plt.subplot(2,2,2)
geplot.plot(x_data,y_data,'.',xplot,f(param,xplot),xplot,y_data-poly3(param[0:4],xplot),'.',xplot,poly3(param[0:4],xplot))
geplot.legend(["Messdaten","Fit","Nutzsignal","Baseline"])
geplot.set_title("Gedaempft")

# Konvergenzverhalten
errplot = plt.subplot(2,2,3)
errplot.plot(err_ungedaempft,'o',err_gedaempft,'.')
errplot.legend(["Ungedaepft","Gedaempft"])
plt.show()

