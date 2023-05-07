import numpy as np
# Variabeln defineiren
g = 9.81  #Erdbeschleunigung
ks = 0.05 #Federkonstante
m = 0.05  #Masse der Kugel
r = 0.1   #Durchmesser der Kugel
roh = 1.184
Ca = 0.45
v0 = 1    #Srömungsgeschwindigkeit
k = 0.5*Ca*roh*np.power(r,2)*np.pi #Zusammenfassug der Fluiddynamischen widerstände
h = 0.1 #Schrittgrösse

# Diff ODE
"""
x'0 = x1
x'1 = g- (ks/m)*x0-(k/m)*(x1+v0)*|x1+v0|
"""
def f(t,x):
    Hesse = np.array([[0, 1],[-ks/m, -2*k/m*np.sign(x[1]+v0)*(x[1]+v0)]])
    #Ableitung
    return Hesse

r1 = lambda t,x0,x1 : f(t,np.array([x0,x1]))
r1 = lambda t,x0,x1 : f(t,np.array([x0,x1]))
print(f(0,0.4))
