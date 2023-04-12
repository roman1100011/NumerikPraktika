import numpy as np
import matplotlib.pyplot as plt
from Praktikum08_LoesungA1bisA5 import implizitEuler, explizitEuler #, absError, Konvergenzordnungskontrolle


# ------------ Implementation des AWP's und dessen analyitischer Lösung --------------
# Implementation der analytischen Lösung
def ya(x):
    return np.sqrt(16 - (2/3) * x**3)

def f(x,y):
    return - x**2 * (1/y)

def df(x,y):
    return x**2/y**2


# ------------ AWP mit beiden Verfahren berechnen und darstellen --------------
xe, ye = explizitEuler(2, 0.1, 4, f)
xi, yi = implizitEuler(2, 0.1, 4, f, df)
xp = np.linspace(0,2,100)
plt.figure('Lösung des AWP')
plt.plot(xp, ya(xp),'-', label='analytische Lösung')
plt.plot(xe, ye,'-', label='explizite Lösung')
plt.plot(xi, yi,'-', label='implizite Lösung')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


# ------------ absoluten Fehler für verschiedene Schrittweiten berechnen und darstellen --------------
hs = []
for j in np.linspace(1,8,8):
    hs.append(2/(3**j))         # die einzelnen h's berechnen, wie im Praktikumsbeschrieb vorgegeben

err_exp = []
err_imp = []
for h in hs:
    xe, ye = explizitEuler(2, h, 4, f)
    err_exp.append(np.linalg.norm(ye - ya(xe), np.inf))  # ya(xe) ist die exakte Lösung am Punkt xe
    xi, yi = implizitEuler(2, h, 4, f, df)
    err_imp.append(np.linalg.norm(yi - ya(xi), np.inf))  # ya(xi) ist die exakte Lösung am Punkt xi

plt.figure('Konvergenzordnung')
plt.semilogx(hs,err_exp,'-', label='explizit')
plt.semilogx(hs,err_imp,'-', label='implizit')
plt.xlabel('h')
plt.ylabel('absoluter Fehler')
plt.legend()
plt.grid()
plt.show()




# ------------ Lösung im Richtungsfeld visualisieren, für N=3^8 --------------
h = 2/(3**8)
xe, ye = explizitEuler(2, h, 4, f)
xi, yi = implizitEuler(2, h, 4, f, df)
xp = np.linspace(0,2,100)
plt.figure('Lösung im Richtungsfeld')
plt.plot(xp, ya(xp),'-', label='analytische Lösung')
plt.plot(xe, ye,'-', label='explizite Lösung')
plt.plot(xi, yi,'-', label='implizite Lösung')
xq,yq = np.meshgrid(np.linspace(0,2,int(2/.05)),np.linspace(np.min(ya(xp)),np.max(ya(xp)),10))
plt.quiver(xq,yq,np.ones_like(xq),f(xq,yq),angles='xy')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
