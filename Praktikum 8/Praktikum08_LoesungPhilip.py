import numpy as np
import matplotlib.pyplot as plt
import simpy as sp

# ------------ Implementation des Modells und dessen analyitischer Lösung --------------
# Implementation der analytischen Lösung
def ya(x):
    return np.exp(-4*x)

def f(x,y):
    return -4*y     # stimmt das hier so?


# ------------ Definition des expliziten Eulerverfahrens -------------------------------
def explizitEuler(xend, h, y0, f):
    x = [0.]
    y = [y0]
    xalt = 0
    yalt = y0

    while x[-1] < xend-h/2:
        # explizites Eulerverfahren
        yneu = yalt + h*f(xalt, yalt)
        xneu = xalt + h

        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)

        yalt = yneu
        xalt = xneu
    return np.array(x), np.array(y)


# # ------------ Definition des impliziten Eulerverfahrens -------------------------------
# def implizitEuler(xend, h, y0, f, df):
#     x = [0.]
#     y = [y0]
#
#     # Verfahrensfunktion für implizit Euler
#     def G(s, xk, yk):
#         return s - yk - h * f(xk, s)    # stimmt hier X_k im Funktionsaufruf? Gemäss Praktikumsbeschrieb müsste es X_k+1 sein....
#
#     # Partielle Ableitung nach s der Verfahrensfunktion
#     def dG(s, xk, yk):
#         return sp.diff(G,s)    # passt das so, wenn ich hier einfach mit simpy ableite?
#
#     def newton(s, xk, yk, tol=1e-12, maxIter=20):
#         k=0
#         delta = 10*tol
#         while np.abs(delta) > tol and k < maxIter:
#             delta = #<<snipp>>
#             s -= delta
#             k += 1
#         return s
#     while x[-1] < xend-h/2:
#         y.append(newton(y[-1],x[-1],y[-1]))
#         x.append(x[-1]+h)
#     return np.array(x), np.array(y)



# ------------ Kontrolle der Konvergenzordnung -------------------------------------
n = 10**np.linspace(2,5)
hs = 2/n
err = []
for h in hs:
    x, y = explizitEuler(2,h,1,f)
    err.append(np.linalg.norm(y-ya(x),np.inf)) # ya(x) ist die exakte Lösung

plt.loglog(hs,err,'-')
plt.xlabel('h')
plt.ylabel(r'$\max_k \|e(x_k,h)\|$')
plt.grid()
plt.show()