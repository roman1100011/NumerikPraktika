import numpy as np
import matplotlib.pyplot as plt

# ------------ Implementation des Modells und dessen analyitischer Lösung --------------
# Implementation der analytischen Lösung
def ya(x):
    return np.exp(-4*x)

def f(x,y):
    return -4*y

def df(x,y):
    return -4

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


def implizitEuler(xend, h, y0, f, df):
    x = [0.]
    y = [y0]

    # Verfahrensfunktion für implizit Euler
    def G(s, xk, yk):
        return s - yk - h * f(xk + h, s)

    # Partielle Ableitung nach s der Verfahrensfunktion
    def dG(s, xk, yk):
        return 1 - h * df(xk + h, s)

    def newton(s, xk, yk, tol=1e-12, maxIter=20):
        k = 0
        delta = 10 * tol
        while np.abs(delta) > tol and k < maxIter:
            delta = G(s, xk, yk) / dG(s, xk, yk)
            s -= delta
            k += 1
        return s

    while x[-1] < xend - h / 2:
        y.append(newton(y[-1], x[-1], y[-1], tol=1e-12, maxIter=20))
        x.append(x[-1] + h)

    return np.array(x), np.array(y)




# ------------ Berechnung des absoluten Fehlers -------------------------------------
def absError(f, df, ya):
    xp = np.linspace(0,2,100)
    xe, ye = explizitEuler(2, 0.01, 1, f)
    xi, yi = implizitEuler(2, 0.01, 1, f, df)

    plt.figure('absoluter Fehler')
    plt.plot(xp, ya(xp),'-', label='analytische Lösung')
    plt.plot(xe, ye,'-', label='explizite Lösung')
    plt.plot(xi, yi,'-', label='implizite Lösung')
    plt.plot(xe, np.abs(ye - ya(xe)),'-', label='Fehler expl. Lösung')
    plt.plot(xe, np.abs(yi - ya(xi)),'-', label='Fehler impl. Lösung')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

#absError(f, df, ya)

# ------------ Kontrolle der Konvergenzordnung -------------------------------------

def Konvergenzordnungskontrolle(f, df, ya):
    n = 10**np.linspace(1,5)
    hs = 2/n
    err_exp = []
    err_imp = []
    for h in hs:
        xe, ye = explizitEuler(2, h, 1, f)
        err_exp.append(np.linalg.norm(ye - ya(xe), np.inf))  # ya(xe) ist die exakte Lösung am Punkt xe

        xi, yi = implizitEuler(2,h,1,f,df)
        err_imp.append(np.linalg.norm(yi-ya(xi),np.inf)) # ya(xi) ist die exakte Lösung am Punkt xi

    plt.figure('Konvergenzordnung')
    plt.loglog(hs,err_exp,'-', label='explizit')
    plt.loglog(hs,err_imp,'-', label='implizit')
    plt.xlabel('h')
    plt.ylabel(r'$\max_k \|e(x_k,h)\|$')
    plt.legend()
    plt.grid()
    plt.show()

#Konvergenzordnungskontrolle(f, df, ya)
