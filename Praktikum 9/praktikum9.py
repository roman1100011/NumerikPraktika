import numpy as np
import matplotlib.pyplot as plt


# ------------ Implementation des ersten Modells und dessen analyitischer Lösung --------------
def ya1(x):
    return np.exp(-4*x)

def f1(x,y):
    return -4*y

# ------------ Implementation des zweiten Modells und dessen analyitischer Lösung --------------
def ya2(x):
    return - np.sqrt(16 - ((2*(x**3))/3))

def f2(x,y):
    return -x**2/y

# ------------ Definition Runge Verfahrens-------------------------------
def Runge(xend, h, y0, f):
    x = [0.]
    y = [y0]
    xalt = 0
    yalt = y0

    while x[-1] < xend-h/2:
        # Runge Verfahren Schritt
        r1 = f(xalt, yalt)
        r2 = f(xalt+h/2,yalt+(h/2)*r1)
        yneu = yalt + h*r2
        xneu = xalt + h

        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)

        yalt = yneu
        xalt = xneu
    return np.array(x), np.array(y)

# ------------ Definition Heun Verfahren-------------------------------
def Heun(xend, h, y0, f):
    x = [0.]
    y = [y0]
    xalt = 0
    yalt = y0

    while x[-1] < xend-h/2:
        # Heun Verfahren Schritt
        k1 = yalt
        k2 = yalt + h * f(xalt,yalt)
        yneu = yalt + 0.5 * h * (f(xalt,k1) + f(xalt+h,k2))
        xneu = xalt + h

        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)

        yalt = yneu
        xalt = xneu
    return np.array(x), np.array(y)

# ------------ Definition Runge-Kutta Verfahren-------------------------------
def Runge_Kutta(s, xend, h, y0, f):
    x = [0.]
    y = [y0]
    xalt = 0
    yalt = y0
    r = np.zeros(s)
    a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
    c = np.array([0,0.5,0.5,1])
    b = np.array([1/6,1/3,1/3,1/6])



    while x[-1] < xend-h/2:
        # Runge-Kutta Verfahren Schritt
        for i in range(s):
            r[i] = f(xalt + c[i] * h, yalt + h * sum(np.multiply(a[i],r)))
        yneu = yalt + h * sum(np.multiply(b,r))
        xneu = xalt + h

        # Speichern des Resultats
        y.append(yneu)
        x.append(xneu)

        yalt = yneu
        xalt = xneu
        r = np.zeros(s)
    return np.array(x), np.array(y)

# ------------ Berechnung des absoluten Fehlers -------------------------------------
def absError(f, ya, y0):
    h_plot = 0.1
    xend_plot = 1
    y0_plot = y0
    xp = np.linspace(0,xend_plot,100)
    xr, yr = Runge(xend_plot,h_plot, y0_plot, f)
    xh, yh = Heun(xend_plot,h_plot,y0_plot,f)
    xrk, yrk = Runge_Kutta(4, xend_plot,h_plot,y0_plot,f)

    plt.figure('Runge/Heun Verfahren')
    plt.title('Runge/Heun/Runge-Kutta Verfahren')
    plt.plot(xp, ya(xp),'-', label='analytische Lösung')
    plt.plot(xr, yr,'--', label='Runge Verfahren')
    plt.plot(xh, yh,'-', label='Heun Verfahren')
    plt.plot(xrk, yrk,'--', label='Runge Kutta (s=4) Verfahren')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure('Fehler Runge/Heun verfahren')
    plt.title('Fehler Runge/Heun/Runge-Kutta Verfahren')
    plt.plot(xr, np.abs(yr - ya(xr)),'o', label='Fehler Runge Verfahren')
    plt.plot(xh, np.abs(yh - ya(xh)),'.', label='Fehler Heun Verfahren')
    plt.plot(xrk, np.abs(yrk - ya(xrk)),'.', label='Fehler Runge-Kutta (s=4) Verfahren')
    print(max(np.abs(yrk - ya(xrk))))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()
    
absError(f1, ya1,1)
absError(f2, ya2,-4)

# Konvergenz überprüfen
def Konvergenzordnungskontrolle(f, ya):
    n = 10**np.linspace(1,5)
    hs = 2/n
    err_run = []
    err_heu = []
    err_ruk = []
    for h in hs:
        xe, ye = Runge(2, h, 1, f)
        err_run.append(np.linalg.norm(ye - ya(xe), np.inf))  # ya(xe) ist die exakte Lösung am Punkt xe

        xi, yi = Heun(2,h,1,f)
        err_heu.append(np.linalg.norm(yi-ya(xi),np.inf)) # ya(xi) ist die exakte Lösung am Punkt xi

        xrk, yrk = Runge_Kutta(4,2,h,1,f)
        err_ruk.append(np.linalg.norm(yrk-ya(xrk),np.inf)) # ya(xi) ist die exakte Lösung am Punkt xi

    plt.figure('Konvergenzordnung')
    plt.title('Konvergenzordnung Verfahren')
    plt.loglog(hs,err_run,'-', label='Runge')
    plt.loglog(hs,err_heu,'--', label='Heun')
    plt.loglog(hs,err_ruk,'-', label='Runge Kutta')
    plt.xlabel('h')
    plt.ylabel(r'$\max_k \|e(x_k,h)\|$')
    plt.legend()
    plt.grid()
    plt.show()

Konvergenzordnungskontrolle(f1,ya1)