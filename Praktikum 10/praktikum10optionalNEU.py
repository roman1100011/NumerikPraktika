import numpy as np
import matplotlib.pyplot as plt

# ------------ Implementation des Modells und dessen analytischer Lösung --------------
def ya(x):     # analytische Lösung
    return - (16-(2/3)*x**3)**0.5  # für alle x grösser oder gleich 2*3**(1/3)

def f(x,y):
    return -(x**2/y)

def df(x,y):
    return x**2/y**2

y0 = -4

# ------------ Definition Runge-Kutta Verfahren explizit -------------------------------
def Runge_Kutta(a, b, c, xend, h, y0, f):
    x = [0.]
    y = [y0]
    s = np.size(b)
    r = np.zeros(s)
    xalt = 0
    yalt = y0

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



# ------------ Definition Runge-Kutta Verfahren explizit mit Butcher Tableau 4. Ordnung -------------------------------
def Runge_Kutta_4(xend, h, y0, f):
    a = np.array([[0,       0,       0,      0,     0,    0],
                  [2/9,     0,       0,      0,     0,    0],
                  [1/12,    1/4,     0,      0,     0,    0],
                  [69/128, -243/128, 135/64, 0,     0,    0],
                  [-17/12,  27/4,   -27/5,   16/15, 0,    0],
                  [65/432, -5/16,    13/16,  4/27,  5/144,0]])
    b = np.array([1/9, 0, 9/20, 16/45, 1/12, 0])
    c = np.array([0, 2/9, 1/3, 3/4, 1, 5/6])

    return Runge_Kutta(a, b, c, xend, h, y0, f)


# ------------ Definition Runge-Kutta Verfahren explizit mit Butcher Tableau 5. Ordnung -------------------------------
def Runge_Kutta_5(xend, h, y0, f):
    a = np.array([[0,       0,       0,      0,     0,    0],
                  [2/9,     0,       0,      0,     0,    0],
                  [1/12,    1/4,     0,      0,     0,    0],
                  [69/128, -243/128, 135/64, 0,     0,    0],
                  [-17/12,  27/4,   -27/5,   16/15, 0,    0],
                  [65/432, -5/16,    13/16,  4/27,  5/144,0]])
    b = np.array([47/450, 0, 12/25, 32/225, 1/30, 6/25])
    c = np.array([0, 2/9, 1/3, 3/4, 1, 5/6])

    return Runge_Kutta(a, b, c, xend, h, y0, f)


# ------------ Berechnung des absoluten Fehlers -------------------------------------
def absError(f, ya, y0):
    xend = 2

    h = []
    err_rk4 = []
    err_rk5 = []
    for j in range(0, 12, 1):
        hnew = 2 / (2 ** ((j+1) - 1))

        xrk4, yrk4 = Runge_Kutta_4(xend, hnew, y0, f)
        xrk5, yrk5 = Runge_Kutta_5(xend, hnew, y0, f)

        h.append(hnew)
        err_rk4.append(abs(ya(xend) - yrk4[-1]))
        err_rk5.append(abs(ya(xend) - yrk5[-1]))

    # Darstellung des Absoluten Fehlers bei Variabler Schrittweite
    plt.figure('Fehler Runge-Kutta Verfahren')
    plt.title('Fehler Runge-Kutta Verfahren bei verschiedenen Schrittweiten')
    plt.loglog(h, err_rk4, 'o', label='Fehler Runge-Kutta 4. Ordnung')
    plt.loglog(h, err_rk5, 'o', label='Fehler Runge-Kutta 5. Ordnung')
    print(max(err_rk4))
    print(max(err_rk5))
    plt.xlabel('h')
    plt.ylabel('absoluter Fehler')
    plt.legend()
    plt.grid()
    plt.show()

    return err_rk4, err_rk5


err_rk4, err_rk5 = absError(f, ya, y0)



# ------------ Definition RK45 Verfahren mit Butcher Tableau -------------------------------
def Runge_Kutta_45(xend, h, y0, f, tol=1e-6, hmin=1e-8, hmax=1.0):
    a = np.array([[0,       0,       0,      0,     0,    0],
                  [2/9,     0,       0,      0,     0,    0],
                  [1/12,    1/4,     0,      0,     0,    0],
                  [69/128, -243/128, 135/64, 0,     0,    0],
                  [-17/12,  27/4,   -27/5,   16/15, 0,    0],
                  [65/432, -5/16,    13/16,  4/27,  5/144,0]])
    b4 = np.array([1/9, 0, 9/20, 16/45, 1/12, 0])
    b5 = np.array([47/450, 0, 12/25, 32/225, 1/30, 6/25])
    c = np.array([0, 2/9, 1/3, 3/4, 1, 5/6])

    x = [0.]
    y = [y0]
    #h = min(hmax, h)
    #h = max(hmin, h)

    while x[-1] < xend - h / 2:
        # Schrittweite anpassen
        if xend - x[-1] < 1.5 * h:
            h = xend - x[-1]

        # 4. Ordnung
        k4 = np.zeros(len(a))
        for i in range(len(a)):
            k4[i] = f(x[-1] + c[i] * h, y[-1] + h * sum(a[i, :len(a[i])] * k4[:len(a[i])]))

        y4 = y[-1] + h * sum(b4 * k4)

        # 5. Ordnung
        k5 = np.zeros(len(a))
        for i in range(len(a)):
            k5[i] = f(x[-1] + c[i] * h, y[-1] + h * sum(a[i, :len(a[i])] * k5[:len(a[i])]))

        y5 = y[-1] + h * sum(b5 * k5)

        # Fehlerabschätzung
        err = np.abs(y5 - y4) / h

        # Schrittweite anpassen
        if err < tol:
            yneu = y5
            xneu = x[-1] + h
            y.append(yneu)
            x.append(xneu)
        else:
            h *= 0.9 * (tol / err) ** 0.25
            h = min(hmax, h)
            continue

        # Schrittweite anpassen
        if err == 0.0:
            h *= 2.0
        else:
            h *= 0.9 * (tol / err) ** 0.2

        h = min(hmax, h)
        h = max(hmin, h)
    return np.array(x), np.array(y)

for j in range(1, 13, 1):
    tol = 10**-j

    xrk45, yrk45 = Runge_Kutta_45(2, 0.1, y0, f, tol)

    max_err = 0
    for k in range(len(xrk45)):
        max_err = max(max_err, abs(ya(xrk45[k]) - yrk45[k]))

    print(max_err)

