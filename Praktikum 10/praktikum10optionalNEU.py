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

# ------------ Definition allgemeines Runge-Kutta Verfahren explizit -------------------------------
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
            r[i] = f(xalt + c[i] * h, yalt + h * sum(a[i]*r))
        yneu = yalt + h * sum(b*r)
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
    # Implementation des Butcher-Tableaus
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
    # Implementation des Butcher-Tableaus
    a = np.array([[0,       0,       0,      0,     0,    0],
                  [2/9,     0,       0,      0,     0,    0],
                  [1/12,    1/4,     0,      0,     0,    0],
                  [69/128, -243/128, 135/64, 0,     0,    0],
                  [-17/12,  27/4,   -27/5,   16/15, 0,    0],
                  [65/432, -5/16,    13/16,  4/27,  5/144,0]])
    b = np.array([47/450, 0, 12/25, 32/225, 1/30, 6/25])
    c = np.array([0, 2/9, 1/3, 3/4, 1, 5/6])

    return Runge_Kutta(a, b, c, xend, h, y0, f)


# ------------ Berechnung und Darstellung des absoluten Fehlers -------------------------------------
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
def Runge_Kutta_45(xend, h, y0, f, tol):
    # Implementation des Butcher-Tableaus
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
    xalt = 0
    yalt = y0
    s = 6
    r = np.zeros(s)

    while x[-1] < xend - h/2:
        # Schritt mit RK4
        r4 = np.zeros(s)
        for i in range(s):
            r4[i] = f(xalt + c[i]*h, yalt + h*np.sum(a[i]*r4))
        y4 = yalt + h*np.sum(b4*r4)

        # Schritt mit RK5
        r5 = np.zeros(s)
        for i in range(s):
            r5[i] = f(xalt + c[i]*h, yalt + h*np.sum(a[i]*r5))
        y5 = yalt + h*np.sum(b5*r5)

        # Fehler abschätzen
        Delta = np.abs(y4 - y5)

        # Schrittweite anpassen
        if Delta < tol/20:
            h_neu = 2*h
        elif Delta <= tol:
            h_neu = h
        else:
            h_neu = h/2
            h = h_neu
            continue  # y_{k+1} nicht aktzeptieren und wiederholen mit halber Schrittweite

        # Resultate speichern
        x.append(xalt + h)
        y.append(y5)
        xalt = x[-1]
        yalt = y[-1]
        r = r5
        h = h_neu

    return np.array(x), np.array(y)



# ------------ Lösen des Modellproblems mit dem RK45-Verfahren für versch. Toleranzen -------------------------------
for j in range(1, 13, 1):
    # Toleranz festlegen und RK45 mit dieser Toleranz aufrufen
    tol = 10**-j
    xrk45, yrk45 = Runge_Kutta_45(2, 0.1, y0, f, tol)

    # Maximalen absoluten Fehler berechnen und ausgeben
    max_err = 0
    for k in range(len(xrk45)):
        max_err = max(max_err, abs(ya(xrk45[k]) - yrk45[k]))
    print(max_err)

