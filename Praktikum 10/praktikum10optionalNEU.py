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
    h_plot = 0.1
    xend_plot = 1
    y0_plot = y0
    xp = np.linspace(0, xend_plot, 100)
    xrk4, yrk4 = Runge_Kutta_4(xend_plot, h_plot, y0_plot, f)
    xrk5, yrk5 = Runge_Kutta_5(xend_plot, h_plot, y0_plot, f)

    plt.figure('Runge-Kutta Verfahren')
    plt.title('Runge-Kutta Verfahren')
    plt.plot(xp, ya(xp), '-', label='analytische Lösung')
    plt.plot(xrk4, yrk4, '--', label='Runge-Kutta 4. Ordnung')
    plt.plot(xrk5, yrk5, '--', label='Runge Kutta 5. Ordnung')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure('Fehler Runge-Kutta Verfahren')
    plt.title('Fehler Runge-Kutta Verfahren')
    plt.plot(xrk4, np.abs(yrk4 - ya(xrk4)), 'o', label='Fehler Runge-Kutta 4. Ordnung')
    plt.plot(xrk5, np.abs(yrk5 - ya(xrk5)), '.', label='Fehler Runge-Kutta 5. Ordnung')
    print(max(np.abs(yrk4 - ya(xrk4))))
    print(max(np.abs(yrk5 - ya(xrk5))))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()


absError(f, ya, y0)