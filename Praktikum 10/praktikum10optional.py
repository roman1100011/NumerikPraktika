import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

# ------------ Implementation des ersten Modells und dessen analyitischer Lösung --------------
def ya1(x):
    return - (16-(2/3)*x**3)**0.5  # für alle x grösser oder gleich 2*3**(1/3)

def f1(x,y):
    return -(x**2/y)

def df1(x,y):
    return x**2/y**2


# ------------ Implementation des klassischen Runge-Kutta-Verfahren mit Ordnung 4 --------------
def explicit_RK4_konstant(h, x_end, y0, f):

    n = int(x_end / h)      # Anzahl Schritte
    x = np.linspace(0, x_end, n+1)
    y = np.zeros(n+1)
    y[0] = y0

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h/2, y[i] + h/2 * k1)
        k3 = f(x[i] + h/2, y[i] + h/2 * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    return x, y


# ------------ Implementation des Runge-Kutta-Verfahrens 4. Ordnung (RK4) --------------
def rk4(f, x0, y0, x_end, h):
    # Anzahl der Schritte berechnen
    N = int((x_end - x0) / h)

    # Arrays für x und y initialisieren
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0

    # RK4 Verfahren anwenden
    for i in range(N):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

    return x, y

# ------------ Implementation des Fehlberg-Verfahrens (RK5) --------------
def rk5(f, x0, xf, y0, h):
    a = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    b = np.array([
        [0, 0, 0, 0, 0],
        [1/4, 0, 0, 0, 0],
        [3/32, 9/32, 0, 0, 0],
        [1932/2197, -7200/2197, 7296/2197, 0, 0],
        [439/216, -8, 3680/513, -845/4104, 0],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ])
    c1 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    c2 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    x = np.arange(x0, xf+h, h)
    y = np.zeros((len(x), len(y0)))
    y[0] = y0
    for i in range(len(x)-1):
        k = np.zeros((6, len(y0)))
        for j in range(6):
            k[j] = f(x[i] + h * a[j], y[i] + h * np.dot(b[j], k))
        y[i+1] = y[i] + h * np.dot(c1, k)
        y_hat = y[i] + h * np.dot(c2, k)
        delta = np.max(np.abs(y_hat - y[i+1]))
        tol = 0.01 * h
        if delta > tol:
            h *= 0.9 * (tol / delta)**0.2
            x_new = x[i]
            y_new = y[i]
            continue
        h *= 0.9 * (tol / delta)**0.25
    return x, y



# ------------ Implementation des Runge-Kutta-Fehlberg-Verfahrens (RKF45) --------------
def rk45(f, x0, y0, x_end, h0, atol, rtol):
    def runge_kutta_step(f, xn, yn, h):
        k1 = h * f(xn, yn)
        k2 = h * f(xn + h / 4, yn + k1 / 4)
        k3 = h * f(xn + 3 * h / 8, yn + 3 * k1 / 32 + 9 * k2 / 32)
        k4 = h * f(xn + 12 * h / 13, yn + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197)
        k5 = h * f(xn + h, yn + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104)
        k6 = h * f(xn + h / 2, yn - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40)
        y_np1 = yn + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
        y_np1_hat = yn + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55
        return y_np1, y_np1_hat

    xn = x0
    yn = y0
    h = h0
    x = [x0]
    y = [y0]
    while xn < x_end:
        y_np1, y_np1_hat = runge_kutta_step(f, xn, yn, h)
        err = np.abs(y_np1 - y_np1_hat)
        if err < atol + rtol * np.abs(y_np1):
            xn = xn + h
            yn = y_np1
            x.append(xn)
            y.append(yn)
        if err != 0:
            h = 0.84 * h * (atol / err) ** 0.25
        else:
            h = 2 * h
    return x, y



