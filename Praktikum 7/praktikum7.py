import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

imported_data = np.loadtxt("dataP7.txt")

def u(t,x):
    u0, tau, omega, phi = x
    return u0 * sp.exp(- tau * t) * sp.sin(omega * t + phi)

def u_(t,x):
    u0, tau, omega, phi = x
    return [sp.exp(- tau * t) * sp.sin(omega * t + phi), 
            - t * u0 * sp.exp(- tau * t) * sp.sin(omega * t + phi),
            t * u0 * sp.exp(- tau * t) * sp.cos(omega * t + phi),
            u0 * sp.exp(- tau * t) * sp.cos(omega * t + phi)]

def GaussNewton(t,x0):
    return 0

