import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Lorentz function
def lorentz(x, a, x0, sigma):
    return a / (1 + ((x - x0)/(sigma/2))**2)

# Generate some test data
imported_data = np.loadtxt("data.txt")
x,y= imported_data[:,0],imported_data[:,1]
# Estimate the baseline using a polynomial fit
baseline = np.polyval(np.polyfit(x, y, deg=3), x) #Hier noch eine lineare ausgleichsrechnung!!!

# Subtract the baseline from the data
residual = y - baseline

# Fit the residual with the Lorentz function
popt, pcov = curve_fit(lorentz, x, residual, p0=[0.5, 80e3, 100])

# Subtract the Lorentz peak from the residual to obtain the baseline-corrected data
corrected = residual - lorentz(x, *popt)

# Plot the original data, estimated baseline, and corrected data
plt.plot(x, y, label='Original Data')
plt.plot(x, baseline, label='Estimated Baseline')
plt.plot(x, corrected, label='Baseline-Corrected Data')
plt.legend()
plt.show()