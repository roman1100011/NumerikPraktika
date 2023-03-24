import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
# Define the Lorentz function
def lorentz(x, a, x0, sigma):
    return a / (1 + ((x - x0)/(sigma/2))**2)




imported_data = np.loadtxt("data.txt")
x,y= imported_data[:,0],imported_data[:,1]


#---------------------Ausgleichsrechnung--------------------------------------------
n = 4
A = np.array(np.zeros([n+1,len(x)]))
A[0,]= 1
for o in range(1,n):
    A[o,0:] = np.power(x[:], (4-o))
A[4,0:]=lorentz(x[:],1,80.3e3,100)

A = A.T
#-----A^TA ausrechnen---------------------
A_dig = A.T @ A
b_dig = A.T @ y
#Cholesky zerlegung
L = cholesky(A_dig,lower = True)

#     Ly = A^T *b nach y auflösen
los = solve_triangular(L,b_dig,lower = True)
los = solve_triangular(L.T ,los,lower = False)



fit = np.array(np.zeros(len(x)))
fit[:] = (A[0:,0:4] @ los[0:4])
baseline = fit
#-----------------------------------------------------------------------------------

# Subtract the baseline from the data
residual = y - baseline

# Fit the residual with the Lorentz function
#popt, pcov = curve_fit(lorentz, x, residual, p0=[1, 80.3e3, 100])
#eigener fit:
#corrected = lorentz(x, *popt)
corrected = lorentz(x[:],los[4],80.3e3,100)
# Plot the original data, estimated baseline, and corrected data
plt.plot(x, y, label='Original Data')
plt.plot(x, baseline, label='Estimated Baseline')
plt.plot(x, corrected, label='estimated Data')
plt.plot(x, residual, label='Residual')
plt.legend()
plt.show()