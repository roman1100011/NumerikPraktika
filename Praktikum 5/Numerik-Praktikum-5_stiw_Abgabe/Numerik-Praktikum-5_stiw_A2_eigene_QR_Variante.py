import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp

#Kronecker selber implementieren um Probleme mit Transponieren zu vermeiden
def Kronecker(w):
    res = np.zeros([len(w),len(w)],dtype=float)
    for i in range(len(w)):
        res[:,i] = w * w[i]
    return(res)

#Lorentz shape function
def lorentz_shape(x):
    s_0 = 0.2
    x_0 = 0.35
    res = 1 / (1 + ((x - x_0) /(s_0/2))**2)
    return res

#Householder transformation implementieren 
def HouseholderTransformation(w):
    H = np.eye(len(w)) - (Kronecker(w) / (0.5 * np.dot(w,w)))
    return H

#mysign Funktion definieren da bei 0 auch 1 returned werden muss
def mysign(x): # numpy sign liefert 0 für 0
    if x >= 0:
        return 1
    else:
        return -1
    
#Einheitsvektor generieren
def e(n):
    return np.array([1]+[0 for k in range(n-1)])

#Array vorgeben
A = np.array([[-1,  7, -8, -9,  6],
       [-6, -8,  0,  3,  8],
       [-4, -2,  8,  0, -2],
       [-1, -9,  4, -8,  2],
       [-3, -5, -5,  7, -4],
       [-7, -4,  7, -1,  5],
       [-9, -7,  6, -5, -8],
       [-4, -3, -5,  3, -6],
       [ 5,  7,  5, -4, -5],
       [ 4, -6, -8, -2, -5]],dtype=float)



def QR(A):
# Das ganze Array durchlaufen
    m,n = A.shape
    Anew = A.copy()
    onesmax = np.eye(m)
    Q = np.eye(m)

    for k in range(n):
        y = Anew[k:,k]
        w = y.T + mysign(y[0]) * np.linalg.norm(y) * e(len(y))
        Qk = HouseholderTransformation(w)
        Q = Q@(onesmax + np.pad(Qk,[(k,0),(k,0)]) - np.pad(np.eye(m-k),[(k,0),(k,0)]))
        Anew[k:,k:] = Qk@Anew[k:,k:]

    R = Anew[:n,:n]
    Q = Q[:n,:]

    #Matrizen zurückgeben
    return Q,R

#Q1,R1 = QR(A)
#Q2,R2 = np.linalg.qr(A)

#print(np.round(R1-R2,4))





#Aufgabe 2
imported_data = np.loadtxt("Praktikum 7\dataP7.txt")
t,b= imported_data[:,0],imported_data[:,1]
t = t-80000
t = t/1000

#---------------------Ausgleichsrechnung--------------------------------------------
n = 4
A = np.array(np.zeros([n+1,len(t)]))

A[0,]= 1.
for o in range(1,n):
    A[o,0:] = np.power(t[:], (4-o))
A[n,] = lorentz_shape(t[:])

A = A.T

#QR-Zerlegung von Numpy
Qdata, Rdata = np.linalg.qr(A) 
print('Condition of R = ',np.linalg.cond(A))

#B-Vektor errechnen, lösen und plotten
bsolve = Qdata.T@b

x = sp.linalg.solve_triangular(Rdata,bsolve,unit_diagonal=False)
print(x)


plt.plot(t,b,label='Messdaten')
plt.plot(t,x[0]+x[3]*t+x[2]*t**2+x[1]*t**3+x[n]*lorentz_shape(t),label='Berechnete Daten')
plt.plot(t,x[0]+x[3]*t+x[2]*t**2+x[1]*t**3,label='Polynom 3. Ordnung')
plt.plot(t,b-(x[0]+x[3]*t+x[2]*t**2+x[1]*t**3),label='Subtrahierte Lorentz-Kurve')
plt.plot(t,x[n]*lorentz_shape(t),label='Approximierte Lorentz-Kurve')
plt.grid()
plt.legend()
plt.title('Approximation mit QR-Zerlegung von NumPy')
plt.show()

#Eigene QR-Zerlegung
Qdata, Rdata = QR(A) 
Qdata = Qdata.T
print('Condition of R = ',np.linalg.cond(Qdata@Rdata))

#B-Vektor errechnen, lösen und plotten
bsolve = Qdata.T@b

x = sp.linalg.solve_triangular(Rdata,bsolve,unit_diagonal=False)
print(x)


plt.plot(t,b,label='Messdaten')
plt.plot(t,x[0]+x[3]*t+x[2]*t**2+x[1]*t**3+x[n]*lorentz_shape(t),label='Berechnete Daten')
plt.plot(t,x[0]+x[3]*t+x[2]*t**2+x[1]*t**3,label='Polynom 3. Ordnung')
plt.plot(t,b-(x[0]+x[3]*t+x[2]*t**2+x[1]*t**3),label='Subtrahierte Lorentz-Kurve')
plt.plot(t,x[n]*lorentz_shape(t),label='Approximierte Lorentz-Kurve')
plt.grid()
plt.legend()
plt.title('Approximation mit eigener QR-Zerlegung')
plt.show()







