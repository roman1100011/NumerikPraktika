# %% imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data = np.genfromtxt('data.txt')

print("Anzahl Messpunkte: ",len(data[:,1]))

"""
plt.plot(data[:,0],data[:,1],'.')
plt.grid()
plt.show()
"""

# 2. Wie man sieht ist die höchste Frequenz etwa eine Periode von 1.5s was einer Frequenz von 0.66Hz entspricht, die Grundfrequenz beträgt wohl etwas um die 0.2Hz. Die Samplingzeit beträgt 0.01s, die Abtastrate beträgt also 100Hz.

# 3. Die Systemmatrix ist eine (1 + 2n) * 1501 Matrix. Weil man 1 + 2n unbekannte in der gegebenen Funktion hat und 1501 Messwerte gegeben sind die jeweils im Vektor b einen Eintrag erhalten. Damit die Dimensionen stimmen muss die Matrix A also eine (1 + 2n) * 1501 Matrix sein.

# 4. Die Systemmatrix hat folgende Eigenschaften: die Erste Spalte besteht nur aus den Vorfaktoren von $a_0$, dieser ist konstant und gegeben als $1/2$. Die folgenden Spalten bestehen jeweils abwechselnd aus den Vorfaktoren von $a_k$ bzw. $b_k$. Diese sind definiert als $cos(\omega k t)$ bzw. $sin(\omega k t)$. Die gesuchte Matrix ist also die folgende: 

n = 5
w=1.0
t = data[:,0]

A = np.zeros([len(t),1+2*n],dtype=float)
A[:,0] = 0.5
for k in range(1,n+1):
    A[:,1+(k-1)*2] = np.cos(w*k*t)
    A[:,2+(k-1)*2] = np.sin(w*k*t)


# 5. Die Normalengleichung hat die Dimension 11 dh. es sind 11 Gleichungen die gelöst werden müssen.

# $$A^T * A * y = A^T * b$$

b = data[:,1]
Asolve = A.T@A
bsolve = A.T@b


L = np.linalg.cholesky(Asolve)
ysolve = sp.linalg.solve_triangular(L,bsolve,unit_diagonal=False,lower=True)
x = sp.linalg.solve_triangular(L.T,ysolve,unit_diagonal=False)


def fn(t):
    val = np.zeros(len(t))
    for k in range(n):
          val += np.cos(w*t*(k+1))*x[k*2+1] + np.sin(w*t*(k+1))*x[k*2+2]
    val += 0.5*x[0] 
    return val

plt.plot(data[:,0],fn(t),'k',label="interpolierte Daten")
plt.plot(data[:,0],data[:,1],'.',label="Messdaten")
plt.grid()
plt.legend()
plt.show()

print("a0 = ",x[0])
for i in range(n):
     print("a",i+1," = ",x[i*2+1])
     print("b",i+1," = ",x[i*2+2])



##Summe der Fehlerquadrate berechnen
sumfehl = np.sum((fn(t)-b)**2)
print(sumfehl)

# 8. Die Summe der kleinsten Fehlerquadrate beträgt $\approx 59.26$.