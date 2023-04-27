import numpy as np
from numpy.linalg import norm


# Kronecker selber implementieren um Probleme mit Transponieren zu vermeiden
def Kronecker(w):
    res = np.zeros([len(w), len(w)], dtype=float)
    for i in range(len(w)):
        res[:, i] = w * w[i]
    return (res)


# Householder transformation implementieren
def HouseholderTransformation(w):
    H = np.eye(len(w)) - (Kronecker(w) / (0.5 * np.dot(w, w)))
    return H


# mysign Funktion definieren da bei 0 auch 1 returned werden muss
def mysign(x):  # numpy sign liefert 0 für 0
    if x >= 0:
        return 1
    else:
        return -1


# Einheitsvektor generieren
def e(n):
    return np.array([1] + [0 for k in range(n - 1)])


# Array vorgeben
imported_data = np.loadtxt("Praktikum 7\dataP7.txt")
x,y= imported_data[:,0],imported_data[:,1]


#---------------------Ausgleichsrechnung--------------------------------------------
n = 4
A = np.array(np.zeros([n,len(x)]))
A[0,]= 1
for o in range(1,n):
    A[o,0:] = np.power(x[:], (4-o))

m, n = A.shape

# Das ganze Array durchlaufen
Anew = A.copy()
onesmax = np.eye(m)
Q = np.eye(m)

for k in range(n):
    print('Spalte ' + str(k + 1))
    y = Anew[k:, k]
    w = y.T + mysign(y[0]) * np.linalg.norm(y) * e(len(y))
    Qk = HouseholderTransformation(w)
    Q = Q @ (onesmax + np.pad(Qk, [(k, 0), (k, 0)]) - np.pad(np.eye(m - k), [(k, 0), (k, 0)]))
    Anew[k:, k:] = Qk @ Anew[k:, k:]

# R = Anew[:n,:n]
# Q = Q[:n,:]
R = Anew
print(np.round(R, 4))
print(Q)

print(np.round(Q @ R - A, 4))