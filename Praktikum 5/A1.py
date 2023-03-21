import numpy as np
from numpy.linalg import norm

#Kronecker selber implementieren um Probleme mit Transponieren zu vermeiden
def Kronecker(w):
    res = np.zeros([len(w),len(w)],dtype=float)
    for i in range(len(w)):
        res[:,i] = w * w[i]
    return(res)

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
m,n = A.shape

# Das ganze Array durchlaufen
Anew = A.copy()
onesmax = np.eye(m)
Q = np.eye(m)

for k in range(n):
    print('Spalte '+str(k+1))
    y = Anew[k:,k]
    w = y.T + mysign(y[0]) * np.linalg.norm(y) * e(len(y))
    Qk = HouseholderTransformation(w)
    Q = Q@(onesmax + np.pad(Qk,[(k,0),(k,0)]) - np.pad(np.eye(m-k),[(k,0),(k,0)]))
    Anew[k:,k:] = Qk@Anew[k:,k:]

#R = Anew[:n,:n]
#Q = Q[:n,:]
R = Anew
print(np.round(R,4))
print(Q)

print(np.round(Q@R - A,4))

