# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from scipy.linalg import solve_triangular
plt.style.use('_mpl-gallery')

#---------------------------------------Data importieren--------------------------------
data = np.genfromtxt("data.txt")
# plot


#---------------------------------------Achsen separieren--------------------------------
x=data[:,0]
y=data[:,1]
#--------------------------------------Abtastrate---------------------------------------------
Ts=np.amax(x)/(len(x)-1)
#--------------------------------------System---------------------------------------------
n = 1+5*2 #anzahl Schwingungen
A = np.array(np.zeros([n,len(x)]))

A[0,]= 0.5
for o in range(1,n):
    if( np.mod(o,2)==1):
        A[o,1:] = np.cos(o*x[1:])
    if (np.mod(o,2) == 0):
        A[o,1:] = np.sin(o * x[1:])

#-----A^TA ausrechnen---------------------
A_dig = A@ A.T
b_dig = y @ A.T
#Cholesky zerlegung
L = np.linalg.cholesky(A_dig)
#     Ly = A^T *b nach y auflösen
los = solve_triangular(L,b_dig)/len(b_dig)



fit = np.array(np.zeros(len(x)))
fit[:] = (A.T @ los)




#--------------------------------------Plot---------------------------------------------
fig, ax = plt.subplots()
ax.plot(x,y,x,fit)
ax.set(xlim=(0,np.amax(x)), xticks=np.arange(0, np.amax(x)),
       ylim=(np.amin(fit)-0.1, np.amax(fit)+0.1), yticks=np.arange(np.amin(fit)-0.1, np.amax(fit)+0.1))
ax.set_xlabel("time [s]")
ax.set_xlabel("Amplitude [V]")

plt.show()