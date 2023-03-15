# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
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
A = np.array(np.zeros([n,len(r)]))
#symbolische def (provisorisch
a_0,a_1,a_2,a_3,a_4,a_5= 0,0,0,0,0,0
b_1,b_2,b_3,b_4,b_5 = 0,0,0,0,0
r = np.array([[a_0,0,0,0,0],[a_1,a_2,a_3,a_4,a_5],[b_1,b_2,b_3,b_4,b_5]])
r = r.T
A[0,:]= 0.5
for o in range(n):
    if( np.mod(o,2)<1):
        A[o,:] = [np.cos(o*x[:]),]


#--------------------------------------Plot---------------------------------------------
fig, ax = plt.subplots()
ax.plot(x,y)
ax.set(xlim=(0,np.amax(x)), xticks=np.arange(0, np.amax(x)),
       ylim=(np.amin(y)-0.1, np.amax(y)+0.1), yticks=np.arange(np.amin(y)-0.1, np.amax(y)+0.1))
ax.set_xlabel("time [s]")
ax.set_xlabel("Amplitude [V]")

plt.show()