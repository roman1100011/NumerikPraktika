# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')
data = np.genfromtxt("data.txt")
# plot



x=data[:,0]
y=data[:,1]


fig, ax = plt.subplots()
ax.plot(x,y,linewidth=2.0)

ax.set(xlim=(0,len(x)), xticks=np.arange(0, len(x)),
       ylim=(-2, 3), yticks=np.arange(-2,3))

plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
