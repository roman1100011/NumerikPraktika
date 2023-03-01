# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
LR = np.array(np.random.rand(3,3))
B = np.array(np.random.rand(3))
print("LR:",LR,"\n")

def forwardEl(A,b):
    a = A
    n = len(b)
    l= A#np.zeros((n,n))
    ln = np.empty([n,n])
    m = 0.0
    xs= 0
    m = l[:,xs]/l[0,0]
    m = np.delete(m,0)
    for k in range(1,n):
        l[k,:] -= m[k-1]*l[0,:]
    print(l)
    return l

forwardEl(LR,B)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
