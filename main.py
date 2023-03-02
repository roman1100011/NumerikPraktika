# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
def step(A,b):

    assert (len(A)==len(b)), "dimension error"

    a = A
    n = len(b)
    m = 0.0
    m = a[:,0]/a[0,0]
    m = np.delete(m,0)
    for k in range(1,n):
        a[k,:] -= m[k-1]*a[0,:]
    return a
def matrix_einf(corner, bigMatrix ,smalMatrix):
    #Check dimensions
    assert (len(corner) == 2), "array für den Eckpunkt hat die falsche Dimensin"
    assert (len(bigMatrix) >= len(smalMatrix)), "array für den Eckpunkt hat die falsche Dimensin"
    bMat = bigMatrix
    sMat = smalMatrix
    bMat[corner[0]:,corner[1]:] = sMat[0:,0:]
    return bMat
def matrix_resize(new_pivot, orgiMatrix): #gibt eine kleinere Matrix zurück
    return orgiMatrix[new_pivot[0]:,new_pivot[1]:]
"""----------------------------------------------------------------------------------"""

LR = np.array(np.random.rand(100,100))
B = np.array(np.random.rand(100))






for x in range(len(B)-1):
    H = matrix_resize([x, x], LR)
    #print("H",x,":", H, "\n")
    H = step(H, B[0:(len(B)-x)]) #temporär Fix
    LR = matrix_einf([x, x], LR, H)

LR = LR
print("LR:",LR,"\n")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
