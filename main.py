# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
def step(A,B):


    b = B
    a = A
    n = len(b)
    assert (len(A) == n), "dimension error"
    m = a[:,0]/a[0,0]
    m = np.delete(m,0)
    for k in range(1,n):
        a[k,:] -= m[k-1]*a[0,:]
        b[k] -= m[k - 1] * b[0]
    return a,b
def matrix_einf(corner, bigMatrix ,smalMatrix):
    #Check dimensions
    assert (len(corner) == 2), "array für den Eckpunkt hat die falsche Dimensin"
    assert (len(bigMatrix) >= len(smalMatrix)), "array für den Eckpunkt hat die falsche Dimensin"
    bMat = bigMatrix
    sMat = smalMatrix
    bMat[corner[0]:,corner[1]:] = sMat[0:,0:]
    return bMat
def vector_einf(start, bigVector ,smalVector):
    #Check dimensions
    assert (len(start) == 2), "array für den Eckpunkt hat die falsche Dimensin"
    assert (len(bigVector) >= len(smalVector)), "array für den Eckpunkt hat die falsche Dimensin"
    bVec = bigVector
    sVec = smalVector
    bVec[start[0]:] = sVec[0:]
    return bVec
def matrix_resize(new_pivot, orgiMatrix): #gibt eine kleinere Matrix zurück
    return orgiMatrix[new_pivot[0]:,new_pivot[1]:]
def vector_resize(new_pivot, orgiMatrix): #gibt eine kleinere Matrix zurück
    return orgiMatrix[new_pivot[0]:]

"""----------------------------------------------------------------------------------"""

LR = np.array(np.random.rand(3,3))
B = np.array(np.random.rand(3))


print("LR:",LR,"B: ",B,"\n")



for x in range(len(B)-1):
    H = matrix_resize([x, x], LR)
    #print("H",x,":", H, "\n")
    H,G = step(H, vector_resize([x,0],B)) #temporär Fix
    LR = matrix_einf([x, x], LR, H)
    B = vector_einf([x,0],B,G)

LR = LR
print("LR:",LR,"B: ",B,"\n")




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
