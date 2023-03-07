# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""---------Es gibt noch einen fehler bei aufgabe 3 aus ich versuche einen egenen Algorythmuss aus"""
import numpy as np
def step(A,X):
    x = X
    a = A
    n = len(x)
    assert (len(A) == n), "dimension error"
    l = a[:,0]/a[0,0]
    l = np.delete(l,0)
    for k in range(1,n):
        a[k,:] -= l[k-1]*a[0,:]
        x[k] -= l[k - 1] * x[0]
    return a,x,l
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
def forward_s(l,x):
    l=l
    x=x
    b = np.array(np.zeros([len(x), len(x)]))
    for i in range(len(x)):
        xj = 0
        for j in range(i):
            xj += l[i,j]*x[j]*-1
        b[i]=(l[i,i]+xj)/l[i,i]
    print("b:", b, "\n")
    return b

"""----------------------------------------------------------------------------------"""

B = np.array(np.random.rand(4))


#print("LR:", Upper, "B: ", B, "\n")


def LU_zerlegung(A,x):
    mode = 1;           #mode 1 returns (upper,lower,x_vector)
                        #mode 0 returns (b lösungsvektor)
                        # mode 2 Ivan test
    lower = np.array(np.zeros([len(A), len(A)]))
    upper = A
    x_vector = x
    for k in range(len(x_vector)-1):
        H = matrix_resize([k, k], upper)
        #print("H",x,":", H, "\n")
        H,G,lower[k+1:,k] = step(H, vector_resize([k,0],x_vector)) #temporär Fix
        lower[k,k]=1
        upper = matrix_einf([k, k], upper, H)
        x_vector = vector_einf([k,0],x_vector,G)
    lower[len(x_vector)-1, len(x_vector)-1] = 1

    print("Mein Test:\n")
    print("U@L-A:", np.linalg.norm((upper@lower)-A), "\n")
    print("Upper:", upper, "\n")
    print("Lower:",lower,"\n")
    print("x:", x, "\n")
    print("lower@upper@x:",lower@upper@x, "\n")
    if mode == 1:
        return upper,lower,x_vector
    if mode == 0:
        l = lower
        x = x_vector
        b = np.array(np.zeros([len(x)]))
        for i in range(len(x)):
            xj = 0
            for j in range(i):
                xj += l[i, j] * x[j] * -1
            b[i] = (l[i, i] + xj) / l[i, i]
        print("L@U@b=x:", lower@upper@b,"=",x, "\n")
        return b
    if mode == 2:
        ones = np.array(np.zeros([len(x_vector), len(x_vector)]))
        for i in range(len(x_vector)):
            ones[i-1,i-1] = 1
        print("ones:", ones, "\n")
        return upper,ones,x_vector



"""-------------------------FSubs-----------------------------------------"""
def fbSubsBerti(LR,b):
    assert(len(LR) == len(b)), "Dimension error"
    x = b.copy()
    n= len(b)
    for i in range(0,len(b)-1):
        x[i+1] = sum(np.multiply(LR[i+1,:i],x[:i]))/LR[i+1,i+1]
    print("mien X:", x,"\n")
    return x
"""-------------------------relativer fehler----------------------------------"""
"""-------------------------IWAN----------------------------------------------"""
import numpy.linalg as lin
import numpy.random as rnd

# random orthogonal matrix
def rndOrtho(n):
    S = rnd.rand(n,n)
    S = S - S.T
    O = lin.solve(S - np.identity(n), S + np.identity(n))
    return O

# random matrix with specified condition number
def rndCond(n, cond):
    d = np.logspace(-np.log10(cond)/2, np.log10(cond)/2,n);
    A = np.diag(d)
    U,V = rndOrtho(n), rndOrtho(n)
    return U@A@V.T
"""-------------------------Berti----------------------------------------------"""
N = 100
for k in range(N):
    A = rndCond(3, 1e14)
def relError(A):
    A = A.copy()
    n = len(A)
    b = np.random.rand(n)
    x = linsolve(A, b)
    b1 = np.linalg.inv(A) @ x
    print("orginal rhs: ",rhs,"\n")
    print("fehler rhs: ", rhs1, "\n")
    return rhs
v =relError(A)

"""---------------------Test von Iwan----------------------------------------"""
n = 3  # Dimension der Koeffizientenmatrix
for k in range(1000):  # Testläufe
    LR = np.array(np.random.rand(n, n))  # zufällige Matrix LR
    rhs = np.array(np.random.rand(n))  # zufällige rechte Seite des LGS
    x = fbSubsBerti(LR, rhs)  # Aufruf der eigenen Funktion

    L, R = np.tril(LR, -1) + np.identity(n), np.triu(LR)  # L und R extrahieren
    print("Iwans result:\n")
    print("rhs-ivan:", rhs, "\n")
    print("R-ivan:", R, "\n")
    print("L-ivan:", L, "\n")
    #print("L@R@x:", L@R@x, "\n")
    print("result:", L @ R @ x, "\n")
    print("Test number:", k, "\n")
    assert (np.linalg.norm(rhs - L @ R @ x) < 1e-10)  # Test, mit numerischer Toleranz


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
