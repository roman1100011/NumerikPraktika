import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

# ------------ Implementation des Modellproblems --------------

g = 9.81
v0 = 1
k = 0.5 * 0.45 * 1.184 * 0.1**2 * np.pi
m = 0.05
ks = 0.05

def f(t,x):
    x0 = x[0]
    x1 = x[1]
    return np.array([x1,g-(k/m)*(x1+v0)*np.abs(x1+v0) - x0*ks/m],dtype=float)

def df(t,x):
    x0 = x[0]
    x1 = x[1]
    return np.array([[0,1],[-ks/m,-2*k/m*np.abs(x1+v0)]],dtype=float)

# ------------ Matrizen für die Verfahren definieren -------------------------------
aRK4 = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]],dtype=float)
bRK4 = np.array([1/6,1/3,1/3,1/6],dtype=float)
cRK4 = np.array([0,0.5,0.5,1],dtype=float)



# --------------- implizites Verfahren ---------------------
def implizitesVerfahren(a,b,c,xEnd,h,y0,f,df):
    x = np.linspace(0,xEnd,num=int(xEnd/h))
    y = np.outer(np.ones_like(x),y0)
    r = np.zeros((len(y0),len(a)),dtype=float)

    def Galt(r,xk,yk,a,b,c):                                               ## um die Verfahrensfunktion danach einfach ableiten zu können wird das ganze nicht Matrix-, sondern Vektorwertig gerechnet
        retArray = np.ones_like(r)
        print(r[:,1])
        for i in range(np.shape(r)[1]):
            retArray[:,i] = f(xk+c[i]*h,yk+h*(np.sum(np.outer(a[i,:],r[:,i]),axis=0))) - r[:,i]
        return retArray
    
    def G(r,xk,yk,a,b,c):                                               ## um die Verfahrensfunktion danach einfach ableiten zu können wird das ganze nicht Matrix-, sondern Vektorwertig gerechnet
        r = np.ravel(r,order='F')
        retArray = np.ones_like(r)
        for i in range(len(r)//2):
            retArray[i], retArray[i+1] = (yk + h*f(xk,np.array([r[0::2],r[1::2]])@a[i,:]))
        return  retArray - r
    
    def dGalt(r,xk,yk,a,b,c):
        retArray = np.ones((2,2,2))
        print(retArray)
        for i in range(np.shape(r)[1]):
            retArray[:,:,i] = df(xk+c[i]*h,yk+h*(np.sum(np.outer(a[i,:],r[:,i]),axis=0))) - np.eye(np.shape(r)[0])

    def dG(r,xk,yk,a,b,c):
        r = np.ravel(r,order='F')
        retArray = np.ones((len(r),len(r)))
        for i in range(len(r)//2):
            for j in range(len(r)//2):
                retArray[i*2:i*2+2,j*2:j*2+2] = df(xk,h*(np.array([r[0::2],r[1::2]])@a[i,:]))*h
        return  retArray - np.eye(len(r))

    def GaussNewton(xk,yk,r0,F,dF,a_,b_,c,maxIter=100,tol=1e-12,damped=False,delta_min=0.001,maxDampingIter=10):  ##leicht abgewandelte Form des GaussNewton verfahrens aus dem Vorletzten Praktikum
        r_input = r0.copy()                                   ## copy Start values into param, to not alter anything outside of fct
        for k in range(maxIter):                            ## Iterates for a max of int given in maxIter
            A = dF(r_input,xk,yk,a_,b_,c)                          ## generates dF matrix with paramters given and x of data
            b = F(r_input,xk,yk,a_,b_,c)                          ## generates F matrix with paramters given and x and y of data
            q,r = np.linalg.qr(A)                           ## q-r-Deconstruction to get a solvable system
            delta = 1                                       ## sets starting dampening factor to 1
            diter = 0                                       ## resets number of dampening iterations
            s = solve_triangular(r,q.T@b)                   ## solves system to get correction vector s
            if (damped==True):                              ## only applies if dampening is activated
                while (np.linalg.norm(F(r_input - delta * s)) > np.linalg.norm(F(r_input)) and (delta > delta_min) and (diter < maxDampingIter)): ##ensures that corrected parameters dont lead to overcorrection and all logical conditions apply (iterations,dampening)
                    delta /= 2                              ## corrects dampening factor
                    diter += 1                              ## increases number of iterations
                r_input -= delta*s                          ## caclulates dampened correction
            else:
                r_input -= s.reshape((len(y0),len(b_)))                                  ## calculates correction
            return r_input
        
    for i in range(len(x)-1):
        y[i+1] = y[i] + h * np.dot(GaussNewton(x[i],y[i],r,G,dG,a,b,c),b)

    return x,y

# --------------- Ausführung und Plot ------------------
xRK4, yRK4 = implizitesVerfahren(aRK4,bRK4,cRK4,10,0.001,np.array([0,1]),f,df)

plt.plot(xRK4,yRK4[:,0],label='0')
plt.plot(xRK4,yRK4[:,1],label='1')
plt.legend()
plt.show()