import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

# ------------ Implementation des ersten Modells und dessen analyitischer Lösung --------------
def ya1(x):
    return np.exp(-4*x)

def f1(x,y):
    return -4*y

def df1(x,y):
    return -4*np.ones_like(y)

# ------------ Implementation des zweiten Modells und dessen analyitischer Lösung --------------
def ya2(x):
    return - np.sqrt(16 - ((2*(x**3))/3))

def f2(x,y):
    return -x**2/y

def df2(x,y):
    return x**2/y**2

# ------------ Implizites Verfahren für irgendeine Matrix a und vektoren b und c
def implizitesVerfahren(a_,b_,c_,xend,h,y0,f,df):
    x = np.linspace(0,xend,num=int(xend/h))
    y = np.ones_like(x) * y0
    r0 = np.zeros(len(a_))
    # Partielle Ableitung nach r der Verfahrensfunktion
    def dG(r, data, param):
        xk, yk = data
        a_,b_,c_ = param
        a_ = np.pad(a_,((0,1),(0,1)))                                         #damit die Funktion auch für arrays der grösse 1 funktioniert
        r = np.append(r,0)  
        test =  np.dot(df(xk+c_*h,yk+h*(0.5*r)),(h*np.diag(a_))) - np.eye(len(r))                                               #damit die Funktion auch für arrays der grösse 1 funktioniert
        return test[0:-1,0:-1]

    # Verfahrensfunktion G(r) = r-blabla = 0
    def G(r, data, param):
        xk, yk = data
        a_,b_,c_ = param
        a_ = np.pad(a_,((0,1),(0,1)))                                         #damit die Funktion auch für arrays der grösse 1 funktioniert
        r = np.append(r,0)                                                  #damit die Funktion auch für arrays der grösse 1 funktioniert
        return f(xk+c_*h,(yk+h*np.apply_along_axis(np.dot,1,a_,r))[0:-1]) - r[0:-1]
    
    def GaussNewton(data,x0,F,dF,mat,maxIter=100,tol=1e-12,damped=False,delta_min=0.001,maxDampingIter=10):
        param = x0.copy()                                   ## copy Start values into param, to not alter anything outside of fct
        for k in range(maxIter):                            ## Iterates for a max of int given in maxIter
            A = dF(param,data,mat)                              ## generates dF matrix with paramters given and x of data
            b = F(param,data,mat)                               ## generates F matrix with paramters given and x and y of data
            q,r = np.linalg.qr(A)                           ## q-r-Deconstruction to get a solvable system
            delta = 1                                       ## sets starting dampening factor to 1
            diter = 0                                       ## resets number of dampening iterations
            s = solve_triangular(r,q.T@b)                   ## solves system to get correction vector s
            if (damped==True):                              ## only applies if dampening is activated
                while (np.linalg.norm(F(param - delta * s)) > np.linalg.norm(F(param)) and (delta > delta_min) and (diter < maxDampingIter)): ##ensures that corrected parameters dont lead to overcorrection and all logical conditions apply (iterations,dampening)
                    delta /= 2                              ## corrects dampening factor
                    diter += 1                              ## increases number of iterations
                param -= delta*s                            ## caclulates dampened correction
            else:
                param -= s                                  ## calculates correction
            return param

    mat = a_,b_,c_
    for i in range(len(x)-1):
        data_ = np.array([x[i],y[i]],dtype=float)
        y[i+1] = y[i] + h * np.dot(GaussNewton(data_,r0,G,dG,mat),b_)

    return x,y
    

# -------------- erstes Modellproblem -----------------------

a = np.array([[0.5]])
b = np.array([1])
c = np.array([0.5])
x_plot1_mittelpunkt,y_plot1_mittelpunkt = implizitesVerfahren(a,b,c,1.,0.1,1.,f1,df1)

a = np.array([[0,0],[0.5,0.5]])
b = np.array([0.5,0.5])
c = np.array([0,1])
x_plot1_trapez,y_plot1_trapez = implizitesVerfahren(a,b,c,1.,0.1,1.,f1,df1)

x_anal = np.linspace(0,1,100)
y_anal = ya1(x_anal)

plt.plot(x_plot1_mittelpunkt,y_plot1_mittelpunkt, label = 'implizite Mittelpunktregel')
plt.plot(x_plot1_trapez,y_plot1_trapez, label = 'implizite Trapezregel')
plt.plot(x_anal,y_anal, label = 'Analytisch')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

