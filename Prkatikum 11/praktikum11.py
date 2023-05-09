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

# ------------ Implizites Verfahren für irgendeine Matrix a und vektoren b und c
def implizitesVerfahren(a_,b_,c_,xend,h,y0,f,df):
    x = np.linspace(0,xend,num=int(xend/h))
    y = np.tile(np.reshape(y0,(len(y0),1)),len(x))
    r0 = np.zeros((len(y0),len(a_)),dtype=float)
    # Partielle Ableitung nach r der Verfahrensfunktion
    def dG(r, xk, yk, param):
        a_,b_,c_ = param                                      
        test =  np.dot(df(xk+c_*h,yk+h*(0.5*r)),(h*np.diag(a_))) - np.eye(len(r))                                               #damit die Funktion auch für arrays der grösse 1 funktioniert
        return test

    # Verfahrensfunktion G(r) = r-blabla = 0
    def G(r, xk, yk, param):
        a_,b_,c_ = param
        return f(xk+c_*h,(yk+h*a_)) - r
    
    def GaussNewton(datax,datay,r0,F,dF,mat,maxIter=100,tol=1e-12,damped=False,delta_min=0.001,maxDampingIter=10):  ##leicht abgewandelte Form des GaussNewton verfahrens aus dem Vorletzten Praktikum
        param = r0.copy()                                   ## copy Start values into param, to not alter anything outside of fct
        for k in range(maxIter):                            ## Iterates for a max of int given in maxIter
            A = dF(param,datax,datay,mat)                          ## generates dF matrix with paramters given and x of data
            b = F(param,datax,datay,mat)                           ## generates F matrix with paramters given and x and y of data
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
        datax = np.array(x[i],dtype=float)
        datay = np.array(y[:,i],dtype=float)
        y[:,i+1] = y[:,i] + h * GaussNewton(datax,datay,r0,G,dG,mat)@b_

    return x,y
    
xRK4, yRK4 = implizitesVerfahren(aRK4,bRK4,cRK4,10,0.01,np.array([0,1]),f,df)

plt.plot(xRK4,yRK4)
plt.show()

""""
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
plt.title('Modellproblem 1')
plt.grid()
plt.show()

# -------------- zweites Modellproblem -----------------------

a = np.array([[0.5]])
b = np.array([1])
c = np.array([0.5])
x_plot1_mittelpunkt,y_plot1_mittelpunkt = implizitesVerfahren(a,b,c,2.,0.1,-4.,f2,df2)

a = np.array([[0,0],[0.5,0.5]])
b = np.array([0.5,0.5])
c = np.array([0,1])
x_plot1_trapez,y_plot1_trapez = implizitesVerfahren(a,b,c,2.,0.1,-4.,f2,df2)

x_anal = np.linspace(0,2,100)
y_anal = ya2(x_anal)

plt.plot(x_plot1_mittelpunkt,y_plot1_mittelpunkt, label = 'implizite Mittelpunktregel')
plt.plot(x_plot1_trapez,y_plot1_trapez, label = 'implizite Trapezregel')
plt.plot(x_anal,y_anal, label = 'Analytisch')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Modellproblem 2')
plt.grid()
plt.show()

# --------------- Konvergenzordnung bestimmen ----------------------
n = 3**np.linspace(1,8,num=8)
hs = 2/n


# für die implizite Mittelpunktregel
a = np.array([[0.5]])
b = np.array([1])
c = np.array([0.5])

errm = []

for h in hs:
    xm,ym = implizitesVerfahren(a,b,c,2.,h,-4.,f2,df2)
    errm.append(np.linalg.norm(ym[-1] - ya2(xm[-1])))

# für die implizite Trapezregel

a = np.array([[0,0],[0.5,0.5]])
b = np.array([0.5,0.5])
c = np.array([0,1])

errt = []

for h in hs:
    xt,yt = implizitesVerfahren(a,b,c,2.,h,-4.,f2,df2)
    errt.append(np.linalg.norm(yt[-1] - ya2(xt[-1])))

plt.loglog(hs,errm,marker='*',label='implizite Mittelpunktregel')
plt.loglog(hs,errt,marker='*',label='implizite Trapezregel')
plt.xlabel('h')
plt.ylabel('Fehler and er Endstelle')
plt.legend()
plt.title('Konvergenz')
plt.grid()
plt.show()
"""