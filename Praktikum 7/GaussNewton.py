## imports 
import numpy as np
from scipy.linalg import solve_triangular

## -------------------------------
## Applies Gauss Newton Algorithm with or without dampening
## input: data,2*n array; x0, list of start parameters; F(parameters,data), main Function; dF(parameters,data), differentiated function; maxIter, int for max number of iterations, tol, accuracy of final result wanted; damped, true/false; delta_min, smallest dampening coefficient; maxDampingIter, max number of iterations for dampening coefficient
## output: calculated list of parameters param
## -------------------------------

def GaussNewton(data,x0,F,dF,maxIter=100,tol=1e-12,damped=False,delta_min=0.001,maxDampingIter=10):
    param = x0.copy()                                   ## copy Start values into param, to not alter anything outside of fct
    for k in range(maxIter):                            ## Iterates for a max of int given in maxIter
        A = dF(param,data)                              ## generates dF matrix with paramters given and x of data
        b = F(param,data)                               ## generates F matrix with paramters given and x and y of data
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
    
