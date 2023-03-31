import numpy as np
import matplotlib.pyplot as plt

phitestHalf = np.linspace(0,1)
phitestFull = np.linspace(0,2)

l1 = 2.
l2 = 1.

def fAlt(phi1,phi2):
    v = [np.sqrt(l1**2+l2**2-2*l1*l2*np.cos(np.pi-phi2))*np.cos(phi1+phi2),np.sqrt(l1**2+l2**2-2*l1*l2*np.cos(np.pi-phi2))*np.sin(phi1+phi2)]
    return v

def F(phi1,phi2):
    v = np.array([l1 * np.cos(phi1) + l2 * np.cos(phi1+phi2), l1 * np.sin(phi1) + l2 * np.sin(phi1+phi2)])
    return v

def F_strich(phi1,phi2):
    v_strich = np.array([-l1 * np.sin(phi1) - np.sin(phi1+phi2), l1 * np.cos(phi1) + l2 * np.cos(phi1+phi2)])
    return v_strich

def Taylor_F(phi1_0,phi2_0,x,y):
    maxIt = 20
    i = 0
    x_v = np.array([x,y],dtype=float)
    phi_res = np.array([phi1_0,phi2_0],dtype=float)
    while np.linalg.norm((F(*phi_res)-x_v))>1e-12 and (i < maxIt):
        phi_res = - np.divide((F(*phi_res) - x_v), F_strich(*phi_res)) + phi_res
        i += 1
    return(phi_res)

print(Taylor_F(0.1,0.3,3.,0.))

