import numpy as np
import matplotlib.pyplot as plt

l1 = 2.
l2 = 1.

def F(phi1,phi2):
    v = np.array([l1 * np.cos(phi1) + l2 * np.cos(phi1+phi2), l1 * np.sin(phi1) + l2 * np.sin(phi1+phi2)],dtype=float)
    return v

def F_strich(phi1,phi2):
    v_strich = np.array([-l1 * np.sin(phi1) - np.sin(phi1+phi2), l1 * np.cos(phi1) + l2 * 2 * np.cos(phi1+phi2)], dtype=float)
    return v_strich

def F_inner(phi1):
    v = np.array([l1 * np.cos(phi1),l1 * np.sin(phi1)],dtype=float)
    return v

def Taylor_F(phi1_0,phi2_0,x,y):
    maxIt = 50
    i = 0
    x_v = np.array([x,y],dtype=float)
    phi_res = np.array([phi1_0,phi2_0],dtype=float)
    while np.linalg.norm((F(*phi_res)-x_v))>1e-14 and (i < maxIt):
        #print(F(*phi_res))
        phi_res = - np.divide((F(*phi_res) - x_v), F_strich(*phi_res)) + phi_res
        i += 1
        phi_res = np.mod(phi_res,[2*np.pi,2*np.pi])
    return(phi_res)

def Taylor_F_varMax(phi1_0,phi2_0,x,y,iterations):
    maxIt = iterations
    i = 0
    x_v = np.array([x,y],dtype=float)
    x_old = x_v
    phi_res = np.array([phi1_0,phi2_0],dtype=float)
    while (i < maxIt):
        x_old = F(*phi_res)
        phi_res = - np.divide((F(*phi_res) - x_v), F_strich(*phi_res)) + phi_res
        x_v = x_old
        i += 1
    phi_res = np.mod(phi_res,[2*np.pi,2*np.pi])
    return(phi_res)

point = [-1,0]
phi_start = [3,0.3]
phi_calc = Taylor_F(*phi_start,*point)
print(phi_calc)

p1 = F_inner(phi_calc[0])
print(p1)
p2 = F(*phi_calc)
print(p2)

print("Test: ",F(3.1,6))

line1x = np.linspace(0,p1[0])
line2x = np.linspace(p1[0],p2[0])
line1y = np.linspace(0,p1[1])
line2y = np.linspace(p1[1],p2[1])

plt.plot(line1x,line1y,'g')
plt.plot(line2x,line2y,'b')
plt.grid()
plt.ylim(-3,3)
plt.xlim(-3,3)
plt.show()

