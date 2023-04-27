import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

l1 = 2.
l2 = 1.

def F(phi1,phi2):
    v = np.array([l1 * np.cos(phi1) + l2 * np.cos(phi2), l1 * np.sin(phi1) + l2 * np.sin(phi2)],dtype=float)
    return v

def F_strich(phi1,phi2):
    v_strich = np.array([-l1 * np.sin(phi1),l2 * np.cos(phi2)], dtype=float)
    return v_strich

def F_inner(phi1):
    v = np.array([l1 * np.cos(phi1),l1 * np.sin(phi1)],dtype=float)
    return v


def Taylor_F_varMax(phi1_0,phi2_0,x,y,iterations):
    maxIt = iterations
    i = 0
    x_v = np.array([x,y],dtype=float)
    phi_res = np.array([phi1_0,phi2_0],dtype=float)
    while ((i < maxIt) and (np.linalg.norm(F(*phi_res) - x_v) > 1e-14)):
        b = F(*phi_res) - x_v
        #print(i," : ",b)
        A = np.matrix([[F_strich(*phi_res)[0],0],[0,F_strich(*phi_res)[1]]],dtype=float)
        d_phi = sp.linalg.solve_triangular(A,b,unit_diagonal=False)
        phi_res = phi_res - d_phi
        i += 1
        phi_res = np.mod(phi_res,[2*np.pi,2*np.pi])

#    if (phi_res[1] > np.pi) :
#        phi_res[1] -= (2*np.pi)
#    if (phi_res[0] > np.pi) :
#        phi_res[0] -= (2*np.pi)

    return(phi_res)

print(F(2.61799,2.0944))
plt.plot(*F(2*0.5236+np.pi/4,2.0944),'o')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show

#Berechnen eines einzelnen Punktes
point = [-2.,0.]
phi_start = [np.pi,np.pi/2]
phi_calc = Taylor_F_varMax(*phi_start,*point,1000)
print("Phi_calc: ", phi_calc)

p1 = F_inner(phi_calc[0])
print("Inner point", p1)
p2 = F(*phi_calc)
print("Outer point", p2)

line1x = np.linspace(0,p1[0])
line2x = np.linspace(p1[0],p2[0])
line1y = np.linspace(0,p1[1])
line2y = np.linspace(p1[1],p2[1])

plt.plot(line1x,line1y,line2x,line2y,point[0],point[1])
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.grid()
plt.show()

#Abfahren einer Trajektorie
"""
t = np.linspace(0,4,num=401)
d = np.divide(1,np.sqrt(17))* np.array([1,-4],dtype=float)
p = np.array([-2,1],dtype=float)
p = np.tile(p,[len(t),1])
bahn = p + (np.array([t*d[0],t*d[1]])).T
bahn_p_innen = np.zeros([401,2])
bahn_p = np.zeros([401,2])
phi_input = [np.pi-0.1,np.pi-0.2]
plt.ion()
figure, ax = plt.subplots(figsize=(8,8))
ax.set_ylim(-3,3)
ax.set_xlim(-3,3)
data1, = ax.plot(bahn[0,0],bahn[0,1],'ro')
data2, = ax.plot(bahn_p[0,0],bahn_p[0,1],'g')
data3, = ax.plot(bahn_p_innen[0,0],bahn_p_innen[0,1],'b')
bahn_plot, = ax.plot(bahn[:,0],bahn[:,1],'r')
midpoint, = ax.plot(0.,0.,'ko')
plt.title("Animation Test")
for i in range(len(bahn[:,0])):
    phi_input = Taylor_F(phi_input[0],phi_input[1],bahn[i,0],bahn[i,1])
    bahn_p_innen[i,0],bahn_p_innen[i,1] = F_inner(phi_input[0])
    bahn_p[i,0],bahn_p[i,1] = F(*phi_input)
    data1.set_xdata(bahn[i,0])
    data1.set_ydata(bahn[i,1])
    data2.set_xdata([bahn_p_innen[i,0],bahn_p[i,0]])
    data2.set_ydata([bahn_p_innen[i,1],bahn_p[i,1]])
    data3.set_xdata([0,bahn_p_innen[i,0]])
    data3.set_ydata([0,bahn_p_innen[i,1]])
    figure.canvas.draw()
#    figure.savefig("figure" + str(i) + ".png" )
    figure.canvas.flush_events()

#plt.plot(bahn[:,0],bahn[:,1],'r')
#plt.plot(bahn_p_innen[0:,0],bahn_p_innen[:,1],'g.')
#plt.plot(bahn_p[:,0],bahn_p[:,1],'b.')
plt.grid()
plt.ylim(-3,3)
plt.xlim(-3,3)
plt.show()
"""

#Konvergenzverhalten
"""
logsp = [1,5,10,50,100,500,1000,5000,10000,50000,100000,500000]  #mit np.logspace hat es nicht funktioniert?
residuum = np.zeros(len(logsp))
for i in range(len(logsp)):
    print(logsp[i])
    residuum[i] = np.linalg.norm(Taylor_F_varMax(0.1,0.1,3.,0.,logsp[i]))

print(residuum)
plt.plot(logsp,residuum)
plt.xscale("log")
plt.yscale("log")
plt.show()
"""





