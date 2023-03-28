import numpy as np
import scipy as sy
import matplotlib as plt

class position:
    def __int__(self,Phi1,Phi2,L1,L2):
        self.Phi1 =  np.array([Phi1])
        self.Phi2 = np.array([Phi2])
        self.L1 = np.array([L1])
        self.L2 = np.array([L2])
        self.pos_c = self.cart(self)
        self.pos_p = self.polar(self)
    def cart(self):
        v = np.array([self.L1**2+self.L2**2-2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.cos(self.Phi1+self.Phi2),
                      self.L1**2+self.L2**2-2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.sin(self.Phi1+self.Phi2)])
        return v

def positon_from_angles(Phi1, Phi2):
    phi1 = Phi1.copy