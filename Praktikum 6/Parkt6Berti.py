import numpy as np
import scipy as sy
import matplotlib as plt

class Robot:
    def __int__(self,Phi1,Phi2,L1,L2):
        self.Phi1 =  np.array([Phi1]) #Winkel des armes, welcher an dem Uhrspung befestigt ist
        self.Phi2 = np.array([Phi2]) #Winkel des armes, welcher an dem ersten Arm befestigt ist und dessen endpunkt der Greifer ist (winkel ist zwischen der verlängerung des erstem arm und dem 2 ten arm
        self.L1 = np.array([L1])#Länge des armes, welcher an dem Uhrspung befestigt ist
        self.L2 = np.array([L2]) #Länge des armes, welcher an dem ersten Arm befestigt ist und dessen endpunkt der Greifer ist
        self.pos_c = np.array([self.L1*np.cos(self.Phi1)+self.L2*np.cos(self.Phi1+self.Phi2),
                      self.L1*np.sin(self.Phi1)+self.L2*np.sin(self.Phi1+self.Phi2)]) #aktuelle position des Greifers in Kartesischen koordinaten
        self.pos_p = np.array([np.sqrt(self.L1**2+self.L2**2-2*self.L1*self.L2)*np.cos(np.pi-self.Phi2),
                      self.Phi1+self.Phi2])#aktuelle position des Greifers in Polar koordinaten
        #self.jako = np.array([[2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.sin(self.Phi2+self.Phi1),
        #-2*self.L1*self.L2*np.sin(np.pi-self.Phi2)*np.cos(self.Phi2+self.Phi1)+2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.sin(self.Phi2+self.Phi1)],
        #[-2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.sin(self.Phi2+self.Phi1),
        #        -2*self.L1*self.L2*np.sin(np.pi-self.Phi2)*np.sin(self.Phi2+self.Phi1)-2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.cos(self.Phi2+self.Phi1)]])

    def cart(self): #funktion zur bestimmung der Kartesischen koordinaten des greifers aus den 2 winkeln und den Längen der Arme
        v = np.array([self.L1*np.cos(self.Phi1)+self.L2*np.cos(self.Phi1+self.Phi2),
                      self.L1*np.sin(self.Phi1)+self.L2*np.sin(self.Phi1+self.Phi2)])
        return v
    def polar(self): #funktion zur bestimmung der Kartesischen koordinaten des greifers aus den 2 winkeln und den Längen der Arme
        v = np.array([np.sqrt(self.L1**2+self.L2**2-2*self.L1*self.L2)*np.cos(np.pi-self.Phi2),
                      self.Phi1+self.Phi2])
        return v

    def get_angeles_ofPoint(self,x,y):
        if self.L1+self.L2 < np.sqrt(np.power(x,2)+np.power(y,2)):
            return 0
        xn =np.array([self.Phi1,self.Phi2]) #init an aktueller position
        xf = np.array([x,y])#fixpunkt
        jakob = np.array([[-self.L1*np.sin(self.Phi1)-self.L2*np.sin(self.Phi1+self.Phi2),-self.L2*np.sin(self.Phi1+self.Phi2)],
                          [self.L1*np.cos(self.Phi1)+self.L2*np.cos(self.Phi1+self.Phi2),self.L2*np.cos(self.Phi1+self.Phi2)]])
        # Korrekturfaktor delta Berechnen

        delt = sy.linalg.solve(jakob,self.pos_c-xf)
        xn -= delt
        self.Phi1 = xn[0]
        self.Phi2 = xn[1]
        return self
    def Newton(v):
        pos = self
        v = v.copy
        while (np.linalg.norm(v - pos) >= 1e-0):
            self = self.get_angeles_ofPoint(v[0], v[1])
        return self







