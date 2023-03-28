import numpy as np
import scipy as sy
import matplotlib as plt

class Robot:
    def __int__(self,Phi1,Phi2,L1,L2):
        self.Phi1 =  np.array([Phi1]) #Winkel des armes, welcher an dem Uhrspung befestigt ist
        self.Phi2 = np.array([Phi2]) #Winkel des armes, welcher an dem ersten Arm befestigt ist und dessen endpunkt der Greifer ist (winkel ist zwischen der verlängerung des erstem arm und dem 2 ten arm
        self.L1 = np.array([L1])#Länge des armes, welcher an dem Uhrspung befestigt ist
        self.L2 = np.array([L2]) #Länge des armes, welcher an dem ersten Arm befestigt ist und dessen endpunkt der Greifer ist
        self.pos_c = self.cart(self) #aktuelle position des Greifers in Kartesischen koordinaten
        self.pos_p = self.polar(self)#aktuelle position des Greifers in Polar koordinaten
    def cart(self): #funktion zur bestimmung der Kartesischen koordinaten des greifers aus den 2 winkeln und den Längen der Arme
        v = np.array([self.L1**2+self.L2**2-2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.cos(self.Phi1+self.Phi2),
                      self.L1**2+self.L2**2-2*self.L1*self.L2*np.cos(np.pi-self.Phi2)*np.sin(self.Phi1+self.Phi2)])
        return v
    def cart(self): #funktion zur bestimmung der Kartesischen koordinaten des greifers aus den 2 winkeln und den Längen der Arme
        v = np.array([self.L1**2+self.L2**2-2*self.L1*self.L2*np.cos(np.pi-self.Phi2),
                      self.Phi1+self.Phi2])
        return v


