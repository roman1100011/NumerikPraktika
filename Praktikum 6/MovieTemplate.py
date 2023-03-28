# Python Template fuer Movie
import numpy as np
import matplotlib as plt
plt.use("Agg")
from matplotlib.animation import FFMpegWriter
import Parkt6Berti as ber

c = np.array([150,0])
ro1 = ber.Robot
ro1.L1= 150
ro1.L2 = 20
ro1.Phi1 = 1.0
ro1.Phi2 = 0.0
ro1 = ro1.Newton(c)
print(ro1.cart())
metadata = dict(title='Trajektorie', artist='Your Name',
                comment='Movie')
writer = FFMpegWriter(fps=60, metadata=metadata)

fig = plt.figure(figsize=(6,6))
l1, = plt.plot([], [])
l2, = plt.plot([], [],'o')

# p: Funktion zur Berechnung der Punkte auf der Trajektorie
# ti: wird auch für die Berechnung der Winkel benutzt
ti = np.linspace(0,4,int(4/.01+1))
plt.plot(*np.array([p(tii) for tii in ti]).T)

plt.xlim(-3,3)
plt.ylim(-3,3)
plt.gca().set_aspect(1)
plt.gca().add_patch(mpatches.Circle((0,0), 2-1,alpha=0.1))
plt.gca().add_patch(mpatches.Circle((0,0), 2+1,alpha=0.1))
plt.grid()
plt.xlabel('x')
plt.ylabel('y')

# si: Liste der Winkel fuer die Trajektorie
# PG: liefert Drehpunkte und Endpunkt des Roboters. Bsp:
#       array([[0.        , 0.        ],
#              [0.58856217, 1.91143783],
#              [1.        , 1.        ]])
with writer.saving(fig, 'Trajektorie.mp4',400):
    for s in si:
        l1.set_data(*PG(*s).T)
        l2.set_data(*PG(*s).T)

        writer.grab_frame()
