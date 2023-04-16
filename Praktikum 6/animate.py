from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython import display
  
# initializing a figure
fig = plt.figure()
  
# labeling the x-axis and y-axis
axis = plt.axes(xlim=(0, 4),  ylim=(-1.5, 1.5))
  
# initializing a line variable
line, = axis.plot([], [], lw=3)
  
def animate(frame_number):
    x = np.linspace(0, 4, 1000)
  
    # plots a sine graph
    y = np.sin(2 * np.pi * (x - 0.01 * frame_number))
    line.set_data(x, y)
    line.set_color('green')
    return line,
  
  
anim = animation.FuncAnimation(fig, animate, frames=100, 
                               interval=20, blit=True)
fig.suptitle('Sine wave plot', fontsize=14)
  
# converting to an html5 video
video = anim.to_html5_video()
  
# embedding for the video
html = display.HTML(video)
  
# draw the animation
display.display(html)
plt.close()