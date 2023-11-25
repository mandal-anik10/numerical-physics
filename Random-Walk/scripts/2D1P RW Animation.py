# Animation of 2D Random Walk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()
ax.set_aspect('equal')


class Particle:
    def __init__(self, x_position=0, y_position=0):
        self.xpos = x_position
        self.ypos = y_position


dot, = ax.plot([], marker='o', color='red')
tr, = ax.plot([], '-g')
tx = plt.text(15, 18, '')
r = 1
p = Particle()
xx, yy, tt, rms = [], [], [], []


def animate(frame):
    x, y = p.xpos, p.ypos
    xx.append(x)
    yy.append(y)
    dot.set_data((x, y))
    tr.set_data((xx, yy))
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    tx.set_text('Time: ' + str(frame/10) + '\nStep:' + str(frame))

    # # Displacement Fixed
    # th = np.random.uniform(0, 2 * np.pi)
    # p.xpos = p.xpos + r * np.cos(th)
    # p.ypos = p.ypos + r * np.sin(th)

    # Motion in Lattice:
    dir = np.random.randint(0, 2)
    if dir == 0:
        s = np.random.choice([-1, 1])
        p.xpos = p.xpos + s
    else:
        s = np.random.choice([-1, 1])
        p.ypos = p.ypos + s

    # # Brownian Motion:
    # lam = 1
    # r = np.random.random()
    # th = np.random.uniform(0, 2 * np.pi)
    # p.xpos = p.xpos + r * np.cos(th)
    # p.ypos = p.ypos + r * np.sin(th)


ani = FuncAnimation(fig, animate, frames=600, interval=100)
# title_txt = 'Maximum possible displacement: 1; Collision probability per displacement: uniform'
plt.title('2D 1 Particle Random Walk-Motion in Lattice')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.tight_layout()

Writer = FFMpegWriter(fps=10)
ani.save(r'2D1P_RW_Lattice.mp4', writer=Writer)

plt.show()

