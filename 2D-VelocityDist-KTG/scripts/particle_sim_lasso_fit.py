'''
Simulating particles in a box to find velocity distribution law with lasso fit

Note : you need to have FFmpeg installed in your pc for saving the animation.
       Install FFmpeg in ubuntu using `$ sudo apt install ffmpeg`

Author : Anik Mandal
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter	# FuncAnimation for animation and FFMpegWriter for writing/saving video file
from utils import *
import time
from warnings import filterwarnings
filterwarnings('ignore')

# defining plot axes
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_subplot(1, 2, 1)	# ax1 for particle simulation
ax2 = fig.add_subplot(1, 2, 2)	# ax2 for velocity distribution

Temp = 273+30       		# in absolute scale
R = 8.314			# ideal gas constant
M = 0.015              		# Molar mass (in kg per mole)
v_avg = ((np.pi * R * Temp) / (2 * M)) ** 0.5  		# in m/s (initially, all particle will have same avg velocity)

xmin, ymin, xmax, ymax = 0, 0, 10, 10			# box limits

n = 250  			# number of particles

dv = 20				# bin size in velocity distribution plot
v_list = np.array([dv * i for i in range(60)]) 		# velocity bins
pp = []				# list to store particles
dt = 0.00005  			# in s (increment in time with frames)
trail_x, trail_y = [], []	# to store trails of a particle
sum_n = np.array([0 for i in range(len(v_list))]) 	# to store sum of the no of particles


# Defining a Particle class with some attributes
class Particle:
    '''
    Partcle class object will have a specific id, postion, velocity, mass and redius(default 0.1)
    '''
    def __init__(self, id, position_vector, velocity_vector, mass=M/(6.023*10**(26)), radius=0.1):
        self.id, self.pos, self.vel, self.mass, self.rad = id, position_vector, velocity_vector, mass, radius

# Defining boundaries of the box
def Boundary(x_min, y_min, x_max, y_max):
    '''
    Defining boundaries of the box with input minimum and maximum limits in x and y
    '''
    ax1.plot([x_min, x_max], [y_min, y_min], '-k')
    ax1.plot([x_max, x_max], [y_min, y_max], '-k')
    ax1.plot([x_max, x_min], [y_max, y_max], '-k')
    ax1.plot([x_min, x_min], [y_max, y_min], '-k')


def Wall_Collision_Det(axis, particle, min, max):
    '''
    This function checks collision of the particles with the box walls
    using continuous collision detection method and deflect particles from 
    the walls.
    -----Inputs-----------------------------------------------------------
    axis : string
        Axis of the wall
    particle : Particle object
        Particle to be considered for wall collison check
    min : float
    	Minimum position
    max : float
    	Maximum position
    '''
    if axis == 'x':
        pi = particle.pos[0][0]
        v = particle.vel[0][0]
        dx = v * dt
        pf = particle.pos[0][0] + dx
        if pi >= particle.rad - min > pf:
            particle.pos[0][0] = 2 * particle.rad - particle.pos[0][0]
            particle.vel[0][0] = -particle.vel[0][0]

        elif pi <= max - particle.rad < pf:
            particle.pos[0][0] = 2 * (max - particle.rad) - particle.pos[0][0]
            particle.vel[0][0] = -particle.vel[0][0]
    elif axis == 'y':
        pi = particle.pos[1][0]
        v = particle.vel[1][0]
        dx = v * dt
        pf = particle.pos[1][0] + dx
        if pi >= particle.rad - min > pf:
            particle.pos[1][0] = 2 * particle.rad - particle.pos[1][0]
            particle.vel[1][0] = -particle.vel[1][0]

        elif pi <= max - particle.rad < pf:
            particle.pos[1][0] = 2 * (max - particle.rad) - particle.pos[1][0]
            particle.vel[1][0] = -particle.vel[1][0]


def PP_Collision_Det(particle_a, particle_b):
    '''
    This function checks collision between two individual particles and
    deflects them accordingly.
    -----Inputs-----------------------------------------------------------
    particle_a : particle object
        First partcle
    particle_b : Particle object
        Second particle
    '''
    dis_v = V_Subtract(particle_b.pos, particle_a.pos) 		# distance vector between two particles
    dis = V_Mod(dis_v)						# distance between two particles
    a_nxt, b_nxt = [], []
    for i in range(2):
        a_nxt.append([particle_a.pos[i][0] + particle_a.vel[i][0] * dt])
        b_nxt.append([particle_b.pos[i][0] + particle_b.vel[i][0] * dt])
    dis_nxt = V_Mod(V_Subtract(a_nxt, b_nxt))

    if 2 * particle_a.rad > dis > dis_nxt: 			# if collision happens rectify the trajectories
        v_rel_ba = V_Subtract(particle_b.vel, particle_a.vel)
        p_rel_ba = V_Subtract(particle_b.pos, particle_a.pos)
        unit_rel_ba = V_Unit(p_rel_ba)
        Comp = V_Dot(v_rel_ba, unit_rel_ba)
        
        particle_a.vel = V_Subtract(particle_a.vel, V_Scale(V_Neg(unit_rel_ba), Comp))
        particle_b.vel = V_Subtract(particle_b.vel, V_Scale(unit_rel_ba, Comp))


def Set_axis(ax1, ax2, frame):
    '''
    This function configures plot axes based on animation frame
    -----Inputs-----------------------------------------------------------
    ax1 : subplot object
        First subplot
    ax2 : subplot object
        Second subplot
    frame : int
        Frame number of animation
    '''
    ax1.cla()
    ax2.cla()
    ax1.set_aspect('equal')
    ax1.set_title('Particles inside a 2D box\nNumber of particles: ' + str(n)+'; Sim_speed: '+str(dt/0.01)+'s')
    ax1.set_xlabel('x-position')
    ax1.set_ylabel('y-position')
    ax2.set_ylim([0,20])
    ax2.set_title('Velocity Distribution\nTemp: '+str(Temp-273)+'℃')
    ax2.set_xlabel('Velocity(m/s)')
    ax2.set_ylabel('number of particles')
    time_text = ax2.text(800, 16, 'Simulation Time %.2fms' % (frame * 0.05))


# Creating particles using Particle class:
for i in range(n):
    px = xmax * np.random.uniform(0.01, 0.99)
    py = ymax * np.random.uniform(0.01, 0.99)
    angle = np.random.uniform(0, 2 * np.pi)
    [vx, vy] = [v_avg * np.cos(angle), v_avg * np.sin(angle)]
    p = Particle(id=i+1, position_vector=[[px],[py]], velocity_vector=[[vx], [vy]])
    pp.append(p)


def animate(frame):
    # Main animation function, every thing is done here based on animation frame number
    
    Set_axis(ax1, ax2, frame)		# Configuring subplot axes
    Boundary(xmin, ymin, xmax, ymax)	# Defining boundaries
    
    
    xx, yy, vv, nn = [], [], [], []	# lists of frame dependent axes coordinates
    
    # storing trails for 1st particle
    trail_x.append(pp[0].pos[0][0])	
    trail_y.append(pp[0].pos[1][0])

    for i in range(n):
        for j in range(i + 1, n):
            PP_Collision_Det(pp[i], pp[j]) 		# checking mutual particle-particle collision

        Wall_Collision_Det('x', pp[i], xmin, xmax)	# checking wall collision along x axis
        Wall_Collision_Det('y', pp[i], ymin, ymax)	# checking wall collision along y axis
        
        # particles cooedinates
        xx.append(pp[i].pos[0][0])
        yy.append(pp[i].pos[1][0])

        # particle motion: moving particle with time per frame to new position
        pp[i].pos[0][0] = pp[i].pos[0][0] + pp[i].vel[0][0] * dt
        pp[i].pos[1][0] = pp[i].pos[1][0] + pp[i].vel[1][0] * dt

    # checking and storing no of particles in each velocity bins
    for i in range(len(v_list)):		
        c = 0
        for j in range(n):
            v = V_Mod(pp[j].vel)
            if v_list[i] <= v < v_list[i] + dv:
                c = c + 1
        nn.append(c)

    ax1.plot(xx, yy, '.r')				# plotting particles in the box
    ax1.plot(trail_x, trail_y, '-c')			# plotting trail of a particle
    ax2.bar(v_list, nn, width=dv*0.8, align='edge')	# plotting velocity distribution histogram
    
    # Creating Lasso regression fit over the instantanous velocity distribution
    x_data = v_list[:, None]
    y_data = np.array(nn)[:, None]
    
    n_p, max_degree, alpha = 216, 10, 0.1		# Lasso fit configurations
    v_data, n_data = LassoRegression(x_data, y_data, max_degrees=max_degree, alpha_value=alpha, num_points=n_p)		# fit datas
    ax2.plot(v_data, n_data,'-y')			# plotting the lasso fit
    ax2.legend(['Lasso Regression\nalpha= 0.1; max_degree= 10', 'Simulation Data'])
    
    if frame % 300 == 0:				# for checking animation progress
        dur = time.time() - ti
        print(frame / 60, '%\t', dur, 's')


ti = time.time()
print('Running...')
ani = FuncAnimation(fig, animate, frames=6000, interval=10)

wr = FFMpegWriter(fps=100)	# animation writer
ani.save('/home/anik/git-self/numerical-physics/2D-VelocityDist-KTG/outputs/Velocity_Distribution_Lasso_fit.mp4', writer=wr)	# Don't forget to change location

tf = time.time()
print('Completed!\nTotal rendering time: ', (tf - ti) / 60, 'min.')

plt.show()

