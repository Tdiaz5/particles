# cirkel.py
# Jasper Lankhorst
# 2021-12-16
# Stochastic Simulation

import math
import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.core.numeric import _array_equal_dispatcher

N = 10
SCALE = 0.05
A = 1
B = 2
NSTEPS = 10000

def produce_particles(n):
    """
    Produces n particles at random positions in circle, i.e. gives random x and
    y as coordinates. Adds to list. Returns: list of n [x, y] lists.
    """
    xlist, ylist = [], []
    while len(xlist) < n:
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
        r = math.sqrt(x**2 + y**2)
        if r < 1:
            xlist.append(x)
            ylist.append(y)
    return xlist, ylist

def total_energy(xlist, ylist):
    """
    Calculates the total system energy for a given xlist, ylist
    """

    total_energy = 0
    N = len(xlist)

    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = ((xlist[i] - xlist[j]) ** 2 + (ylist[i] - ylist[j]) ** 2) ** (1 / 2)
                total_energy += 1 / r_ij

    # account for double counts (but doesn't really matter)
    total_energy /= 2
    return total_energy

def move_particle(scale, particle):
    """
    Takes particle, moves random length (< scale * r) in random direction,
    returns new position.
    """
    length = scale * random.random()
    direction = 2 * math.pi * random.random()
    x_move = length * math.cos(direction)
    y_move = length * math.sin(direction)
    x = particle[0]
    y = particle[1]
    x_new = x + x_move
    y_new = y + y_move
    return [x_new, y_new]

def move_inside(scale, particle):
    """Calls move_particle function until the particle is inside circle."""
    position = move_particle(scale, particle)
    while math.sqrt(position[0]**2 + position[1]**2) > 1:
        position = move_particle(scale, particle)
    return position

# def move_particles(scale, x_list, y_list):
#     """
#     Takes lists of x-coordinates and y-coordinates, moves all in random
#     direction but does not move outside the circle.
#     """
#     index = 0
#     while index < len(x_list):
#         particle = [x_list[index], y_list[index]]
#         particle = move_inside(scale, particle)
#         x_list[index] = particle[0]
#         y_list[index] = particle[1]
#         index += 1
#     return x_list, y_list

def move_particles(scale, xlist, ylist):
    """
    Takes lists of x-coordinates and y-coordinates, moves all in random
    direction but does not move outside the circle.
    """
    xlist_new = []
    ylist_new = []

    for i in range(len(xlist)):
        xnew, ynew = move_inside(scale, [xlist[i], ylist[i]])
        xlist_new.append(xnew)
        ylist_new.append(ynew)

    return xlist_new, ylist_new

def annealing_step(xlist, ylist, T):
    """Computes one step of the annealing algorithm"""

    # step 1: make move
    xlist_new, ylist_new = move_particles(SCALE, xlist, ylist)
    # print(xlist)
    # sample U
    U = random.random()
    # print(xlist, xlist_new)
    # compute alpha
    # print(xlist_new, ylist_new)
    # print(xlist, ylist)
    h_new = total_energy(xlist_new, ylist_new)
    h = total_energy(xlist, ylist)
    # print(h, h_new)
    alpha = min(np.exp((h - h_new) / T), 1)
    
    # determine which list to return
    if U < alpha:
        return xlist_new, ylist_new
    return xlist, ylist

def annealing_algorithm(a, b, nsteps, xlist, ylist):
    """Computes the total annealing algorithm"""

    for n in range(nsteps):
        T_n = (a) / (np.log(n + b))
        xlist, ylist = annealing_step(xlist, ylist, T_n)
    return xlist, ylist

if __name__ == "__main__":
    xlist, ylist = produce_particles(N)
    # print(total_energy([1, -1], [1, -1]))
    # print(total_energy([0, 0], [1, -1]))

    xlist, ylist = annealing_algorithm(A, B, NSTEPS, xlist, ylist)

    # for i in range(100):
        # xlist, ylist = move_particles(SCALE, xlist, ylist)

    x_circle, y_circle = [], []
    for theta in np.arange(0, 2 * np.pi, 0.01):
        x_circle.append(np.cos(theta))
        y_circle.append(np.sin(theta))
        
    plt.plot(x_circle, y_circle, 'b--')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    
    plt.plot(xlist, ylist, "ro")
    plt.show()
