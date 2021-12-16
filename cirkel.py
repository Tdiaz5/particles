# cirkel.py
# Jasper Lankhorst
# 2021-12-16
# Stochastic Simulation

import math
import numpy as np
import matplotlib.pyplot as plt
import random



N = 2

def produce_particles(n):
    """
    Produces n particles at random positions in circle, i.e. gives random x and
    y as coordinates. Adds to list. Returns: list of n [x, y] lists.
    """
    particle_list = []
    while len(particle_list) < n:
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
        r = math.sqrt(x**2 + y**2)
        if r < 1:
            particle_list.append([x, y])
    return particle_list

def move_particle(scale):
    """
    Takes particle, moves random length in random direction
    """


if __name__ == "__main__":
    system = produce_particles(N)



    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    xlist, ylist = [], []
    for particle in system:
        xlist.append(particle[0])
        ylist.append(particle[1])
    plt.plot(xlist, ylist, "ro")
    plt.show()
