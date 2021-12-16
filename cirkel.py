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

def move_particle(scale):
    """
    Takes particle, moves random length in random direction
    """

def annealing_step(xlist, ylist, T):
    """Computes one step of the annealing algorithm"""

    # step 1: make move
    xlist_new, ylist_new = move_particles(xlist, ylist)
    
    # sample U
    U = random.random()

    # compute alpha
    h_new = total_energy(xlist_new, ylist_new)
    h = total_energy(xlist, ylist)
    alpha = min(np.exp((h_new - h) / T), 1)
    
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

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    
    plt.plot(xlist, ylist, "ro")
    plt.show()
