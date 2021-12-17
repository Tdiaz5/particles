# cirkel.py
# Jasper Lankhorst
# 2021-12-16
# Stochastic Simulation

import math
import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.core.numeric import _array_equal_dispatcher
from numpy.lib.function_base import diff

N = 3
SCALE = 0.05
A = 1
B = 2
NSTEPS = 10000

def produce_particles(n):
    """
    Produces n particles at random positions in circle, i.e. gives random x and
    y as coordinates. Adds to list. Returns: list of n [x, y] lists.
    """
    particles = []
    
    while len(particles) < n:
        x = random.random() * 2 - 1
        y = random.random() * 2 - 1
        r = math.sqrt(x**2 + y**2)
        if r < 1:
            particles.append((x, y))
    
    particles = np.array(particles)
    return particles

def calculate_energy(particles):
    """
    Calculates the total system energy for a given particles array
    """

    total_energy = 0

    for index in range(len(particles)):
        selected_particle = particles[index]
        difference_vectors = particles - selected_particle
        # # exclude the selected particle
        difference_vectors = np.delete(difference_vectors, index, 0)
        # source: https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
        distances = np.sqrt(np.einsum('ij,ij->i', difference_vectors, difference_vectors))

        energy_particle = np.sum(distances ** (-1))
        total_energy += energy_particle
    
    return total_energy

def calculate_force_vectors(particles):
    """Calculates the force vector on each particle"""

    force_vectors = []

    for index in range(len(particles)):
        selected_particle = particles[index]
        difference_vectors = particles - selected_particle
        # # exclude the selected particle
        difference_vectors = np.delete(difference_vectors, index, 0)
        # source: https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
        distances = np.sqrt(np.einsum('ij,ij->i', difference_vectors, difference_vectors))
        
        division_array = distances ** 3
        force_vector_particle = -np.sum(difference_vectors / division_array[:,None], 0)
        force_vectors.append(force_vector_particle)
    
    return np.array(force_vectors)

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

def move_particles(scale, xlist, ylist):
    """
    Takes lists of x-coordinates and y-coordinates, moves all in random
    direction but does not move outside the circle.
    """
    xlist_new = []
    ylist_new = []

    for i in range(len(xlist)):
        x_new, y_new = move_inside(scale, [xlist[i], ylist[i]])
        xlist_new.append(x_new)
        ylist_new.append(y_new)

    return xlist_new, ylist_new

def annealing_step(particles, T):
    """Computes one step of the annealing algorithm"""

    # step 1: make move
    particles_new = move_particles(SCALE, particles)
    # sample U
    U = random.random()
    # compute alpha
    h_new = calculate_energy(particles_new)
    h = calculate_energy(particles)

    alpha = min(np.exp((h - h_new) / T), 1)
    
    # determine which list to return
    if U < alpha:
        return particles_new
    return particles

def annealing_algorithm(a, b, nsteps, particles):
    """Computes the total annealing algorithm"""

    for n in range(nsteps):
        T_n = (a) / (np.log(n + b))
        particles = annealing_step(particles, T_n)
    return particles

if __name__ == "__main__":
    particles = produce_particles(N)
    # energy = calculate_energy(particles)
    # force_vectors = calculate_force_vectors(particles)

    # print(particles)
    # print(force_vectors)

    particles = annealing_algorithm(A, B, NSTEPS, particles)

    # for i in range(100):
        # xlist, ylist = move_particles(SCALE, xlist, ylist)

    x_circle, y_circle = [], []
    for theta in np.arange(0, 2 * np.pi, 0.01):
        x_circle.append(np.cos(theta))
        y_circle.append(np.sin(theta))
        
    plt.plot(x_circle, y_circle, 'b--')

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    
    x_list = particles[:,0]
    y_list = particles[:,1]

    plt.plot(x_list, y_list, "ro")
    plt.show()
