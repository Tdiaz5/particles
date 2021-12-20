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

N = 23
SCALE = 0.001
A = 10
B = 0.01
NSTEPS = 1000

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

def move_particles(scale, particles):
    force_vectors = calculate_force_vectors(particles)
    random_step_magnitudes = scale * np.random.rand(N)

    steps = force_vectors * random_step_magnitudes[:,None]
    particles = particles + steps
    
    distance_to_origin = np.sqrt(np.einsum('ij,ij->i', particles, particles))
    particles = np.where(distance_to_origin[:,None] > 1, particles / distance_to_origin[:,None], particles)

    return particles

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
    
    # perform the algorithm
    # particles = annealing_algorithm(A, B, NSTEPS, particles)

    for i in range(NSTEPS):
        particles = move_particles(SCALE, particles)
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

        plt.draw()
        plt.pause(0.001)
        plt.clf()
