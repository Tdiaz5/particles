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

ROTATION_SCALE = 1
NSTEPS = 1000

def logarithmic_cooling(n):
    A = 1000
    B = 2
    return (A) / (np.log(n + B))

def exponential_cooling(n):
    A = 10
    B = 0.9
    return A * (B) ** n

def linear_multiplicative_cooling(n):
    A = 10
    B = 2
    return A / (1 + B*n)

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
    """Calculates the total force vector on each particle"""

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
    """Move the particles for one time step"""

    # first determine vectors of random length in the force vector direction
    force_vectors = calculate_force_vectors(particles)
    random_step_magnitudes = scale * np.random.rand(N)
    steps = force_vectors * random_step_magnitudes[:,None]

    # now add a slight random rotation
    thetas = ROTATION_SCALE * (2 * np.random.rand(N) - 1)
    rotation_matrices = []

    # calculate a random rotation matrix for every vector
    for theta in thetas:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotation_matrices.append(rotation_matrix)

    # apply random rotation matrix to every step vector: random rotation achieved
    for index in range(N):
        steps[index] = rotation_matrices[index].dot(steps[index])

    particles = particles + steps

    # to enforce the boundary, particles at a distance > 1 are normalized
    # (thus setting their distance to 1). This ensures that all movement in the
    # angular direction doesn't change, but the radial part doesn't change    
    distance_to_origin = np.sqrt(np.einsum('ij,ij->i', particles, particles))
    particles = np.where(distance_to_origin[:,None] > 1, particles / distance_to_origin[:,None], particles)

    return particles

def annealing_step(particles, scale, T):
    """Computes one step of the annealing algorithm"""

    # step 1: make move
    particles_new = move_particles(scale, particles)
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

def plotting_step(particles):
    """Plots one step of the iteration"""

    x_circle, y_circle = [], []
    for theta in np.arange(0, 2 * np.pi, 0.01):
        x_circle.append(np.cos(theta))
        y_circle.append(np.sin(theta))
        
    plt.plot(x_circle, y_circle, 'k')

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    
    x_list = particles[:,0]
    y_list = particles[:,1]

    plt.plot(x_list, y_list, "ro")

def annealing_algorithm(nsteps, particles, scale, cooling_algorithm, animation):
    """Computes the total annealing algorithm"""

    for n in range(nsteps):
        T_n = cooling_algorithm(n)
        particles = annealing_step(particles, scale, T_n)

        if animation:
            plotting_step(particles)
            plt.draw()
            plt.pause(0.001)
            plt.clf()
    
    E_final = calculate_energy(particles)
    print(f"Final energy: {E_final}")

    if E_final < 2928:
        plotting_step(particles)
        plt.savefig(f"plots/N{N}_E_{round(E_final, 2)}.png")
        plt.clf()

    return particles

if __name__ == "__main__":
    # set square figure
    plt.rcParams["figure.figsize"] = (8, 8)
    # on higher N, it is necessary to decrease scale. Sometimes it turns into
    # a twitchy mess with a high SCALE and high N
    N = 50
    for i in range(10):
        for scale in np.arange(0.0005, 0.003, 0.0005):
            particles = produce_particles(N)
            particles = annealing_algorithm(NSTEPS, particles, scale, logarithmic_cooling, False)