import numpy as np

# Helper function to compute system energy
# assumes periodic boundary condition
def compute_total_energy(lattice, int_J):
    total_energy = 0
    if lattice.ndim == 2:
        M, N = lattice.shape
        for index, value in np.ndenumerate(lattice):
            # Along axis 1
            total_energy += value * lattice[index[0],  (index[1]+1) % N]
            total_energy += value * lattice[index[0], ((index[1]-1) + N) % N]
            # Along axis 2
            total_energy += value * lattice[((index[0]-1) + M) % M, index[1]]
            total_energy += value * lattice[ (index[0]+1) % M, index[1]]
    elif lattice.ndim == 3:
        M, N, O = lattice.shape
        for index, value in np.ndenumerate(lattice):
            # Along axis 1
            total_energy += value * lattice[index[0],  (index[1]+1) % N, index[2]]
            total_energy += value * lattice[index[0], ((index[1]-1) + N) % N, index[2]]
            # Along axis 2
            total_energy += value * lattice[ (index[0]+1) % M, index[1], index[2]]
            total_energy += value * lattice[((index[0]-1) + M) % M, index[1], index[2]]
            # Along axis 3
            total_energy += value * lattice[index[0], index[1],  (index[2]+1) % O]
            total_energy += value * lattice[index[0], index[1], ((index[2]-1) + O) % O]
    return -int_J * total_energy

# Helper function to compute change in energy
def compute_energy_delta(lattice, flip_coord, int_J):
    nn_sum = 0
    spin_mu = None
    if lattice.ndim == 2:
        M, N = lattice.shape
        nn_sum += lattice[  flip_coord[0],               (flip_coord[1]+1) % N]
        nn_sum += lattice[  flip_coord[0],              ((flip_coord[1]-1) + N) % N]
        nn_sum += lattice[((flip_coord[0]-1) + M) % M,    flip_coord[1]]
        nn_sum += lattice[ (flip_coord[0]+1) % M,         flip_coord[1]]
        spin_mu = lattice[flip_coord[0], flip_coord[1]]
    elif lattice.ndim == 3:
        M, N, O = lattice.shape
        nn_sum += lattice[flip_coord[0], (flip_coord[1]+1) % N, flip_coord[2]]
        nn_sum += lattice[flip_coord[0], ((flip_coord[1]-1) + N) % N, flip_coord[2]]
        nn_sum += lattice[((flip_coord[0]-1) + M) % M,    flip_coord[1], flip_coord[2]]
        nn_sum += lattice[ (flip_coord[0]+1) % M,         flip_coord[1], flip_coord[2]]
        nn_sum += lattice[flip_coord[0], flip_coord[1], ((flip_coord[2]-1) + O) % O]
        nn_sum += lattice[flip_coord[0], flip_coord[1], (flip_coord[2]+1) % O]
        spin_mu = lattice[flip_coord[0], flip_coord[1], flip_coord[2]]
    # Compute energy delta
    energy_delta = 2 * int_J * spin_mu * nn_sum
    return energy_delta

def compute_magnetization(lattice):
    # Compute based on number of sites
    return np.sum(lattice) #abs(np.sum(lattice) / lattice.size)

def compute_magnetization_delta(lattice, flip_coord):
    spin_delta = 0
    if lattice.ndim == 2:
        spin_delta = -2 * lattice[flip_coord[0], flip_coord[1]]
    elif lattice.ndim == 3:
        spin_delta = -2 * lattice[flip_coord[0], flip_coord[1], flip_coord[2]]
    return spin_delta

def perform_single_sweep(lattice, constant_J, flip_coordinates, rand_comparisons, total_energy, magnetization, acceptance_prob):
    N = lattice.shape[0]
    N_2 = N**2
    
    for i in range(N_2):
        # Compute delta
        delta_E = compute_energy_delta(lattice=lattice, flip_coord=flip_coordinates[i], int_J=constant_J)
        delta_M = compute_magnetization_delta(lattice=lattice, flip_coord=flip_coordinates[i])
        # Perform metropolish checks
        if  rand_comparisons[i] < acceptance_prob[delta_E]:
            # Accept the flip, and compute new magetization
            lattice[flip_coordinates[i][0], flip_coordinates[i][1]] *= -1
            total_energy += delta_E
            magnetization += delta_M
    
    return total_energy, magnetization, lattice


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
######
######
######          Parallel Implementation
######
######
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
def compute_energy_delta_parallel(lattice, flip_mask, int_J):
    # Compute nearest neighbor sums
    # for the entire computation mask
    nn_sum = \
        np.roll(lattice,  1, axis=0)[flip_mask] + \
        np.roll(lattice, -1, axis=0)[flip_mask] + \
        np.roll(lattice,  1, axis=1)[flip_mask] + \
        np.roll(lattice, -1, axis=1)[flip_mask]
    # Compute energy deltas for the entire mask
    energy_deltas = 2 * int_J * lattice[flip_mask] * nn_sum
    return energy_deltas

def perform_single_sweep_parallel(lattice, constant_J, flip_coordinates, rand_comparisons, total_energy, magnetization, acceptance_prob):
    # Perform for flip coordinates
    # Compute energy delta
    energy_deltas = compute_energy_delta_parallel(lattice=lattice, flip_mask=flip_coordinates, int_J=constant_J)
    probs = np.array([acceptance_prob[delta_E] for delta_E in energy_deltas])
    # Accept or reject flips
    accept_mask = rand_comparisons[0] < probs
    accepted_flips = flip_coordinates.copy().reshape(-1)
    accepted_flips[accepted_flips] = accept_mask
    accepted_flips = accepted_flips.reshape(lattice.shape)
    # Updated lattice
    lattice[accepted_flips] *= -1
    # Update total energy and magnetization
    total_energy += np.sum(energy_deltas[accept_mask])
    magnetization = np.sum(lattice)
    # Perform for flip inverse coordinates
    # Compute energy delta
    flip_coordinates = np.invert(flip_coordinates.copy())
    energy_deltas = compute_energy_delta_parallel(lattice=lattice, flip_mask=flip_coordinates, int_J=constant_J)
    probs = np.array([acceptance_prob[delta_E] for delta_E in energy_deltas])
    # Accept or reject flips
    accept_mask = rand_comparisons[1] < probs
    accepted_flips = flip_coordinates.copy().reshape(-1)
    accepted_flips[accepted_flips] = accept_mask
    accepted_flips = accepted_flips.reshape(lattice.shape)
    # Updated lattice
    lattice[accepted_flips] *= -1
    # Update total energy and magnetization
    total_energy += np.sum(energy_deltas[accept_mask])
    magnetization = np.sum(lattice)
    return total_energy, magnetization, lattice
