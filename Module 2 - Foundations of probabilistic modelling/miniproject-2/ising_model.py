import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from helper_code import compute_total_energy, compute_magnetization, perform_single_sweep, perform_single_sweep_parallel

class Simulator:
    def __init__(self, size, p_ratio, temp_min, temp_max, temp_delta, constant_J, constant_kB, rand_seed = 32) -> None:
        self.rand_seed = rand_seed
        self.temp_delta = temp_delta
        if isinstance(self.temp_delta, list):
            self.temp_min = self.temp_delta[0]
            self.temp_max = self.temp_delta[-1]
        else:
            self.temp_min = temp_min
            self.temp_max = temp_max
        self.J = constant_J
        self.kB = constant_kB
        self.size = size
        self.p_ratio = p_ratio
        self.lattice = self.gen_lattice(self.size, self.p_ratio, self.rand_seed)
        self.cmap = matplotlib.colors.ListedColormap(["Red", "Green"])

    def gen_lattice(self, size, p_ratio, rand_seed = 64):
        """ Helper function to initialize lattice grid."""
        # Setup random seed
        np.random.seed(rand_seed)
        # Initialize the radom lattice requested
        lattice_dim = [size, size]
        init_random = np.random.random(lattice_dim)
        init_lattice = np.zeros_like(init_random, dtype=np.int8)
        init_lattice[init_random  > p_ratio] = -1
        init_lattice[init_random <= p_ratio] = +1
        return init_lattice

    def run_simulation_sequential(self, n_sweeps = 10_000, plot_update_step = 500, plot = False, out_path = None):
        """Helper function to start the monte-carlo simulation"""
        current_temp = self.temp_min

        # Create plot if requested
        if plot:
            fig_plot, ax_plot = plt.subplots(1, 3, figsize=(24, 8))
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            ax_plot[0].set_box_aspect(1)
            ax_plot[1].set_box_aspect(1)
            ax_plot[2].set_box_aspect(1)
        
        # Pre-compute random states
        flip_coordinates = np.random.randint(0, self.size, size = [n_sweeps, self.size**2, 2])
        rand_comparisons = np.random.random(size=[n_sweeps, self.size**2])
        
        # Precompute total energy and magnetization
        total_energy = compute_total_energy(lattice=self.lattice, int_J=self.J)
        magnetization = compute_magnetization(lattice=self.lattice)

        # An index to keep track of current temperature
        indx = 0

        # Plots and logging variables
        hist_total_engery = np.zeros(n_sweeps, dtype=np.int32)
        hist_magnetization = np.zeros(n_sweeps)
        hist_lattices = np.zeros([n_sweeps, self.size, self.size], dtype=np.int8)

        while current_temp <= self.temp_max:
            print(f"Performing Simulation at T = {current_temp}")

            # Pre-compute exponentials / acceptance probabilities
            current_beta = np.Inf if current_temp == 0 else 1.0 / (current_temp * self.kB)
            acceptance_prob = {
                 -8: min(1, np.exp(-current_beta *  -8)),
                 -4: min(1, np.exp(-current_beta *  -4)),
                  0: min(1, np.exp(-current_beta *   0)),
                  4: min(1, np.exp(-current_beta *   4)),
                  8: min(1, np.exp(-current_beta *   8)),
            }

            # Perform specified number of sweeps at 
            # current temperature to reach equilibrium
            for sweep in range(n_sweeps):
                total_energy, magnetization, self.lattice = perform_single_sweep(
                    lattice=self.lattice, 
                    constant_J=self.J, 
                    flip_coordinates=flip_coordinates[sweep],
                    rand_comparisons=rand_comparisons[sweep],
                    total_energy=total_energy,
                    magnetization=magnetization,
                    acceptance_prob=acceptance_prob
                )

                # Log sweep results
                hist_total_engery[sweep] = total_energy
                hist_magnetization[sweep] = magnetization
                hist_lattices[sweep] = self.lattice

                # Update plots
                if plot:
                    if (sweep % plot_update_step) == 0:
                        print(f"Completed {sweep} sweeps at T = {current_temp}")
                        # Clear all axis
                        ax_plot[0].clear()
                        ax_plot[1].clear()
                        ax_plot[2].clear()
                        ax_plot[0].imshow(self.lattice, interpolation="None", cmap=self.cmap, vmin=-1, vmax=1)
                        ax_plot[1].plot(hist_total_engery[:sweep])
                        ax_plot[2].plot(abs(hist_magnetization[:sweep] / (self.size**2)))
                        plt.pause(0.05)

            # Save results for current temperature run
            if out_path is not None:
                historical_data = {
                    "hist_magnetization": hist_magnetization,
                    "hist_total_engery": hist_total_engery,
                    "hist_lattices": hist_lattices,
                }
                os.makedirs(out_path, exist_ok=True)
                np.savez(os.path.join(out_path, f"temp_{current_temp}_s.npz"), **historical_data)

            # Update current temperature
            indx += 1
            if isinstance(self.temp_delta, list):
                if indx >= len(self.temp_delta):
                    break
                else:
                    current_temp = self.temp_delta[indx]
            else:
                current_temp += self.temp_delta

    def run_simulation_parallelized(self, n_sweeps = 10_000, plot_update_step = 500, plot = False, out_path = None):
        """Helper function to start the monte-carlo simulation"""
        current_temp = self.temp_min

        # Create plot if requested
        if plot:
            fig_plot, ax_plot = plt.subplots(1, 3, figsize=(24, 8))
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            ax_plot[0].set_box_aspect(1)
            ax_plot[1].set_box_aspect(1)
            ax_plot[2].set_box_aspect(1)
        
        ## Pre-compute the checkerd mask for parallel computations
        checkerd_mask = np.zeros(shape=(self.size, self.size), dtype=bool)
        checkerd_mask[ ::2,  ::2] = True
        checkerd_mask[1::2, 1::2] = True

        # Pre-compute random states
        rand_comparisons = np.random.random(size=[n_sweeps, 2, self.size**2 // 2])
        
        # Precompute total energy and magnetization
        total_energy = compute_total_energy(lattice=self.lattice, int_J=self.J)
        magnetization = compute_magnetization(lattice=self.lattice)

        # An index to keep track of current temperature
        indx = 0

        # Plots and logging variables
        hist_total_engery = np.zeros(n_sweeps, dtype=np.int32)
        hist_magnetization = np.zeros(n_sweeps)
        hist_lattices = np.zeros([n_sweeps, self.size, self.size], dtype=np.int8)

        while current_temp <= self.temp_max:
            print(f"Performing Simulation at T = {current_temp}")

            # Pre-compute exponentials / acceptance probabilities
            current_beta = np.Inf if current_temp == 0 else 1.0 / (current_temp * self.kB)
            acceptance_prob = {
                 -8: min(1, np.exp(-current_beta *  -8)),
                 -4: min(1, np.exp(-current_beta *  -4)),
                  0: min(1, np.exp(-current_beta *   0)),
                  4: min(1, np.exp(-current_beta *   4)),
                  8: min(1, np.exp(-current_beta *   8)),
            }

            # Perform specified number of sweeps at 
            # current temperature to reach equilibrium
            for sweep in range(n_sweeps):
                total_energy, magnetization, self.lattice = perform_single_sweep_parallel(
                    lattice=self.lattice, 
                    constant_J=self.J, 
                    flip_coordinates=checkerd_mask,
                    rand_comparisons=rand_comparisons[sweep],
                    total_energy=total_energy,
                    magnetization=magnetization,
                    acceptance_prob=acceptance_prob
                )

                # Log sweep results
                hist_total_engery[sweep] = total_energy
                hist_magnetization[sweep] = magnetization
                hist_lattices[sweep] = self.lattice

                # Update plots
                if plot:
                    if (sweep % plot_update_step) == 0:
                        print(f"Completed {sweep} sweeps at T = {current_temp}")
                        # Clear all axis
                        ax_plot[0].clear()
                        ax_plot[1].clear()
                        ax_plot[2].clear()
                        ax_plot[0].imshow(self.lattice, interpolation="None", cmap=self.cmap, vmin=-1, vmax=1)
                        ax_plot[1].plot(hist_total_engery[:sweep])
                        ax_plot[2].plot(abs(hist_magnetization[:sweep] / (self.size**2)))
                        plt.pause(0.05)

            # Save results for current temperature run
            if out_path is not None:
                historical_data = {
                    "lattice_size": self.size,
                    "hist_magnetization": hist_magnetization,
                    "hist_total_engery": hist_total_engery,
                    "hist_lattices": hist_lattices,
                }
                os.makedirs(out_path, exist_ok=True)
                np.savez(os.path.join(out_path, f"temp_{current_temp}_p.npz"), **historical_data)

            # Update current temperature
            indx += 1
            if isinstance(self.temp_delta, list):
                if indx >= len(self.temp_delta):
                    break
                else:
                    current_temp = self.temp_delta[indx]
            else:
                current_temp += self.temp_delta



if __name__ == "__main__":
    # All simulation temperatures
    simulation_temps = [round(0.25*i, 2) for i in range(0, 8, 1)] + [round(0.05*i, 2) for i in range(40, 60, 1)] + [round(0.25*i, 2) for i in range(12, 21, 1)]
    
    # t1 = time.time()
    # sim = Simulator(size=100, p_ratio=1.0, temp_min=0.0, temp_max=5.0, temp_delta=[2.5], constant_J=1.0, constant_kB=1.0, rand_seed=32)
    # sim.run_simulation_sequential(n_sweeps=10_000, plot_update_step=100, plot=True, out_path="Module 2 - Foundations of probabilistic modelling/miniproject-2/results/")
    # print(f"Sequential Ran in: {time.time() - t1}")

    t1 = time.time()
    sim = Simulator(size=100, p_ratio=1.0, temp_min=0.0, temp_max=5.0, temp_delta=simulation_temps, constant_J=1.0, constant_kB=1.0, rand_seed=64)
    sim.run_simulation_parallelized(n_sweeps=25_000, plot_update_step=50, plot=False, out_path="Module 2 - Foundations of probabilistic modelling/miniproject-2/results/")
    print(f"Parallel Ran in: {time.time() - t1}")

    plt.show()