import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

# CUDA kernel to compute spin components (sx, sy, sz) from spherical coordinates (phi, theta)
@cuda.jit
def compute_spin_components(phi, theta, sx, sy, sz, N):
    i, j = cuda.grid(2)
    if i < N and j < N:
        phi_ij = phi[i, j]
        theta_ij = theta[i, j]
        sx[i, j] = 1000.0 * math.sin(phi_ij) * math.cos(theta_ij)  # x-component of spin
        sy[i, j] = 1000.0 * math.sin(phi_ij) * math.sin(theta_ij)  # y-component of spin
        sz[i, j] = 1000.0 * math.cos(phi_ij)                       # z-component of spin

# CUDA device function to compute local energy at site (i, j)
@cuda.jit(device=True)
def compute_local_energy(i, j, phi, theta, N, Jx, Jy, Dx, Dy, K1, K2, B_field):
    # Current spin components
    sx = 1000.0 * math.sin(phi[i, j]) * math.cos(theta[i, j])
    sy = 1000.0 * math.sin(phi[i, j]) * math.sin(theta[i, j])
    sz = 1000.0 * math.cos(phi[i, j])
    
    # Right neighbor (periodic boundary)
    right_i = (i + 1) % N
    sx_r = 1000.0 * math.sin(phi[right_i, j]) * math.cos(theta[right_i, j])
    sy_r = 1000.0 * math.sin(phi[right_i, j]) * math.sin(theta[right_i, j])
    sz_r = 1000.0 * math.cos(phi[right_i, j])
    
    # Upper neighbor (periodic boundary)
    up_j = (j + 1) % N
    sx_u = 1000.0 * math.sin(phi[i, up_j]) * math.cos(theta[i, up_j])
    sy_u = 1000.0 * math.sin(phi[i, up_j]) * math.sin(theta[i, up_j])
    sz_u = 1000.0 * math.cos(phi[i, up_j])
    
    # Exchange energy: -J * (S_i * S_j) for x and y directions
    ex_energy = -Jx * (sx*sx_r + sy*sy_r + sz*sz_r) \
                -Jy * (sx*sx_u + sy*sy_u + sz*sz_u)
    
    # DMI energy: Dx * (Sz * Sx' - Sx * Sz') + Dy * (Sy * Sz' - Sz * Sy')
    dmi_energy = -Dx * (sz*sx_r - sx*sz_r) \
                 + Dy * (sy*sz_u - sz*sy_u)
    
    # Anisotropy energy: +K * (Sz)^2 (positive for perpendicular anisotropy)
    K_energy = K1 * (sz ** 4+sx ** 4+sy ** 4)- K2*(sx*sx_u+sy*sy_r)
    
    # Zeeman energy: -B_field[i, j] * Sz
    zeeman_energy = -B_field[i, j] * sz
    
    # Total local energy
    return ex_energy + dmi_energy + K_energy + zeeman_energy

# CUDA kernel for one Monte Carlo step using Metropolis algorithm
@cuda.jit
def monte_carlo_step(phi, theta, N, T, rng_states, parity, Jx, Jy, Dx, Dy, K1,K2, B_field):
    i, j = cuda.grid(2)
    if i < N and j < N and (i + j) % 2 == parity:  # Checkerboard update (odd/even parity)
        idx = i * N + j
        old_phi = phi[i, j]
        old_theta = theta[i, j]
        # Compute current energy
        old_energy = compute_local_energy(i, j, phi, theta, N, Jx, Jy, Dx, Dy, K1,K2, B_field)
        # Propose new angles with small random perturbations
        phi_new = old_phi + (xoroshiro128p_uniform_float32(rng_states, idx) - 0.5) * 0.05
        theta_new = old_theta + (xoroshiro128p_uniform_float32(rng_states, idx) - 0.5) * 0.05
        phi[i, j] = phi_new
        theta[i, j] = theta_new
        # Compute new energy
        new_energy = compute_local_energy(i, j, phi, theta, N, Jx, Jy, Dx, Dy, K1,K2, B_field)
        delta_E = new_energy - old_energy
        # Metropolis acceptance criterion
        if delta_E > 0 and xoroshiro128p_uniform_float32(rng_states, idx) >= math.exp(-delta_E / T):
            phi[i, j] = old_phi    # Reject: revert to old angles
            theta[i, j] = old_theta

# Class to manage the spin system and simulation
class SpinSystem:
    def __init__(self, N, Jx=1.0, Jy=1.0, Dx=1.5, Dy=1.5, K1=0.2, K2=0.3, B=0.5, r0=0.1):
        # Initialize system parameters
        self.N = N                     # Grid size (N x N)
        self.Jx = Jx                   # Exchange coupling (x-direction)
        self.Jy = Jy                   # Exchange coupling (y-direction)
        self.Dx = Dx                   # DMI strength (x-direction)
        self.Dy = Dy                   # DMI strength (y-direction)
        self.K1 = K1
        self.K2 = K2                   # magnetic anisotropy
        self.B = B                     # Magnetic field magnitude
        self.r0 = r0                   # Ratio of downward field radius to grid size
        
        # Allocate CUDA arrays for spin angles and components
        self.phi = cuda.device_array((N, N), dtype=np.float32)    # Polar angle
        self.theta = cuda.device_array((N, N), dtype=np.float32)  # Azimuthal angle
        self.rng_states = create_xoroshiro128p_states(N*N, seed=np.random.randint(0, 10000))  # Random number states
        self.d_sx = cuda.device_array((N, N), dtype=np.float32)   # Spin x-component
        self.d_sy = cuda.device_array((N, N), dtype=np.float32)   # Spin y-component
        self.d_sz = cuda.device_array((N, N), dtype=np.float32)   # Spin z-component
        self.d_energy = cuda.device_array(1, dtype=np.float32)     # Total energy

        # Define the spatially varying magnetic field
        cx = N // 2  # Center x-coordinate
        cy = N // 2  # Center y-coordinate
        R = r0 * N   # Radius of downward field region
        B_field_host = np.full((N, N), B, dtype=np.float32)  # Default: upward field (B)
        for i in range(N):
            for j in range(N):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                if dist <= R:
                    B_field_host[i, j] = -B  # Downward field (-B) within radius
        self.B_field = cuda.to_device(B_field_host)  # Copy to device

    # Initialize spins randomly
    def random_initialize(self):
        threads = (16, 16)
        blocks = ((self.N + 15)//16, (self.N + 15)//16)
        @cuda.jit
        def init_kernel(phi, theta, rng_states, N):
            i, j = cuda.grid(2)
            if i < N and j < N:
                idx = i * N + j
                phi[i,j] = xoroshiro128p_uniform_float32(rng_states, idx) * math.pi      # Random phi (0 to pi)
                theta[i,j] = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * math.pi  # Random theta (0 to 2pi)
        init_kernel[blocks, threads](self.phi, self.theta, self.rng_states, self.N)
        cuda.synchronize()

    # Initialize with a single Skyrmion at specified center
    def skyrmion_initialize(self, center_x, center_y, radius=8.0, chirality=1):
        threads = (16, 16)
        blocks = ((self.N + 15)//16, (self.N + 15)//16)
        @cuda.jit
        def init_skyrmion_kernel(phi, theta, N, cx, cy, R, chi):
            i, j = cuda.grid(2)
            if i < N and j < N:
                x = j - cx
                y = i - cy
                r = math.sqrt(x*x + y*y)
                # Smooth tanh profile for Skyrmion
                phi_angle = math.pi * (1 - 0.5 * (1 + math.tanh((r - R) / 2.0)))  # Transitions from pi (core) to 0 (background)
                theta_angle = chi * math.atan2(y, x) + math.pi / 2  # Neel-type Skyrmion (adjust for chirality)
                phi[i, j] = phi_angle
                theta[i, j] = theta_angle
        init_skyrmion_kernel[blocks, threads](self.phi, self.theta, self.N, center_x, center_y, radius, chirality)
        cuda.synchronize()
        
    def ferro_initialize(self):
        threads = (16, 16)
        blocks = ((self.N + 15)//16, (self.N + 15)//16)
        @cuda.jit
        def init_ferro_kernel(phi, theta, N):
            i, j = cuda.grid(2)
            if i < N and j < N:
                phi[i, j] = 0.0  # Spins along +z (phi=0)
                theta[i, j] = 0.0  # Theta irrelevant when phi=0
        init_ferro_kernel[blocks, threads](self.phi, self.theta, self.N)
        cuda.synchronize()
        
    # Compute topological charge Q
    def compute_topological_charge(self):
        phi = self.phi.copy_to_host()
        theta = self.theta.copy_to_host()
        Q = 0.0
        for i in range(self.N):
            for j in range(self.N):
                ip = (i + 1) % self.N
                jp = (j + 1) % self.N
                # Spin vectors at current and neighboring sites (normalized)
                s1 = np.array([math.sin(phi[i,j]) * math.cos(theta[i,j]),
                               math.sin(phi[i,j]) * math.sin(theta[i,j]),
                               math.cos(phi[i,j])])
                norm_s1 = np.linalg.norm(s1)
                s1 /= norm_s1 if norm_s1 > 0 else 1.0
                s2 = np.array([math.sin(phi[ip,j]) * math.cos(theta[ip,j]),
                               math.sin(phi[ip,j]) * math.sin(theta[ip,j]),
                               math.cos(phi[ip,j])])
                norm_s2 = np.linalg.norm(s2)
                s2 /= norm_s2 if norm_s2 > 0 else 1.0
                s3 = np.array([math.sin(phi[i,jp]) * math.cos(theta[i,jp]),
                               math.sin(phi[i,jp]) * math.sin(theta[i,jp]),
                               math.cos(phi[i,jp])])
                norm_s3 = np.linalg.norm(s3)
                s3 /= norm_s3 if norm_s3 > 0 else 1.0
                # Topological charge density: S * (dS/dx x dS/dy)
                Q += np.dot(s1, np.cross(s2, s3))
        Q /= (4 * math.pi)  # Normalize to get integer charge for Skyrmions
        return Q

    # Perform simulated annealing
    def simulate(self, T_start=0.05, T_end=0.01, steps=500, sweeps=2000):
        threads = (16, 16)
        blocks = ((self.N + 15)//16, (self.N + 15)//16)
        energies = []
        charges = []
        for step in range(steps):
            # Linear temperature schedule
            T = T_start + (T_end - T_start) * step / (steps - 1)
            for _ in range(sweeps):
                # Perform two Monte Carlo steps (odd and even parity for checkerboard update)
                monte_carlo_step[blocks, threads](
                    self.phi, self.theta, self.N, T, self.rng_states, 
                    0, self.Jx, self.Jy, self.Dx, self.Dy, self.K1, self.K2, self.B_field)
                monte_carlo_step[blocks, threads](
                    self.phi, self.theta, self.N, T, self.rng_states,
                    1, self.Jx, self.Jy, self.Dx, self.Dy, self.K1, self.K2, self.B_field)
            # Record energy and topological charge
            energy = self.energy()
            charge = self.compute_topological_charge()
            energies.append(energy)
            charges.append(charge)
            print(f"Step {step+1}/{steps}, Temperature={T:.3f}, Energy={energy:.4f}, Topological Charge={charge:.3f}")
        
        # Plot energy and topological charge vs. temperature step
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        ax1.plot(energies)
        ax1.set_xlabel('Temperature Step')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Temperature')
        ax2.plot(charges)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Topological Charge')
        ax2.set_title('Topological Charge vs Temperature')
        plt.tight_layout()
        plt.show()

    # Compute total energy of the system
    def energy(self):
        self.d_energy[0] = 0
        threads = (16, 16)
        blocks = ((self.N + 15)//16, (self.N + 15)//16)
        # Compute spin components
        compute_spin_components[blocks, threads](
            self.phi, self.theta, self.d_sx, self.d_sy, self.d_sz, self.N)
        
        # CUDA kernel to compute total energy
        @cuda.jit
        def energy_kernel(sx, sy, sz, N, Jx, Jy, Dx, Dy, K1, K2, B_field, energy):
            i, j = cuda.grid(2)
            if i < N and j < N:
                right = (i + 1) % N, j
                up = i, (j + 1) % N
                # Exchange energy
                ex = -Jx*(sx[i,j]*sx[right] + sy[i,j]*sy[right] + sz[i,j]*sz[right]) \
                     -Jy*(sx[i,j]*sx[up] + sy[i,j]*sy[up] + sz[i,j]*sz[up])
                # DMI energy
                dmi = Dx*(sz[i,j]*sx[right] - sx[i,j]*sz[right]) \
                      + Dy*(sy[i,j]*sz[up] - sz[i,j]*sy[up])
                # Anisotropy and Zeeman energy
                other = K1*(sz[i,j]**4+sy[i,j]**4+sx[i,j]**4) - K2*(sx[i,j]*sx[up]+sy[i,j]*sy[right])-B_field[i,j]*sz[i,j]
                # Accumulate total energy atomically
                cuda.atomic.add(energy, 0, ex + dmi + other)
        
        energy_kernel[blocks, threads](
            self.d_sx, self.d_sy, self.d_sz, self.N,
            self.Jx, self.Jy, self.Dx, self.Dy, self.K1,self.K2, self.B_field, self.d_energy)
        return self.d_energy.copy_to_host()[0]

    # Visualize spin configuration and optionally topological charge density
    def visualize(self, title="Spin Configuration", show_charge_density=False):
        phi = self.phi.copy_to_host()
        theta = self.theta.copy_to_host()
        sx = 1000.0 * np.sin(phi) * np.cos(theta)
        sy = 1000.0 * np.sin(phi) * np.sin(theta)
        sz = 1000.0 * np.cos(phi)

        if show_charge_density:
            fig = plt.figure(figsize=(12, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

            # Plot spin configuration (sz with sx, sy arrows)
            ax1 = fig.add_subplot(gs[0])
            im = ax1.imshow(sz/1000.0, cmap='RdBu', vmin=-1, vmax=1, origin='lower')  # Normalize for display
            x, y = np.meshgrid(np.arange(self.N), np.arange(self.N))
            ax1.quiver(x, y, sx/1000.0, sy/1000.0, scale=20, scale_units='inches')  # Normalize for arrows
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.set_xticks(np.arange(0, self.N, 5))
            ax1.set_yticks(np.arange(0, self.N, 5))
            ax1.set_xlabel('j (Horizontal)', fontsize=12)
            ax1.set_ylabel('i (Vertical)', fontsize=12)
            ax1.set_title(
                f"{title}\n(Spin Configuration: N={self.N}, Dx={self.Dx:.1e}, B={self.B:.1f}, \n E={self.energy():.2f}, Q={self.compute_topological_charge():.2f})",
                fontsize=14, fontweight='bold', pad=10
            )
            cbar = plt.colorbar(im, ax=ax1, label='Sz Component (Normalized)', ticks=[-1, 0, 1])
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Sz Component (Normalized)', fontsize=12)

            # Plot topological charge density
            ax2 = fig.add_subplot(gs[1])
            charge_density = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    ip = (i + 1) % self.N
                    jp = (j + 1) % self.N
                    s1 = np.array([math.sin(phi[i,j]) * math.cos(theta[i,j]),
                                   math.sin(phi[i,j]) * math.sin(theta[i,j]),
                                   math.cos(phi[i,j])])
                    norm_s1 = np.linalg.norm(s1)
                    s1 /= norm_s1 if norm_s1 > 0 else 1.0
                    s2 = np.array([math.sin(phi[ip,j]) * math.cos(theta[ip,j]),
                                   math.sin(phi[ip,j]) * math.sin(theta[ip,j]),
                                   math.cos(phi[ip,j])])
                    norm_s2 = np.linalg.norm(s2)
                    s2 /= norm_s2 if norm_s2 > 0 else 1.0
                    s3 = np.array([math.sin(phi[i,jp]) * math.cos(theta[i,jp]),
                                   math.sin(phi[i,jp]) * math.sin(theta[i,jp]),
                                   math.cos(phi[i,jp])])
                    norm_s3 = np.linalg.norm(s3)
                    s3 /= norm_s3 if norm_s3 > 0 else 1.0
                    charge_density[i,j] = np.dot(s1, np.cross(s2, s3)) / (4 * math.pi)
            im2 = ax2.imshow(charge_density, cmap='viridis', interpolation='bilinear', origin='lower')
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_xticks(np.arange(0, self.N, 5))
            ax2.set_yticks(np.arange(0, self.N, 5))
            ax2.set_xlabel('j (Horizontal)', fontsize=12)
            ax2.set_ylabel('i (Vertical)', fontsize=12)
            ax2.set_title('Topological Charge Density', fontsize=14, fontweight='bold', pad=10)
            cbar2 = plt.colorbar(im2, ax=ax2, label='Charge Density')
            cbar2.ax.tick_params(labelsize=10)
            cbar2.set_label('Charge Density', fontsize=12)
        else:
            # Plot only spin configuration
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(sz/1000.0, cmap='RdBu', vmin=-1, vmax=1, interpolation='bilinear', origin='lower')  # Normalize for display
            x, y = np.meshgrid(np.arange(self.N), np.arange(self.N))
            ax.quiver(x, y, sx/1000.0, sy/1000.0, scale=30, width=0.0015, color='black')  # Normalize for arrows
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks(np.arange(0, self.N, 5))
            ax.set_yticks(np.arange(0, self.N, 5))
            ax.set_xlabel('j (Horizontal)', fontsize=12)
            ax.set_ylabel('i (Vertical)', fontsize=12)
            ax.set_title(
                f"{title}\n(N={self.N}, Dx={self.Dx:.1e}, B={self.B:.1f}, E={self.energy():.2f}, Q={self.compute_topological_charge():.2f})",
                fontsize=14, fontweight='bold', pad=10
            )
            cbar = plt.colorbar(im, ax=ax, label='Sz Component (Normalized)', ticks=[-1, 0, 1])
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Sz Component (Normalized)', fontsize=12)

        plt.tight_layout()
        plt.show()

# Main simulation
if __name__ == "__main__":
    # Initialize system with parameters tuned for Skyrmion formation
    system = SpinSystem(
        N=65,          # Grid size#      
        Jy=6.0e-3,     # Ferromagnetic exchange
        Dx=1.4e-3,    # DMI (increased for chirality)
        Dy=1.4e-3,
        K1=0,
        K2=0,# Perpendicular anisotropy
        B=3,         # Magnetic field magnitude
        r0=0.1         # Downward field radius
    )


    
    # Initialize with a single Skyrmion
    system.random_initialize()
    
    print("Initial state:")
    initial_charge = system.compute_topological_charge()
    print(f"Initial topological charge: {initial_charge:.3f}")
    system.visualize(title="Initial Spin Configuration", show_charge_density=True)
    
    print("Starting simulated annealing")
    # Run simulated annealing
    system.simulate(T_start=1.0, T_end=0.01, steps=10, sweeps=10000)
    
    print("Final state:")
    final_energy = system.energy()
    final_charge = system.compute_topological_charge()
    print(f"Final energy: {final_energy:.3f}")
    print(f"Final topological charge: {final_charge:.3f}")
    system.visualize(title="Optimized Spin Configuration", show_charge_density=True)