import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

# CUDA device function to compute local energy at site (i, j) for a specified layer
@cuda.jit(device=True)
def compute_local_energy(layer, i, j, phi1, theta1, phi2, theta2, N, Jx1, Jy1, Dx1, Dy1, K1, B_field1, Jx2, Jy2, Dx2, Dy2, K2, B_field2, J_AFM):
    if layer == 0:
        Jx, Jy, Dx, Dy, K, B_field = Jx1, Jy1, Dx1, Dy1, K1, B_field1
        phi, theta = phi1, theta1
        phi_other, theta_other = phi2, theta2
    else:
        Jx, Jy, Dx, Dy, K, B_field = Jx2, Jy2, Dx2, Dy2, K2, B_field2
        phi, theta = phi2, theta2
        phi_other, theta_other = phi1, theta1

    sx = math.sin(phi[i, j]) * math.cos(theta[i, j])
    sy = math.sin(phi[i, j]) * math.sin(theta[i, j])
    sz = math.cos(phi[i, j])

    sx_other = math.sin(phi_other[i, j]) * math.cos(theta_other[i, j])
    sy_other = math.sin(phi_other[i, j]) * math.sin(theta_other[i, j])
    sz_other = math.cos(phi_other[i, j])

    right_i = (i + 1) % N
    sx_r = math.sin(phi[right_i, j]) * math.cos(theta[right_i, j])
    sy_r = math.sin(phi[right_i, j]) * math.sin(theta[right_i, j])
    sz_r = math.cos(phi[right_i, j])

    up_j = (j + 1) % N
    sx_u = math.sin(phi[i, up_j]) * math.cos(theta[i, up_j])
    sy_u = math.sin(phi[i, up_j]) * math.sin(theta[i, up_j])
    sz_u = math.cos(phi[i, up_j])

    ex_energy = -Jx * (sx * sx_r + sy * sy_r + sz * sz_r) - Jy * (sx * sx_u + sy * sy_u + sz * sz_u)
    dmi_energy = -Dx * (sz * sx_r - sx * sz_r) + Dy * (sy * sz_u - sz * sy_u)
    anis_energy = -K * (sz * sz)  # Corrected for perpendicular anisotropy
    zeeman_energy = -B_field[i, j] * sz
    coupling_energy = -J_AFM * (sx * sx_other + sy * sy_other + sz * sz_other)

    return ex_energy + dmi_energy + anis_energy + zeeman_energy + coupling_energy

# CUDA kernel for one Monte Carlo step
@cuda.jit
def monte_carlo_step(layer, phi1, theta1, phi2, theta2, N, T, rng_states, parity, Jx1, Jy1, Dx1, Dy1, K1, B_field1, Jx2, Jy2, Dx2, Dy2, K2, B_field2, J_AFM):
    i, j = cuda.grid(2)
    if i < N and j < N and (i + j) % 2 == parity:
        idx = i * N + j
        phi = phi1 if layer == 0 else phi2
        theta = theta1 if layer == 0 else theta2
        old_phi = phi[i, j]
        old_theta = theta[i, j]
        old_energy = compute_local_energy(layer, i, j, phi1, theta1, phi2, theta2, N, Jx1, Jy1, Dx1, Dy1, K1, B_field1, Jx2, Jy2, Dx2, Dy2, K2, B_field2, J_AFM)
        phi_new = old_phi + (xoroshiro128p_uniform_float32(rng_states, idx) - 0.5) * 0.05
        theta_new = old_theta + (xoroshiro128p_uniform_float32(rng_states, idx) - 0.5) * 0.05
        phi[i, j] = phi_new
        theta[i, j] = theta_new
        new_energy = compute_local_energy(layer, i, j, phi1, theta1, phi2, theta2, N, Jx1, Jy1, Dx1, Dy1, K1, B_field1, Jx2, Jy2, Dx2, Dy2, K2, B_field2, J_AFM)
        delta_E = new_energy - old_energy
        if delta_E > 0 and xoroshiro128p_uniform_float32(rng_states, idx) >= math.exp(-delta_E / T):
            phi[i, j] = old_phi
            theta[i, j] = old_theta

# CUDA kernel to compute total energy
@cuda.jit
def energy_kernel(phi1, theta1, phi2, theta2, N, Jx1, Jy1, Dx1, Dy1, K1, B_field1, Jx2, Jy2, Dx2, Dy2, K2, B_field2, J_AFM, energy):
    i, j = cuda.grid(2)
    if i < N and j < N:
        # Layer 1
        sx1 = math.sin(phi1[i, j]) * math.cos(theta1[i, j])
        sy1 = math.sin(phi1[i, j]) * math.sin(theta1[i, j])
        sz1 = math.cos(phi1[i, j])
        right_i = (i + 1) % N
        sx1_r = math.sin(phi1[right_i, j]) * math.cos(theta1[right_i, j])
        sy1_r = math.sin(phi1[right_i, j]) * math.sin(theta1[right_i, j])
        sz1_r = math.cos(phi1[right_i, j])
        up_j = (j + 1) % N
        sx1_u = math.sin(phi1[i, up_j]) * math.cos(theta1[i, up_j])
        sy1_u = math.sin(phi1[i, up_j]) * math.sin(theta1[i, up_j])
        sz1_u = math.cos(phi1[i, up_j])
        ex1 = -Jx1 * (sx1 * sx1_r + sy1 * sy1_r + sz1 * sz1_r) - Jy1 * (sx1 * sx1_u + sy1 * sy1_u + sz1 * sz1_u)
        dmi1 = -Dx1 * (sz1 * sx1_r - sx1 * sz1_r) + Dy1 * (sy1 * sz1_u - sz1 * sy1_u)
        anis1 = -K1 * (sz1 * sz1)
        zeeman1 = -B_field1[i, j] * sz1

        # Layer 2
        sx2 = math.sin(phi2[i, j]) * math.cos(theta2[i, j])
        sy2 = math.sin(phi2[i, j]) * math.sin(theta2[i, j])
        sz2 = math.cos(phi2[i, j])
        sx2_r = math.sin(phi2[right_i, j]) * math.cos(theta2[right_i, j])
        sy2_r = math.sin(phi2[right_i, j]) * math.sin(theta2[right_i, j])
        sz2_r = math.cos(phi2[right_i, j])
        sx2_u = math.sin(phi2[i, up_j]) * math.cos(theta2[i, up_j])
        sy2_u = math.sin(phi2[i, up_j]) * math.sin(theta2[i, up_j])
        sz2_u = math.cos(phi2[i, up_j])
        ex2 = -Jx2 * (sx2 * sx2_r + sy2 * sy2_r + sz2 * sz2_r) - Jy2 * (sx2 * sx2_u + sy2 * sy2_u + sz2 * sz2_u)
        dmi2 = -Dx2 * (sz2 * sx2_r - sx2 * sz2_r) + Dy2 * (sy2 * sz2_u - sz2 * sy2_u)
        anis2 = -K2 * (sz2 * sz2)
        zeeman2 = -B_field2[i, j] * sz2

        # Coupling energy
        coupling = -J_AFM * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
        total = ex1 + dmi1 + anis1 + zeeman1 + ex2 + dmi2 + anis2 + zeeman2 + coupling
        cuda.atomic.add(energy, 0, total)

class BilayerSpinSystem:
    def __init__(self, N, Jx1, Jy1, Dx1, Dy1, K1, B_field1, Jx2, Jy2, Dx2, Dy2, K2, B_field2, J_AFM):
        self.N = N
        self.Jx1, self.Jy1, self.Dx1, self.Dy1, self.K1 = Jx1, Jy1, Dx1, Dy1, K1
        self.B_field1 = cuda.to_device(B_field1)
        self.Jx2, self.Jy2, self.Dx2, self.Dy2, self.K2 = Jx2, Jy2, Dx2, Dy2, K2
        self.B_field2 = cuda.to_device(B_field2)
        self.J_AFM = J_AFM
        self.phi1 = cuda.device_array((N, N), dtype=np.float32)
        self.theta1 = cuda.device_array((N, N), dtype=np.float32)
        self.phi2 = cuda.device_array((N, N), dtype=np.float32)
        self.theta2 = cuda.device_array((N, N), dtype=np.float32)
        self.rng_states = create_xoroshiro128p_states(N * N, seed=np.random.randint(0, 10000))
        self.d_energy = cuda.device_array(1, dtype=np.float32)

    def random_initialize(self):
        threads = (16, 16)
        blocks = ((self.N + 15) // 16, (self.N + 15) // 16)

        @cuda.jit
        def init_kernel(phi, theta, rng_states, N):
            i, j = cuda.grid(2)
            if i < N and j < N:
                idx = i * N + j
                phi[i, j] = xoroshiro128p_uniform_float32(rng_states, idx) * math.pi
                theta[i, j] = xoroshiro128p_uniform_float32(rng_states, idx) * 2 * math.pi

        init_kernel[blocks, threads](self.phi1, self.theta1, self.rng_states, self.N)
        init_kernel[blocks, threads](self.phi2, self.theta2, self.rng_states, self.N)
        cuda.synchronize()

    def skyrmion_initialize(self, center_x, center_y, r):
        threads = (16, 16)
        blocks = ((self.N + 15) // 16, (self.N + 15) // 16)

        @cuda.jit
        def init_kernel(phi1, theta1, phi2, theta2, N, cx, cy, r):
            i, j = cuda.grid(2)
            if i < N and j < N:
                dist = math.sqrt((i - cy) ** 2 + (j - cx) ** 2)
                if dist < r:
                    phi1[i, j] = 0.0  # Up for layer 1
                    phi2[i, j] = math.pi  # Down for layer 2
                else:
                    phi1[i, j] = math.pi  # Down for layer 1
                    phi2[i, j] = 0.0  # Up for layer 2
                theta1[i, j] = 0.0
                theta2[i, j] = 0.0

        init_kernel[blocks, threads](self.phi1, self.theta1, self.phi2, self.theta2, self.N, center_x, center_y, r)
        cuda.synchronize()

    def simulate(self, T_start=1.0, T_end=0.01, steps=500, sweeps=2000):
        threads = (16, 16)
        blocks = ((self.N + 15) // 16, (self.N + 15) // 16)
        energies = []
        charges1 = []
        charges2 = []
        for step in range(steps):
            T = T_start + (T_end - T_start) * step / (steps - 1)
            for _ in range(sweeps):
                for parity in [0, 1]:
                    for layer in [0, 1]:
                        monte_carlo_step[blocks, threads](
                            layer, self.phi1, self.theta1, self.phi2, self.theta2, self.N, T, self.rng_states, parity,
                            self.Jx1, self.Jy1, self.Dx1, self.Dy1, self.K1, self.B_field1,
                            self.Jx2, self.Jy2, self.Dx2, self.Dy2, self.K2, self.B_field2, self.J_AFM
                        )
            energy = self.energy()
            charge1 = self.compute_topological_charge(0)
            charge2 = self.compute_topological_charge(1)
            energies.append(energy)
            charges1.append(charge1)
            charges2.append(charge2)
            print(f"Step {step+1}/{steps}, T={T:.3f}, E={energy:.4f}, Q1={charge1:.3f}, Q2={charge2:.3f}")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        ax1.plot(energies)
        ax1.set_title('Energy vs Temperature Step')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Energy')
        ax2.plot(charges1)
        ax2.set_title('Q1 vs Temperature Step')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Q Layer 1')
        ax3.plot(charges2)
        ax3.set_title('Q2 vs Temperature Step')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Q Layer 2')
        plt.tight_layout()
        plt.savefig('simulation_results.png')
        plt.close()

    def energy(self):
        self.d_energy[0] = 0.0
        threads = (16, 16)
        blocks = ((self.N + 15) // 16, (self.N + 15) // 16)
        energy_kernel[blocks, threads](
            self.phi1, self.theta1, self.phi2, self.theta2, self.N,
            self.Jx1, self.Jy1, self.Dx1, self.Dy1, self.K1, self.B_field1,
            self.Jx2, self.Jy2, self.Dx2, self.Dy2, self.K2, self.B_field2,
            self.J_AFM, self.d_energy
        )
        return self.d_energy.copy_to_host()[0]

    def compute_topological_charge(self, layer):
        phi = self.phi1.copy_to_host() if layer == 0 else self.phi2.copy_to_host()
        theta = self.theta1.copy_to_host() if layer == 0 else self.theta2.copy_to_host()
        Q = 0.0
        for i in range(self.N):
            for j in range(self.N):
                ip = (i + 1) % self.N
                jp = (j + 1) % self.N
                s1 = np.array([math.sin(phi[i, j]) * math.cos(theta[i, j]),
                              math.sin(phi[i, j]) * math.sin(theta[i, j]),
                              math.cos(phi[i, j])])
                s2 = np.array([math.sin(phi[ip, j]) * math.cos(theta[ip, j]),
                              math.sin(phi[ip, j]) * math.sin(theta[ip, j]),
                              math.cos(phi[ip, j])])
                s3 = np.array([math.sin(phi[i, jp]) * math.cos(theta[i, jp]),
                              math.sin(phi[i, jp]) * math.sin(theta[i, jp]),
                              math.cos(phi[i, jp])])
                Q += np.dot(s1, np.cross(s2, s3))
        return Q / (4 * math.pi)

    def visualize(self, layer, title=""):
        phi = self.phi1.copy_to_host() if layer == 0 else self.phi2.copy_to_host()
        theta = self.theta1.copy_to_host() if layer == 0 else self.theta2.copy_to_host()
        sz = np.cos(phi)
        sx = np.sin(phi) * np.cos(theta)
        sy = np.sin(phi) * np.sin(theta)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(sz, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
        x, y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        ax.quiver(x, y, sx, sy, scale=20)
        ax.set_title(f"{title} (Q={self.compute_topological_charge(layer):.2f})")
        ax.set_xlabel('j')
        ax.set_ylabel('i')
        plt.colorbar(im, ax=ax, label='Sz')
        plt.savefig(f'{title.lower().replace(" ", "_")}.png')
        plt.close()

def simulate_system(num=1):
    N = 64
    r = 10
    B_field1 = np.full((N, N), 2.0, dtype=np.float32)
    if num == 1:
        system = BilayerSpinSystem(
            N=N, Jx1=4.0, Jy1=4.0, Dx1=3.05, Dy1=3.05, K1=0.0, B_field1=B_field1,
            Jx2=0.0, Jy2=0.0, Dx2=0.0, Dy2=0.0, K2=0.0, B_field2=np.zeros((N, N), dtype=np.float32),
            J_AFM=0.0
        )
    elif num == 2:
        B_field2 = np.zeros((N, N), dtype=np.float32)
        system = BilayerSpinSystem(
            N=N, Jx1=4.0, Jy1=4.0, Dx1=3.05, Dy1=3.05, K1=0.0, B_field1=B_field1,
            Jx2=4.0, Jy2=4.0, Dx2=-3 * 3.05, Dy2=-3 * 3.05, K2=0.0032, B_field2=B_field2,
            J_AFM=-1.6
        )
    else:
        raise ValueError("num must be 1 or 2")

    system.skyrmion_initialize(N // 2, N // 2, r)
    system.visualize(0, "Initial Layer 1")
    if num == 2:
        system.visualize(1, "Initial Layer 2")
    system.simulate()
    system.visualize(0, "Final Layer 1")
    if num == 2:
        system.visualize(1, "Final Layer 2")

if __name__ == "__main__":
    simulate_system(num=2)
