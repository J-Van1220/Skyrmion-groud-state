import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import time

#def vector spin as function of phi and theta 
@cuda.jit(device=True)
def S(x, y, phi, theta):
    sx = math.sin(phi[x, y]) * math.cos(theta[x, y])
    sy = math.sin(phi[x, y]) * math.sin(theta[x, y])
    sz = math.cos(phi[x, y])
    return sx, sy, sz

@cuda.jit
def compute_layer_energy(phi, theta, N, Jx, Jy, Dx, Dy, K, A, Bx, By, Bz, energy):
    i, j = cuda.grid(2)#all i,j in 64*64 grid
    if i < N and j < N:
        sx_i, sy_i, sz_i = S(i, j, phi, theta)
        sx_ip1, sy_ip1, sz_ip1 = S((i + 1) % N, j, phi, theta)
        sx_jp1, sy_jp1, sz_jp1 = S(i, (j + 1) % N, phi, theta)#periodic boundary condition
        
        ex = -Jx * (sx_i * sx_ip1 + sy_i * sy_ip1 + sz_i * sz_ip1)
        ex += -Jy * (sx_i * sx_jp1 + sy_i * sy_jp1 + sz_i * sz_jp1)
        # Néel-type interfacial DMI (discrete):
        # E_DMI = Dx * ( sz_i * sx_{i+1} - sx_i * sz_{i+1} ) + Dy * ( sz_i * sy_{j+1} - sy_i * sz_{j+1} )
        dmi = Dx * (sz_i * sx_ip1 - sx_i * sz_ip1) + Dy * (sz_i * sy_jp1 - sy_i * sz_jp1)
        anis = K * sz_i * sz_i
        ani = -A *(sy_i*sy_ip1+sx_i*sx_jp1)
        zeeman = -(Bx[i, j] * sx_i + By[i, j] * sy_i + Bz[i, j] * sz_i)
        cuda.atomic.add(energy, 0, ex + dmi + anis + zeeman+ani)#adding all energy to the first element then return

        #pure math, partial partial partial
@cuda.jit
def compute_layer_gradient(phi, theta, N, Jx, Jy, Dx, Dy, K, A, Bx, By, Bz, grad_phi, grad_theta):
    i, j = cuda.grid(2)
    if i < N and j < N:
        sx, sy, sz = S(i, j, phi, theta)
        dsx_dphi = math.cos(phi[i, j]) * math.cos(theta[i, j])
        dsy_dphi = math.cos(phi[i, j]) * math.sin(theta[i, j])
        dsz_dphi = -math.sin(phi[i, j])
        dsx_dtheta = -math.sin(phi[i, j]) * math.sin(theta[i, j])
        dsy_dtheta = math.sin(phi[i, j]) * math.cos(theta[i, j])
        dsz_dtheta = 0.0
        
        dsx_dphi_ip1 = math.cos(phi[(i+1)%N, j]) * math.cos(theta[(i+1)%N, j])
        dsy_dphi_ip1 = math.cos(phi[(i+1)%N, j]) * math.sin(theta[(i+1)%N, j])
        dsz_dphi_ip1 = -math.sin(phi[(i+1)%N, j])
        dsx_dtheta_ip1 = -math.sin(phi[(i+1)%N, j]) * math.sin(theta[(i+1)%N, j])
        dsy_dtheta_ip1 = math.sin(phi[(i+1)%N, j]) * math.cos(theta[(i+1)%N, j])
        dsz_dtheta_ip1 = 0.0
        
        dsx_dphi_jp1 = math.cos(phi[i, (j+1)%N]) * math.cos(theta[i, (j+1)%N])
        dsy_dphi_jp1 = math.cos(phi[i, (j+1)%N]) * math.sin(theta[i, (j+1)%N])
        dsz_dphi_jp1 = -math.sin(phi[i, (j+1)%N])
        dsx_dtheta_jp1 = -math.sin(phi[i, (j+1)%N]) * math.sin(theta[i, (j+1)%N])
        dsy_dtheta_jp1 = math.sin(phi[i, (j+1)%N]) * math.cos(theta[i, (j+1)%N])
        dsz_dtheta_jp1 = 0.0
        
        sx_ip1, sy_ip1, sz_ip1 = S((i + 1) % N, j, phi, theta)
        sx_im1, sy_im1, sz_im1 = S((i - 1) % N, j, phi, theta)
        sx_jp1, sy_jp1, sz_jp1 = S(i, (j + 1) % N, phi, theta)
        sx_jm1, sy_jm1, sz_jm1 = S(i, (j - 1) % N, phi, theta)
        
        S_neigh_x = Jx * (sx_ip1 + sx_im1) + Jy * (sx_jp1 + sx_jm1)
        S_neigh_y = Jx * (sy_ip1 + sy_im1) + Jy * (sy_jp1 + sy_jm1)
        S_neigh_z = Jx * (sz_ip1 + sz_im1) + Jy * (sz_jp1 + sz_jm1)
        
        grad_phi_ex = -(S_neigh_x * dsx_dphi + S_neigh_y * dsy_dphi + S_neigh_z * dsz_dphi)
        grad_theta_ex = -(S_neigh_x * dsx_dtheta + S_neigh_y * dsy_dtheta + S_neigh_z * dsz_dtheta)
        # --- Néel DMI gradient ---
        # dE/dSx = Dx * (sz_im1 - sz_ip1)
        # dE/dSy = Dy * (sz_jm1 - sz_jp1)
        # dE/dSz = Dx * (sx_ip1 - sx_im1) + Dy * (sy_jp1 - sy_jm1)
        dE_dSx = Dx * (sz_im1 - sz_ip1)
        dE_dSy = Dy * (sz_jm1 - sz_jp1)
        dE_dSz = Dx * (sx_ip1 - sx_im1) + Dy * (sy_jp1 - sy_jm1)
        #grad_phi_dmi = dE_dSx * dsx_dphi + dE_dSy * dsy_dphi + dE_dSz * dsz_dphi
        #grad_theta_dmi = dE_dSx * dsx_dtheta + dE_dSy * dsy_dtheta + dE_dSz * dsz_dtheta
        grad_phi_dmi = -Dx * (dsx_dphi * (sz_ip1-sz_im1) + sx * (sz_ip1-sz_im1) * dsz_dphi - sz * (sx_ip1-sx_im1) * dsx_dphi - dsz_dphi * (sx_ip1-sx_im1))
        grad_phi_dmi += Dy * (dsy_dphi * (sz_jp1-sz_jm1) + sy * (sz_jp1-sz_jm1) * dsz_dphi - sz * (sy_jp1-sy_jm1) * dsy_dphi - dsz_dphi * (sy_jp1-sy_jm1))
        grad_theta_dmi = -Dx * (dsx_dtheta * (sz_ip1-sz_im1) + sx * (sz_ip1-sz_im1) * dsz_dtheta - sz * (sx_ip1-sx_im1) * dsx_dtheta - dsz_dtheta * (sx_ip1-sx_im1))
        grad_theta_dmi += Dy * (dsy_dtheta * (sz_jp1-sz_jm1) + sy * (sz_jp1-sz_jm1) * dsz_dtheta - sz * (sy_jp1-sy_jm1) * dsy_dtheta - dsz_dtheta * (sy_jp1-sy_jm1))
       
        grad_phi_K = 2 * K * sz * dsz_dphi
        grad_theta_K = 0.0
        
        grad_phi_A= -A*(dsx_dphi*sx_jp1+dsx_dphi_jp1*sx+dsy_dphi*sy_ip1+dsy_dphi_ip1*sy)
        grad_theta_A= -A*(dsx_dtheta*sx_jp1+dsx_dtheta_jp1*sx+dsy_dtheta*sy_ip1+dsy_dtheta_ip1*sy)
        
        grad_phi_zeeman = -(Bx[i, j] * dsx_dphi + By[i, j] * dsy_dphi + Bz[i, j] * dsz_dphi)
        grad_theta_zeeman = -(Bx[i, j] * dsx_dtheta + By[i, j] * dsy_dtheta + Bz[i, j] * dsz_dtheta)
        
        grad_phi[i, j] = grad_phi_ex + grad_phi_dmi + grad_phi_K + grad_phi_zeeman+grad_phi_A
        grad_theta[i, j] = grad_theta_ex + grad_theta_dmi + grad_theta_K + grad_theta_zeeman+grad_theta_A

#coupling energy
@cuda.jit
def compute_coupling_energy(phi1, theta1, phi2, theta2, N, J_AFM, energy):
    i, j = cuda.grid(2)
    if i < N and j < N:
        sx1, sy1, sz1 = S(i, j, phi1, theta1)
        sx2, sy2, sz2 = S(i, j, phi2, theta2)
        coupling = -J_AFM * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
        cuda.atomic.add(energy, 0, coupling)
# partial partial partial 
@cuda.jit
def compute_coupling_gradient_kernel(phi1, theta1, phi2, theta2, N, J_AFM, grad_phi1, grad_theta1, grad_phi2, grad_theta2):
    i, j = cuda.grid(2)
    if i < N and j < N:
        sx1, sy1, sz1 = S(i, j, phi1, theta1)
        dsx1_dphi = math.cos(phi1[i, j]) * math.cos(theta1[i, j])
        dsy1_dphi = math.cos(phi1[i, j]) * math.sin(theta1[i, j])
        dsz1_dphi = -math.sin(phi1[i, j])
        dsx1_dtheta = -math.sin(phi1[i, j]) * math.sin(theta1[i, j])
        dsy1_dtheta = math.sin(phi1[i, j]) * math.cos(theta1[i, j])
        dsz1_dtheta = 0.0
        
        sx2, sy2, sz2 = S(i, j, phi2, theta2)
        dsx2_dphi = math.cos(phi2[i, j]) * math.cos(theta2[i, j])
        dsy2_dphi = math.cos(phi2[i, j]) * math.sin(theta2[i, j])
        dsz2_dphi = -math.sin(phi2[i, j])
        dsx2_dtheta = -math.sin(phi2[i, j]) * math.sin(theta2[i, j])
        dsy2_dtheta = math.sin(phi2[i, j]) * math.cos(theta2[i, j])
        dsz2_dtheta = 0.0
        
        grad_phi_coupling1 = -J_AFM * (sx2 * dsx1_dphi + sy2 * dsy1_dphi + sz2 * dsz1_dphi)
        grad_theta_coupling1 = -J_AFM * (sx2 * dsx1_dtheta + sy2 * dsy1_dtheta + sz2 * dsz1_dtheta)
        grad_phi_coupling2 = -J_AFM * (sx1 * dsx2_dphi + sy1 * dsy2_dphi + sz1 * dsz2_dphi)
        grad_theta_coupling2 = -J_AFM * (sx1 * dsx2_dtheta + sy1 * dsy2_dtheta + sz1 * dsz2_dtheta)
        
        grad_phi1[i, j] = grad_phi_coupling1
        grad_theta1[i, j] = grad_theta_coupling1
        grad_phi2[i, j] = grad_phi_coupling2
        grad_theta2[i, j] = grad_theta_coupling2
        
#class Layer
class Layer:
    #construct a layer with parameters 
    def __init__(self, N, Jx, Jy, Dx, Dy, K, A, Bx, By, Bz):
        self.N = N
        self.phi = np.zeros((N, N), dtype=np.float64) #initialize N*N matrix to store phi
        self.theta = np.zeros((N, N), dtype=np.float64) #initialize N*N matrix to store theta 
        self.Jx = Jx#exchange
        self.Jy = Jy
        self.Dx = Dx#DMI
        self.Dy = Dy
        self.K = K#ani_z
        self.A =A#ani_xy
        self.Bx = Bx
        self.By = By#for future develop, no use in current condition
        self.Bz = Bz#z compo of magnetic field

#transient parameters, distrubute resources, compute layer energy
    def compute_energy(self):
        phi_d = cuda.to_device(self.phi)
        theta_d = cuda.to_device(self.theta)
        Bx_d = cuda.to_device(self.Bx)
        By_d = cuda.to_device(self.By)
        Bz_d = cuda.to_device(self.Bz)#transient parameters
        energy_d = cuda.to_device(np.array([0.0]))#initialize to store
        #thread_x_total=block_x*16+thread_x_local
        threadsperblock = (32, 32)#one block contain 16*16 thread
        blockspergrid_x = math.ceil(self.N / threadsperblock[0])#grid dimension x
        blockspergrid_y = math.ceil(self.N / threadsperblock[1])#grid dimension y
        blockspergrid = (blockspergrid_x, blockspergrid_y)#grid size,contain how many blocks
        
        #compute layer energy
        compute_layer_energy[blockspergrid, threadsperblock](
            phi_d, theta_d, self.N, self.Jx, self.Jy, self.Dx, self.Dy, self.K, self.A,
            Bx_d, By_d, Bz_d, energy_d
        )
        
        return energy_d.copy_to_host()[0]#return the first element of energy_d

#transient parameters, distrubute resources, compute layer energy gradient
    def compute_gradient(self):
        phi_d = cuda.to_device(self.phi)
        theta_d = cuda.to_device(self.theta)
        Bx_d = cuda.to_device(self.Bx)
        By_d = cuda.to_device(self.By)
        Bz_d = cuda.to_device(self.Bz)
        grad_phi_d = cuda.device_array((self.N, self.N), dtype=np.float64)
        grad_theta_d = cuda.device_array((self.N, self.N), dtype=np.float64)
        
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(self.N / threadsperblock[0])
        blockspergrid_y = math.ceil(self.N / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        compute_layer_gradient[blockspergrid, threadsperblock](
            phi_d, theta_d, self.N, self.Jx, self.Jy, self.Dx, self.Dy, self.K, self.A,
            Bx_d, By_d, Bz_d, grad_phi_d, grad_theta_d
        )
        
        return grad_phi_d.copy_to_host(), grad_theta_d.copy_to_host()
    
# class multilayersystem
class MultilayerSystem:
    def __init__(self, layers, J_AFM=0.0):
        self.layers = layers
        self.J_AFM = J_AFM#coupling effect
#adding coupling energy to layer if there are more than one layer and J_AFM not equal to
    def compute_total_energy(self):
        total_energy = sum(layer.compute_energy() for layer in self.layers)
        if len(self.layers) > 1 and self.J_AFM != 0:
            total_energy += self.compute_coupling_energy()
        return total_energy
#similar to compute_layer energy
    def compute_coupling_energy(self):
        energy_d = cuda.to_device(np.array([0.0]))
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(self.layers[0].N / threadsperblock[0])
        blockspergrid_y = math.ceil(self.layers[0].N / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        compute_coupling_energy[blockspergrid, threadsperblock](
            self.layers[0].phi, self.layers[0].theta,
            self.layers[1].phi, self.layers[1].theta,
            self.layers[0].N, self.J_AFM, energy_d
        )
        return energy_d.copy_to_host()[0]
    

#the code is not elegant here, becuase I write the single layer first, based on which I wrtie multilayer  #idea: all trans into 1D array      
    
# adding coupling gradient to layer gradient if more than one layer and J_AFM not equal to 0 
    def compute_total_gradient(self):
        gradients = []
        layer_grads = [layer.compute_gradient() for layer in self.layers]
        if len(self.layers) > 1 and self.J_AFM != 0:
            coupling_grads = self.compute_coupling_gradient()
            for i, (grad_phi, grad_theta) in enumerate(layer_grads):
                grad_phi += coupling_grads[i][0]
                grad_theta += coupling_grads[i][1]
                gradients.append(np.concatenate([grad_phi.flatten(), grad_theta.flatten()])) 
        else:
            gradients = [np.concatenate([grad_phi.flatten(), grad_theta.flatten()]) 
                         for grad_phi, grad_theta in layer_grads] 
        return np.concatenate(gradients) 
#
    def compute_coupling_gradient(self):
        phi1_d = cuda.to_device(self.layers[0].phi)
        theta1_d = cuda.to_device(self.layers[0].theta)
        phi2_d = cuda.to_device(self.layers[1].phi)
        theta2_d = cuda.to_device(self.layers[1].theta)
        grad_phi1_d = cuda.device_array((self.layers[0].N, self.layers[0].N), dtype=np.float64)
        grad_theta1_d = cuda.device_array((self.layers[0].N, self.layers[0].N), dtype=np.float64)
        grad_phi2_d = cuda.device_array((self.layers[1].N, self.layers[1].N), dtype=np.float64)
        grad_theta2_d = cuda.device_array((self.layers[1].N, self.layers[1].N), dtype=np.float64)
        
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(self.layers[0].N / threadsperblock[0])
        blockspergrid_y = math.ceil(self.layers[0].N / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        compute_coupling_gradient_kernel[blockspergrid, threadsperblock](
            phi1_d, theta1_d, phi2_d, theta2_d, self.layers[0].N, self.J_AFM,
            grad_phi1_d, grad_theta1_d, grad_phi2_d, grad_theta2_d
        )
        
        return [(grad_phi1_d.copy_to_host(), grad_theta1_d.copy_to_host()),
                (grad_phi2_d.copy_to_host(), grad_theta2_d.copy_to_host())]
    
#initialize an 1D array, the last step phi and theat, on which to manipulate gradient descent 
    def get_parameters(self):
        return np.concatenate([np.concatenate([layer.phi.flatten(), layer.theta.flatten()]) 
                               for layer in self.layers])
    
#distribute new delta to each phi and theta and reconstruct N N matrix
    def set_parameters(self, para):
        N = self.layers[0].N
        for i, layer in enumerate(self.layers):
            start = i * 2 * N * N
            layer.phi = para[start:start + N*N].reshape(N, N)
            layer.theta = para[start + N*N:start + 2*N*N].reshape(N, N)
            
#gradient descent
    def gradient_descent(self, rho=0.95, epsilon=1e-6, max_iter=1000, tol=1e-8,
                         save_frames=False, frame_every=10, frame_dir="frames",
                         animate=False, animation_path="optimization.gif"):
        para = self.get_parameters()
        accum_grad = np.zeros_like(para)
        accum_delta = np.zeros_like(para)
        energy_history = []
        if save_frames:
            import os
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            # 保存初始帧
            for li, layer in enumerate(self.layers):
                plt.figure(figsize=(5,4))
                plt.imshow(np.cos(layer.phi), cmap='RdBu', vmin=-1, vmax=1, origin='lower')
                plt.title(f'Layer {li+1} iter 0')
                plt.colorbar(label='S_z')
                plt.tight_layout()
                plt.savefig(f"{frame_dir}/layer{li+1}_iter0000.png")
                plt.close()
        # 预先计算初始能量
        initial_energy = self.compute_total_energy()
        energy_history.append(initial_energy)
        for it in range(max_iter):
            grad = self.compute_total_gradient()
            accum_grad = rho * accum_grad + (1 - rho) * grad ** 2
            rms_grad = np.sqrt(accum_grad + epsilon)
            delta = - (np.sqrt(accum_delta + epsilon) / rms_grad) * grad
            accum_delta = rho * accum_delta + (1 - rho) * delta ** 2
            para += delta
            self.set_parameters(para)
            # 记录能量
            current_energy = self.compute_total_energy()
            energy_history.append(current_energy)
            # 保存帧
            if save_frames and (it + 1) % frame_every == 0:
                for li, layer in enumerate(self.layers):
                    plt.figure(figsize=(5,4))
                    plt.imshow(np.cos(layer.phi), cmap='RdBu', vmin=-1, vmax=1, origin='lower')
                    plt.title(f'Layer {li+1} iter {it+1}')
                    plt.colorbar(label='S_z')
                    plt.tight_layout()
                    plt.savefig(f"{frame_dir}/layer{li+1}_iter{it+1:04d}.png")
                    plt.close()
            if np.max(np.abs(grad)) < tol:
                print(f"Converged after {it+1} iterations")
                break
            if it == max_iter - 1:
                print("Maximum iterations reached")
        final_energy = energy_history[-1]
        
        return final_energy, energy_history
    
#initalize spin
    def initialize_spins(self, method="skyrmion", r=10, alignment='invert'):
        epsilon = 1e-4
        N = self.layers[0].N
        if method == "skyrmion":
            center = N // 2
            for i in range(N):
                for j in range(N):
                    inside = (i - center)**2 + (j - center)**2 < r**2
                    if inside:
                        self.layers[0].phi[i, j] = 0.0
                        self.layers[0].theta[i, j] = 0.0
                    else:
                        self.layers[0].phi[i, j] = np.pi
                        self.layers[0].theta[i, j] = 0.0
        elif method == "random":
            self.layers[0].phi = np.random.uniform(0, np.pi, (N, N))  # restrict to [0,pi]
            self.layers[0].theta = np.random.uniform(0, 2 * np.pi, (N, N))
        else:
            raise ValueError("Initialization method must be 'skyrmion' or 'random'")

        if len(self.layers) <= 1:
            return

        for li in range(1, len(self.layers)):
            layer = self.layers[li]
            if alignment in ("same", "copy_first"):
                layer.phi = self.layers[0].phi.copy()
                layer.theta = self.layers[0].theta.copy()
            elif alignment == "invert" or alignment == "mirror_dmi":
                phi1 = self.layers[0].phi
                theta1 = self.layers[0].theta
                layer.phi = (np.pi - phi1).copy()
                layer.theta = (theta1 + np.pi) % (2 * np.pi)
                layer.phi = np.clip(layer.phi, epsilon, np.pi - epsilon)
            elif alignment == "opposite_legacy":
                center = N // 2
                for i in range(N):
                    for j in range(N):
                        inside = (i - center)**2 + (j - center)**2 < r**2
                        if method == 'skyrmion':
                            if inside:
                                layer.phi[i, j] = np.pi if self.layers[0].phi[i, j] == 0.0 else 0.0
                                layer.theta[i, j] = 0.0
                            else:
                                layer.phi[i, j] = 0.0 if self.layers[0].phi[i, j] == np.pi else np.pi
                                layer.theta[i, j] = 0.0
                        else:  # random case -> simple inversion
                            layer.phi[i, j] = np.pi - self.layers[0].phi[i, j]
                            layer.theta[i, j] = (self.layers[0].theta[i, j] + np.pi) % (2 * np.pi)
                layer.phi = np.clip(layer.phi, epsilon, np.pi - epsilon)
            else:
                raise ValueError(f"Unknown alignment '{alignment}'")

            if alignment == "mirror_dmi":
                layer.Dx = -self.layers[0].Dx
                layer.Dy = -self.layers[0].Dy
        # End for each layer>0
    #submit to class    
    def compute_topological_charge(self, layer_idx):
        return compute_topological_charge(self.layers[layer_idx].phi, self.layers[layer_idx].theta, self.layers[layer_idx].N)
    def visualize(self, layer_idx, title_prefix=""):
        return visualize_spins_and_density(self.layers[layer_idx].phi, self.layers[layer_idx].theta, self.layers[layer_idx].N, f"{title_prefix} Layer {layer_idx+1}")

# construct magnetic field
def get_magnetic_field(N, B):
    Bx = np.zeros((N, N), dtype=np.float64)
    By = np.zeros((N, N), dtype=np.float64)
    Bz = np.full((N, N), B, dtype=np.float64)
    return Bx, By, Bz

#no use in the this case, but it's useful if want to simulate "writing" a skyrmion on a layer
def get_magnetic_local(N, B, r):
    Bx = np.zeros((N, N), dtype=np.float64)
    By = np.zeros((N, N), dtype=np.float64)
    Bz = np.full((N, N), B, dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if (i - N//2)**2 + (j - N//2)**2 <= r**2:
                Bz[i, j] = -B
    return Bx, By, Bz

#bascially math, compute topology and topological center
def compute_topological_charge(phi, theta, N):
    Q = 0.0
    density = np.zeros((N, N), dtype=np.float64)
    sum_x = sum_y = sum_abs_q = 0.0
    for i in range(N):
        for j in range(N):
            ip = (i + 1) % N
            jp = (j + 1) % N
            s1 = np.array([math.sin(phi[i,j]) * math.cos(theta[i,j]),
                           math.sin(phi[i,j]) * math.sin(theta[i,j]),
                           math.cos(phi[i,j])])
            s1 /= np.linalg.norm(s1) if np.linalg.norm(s1) > 0 else 1.0
            s2 = np.array([math.sin(phi[ip,j]) * math.cos(theta[ip,j]),
                           math.sin(phi[ip,j]) * math.sin(theta[ip,j]),
                           math.cos(phi[ip,j])])
            s2 /= np.linalg.norm(s2) if np.linalg.norm(s2) > 0 else 1.0
            s3 = np.array([math.sin(phi[i,jp]) * math.cos(theta[i,jp]),
                           math.sin(phi[i,jp]) * math.sin(theta[i,jp]),
                           math.cos(phi[i,jp])])
            s3 /= np.linalg.norm(s3) if np.linalg.norm(s3) > 0 else 1.0
            q = np.dot(s1, np.cross(s2, s3))
            density[i, j] = q / (4 * math.pi)
            Q += q
            abs_q = abs(q)
            sum_x += i * abs_q
            sum_y += j * abs_q
            sum_abs_q += abs_q
    Q /= (4 * math.pi)
    x_c = sum_x / sum_abs_q if sum_abs_q > 1e-10 else np.nan
    y_c = sum_y / sum_abs_q if sum_abs_q > 1e-10 else np.nan
    return Q, density, (x_c, y_c)

#visualization: polish by Grok (AI)
def visualize_spins_and_density(phi, theta, N, title=""):
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    U = np.sin(phi) * np.cos(theta)
    V = np.sin(phi) * np.sin(theta)
    W = np.cos(phi)
    
    Q, density, (x_c, y_c) = compute_topological_charge(phi, theta, N)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(W, cmap='RdBu', vmin=-1, vmax=1, origin='lower')
    ax.quiver(X, Y, U, V, scale=20, scale_units='inches')
    ax.set_title(f"{title} Magnetic Moment Distribution (Q = {Q:.6f})")
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    plt.colorbar(im, ax=ax, label='S_z')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_magnetic_moment.png')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(density, cmap='viridis', interpolation='bilinear', origin='lower')
    if not np.isnan(x_c):
        ax.plot(y_c, x_c, 'kx', markersize=10, label=f'Center ({x_c:.2f}, {y_c:.2f})')
        ax.legend()
    ax.set_title(f"{title} Topological Charge Density")
    ax.set_xlabel('j')
    ax.set_ylabel('i')
    plt.colorbar(im, ax=ax, label='q')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_topological_charge.png')
    plt.close()
    
    print(f"{title} Topological Charge: {Q:.6f}, Center: ({x_c:.2f}, {y_c:.2f})")
    return Q, (x_c, y_c)


def simulate_system(num=1, alignment='invert'):
    if num==1:
        print(f"Simulating Single Layer System with Manuscript Parameters:")
    elif num==2:
        print(f"Simulating Bilayer System with Manuscript Parameters (alignment={alignment}):")


    N = 128
    r = 10  

    if num == 1:
        Bx1, By1, Bz1 = get_magnetic_field(N, 2)#setting paramters according to report
        layer1 = Layer(N, Jx=1.0, Jy=1.0, Dx=2.4, Dy=2.4, K=-0.5, A=1, Bx=Bx1, By=By1, Bz=Bz1)#setting paramters according to report
        system = MultilayerSystem([layer1], J_AFM=0.0)#setting paramters according to report
        system.initialize_spins("random", r)#legacy single-layer unaffected
    elif num == 2:
        Bx1, By1, Bz1 = get_magnetic_field(N, 0.0)
        Bx2, By2, Bz2 = get_magnetic_field(N, 0.0)
        layer1 = Layer(N, Jx=1.0, Jy=1.0, Dx=0.03,Dy=0.03, K=-0.01, A=0, Bx=Bx1, By=By1, Bz=Bz1)#setting paramters according to report
        layer2 = Layer(N, Jx=1.0, Jy=1.0, Dx=0.03, Dy=0.03, K=-0.01, A=0, Bx=Bx2, By=By2, Bz=Bz2)#setting paramters according to report
        system = MultilayerSystem([layer1, layer2], J_AFM=-0.01)#setting paramters according to report
        # pass alignment into initializer
        system.initialize_spins("skyrmion", r, alignment='opposite_legacy')
    
    for i in range(len(system.layers)):
        system.visualize(i, f"Initial Layer {i+1}")
        
    final_energy, energy_history = system.gradient_descent(max_iter=1000, save_frames=True, frame_every=100,
                                                           animate=True, animation_path="optimization.gif")
    print(f"Final Energy: {final_energy}")
    # 绘制能量曲线
    plt.figure(figsize=(6,4))
    plt.plot(energy_history, label='Energy')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('Energy vs Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_curve.png')
    plt.close()
    
    centers = []
    #visualization
    for i in range(len(system.layers)):
        Q, center = system.visualize(i, f"Final Layer {i+1}")
        centers.append(center)
        
    #calculate distance between centers
    if num == 2:
        x1, y1 = centers[0]
        x2, y2 = centers[1]
        if not np.isnan(x1) and not np.isnan(x2):
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            print(f"dstance: {distance:.2f} grid size")

# main
#simulate_system(num=1)
start_time=time.time();
simulate_system(num=2, alignment='same');
end_time=time.time();
print(f"{end_time-start_time:.6f}second")




