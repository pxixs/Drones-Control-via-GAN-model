



######################################################################
#               DRONES CONTROL VIA GAN MODEL
# Participants:
# Mohamed-Reda Salhi: mohamed-reda.salhi@polytehnique.edu
# Joseph Combourieu: joseph.combourieu@polytechnique.edu
# Mohssin Bakraoui : mohssin.bakraoui@polytechnique.edu
# Andrea Bourelly: andrea.bourelly@polytechnique.edu
######################################################################




######################################################################
# Block 1: Imports and Global Settings
######################################################################
import math as m
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Block 2: Define Basic Building Blocks and Networks
######################################################################
class ResBlock(nn.Module):
    """
    A residual block that applies a linear transformation, an activation,
    and adds a weighted skip connection.
    """
    def __init__(self, in_features, out_features, activation=nn.ReLU(), skip_weight=0.5):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.skip_weight = skip_weight

    def forward(self, x):
        return self.activation(self.linear(x)) + self.skip_weight * x


class ResNet(nn.Module):
    """
    A simple residual network with an input linear layer,
    three residual blocks, and an output layer.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=100, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock2 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock3 = ResBlock(hidden_dim, hidden_dim, activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Using tanh activation for the input layer, per your original code.
        x = torch.tanh(self.input_layer(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return self.output_layer(x)


# Networks for the GAN-like mean-field game control formulation.
# NOmega approximates the value function (phi network)
class NOmega(nn.Module):
    def __init__(self):
        super(NOmega, self).__init__()
        # Input: 3 (state) + 1 (time) = 4; Output: scalar
        self.net = ResNet(input_dim=4, output_dim=1, activation=nn.Tanh())

    def forward(self, x, t):
        input_data = torch.cat([x, t], dim=-1)
        return self.net(input_data)


# NTheta approximates the generator.
class NTheta(nn.Module):
    def __init__(self):
        super(NTheta, self).__init__()
        # Input: 3 (latent) + 1 (time) = 4; Output: 3 (state)
        self.net = ResNet(input_dim=4, output_dim=3, activation=nn.ReLU())

    def forward(self, z, t):
        input_data = torch.cat([z, t], dim=-1)
        return self.net(input_data)


######################################################################
# Block 3: Boundary Conditions and Helper Functions
######################################################################

# ----------------------------------------
# Génération de points en formation de vague
# ----------------------------------------
A = 0.1

def generate_wave(n_samples) :
    k = 2*m.pi/0.5
    x = torch.linspace(-0.5, 0.5, n_samples)
    y = A * torch.sin(k * x)
    z = torch.ones_like(x) * 0

    return torch.stack([x, y, z], dim=1)

# ----------------------------------------
# Génération de la densité cible (somme de Gaussiennes)
# ----------------------------------------

variance = 0.003

def generate_density_wave(x) :

    x_centered = x - x.mean(dim=0, keepdim=True)
    sigma = np.sqrt(variance)

    def density_estimated(pts):
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)  # (M, N, 3)
        dist2 = (diff ** 2).sum(dim=-1)  # (M, N)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))  # (M, N)
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)

    return density_estimated

density_real = generate_density_wave(generate_wave(100))
x_target = torch.tensor([0,1,0])

def density_final(pts) :
    return density_real(pts - x_target) 

# ----------------------------------------
# Estimateur de densité Gaussien pour les points générés
# ----------------------------------------

class GaussianTorch:
    def __init__(self, data, bandwidth):
        self.data = data  # (M, 3)
        self.bandwidth = bandwidth
        self.length = data.size(0)

    def profile(self, t):
        return torch.exp(-t)  # Exponentielle classique

    def density(self, xyz):  # xyz: (N, 3)
        diff = xyz.unsqueeze(1) - self.data.unsqueeze(0)  # (N, M, 3)
        dist2 = (diff ** 2).sum(dim=-1)  # (N, M)
        kernel_vals = self.profile(dist2 / (self.bandwidth ** 2))  # (N, M)
        density = kernel_vals.sum(dim=1) / (torch.sqrt(torch.tensor(2 * torch.pi, device=xyz.device)) * self.bandwidth * self.length)
        return density  # (N,)

# ----------------------------------------
# Distance L1 entre deux densités différentiable (sur grille 3D)
# ----------------------------------------

def distance_L1_torch(p_func, q_func, n_grid, a=-1.0, b=2.0, device=torch.device("cpu")):
    coords = torch.linspace(a, b, n_grid, device=device)
    dx = (b - a) / n_grid
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)  # (n, n, n, 3)
    flat_grid = grid.view(-1, 3)  # (n^3, 3)

    p_vals = p_func(flat_grid)  # (n^3,)
    q_vals = q_func(flat_grid)  # (n^3,)

    return torch.sum(torch.abs(p_vals - q_vals)) * dx ** 3

def g(x):
    """
    Terminal function used to enforce the boundary condition on phi.
    (For example, it can penalize the distance from the origin.)
    """
    return torch.norm(x.mean(dim=0) - x_target.to(device))


def phi_omega(x, t, N_omega):
    """
    Constructs the value function with boundary condition:
    φ_ω(x, t) = (1 - t) * N_omega(x, t) + t * g(x)
    """
    return (1 - t) * N_omega(x, t) + t * g(x)


def G_theta(z, t, N_theta):
    """
    Constructs the generator with boundary condition:
    G_θ(z, t) = (1 - t) * z + t * N_theta(z, t)
    """
    return (1 - t) * z + t * N_theta(z, t)


def f_collision(x_batch):
    """
    Computes a collision penalty as the mean inverse square of pairwise distances.

    Args:
        x_batch: Tensor [B, 3]

    Returns:
        Mean collision penalty.
    """
    diff = x_batch.unsqueeze(1) - x_batch.unsqueeze(0)  # [B, B, 3]
    dist_sq = torch.sum(diff ** 2, dim=-1)               # [B, B]
    mask = ~torch.eye(dist_sq.size(0), dtype=torch.bool, device=dist_sq.device)
    epsilon = 1e-8  # To avoid division by zero.
    dist_sq_no_diag = dist_sq.masked_select(mask).view(dist_sq.size(0), -1)
    loss_matrix = 1.0 / (dist_sq_no_diag + epsilon)
    if loss_matrix.mean() > 0.03 :
      return 0
    else :
      return loss_matrix.mean()

def f_obstacle(x, obstacles):
      cost = 0
      batch_size = x.size(0)
      for obstacle in obstacles :
          if not isinstance(obstacle, torch.Tensor):
              obstacle_tensor = torch.tensor(obstacle, device=x.device, dtype=x.dtype)
          else:
              obstacle_tensor = obstacle

          for i in range(batch_size):
              # Compute the squared distance to the obstacle.
              Q = torch.norm(x[i] - obstacle_tensor)
              if Q < 0.2:
                  cost += 1.0 / (max(Q-0.1, 0) + 1e-8)   # add a small epsilon to avoid division by zero.
      return cost / batch_size

obstacles=[[(i-1)/2,0.4,(j-5)/10] for i in range(3) for j in range(10)]
'''
If you want to modify the positions of the obstacles you have to put them in obstacles as coordinates (x,y,z) 
'''

# ----------------------------------------
# the Fonction f_formation(x) differentiable
# ----------------------------------------

def f_formation(x, device=torch.device("cpu")):
    # Centrage
    x_centered = x - x.mean(dim=0, keepdim=True)  # (N, 3)

    # Densité estimée à partir de x_centered
    sigma = np.sqrt(variance)
    def density_estimated(pts):
        diff = pts.unsqueeze(1) - x_centered.unsqueeze(0)  # (M, N, 3)
        dist2 = (diff ** 2).sum(dim=-1)  # (M, N)
        gaussians = torch.exp(-dist2 / (2 * sigma**2))  # (M, N)
        norm_const = torch.tensor(2 * torch.pi * variance, device=device)**(3/2)
        return gaussians.sum(dim=1) / (x_centered.shape[0] * norm_const)

    # Distance L1 entre les deux
    d = distance_L1_torch(density_real, density_estimated, n_grid=50, device=device)
    return d


######################################################################
# Block 4: Define Loss Functions for phi and Generator
######################################################################

# Utilise la méthode de rejection sampling pour générer les points selon une densité
# Nécessite de précompute la valeur maximale de la densité
# Dans notre cas: 1/(n*np.sqrt(np.pi*variance))

def sample_from_wave_density(batch_size):

    sigma = np.sqrt(variance)

    k = 2 * m.pi / 0.5  # Nombre d'ondes
    x = torch.rand(batch_size) - 0.5  # x dans [-0.5, 0.5]
    y = A * torch.sin(k * x)
    z = torch.zeros_like(x)

    points = torch.stack([x, y, z], dim=1)

    noise = torch.randn_like(points) * sigma

    return points + noise

def compute_loss_phi(N_omega, N_theta, batch_size, T, lambda_reg):
    """
    Computes the loss for the phi network using derivative and collision terms.
    """
    sigma = np.sqrt(variance)
    # Sample latent variables and time (uniform in [0, T])
    z = sample_from_wave_density(batch_size)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    # Generate states using the generator (applied sample-wise)
    x_list = [G_theta(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega(x, t, N_omega)
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val, (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True
    )
    # Approximate Laplacian: sum of second order derivatives for each spatial dimension
    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i], x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += second_deriv

    H_phi = torch.norm(grad_phi_x, dim=-1, keepdim=True)
    loss_phi_terms = phi_omega(x, torch.zeros_like(t), N_omega) + grad_phi_t \
                     + (sigma**2 / 2) * laplacian + H_phi
    loss_phi_mean = loss_phi_terms.mean()

    # Regularization term penalizing deviation from the HJB residual.
    HJB_residual = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        HJB_residual[i] = torch.norm(
            # Penser à rajouter f_collision
            grad_phi_t[i] + (sigma**2 / 2)*laplacian[i] + H_phi[i]
        )
    loss_HJB = lambda_reg * HJB_residual.mean()

    return loss_phi_mean + loss_HJB + f_collision(x)

def compute_loss_G(N_omega, N_theta, batch_size, T):
    """
    Computes the loss for the generator network.
    """
    sigma = np.sqrt(variance)
    # Sample latent variables and time (uniform in [0, T])
    z = sample_from_wave_density(batch_size)
    t = torch.rand(batch_size, 1, requires_grad=True, device=device) * T
    x_list = [G_theta(z[i:i+1], t[i:i+1], N_theta)[0] for i in range(batch_size)]
    x = torch.stack(x_list)
    x.requires_grad_()

    phi_val = phi_omega(x, t, N_omega)
    phi_val.requires_grad_()
    grad_phi_x, grad_phi_t = torch.autograd.grad(
        phi_val, (x, t),
        grad_outputs=torch.ones_like(phi_val),
        create_graph=True
    )

    laplacian = 0
    for i in range(3):
        second_deriv = torch.autograd.grad(
            grad_phi_x[:, i], x,
            grad_outputs=torch.ones_like(grad_phi_x[:, i]),
            create_graph=True
        )[0][:, i]
        laplacian += second_deriv
    
    H_phi = torch.norm(grad_phi_x, dim=-1, keepdim=True)
    loss_G_terms = grad_phi_t + (sigma**2 / 2)*laplacian + H_phi

    x_final = G_theta(z, torch.ones_like(t), N_theta)
    formation_loss = 0
    for i in range(1,6) :
        sample_x = G_theta(z, torch.ones_like(t)*i/5, N_theta)
        formation_loss += f_formation(sample_x)
    # Penser à rajouter f_collision et f_obstacle
    target_loss = g(x_final)
    print("target_loss: " + str(target_loss))
    print(formation_loss/5)
    return target_loss, loss_G_terms.mean() + 500*target_loss + 70*formation_loss + f_obstacle(x, obstacles) + f_collision(x)


######################################################################
# Block 5: Test Function - Plot Trajectories for 3 Drones over 10s
######################################################################
def test_wave_trajectories(n, N_theta, total_time=10.0, num_steps=100):
    """
    For three drones initialized at the vertices of an equilateral triangle,
    generate and plot their trajectories over a total time period (in seconds).

    Args:
        N_theta: Trained generator network (instance of NTheta).
        total_time: Total simulation time (seconds).
        num_steps: Number of time samples along the trajectory.
    """
    # Define fixed latent vectors for 3 drones (triangle vertices in the xy-plane)
    z_wave = generate_wave(n) # Shape: [3, 3]

    # Prepare a list to hold trajectories for each drone
    trajectories = []  # Each entry: NumPy array of shape [num_steps, 3]

    # Generate equally spaced time instants over the total time.
    times = torch.linspace(0, total_time, num_steps, device=device)

    for i in range(n):  # For each drone
        traj = []
        for t_phys in times:
            # Normalize time to [0, 1] for network input
            t_norm = t_phys / total_time
            t_tensor = torch.tensor([[t_norm]], device=device)
            z = z_wave[i:i+1]  # Shape: [1, 3]
            pos = G_theta(z, t_tensor, N_theta)  # Output: [1, 3]
            traj.append(pos[0])
        traj = torch.stack(traj)  # Shape: [num_steps, 3]
        # Detach before converting to NumPy
        trajectories.append(traj.cpu().detach().numpy())


    # Plot the trajectories
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        traj = trajectories[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
        print("Position finale drône " + str(i) + ": " + str(traj[-1]))
    for i in obstacles:
        ax.plot([i[0]], [i[1]], [i[2]], 'ko', markersize=20)
    ax.set_title("Trajectories of " + str(n) + " Drones Over 10 Seconds")
    ax.plot(x_target[0], x_target[1], x_target[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(-0.5, 0.5)
    ax.legend()
    plt.show()


######################################################################
# Block 6: Main Training and Testing Routine
######################################################################
def main():
    # Hyperparameters (example values; adjust as needed)
 #   print("f_formation:", f_formation.requires_grad)
    batch_size = 50
    T = 1.0              # Normalized training horizon
    epochs = 2500    # Number of training iterations (increase for convergence)
    lambda_reg = 1.0
    n = 10 # Nombre de drones

    learning_rate_phi = 4e-4
    learning_rate_gen = 1e-4

    # Instantiate networks and move them to device
    N_omega = NOmega().to(device)
    N_theta = NTheta().to(device)

    optimizer_phi = optim.Adam(N_omega.parameters(), lr=learning_rate_phi,
                               betas=(0.5, 0.9), weight_decay=1e-4)
    optimizer_theta = optim.Adam(N_theta.parameters(), lr=learning_rate_gen,
                                 betas=(0.5, 0.9), weight_decay=1e-4)

    # Training loop
    target = 2
    target = 900000
    epoch = 0
    while target_loss > 0.1 or cout > 200 :
        optimizer_phi.zero_grad()
        loss_phi_val = compute_loss_phi(N_omega, N_theta, batch_size, T, lambda_reg)
        loss_phi_val.backward()
        optimizer_phi.step()

        optimizer_theta.zero_grad()
        target, loss_gen_val = compute_loss_G(N_omega, N_theta, batch_size, T)
        loss_gen_val.backward()
        cout = loss_gen_val
        optimizer_theta.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss_φ: {loss_phi_val.item():.4f} | Loss_G: {loss_gen_val.item():.4f}")

        epoch += 1

    # After training, test by plotting trajectories of n drones over 20 seconds.
    test_wave_trajectories(n, N_theta, total_time=20.0, num_steps=20)


if __name__ == "__main__":
    main()

