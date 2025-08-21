variance = 0.003
'''
So here the generate wave density serves simply as an example. you can of course choose the density you desire in the following fashion : 
you have point (x_i,y_i,z_i) so you sum normale distributions centered on those points and normalize this will create a denity 
P.S: keep the name "generate_density_wave" for simplicity since the code is built with this as code but  you can change it in each place and it will still be fine 
'''
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
def distance_L1_torch(p_func, q_func, n_grid, a=-1.0, b=2.0, device=torch.device("cpu")):
    coords = torch.linspace(a, b, n_grid, device=device)
    dx = (b - a) / n_grid
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)  # (n, n, n, 3)
    flat_grid = grid.view(-1, 3)  # (n^3, 3)

    p_vals = p_func(flat_grid)  # (n^3,)
    q_vals = q_func(flat_grid)  # (n^3,)

    return torch.sum(torch.abs(p_vals - q_vals)) * dx ** 3
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
