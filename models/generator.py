class NTheta(nn.Module):
    def __init__(self):
        super(NTheta, self).__init__()
        # Input: 3 (latent) + 1 (time) = 4; Output: 3 (state)
        self.net = ResNet(input_dim=4, output_dim=3, activation=nn.ReLU())

    def forward(self, z, t):
        input_data = torch.cat([z, t], dim=-1)
        return self.net(input_data)
def G_theta(z, t, N_theta):
    """
    Constructs the generator with boundary condition:
    G_θ(z, t) = (1 - t) * z + t * N_theta(z, t)
    """
    return (1 - t) * z + t * N_theta(z, t)
