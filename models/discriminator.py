class NOmega(nn.Module):
    def __init__(self):
        super(NOmega, self).__init__()
        # Input: 3 (state) + 1 (time) = 4; Output: scalar
        self.net = ResNet(input_dim=4, output_dim=1, activation=nn.Tanh())

    def forward(self, x, t):
        input_data = torch.cat([x, t], dim=-1)
        return self.net(input_data)

def phi_omega(x, t, N_omega):
    """
    Constructs the value function with boundary condition:
    φ_ω(x, t) = (1 - t) * N_omega(x, t) + t * g(x)
    """
    return (1 - t) * N_omega(x, t) + t * g(x)
