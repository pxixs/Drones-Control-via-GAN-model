
class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(), skip_weight=0.5):
        super(ResBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.skip_weight = skip_weight

    def forward(self, x):
        return self.activation(self.linear(x)) + self.skip_weight * x


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, activation=nn.ReLU()):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.resblock1 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock2 = ResBlock(hidden_dim, hidden_dim, activation)
        self.resblock3 = ResBlock(hidden_dim, hidden_dim, activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return self.output_layer(x)
