class AffineCoupling(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.half = d // 2
        self.net = nn.Sequential(
            nn.Linear(self.half, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * self.half),
        )

    def forward(self, x):
        x_a, x_b = x[:, :self.half], x[:, self.half:]
        s, t = self.net(x_a).chunk(2, dim=-1)
        s = torch.tanh(s)
        y_b = x_b * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([x_a, y_b], dim=-1), log_det

    # inverse: swap + / - and exp(s) / exp(-s)
