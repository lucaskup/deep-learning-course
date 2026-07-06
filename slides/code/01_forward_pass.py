class RedeSimples(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.theta_1 = nn.parameter.Parameter(
                    torch.tensor(-0.5),
                    requires_grad=True)
    self.theta_2 = nn.parameter.Parameter(
                    torch.tensor(0.5),
                    requires_grad=True)
  def forward(self, x):
    z1 = x * self.theta_1
    a1 = torch.sigmoid(z1)
    z2 = a1 * self.theta_2
    a2 = torch.sigmoid(z2)
    return a2
x = torch.tensor(1.0)
y = torch.tensor(1.0)
model = RedeSimples()
y_hat = model(x)
loss = ((y - y_hat)**2)/2
print(f'y_hat: {y_hat}')
print(f'loss: {loss}')
# y_hat: 0.5470529198646545
# loss: 0.10258052498102188