optimizer = SGD(model.parameters(),
                lr=0.1)
x = torch.tensor(1.0)
y = torch.tensor(1.0)
model.train()
optimizer.zero_grad()
y_hat = model(x)
loss = (y - y_hat)**2/2
loss.backward()
print(f'Grad: {model.theta_2.grad}, {model.theta_1.grad}')
optimizer.step()
print(f'Pesos: {model.theta_2.data}, {model.theta_1.data}')
# Grad: -0.04237288236618042, -0.01318769808858633
# Pesos: 0.5042372941970825, -0.4986812174320221