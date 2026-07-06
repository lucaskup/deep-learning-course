model = MLP()
optimizer = SGD(model.parameters(),
                lr=0.1)
x = torch.tensor(1.0)
y = torch.tensor(1.0)
model.train()
optimizer.zero_grad()
y_hat = model(x)
loss = (y - y_hat)**2/2
loss.backward()
optimizer.step()