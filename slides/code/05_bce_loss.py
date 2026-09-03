import torch
import torch.nn as nn

# Trabalha com probabilidades (saída já passada por sigmoid).
# Numericamente menos estável; lança erro se a entrada
# estiver fora de [0, 1].

probs = torch.sigmoid(torch.tensor([2.5, -1.0, 0.3]))
target = torch.tensor([1.0, 0.0, 1.0])

loss_fn = nn.BCELoss()
loss = loss_fn(probs, target)

print(loss.item())  # 0.3155...
