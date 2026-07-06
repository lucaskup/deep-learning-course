import torch
import torch.nn as nn

# Trabalha com probabilidades (saída já passada por sigmoid).
# Numericamente menos estável; falha silenciosamente se a entrada
# não estiver em (0, 1).

probs = torch.sigmoid(torch.tensor([2.5, -1.0, 0.3]))
target = torch.tensor([1.0, 0.0, 1.0])

loss_fn = nn.BCELoss()
loss = loss_fn(probs, target)

# Para classificação multilabel, basta usar a mesma loss
# em saídas com K unidades sigmoide (uma por rótulo).

print(loss.item())  # 0.5390...
