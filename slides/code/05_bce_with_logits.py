import torch
import torch.nn as nn

# Trabalha com logits (saída crua da rede, sem sigmoid)
# Aplica sigmoid + BCE em uma única operação numericamente estável.

logits = torch.tensor([2.5, -1.0, 0.3])
target = torch.tensor([1.0, 0.0, 1.0])

loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(logits, target)

# Equivalente manual (menos estável):
# probs = torch.sigmoid(logits)
# loss = -(target * probs.log() + (1 - target) * (1 - probs).log()).mean()

print(loss.item())  # 0.5390...
