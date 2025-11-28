from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim


def fgsm_attack(inputs: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:

    sign_data_grad = data_grad.sign()
    perturbed_data = inputs + epsilon * sign_data_grad
    return torch.clamp(perturbed_data, 0, 1)


def adversarial_training_step(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    batch_inputs: torch.Tensor,
    batch_labels: torch.Tensor,
    epsilon: float = 0.01,
):

    model.train()
    batch_inputs.requires_grad = True
    outputs = model(batch_inputs)
    loss = loss_fn(outputs, batch_labels)
    model.zero_grad()
    loss.backward()
    data_grad = batch_inputs.grad.data

    adv_inputs = fgsm_attack(batch_inputs, epsilon, data_grad)
    outputs_adv = model(adv_inputs.detach())
    loss_adv = loss_fn(outputs_adv, batch_labels)

    total_loss = (loss + loss_adv) / 2.0
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item()


