import torch

def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    # Flatten the tensors to simplify computation
    input_flat = input.view(-1)
    target_flat = target.view(-1)

    # Calculate the intersection and union
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    # Calculate the Dice coefficient
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

def dice_loss(input: torch.Tensor, target: torch.Tensor):
    # Dice loss is 1 minus the Dice coefficient
    return 1 - dice_coeff(input, target)
