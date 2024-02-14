import torch

def dice_coeff(input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6):
    # Flatten the tensors to simplify computation
    input_flat = input.view(-1)
    target_flat = target.view(-1)
    # print("\n input", torch.mean(input_flat))
    # print(input_flat)
    # print("\n target", torch.mean(target_flat))
    # print(target_flat)


    # Calculate the intersection and union
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    # Calculate the Dice coefficient
    dice_copy = (2. * intersection ) / (union)
    # print("dice loss", dice_copy)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    # print("dice epsilon", dice)

    return dice

def dice_loss(input: torch.Tensor, target: torch.Tensor):
    # Dice loss is 1 minus the Dice coefficient
    return 1 - dice_coeff(input, target)
