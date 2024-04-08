from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage, ToTensor
import torch

def shuffle_image_tensor(image_tensor, patch_size):
    """
    Shuffle the patches of an image tensor and return the shuffled image tensor,
    along with tensors for the shuffled positions and rotations.
    
    Args:
    - image_tensor (Tensor): The image tensor to shuffle, shape [C, H, W].
    - patch_size (int): The size of each square patch.
    
    Returns:
    - shuffled_image_tensor (Tensor): The tensor of the shuffled image, shape [C, H, W].
    - positions (Tensor): The tensor with the final positions of each shuffled patch.
    - rotations (Tensor): The tensor with the rotation class (0, 1, 2, 3) for each patch.
    """
    C, H, W = image_tensor.shape
    num_patches_side = H // patch_size
    num_patches = num_patches_side ** 2

    # Extract patches
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.reshape(C, num_patches_side, num_patches_side, patch_size, patch_size).permute(1, 2, 0, 3, 4)
    patches = patches.reshape(num_patches, C, patch_size, patch_size)

    # Shuffle and assign rotations
    shuffle_indices = torch.randperm(num_patches)
    shuffled_patches = patches[shuffle_indices]
    rotations = torch.randint(0, 4, (num_patches,))

    # Reconstruct shuffled image with rotations applied
    shuffled_image = torch.zeros_like(patches)
    for idx, patch_idx in enumerate(shuffle_indices):
        shuffled_image[idx] = torch.rot90(patches[patch_idx], rotations[patch_idx], [1, 2])
    shuffled_image = shuffled_image.reshape(num_patches_side, num_patches_side, C, patch_size, patch_size)
    shuffled_image = shuffled_image.permute(2, 0, 3, 1, 4).reshape(C, H, W)

    # Directly use shuffle indices and rotations as outputs
    positions_out = shuffle_indices
    rotations_out = rotations

    return shuffled_image, positions_out, rotations_out

class ShuffleAndRotatePatches:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img_tensor):
        original_img_tensor = img_tensor.clone()
        shuffled_img_tensor, positions, rotations = shuffle_image_tensor(img_tensor, self.patch_size)
        return original_img_tensor, shuffled_img_tensor, positions, rotations