from typing import List, Tuple

import numpy as np
import torch
import yacs.config


def cutmix(
    batch: Tuple[torch.Tensor, torch.Tensor], alpha: float
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    data, targets = batch

    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets

# Function added by W210 Team
def w210_cutmix(
    batch: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    data, targets = batch
    
    #print("w210_cutmix:", targets, type(targets), type(targets[0]))
    orig_targets = np.array([np.long(x[0]) for x in targets])
    shuffled_targets = np.array([np.long(x[1]) for x in targets])
    lam = np.array([x[2] for x in targets])

    targets = (torch.from_numpy(orig_targets), torch.from_numpy(shuffled_targets), torch.from_numpy(lam))

    return data, targets

class CutMixCollator:
    def __init__(self, config: yacs.config.CfgNode):
        self.alpha = config.augmentation.cutmix.alpha

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
        batch = torch.utils.data.dataloader.default_collate(batch)
        # Changes by W210 Team begin on line above
        batch = w210_cutmix(batch)
        #batch = cutmix(batch, self.alpha)
        # End changes by W210
        return batch
