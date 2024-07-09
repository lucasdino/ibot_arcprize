# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import torch


def create_channels(image, mask=None):
    """
    Accepts 2-D numpy array and returns original image and the masked image.
    Returns image with 6 channels:
        1) Mask Channel: 1 if pixel is masked, 0 otherwise
        2) Padding Channel: 1 if pixel is from image, 0 if pixel is padding
        3) Value Channels (4x): Binary encoding of pixel value (1-10)
    """
    H, W = image.shape
    
    # Initialize channels
    mask_channel = np.zeros((32, 32), dtype=int)
    padding_channel = np.zeros((32, 32), dtype=int)
    
    # Create binary channels for values
    num_value_channels = 4
    value_channels = np.zeros((num_value_channels, 32, 32), dtype=int)
    
    # Adjust original values to be from 1-10 instead of 0-9
    adjusted_image = image + 1
    
    # Fill original values and padding
    padded_image = np.full((32, 32), -1, dtype=int)
    padded_image[:H, :W] = adjusted_image
    padding_channel[:H, :W] = 1
    
    if mask is not None:
        mask_channel[:H, :W] = mask
        padded_image[mask == 1] = 0  # Setting masked values to 0 for binary encoding

    # Convert each value to binary and set the corresponding channels
    for i in range(32):
        for j in range(32):
            if padding_channel[i, j] == 1 and mask_channel[i, j] == 0:
                binary_value = np.binary_repr(padded_image[i, j], width=num_value_channels)
                for k, bit in enumerate(binary_value):
                    value_channels[k, i, j] = int(bit)
    
    # Stack channels: first mask, then padding, then value channels
    original_image_tensor = np.stack([mask_channel, padding_channel] + [value_channels[k] for k in range(num_value_channels)], axis=0)
    
    # Masked image: Same process but without the original value where mask is 1
    masked_padded_image = padded_image.copy()
    if mask is not None:
        masked_padded_image[mask == 1] = 0
    
    # Convert masked values to binary and set the corresponding channels
    masked_value_channels = np.zeros((num_value_channels, 32, 32), dtype=int)
    for i in range(32):
        for j in range(32):
            if padding_channel[i, j] == 1:
                binary_value = np.binary_repr(masked_padded_image[i, j], width=num_value_channels)
                for k, bit in enumerate(binary_value):
                    masked_value_channels[k, i, j] = int(bit)
    
    masked_image_tensor = np.stack([mask_channel, padding_channel] + [masked_value_channels[k] for k in range(num_value_channels)], axis=0)
    
    return torch.tensor(original_image_tensor, dtype=torch.float32), torch.tensor(masked_image_tensor, dtype=torch.float32)





class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)