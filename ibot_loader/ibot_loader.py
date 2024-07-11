# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

import torch
from torch.utils.data import Dataset, DataLoader



# ============== Location for our dataset ================
def get_data_fp():
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments where __file__ is not defined (e.g., Jupyter notebooks)
        current_file_dir = os.getcwd()
    
    root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    parquet_file = os.path.join(root, 'evaluation', 'ibot_traindata', 'ibot_traindata_aggregate.parquet')
    return parquet_file


# ============== Create our dataset from parquet file ================
class ParquetDataset(Dataset):
    def __init__(self, fp):
        self.fp = fp
        df = pd.read_parquet(fp)
        self.samples = df['data'].tolist()
        self.tensors = [self.convert_to_tensor(sample) for sample in self.samples]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

    @staticmethod
    def convert_to_tensor(sample):
        if isinstance(sample, list):
            sample = [np.array(sublist, dtype=np.float32) if isinstance(sublist, np.ndarray) else sublist for sublist in sample]
            return torch.tensor(sample, dtype=torch.float32)
        elif isinstance(sample, np.ndarray):
            return torch.tensor(sample.tolist(), dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}")


def get_dataset():
    fp = get_data_fp()
    return ParquetDataset(fp)

            
# ============== Data Augmenter and iBOT Dataloader ================
class DataAugmentationiBOT(object):
    """
        Updated from original iBOT implementation.
    """
    def __init__(self, global_crops_number, local_crops_number, pad_to_32=True):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.pad_to_32 = pad_to_32

    def pad_image(self, image):
        """ 
            Returns padded image, height, and width.
        
            Padded image is either 32x32 or the largest multiple of 4 greater than the height and width of the input image,
            with padding set to -1, and image is placed in the top left corner.
        """
        if isinstance(image, np.ndarray):
            height, width = image.shape
        elif isinstance(image, torch.Tensor):
            height, width = image.size()
            image = image.numpy()
        else:
            raise TypeError("Input must be a 2-D NumPy array or a PyTorch tensor.")

        if self.pad_to_32:
            padded_height = 32
            padded_width = 32
        else:
            padded_height = (height + 3) // 4 * 4
            padded_width = (width + 3) // 4 * 4

        padded_image = np.full((padded_height, padded_width), -1, dtype=int)
        padded_image[:height, :width] = image
        return padded_image, height, width

    def globalcrop1(self, image):
        padded_image, height, width = self.pad_image(image)
        padded_height, padded_width = padded_image.shape
        max_shift_y = padded_height - height
        max_shift_x = padded_width - width
        shift_y = random.randint(0, max_shift_y)
        shift_x = random.randint(0, max_shift_x)
        transformed_image = np.full((padded_height, padded_width), -1, dtype=int)
        transformed_image[shift_y:shift_y + height, shift_x:shift_x + width] = padded_image[:height, :width]
        return transformed_image

    def globalcrop2(self, image):
        padded_image, height, width = self.pad_image(image)
        digits = list(range(10))
        random.shuffle(digits)
        mapping = {i: digits[i] for i in range(10)}
        transformed_image = np.array([[mapping[pixel] if pixel != -1 else -1 for pixel in row] for row in padded_image])
        return transformed_image

    def localcrop(self, image):
        """ Placeholder for localcrop, may implement in future. """
        return image

    def __call__(self, image):
        """ Image must be a 2-D NumPy array or a PyTorch tensor """
        if not (isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)):
            raise TypeError("Input must be a 2-D NumPy array or a PyTorch tensor.")
    
        crops = []
        for _ in range(self.global_crops_number):
            crop = self.globalcrop1(image)
            crops.append(torch.tensor(self.globalcrop2(crop), dtype=torch.float32))
        for _ in range(self.local_crops_number):
            pass  # crops.append(self.localcrop(image))
        return crops


class iBOT_DatasetWrapper():
    """ Dataset wrapper that applies data augmentations / transformations while also loading in our data. """
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.augmenter = DataAugmentationiBOT(args.global_crops_number, args.local_crops_number, args.pad_to_32)
        
        self.psz = args.patch_size
        self.pred_ratio = args.pred_ratio
        self.pred_ratio_var = args.pred_ratio_var
        self.pred_start_epoch = args.pred_start_epoch
        self.pred_shape = args.pred_shape


    def __getitem__(self, index):
        """ 
            Gets the next grid (from index), augments it, and converts it into channels (6 channels per grid).
            Returns tuple of (unmasked grid [as 6 channels], masked grid [as 6 channels]).
        """
        data = self.dataset.__getitem__(index)
        data = self.augmenter(data)  # List of 2-D tensors (32 x 32 or shape based on input grid padded to nearest mult of 4 on both dims)
        images = []
        masks = []
        for d in data:
            i, m = self._create_channels(d)
            images.append(i)
            masks.append(m)
    
        images = torch.stack(images)  # Convert list of tensors to a single tensor of shape (n, 5, 32, 32)
        masks = torch.stack(masks)    # Convert list of tensors to a single tensor of shape (n, 1, 32, 32)
    
        return images, masks  # Tuple of tensors


    def __len__(self):
        return len(self.dataset)

    def _generate_mask(self, image):
        """ Placeholder for mask generation function. """
        image = (image % 2 == 0).int()
        image = image.unsqueeze(-1)
        return image

    def _create_channels(self, image):
        """
        Accepts a 2-D torch tensor and returns tuple of original image (e.g., 5 x 32 x 32) with 5 channels and the mask with 1 channel (e.g., 1 x 32 x 32).
        The original image contains:
            1) Padding Channel: 1 if pixel is from image, 0 if pixel is padding
            2) Value Channels (4x): Binary encoding of pixel value (1-10)
        The mask contains:
            1) Mask Channel: 1 if pixel is masked, 0 otherwise
        """
        H, W = image.shape  # 2-D tensor; 32x32 if pad_to_32=True, else next largest multiple of 4 for each dim
        mask = self._generate_mask(image)
        
        # In 'image', all values that are -1 are 'padding'. Set padding channel using this information
        padding_channel = (image != -1).int().unsqueeze(-1)
        image = image + 1  # Add 1 to each -- so now padding is 0 and all int colors are 1-10 (incl.)
        
        # Convert to binary (4 channels)
        num_value_channels = 4
        value_channels = ((image.unsqueeze(-1).byte() >> torch.arange(num_value_channels)) & 1)
        original_image_tensor = torch.cat((padding_channel, value_channels), dim=-1)
        
        # Permute to get the desired shape
        original_image_tensor = original_image_tensor.permute(2, 0, 1)
        mask = mask.permute(2, 0, 1)
        
        return original_image_tensor.float(), mask.int()

def custom_collate_fn(batch):
    """ Custom collate function to return just the batch (list of tuples) rather than concat / stacking into new tensor. """
    return batch[0]