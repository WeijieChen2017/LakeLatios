import os
import torch
import numpy as np
from torch.utils.data import Dataset

# import the data transform class

class slice_npy(Dataset):
    def __init__(self, file_dict_list, required_keys, 
                 is_channel_last=False, return_filename=False,
                 init_verbose=False, transform=False,
                 output_size=512):
        """
        Args:
            file_paths (list of str): List of npy file paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_dict_list = file_dict_list
        self.required_keys = required_keys
        self.transform = transform
        self.is_channel_last = is_channel_last
        self.return_filename = return_filename
        self.init_verbose = init_verbose
        self.output_size = output_size
        if self.transform:
            self.MedicalDataAugmentation = MedicalDataAugmentation()

        if self.init_verbose:
            print("---> slice_npy dataset initialized.")
            print("---> Number of samples: ", len(self.file_dict_list))
            print("---> Required keys: ", self.required_keys)
            print("---> is_channel_last: ", self.is_channel_last)
            print("---> return_filename: ", self.return_filename)

    def __len__(self):
        return len(self.file_dict_list)

    def __getitem__(self, idx):
        data = {}
        for key in self.required_keys:
            loaded_data = np.load(self.file_dict_list[idx][key], allow_pickle=True)
            if self.is_channel_last:
                # change the channel to the first dimension
                loaded_data = loaded_data.transpose(2, 0, 1)
            num_chan, ax, ay = loaded_data.shape
            if ax != self.output_size or ay != self.output_size:
                # resize the data to the output_size
                loaded_data = zoom(loaded_data, (1, self.output_size/ax, self.output_size/ay), order=3)


            data[key] = loaded_data

        # transform the data for all data in the required_keys
        if self.transform:
            rand_seed = np.random.randint(0, 100)
            data = self.MedicalDataAugmentation(data, rand_seed)

        if self.return_filename:
            filename = os.path.basename(self.file_dict_list[idx]["mr"])
            filename = filename.split(".")[0]
            return data, filename
        else:
            return data


import numpy as np
import random
import cv2
from scipy.ndimage import rotate, zoom, shift
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class MedicalDataAugmentation:
    def __init__(self, rotation_angle=30, shift_max=5, zoom_range=(0.9, 1.1), flip_prob=0.5, elastic_deformation=False):
        """
        Initialize the data augmentation class with specified parameters.

        Args:
            rotation_angle (int): Maximum rotation angle in degrees.
            shift_max (int or float): Maximum shift in pixels.
            zoom_range (tuple): Min and max scaling factor.
            flip_prob (float): Probability of applying a horizontal or vertical flip.
            elastic_deformation (bool): If True, applies elastic deformation.
        """
        self.rotation_angle = rotation_angle
        self.shift_max = shift_max
        self.zoom_range = zoom_range
        self.flip_prob = flip_prob
        self.elastic_deformation = elastic_deformation

    def __call__(self, data, rand_seed=None):
        """
        Apply transformations to the input data.

        Args:
            data (dict): Dictionary with keys pointing to NumPy arrays representing images.

        Returns:
            dict: Augmented data.
        """
        if rand_seed is not None:
            np.random.seed(rand_seed)  # Adjusting for NumPy; remove torch.manual_seed(rand_seed) as torch is not used here.

        augmented_data = {}
        for key, array in data.items():
            original_shape = array.shape
            is_single_channel = original_shape[-1] == 1  # Check if the image is single-channel (CT)
            
            if is_single_channel:
                array = array.squeeze(-1)  # Remove the channel dimension for processing
            
            # Apply augmentations here as before, without modifications
            
            # Elastic deformation adjustments for single-channel images
            if self.elastic_deformation and not is_single_channel:
                # Only apply elastic deformation for multi-channel (MR) to avoid channel confusion
                array = self.elastic_transform(array, array.shape[1] * 2, array.shape[1] * 0.08, array.shape[1] * 0.08)

            if is_single_channel:
                array = np.expand_dims(array, axis=-1)  # Restore the channel dimension for single-channel images

            augmented_data[key] = array

        return augmented_data

    def elastic_transform(self, image, alpha, sigma, alpha_affine, random_state=None):
        """
        Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and Recognition, 2003.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z+dz, (-1, 1))

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)