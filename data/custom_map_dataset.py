import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

class CustomMapDataset(BaseDataset):
    """Dataset class that loads satellite-map image pairs and handles segmentation masks.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options and rewrite default values.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase.

        Returns:
            the modified parser.
        """
        parser.set_defaults(input_nc=3, output_nc=3, direction='AtoB')  # 3 channels for RGB
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        self.direction = opt.direction
        
        # Define colors for segmentation (RGB format)
        self.colors = {
            'road': np.array([252, 252, 252]),      # #fcfcfc
            'main_road': np.array([249, 159, 39]),  # #f99f27
            'sm_road': np.array([247, 244, 239]),   # #f7f4ef
            'med_road': np.array([253, 223, 153]),  # Add medium road color
            'vegetation': np.array([203, 223, 174]), # #cbdfae
            'water': np.array([156, 188, 245]),     # #9cbcf5
            'building': np.array([236, 239, 232])   # #ecefe8
        }
        
        # Define tolerances
        self.tolerances = {
            'road': 15,
            'main_road': 50,
            'sm_road': 7,
            'med_road': 5,  # Add medium road tolerance
            'vegetation': 15,
            'water': 45,
            'building': 7
        }

        # Make sure transforms don't include horizontal flips
        opt.no_flip = True  # Force no flipping
        self.transform = get_transform(opt, convert=True)

    def __getitem__(self, index):
        # Read images
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        
        # Split A and B
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        
        # Generate segmentation mask on original B (before transform)
        B_np = np.array(B)
        seg_mask = self.color_segment(torch.from_numpy(B_np.transpose(2, 0, 1)).float() / 255.0)
        
        # Apply transforms for model training
        A = self.transform(A)
        B_transformed = self.transform(B)
        
        # If direction is BtoA, swap A and B
        if self.direction == 'BtoA':
            A, B_transformed = B_transformed, A
        
        return {'A': A, 'B': B_transformed, 'seg_mask': seg_mask, 'A_paths': AB_path, 'B_paths': AB_path}

    def color_segment(self, img_tensor):
        """Convert map to segmentation mask using specific color values."""
        # Convert tensor to numpy array (CHW to HWC)
        img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Create masks for each feature
        building_matrix = np.abs(img_np - self.colors['building']) <= self.tolerances['building']
        road_matrix = np.add(
            np.add(
                np.add(
                    np.abs(img_np - self.colors['road']) <= self.tolerances['road'],
                    np.abs(img_np - self.colors['main_road']) <= self.tolerances['main_road']
                ),
                np.abs(img_np - self.colors['sm_road']) <= self.tolerances['sm_road']
            ),
            np.abs(img_np - self.colors['med_road']) <= self.tolerances['med_road']  # Fixed reference
        )
        
        # Generate masks with building-road conflict resolution
        building_mask = np.all(building_matrix, axis=2)
        road_mask = np.all(np.logical_and(road_matrix, np.logical_not(building_matrix)), axis=2)
        vegetation_mask = np.all(np.abs(img_np - self.colors['vegetation']) <= self.tolerances['vegetation'], axis=2)
        water_mask = np.all(np.abs(img_np - self.colors['water']) <= self.tolerances['water'], axis=2)
        
        # Apply denoising
        building_mask = cv2.medianBlur(building_mask.astype(np.uint8), 5).astype(bool)
        road_mask = cv2.medianBlur(road_mask.astype(np.uint8), 5).astype(bool)
        vegetation_mask = cv2.medianBlur(vegetation_mask.astype(np.uint8), 5).astype(bool)
        water_mask = cv2.medianBlur(water_mask.astype(np.uint8), 3).astype(bool)
        
        # Combine masks into a single tensor
        mask = np.zeros((*img_np.shape[:2], 5), dtype=np.float32)
        mask[:,:,0] = road_mask
        mask[:,:,1] = vegetation_mask
        mask[:,:,2] = water_mask
        mask[:,:,3] = building_mask
        mask[:,:,4] = 1 - np.any(mask[:,:,:4], axis=2)  # Other features
        
        return torch.FloatTensor(mask.transpose(2, 0, 1))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)