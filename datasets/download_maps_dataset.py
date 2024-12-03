import os
import kagglehub
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def get_dataset_path():
    """Download and get the path to the dataset using kagglehub."""
    path = kagglehub.dataset_download("alincijov/pix2pix-maps")
    print(f"Path to dataset files: {path}")
    return path

class CustommapDataset(BaseDataset):
    """Dataset class that loads satellite-map image pairs and handles segmentation masks."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3, direction='AtoB')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags
        """
        BaseDataset.__init__(self, opt)
        
        # Get dataset path from kagglehub
        dataset_path = get_dataset_path()
        
        # Update the directory path to use the downloaded dataset
        self.dir = os.path.join(dataset_path, opt.phase)
        if not os.path.exists(self.dir):
            # If phase-specific directory doesn't exist, use the root directory
            self.dir = dataset_path
            
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        
        if len(self.AB_paths) == 0:
            raise RuntimeError(f"Found 0 images in: {self.dir}\nPlease check the data directory.")
            
        self.direction = opt.direction
        
        # Define colors for segmentation (if needed)
        self.colors = {
            'road': np.array([252, 252, 252]),
            'vegetation': np.array([176, 223, 201]),
            'water': np.array([177, 208, 254])
        }
        
        self.tolerances = {
            'road': 5,
            'vegetation': 6,
            'water': 7
        }

        self.transform = get_transform(opt, convert=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        
        # Split into A and B
        w = AB.size[0]
        w2 = int(w/2)
        
        A = AB.crop((0, 0, w2, AB.size[1]))
        B = AB.crop((w2, 0, w, AB.size[1]))

        # Apply transforms
        A = self.transform(A)
        B = self.transform(B)

        # If direction is BtoA, swap A and B
        if self.direction == 'BtoA':
            A, B = B, A

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def color_segment(self, img_tensor):
        """Convert map to segmentation mask using specific color values."""
        img_np = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        mask = np.zeros((*img_np.shape[:2], 4), dtype=np.float32)
        
        mask[:,:,0] = np.all(np.abs(img_np - self.colors['road']) <= self.tolerances['road'], axis=2)
        mask[:,:,1] = np.all(np.abs(img_np - self.colors['vegetation']) <= self.tolerances['vegetation'], axis=2)
        mask[:,:,2] = np.all(np.abs(img_np - self.colors['water']) <= self.tolerances['water'], axis=2)
        mask[:,:,3] = 1 - np.any(mask[:,:,:3], axis=2)
        
        return torch.FloatTensor(mask.transpose(2, 0, 1))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
