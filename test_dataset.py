import torch
from data import create_dataset
import matplotlib.pyplot as plt
from data.custom_map_dataset import CustomMapDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import kagglehub
import os
import cv2

def get_dataset_path():
    """Download and get the path to the dataset using kagglehub."""
    path = kagglehub.dataset_download("alincijov/pix2pix-maps")
    print(f"Path to dataset files: {path}")
    return path

class Options:
    def __init__(self):
        self.dataroot = "/Users/nandizhang/.cache/kagglehub/datasets/alincijov/pix2pix-maps/versions/1"
        self.phase = 'val'
        self.preprocess = 'resize_and_crop'
        self.load_size = 256
        self.crop_size = 256
        self.input_nc = 3
        self.output_nc = 3
        self.direction = 'AtoB'
        self.no_flip = True
        self.max_dataset_size = float("inf")
        self.isTrain = False

def test_dataset():
    opt = TestOptions()
    print("Creating dataset...")
    dataset = create_dataset(opt)
    print("Dataset created successfully!")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first batch
    data = next(iter(dataset))
    print("Data keys:", data.keys())
    print("Input shape:", data['A'].shape)
    print("Target shape:", data['B'].shape)
    print("Segmentation mask shape:", data['seg_mask'].shape)
    
    # Save samples
    real_A = data['A'][0].numpy()
    real_B = data['B'][0].numpy()
    seg_mask = data['seg_mask'][0].numpy()
    
    # Convert from tensor format to image format
    real_A = (np.transpose(real_A, (1, 2, 0)) + 1) / 2.0 * 255.0
    real_B = (np.transpose(real_B, (1, 2, 0)) + 1) / 2.0 * 255.0
    
    # Create color-coded segmentation visualization
    seg_colors = np.array([
        [255, 0, 0],    # Red for roads
        [0, 255, 0],    # Green for vegetation
        [0, 0, 255],    # Blue for water
        [255, 0, 255],  # Purple for buildings
        [128, 128, 128] # Gray for other
    ])
    
    # Create separate visualizations for each class
    plt.figure(figsize=(20, 4))
    
    plt.subplot(161)
    plt.imshow(real_A.astype(np.uint8))
    plt.title('Input (Satellite)')
    plt.axis('off')
    
    plt.subplot(162)
    plt.imshow(real_B.astype(np.uint8))
    plt.title('Target (Map)')
    plt.axis('off')
    
    plt.subplot(163)
    plt.imshow(seg_mask[0], cmap='gray')
    plt.title('Roads')
    plt.axis('off')
    
    plt.subplot(164)
    plt.imshow(seg_mask[1], cmap='gray')
    plt.title('Vegetation')
    plt.axis('off')
    
    plt.subplot(165)
    plt.imshow(seg_mask[2], cmap='gray')
    plt.title('Water')
    plt.axis('off')
    
    plt.subplot(166)
    plt.imshow(seg_mask[3], cmap='gray')
    plt.title('Buildings')
    plt.axis('off')
    
    plt.savefig('test_sample_with_seg.png')
    print("Sample saved as 'test_sample_with_seg.png'")

def color_segment(img):
    """
    Convert image to segmentation masks
    Returns a tensor where each channel is a binary mask for one class
    """
    # Convert BGR to RGB if needed
    if len(img.shape) == 4:  # If batch of images
        image = img
    else:  # Single image
        image = img[None]
    
    # Define target colors in RGB
    road_color = np.array([252, 252, 252])      
    main_road_color = np.array([249, 159, 39])  
    sm_road_color = np.array([247, 244, 239])   
    med_road_color = np.array([253, 223, 153])  
    vegetation_color = np.array([203, 223, 174]) 
    water_color = np.array([156, 188, 245])     
    building_color = np.array([236, 239, 232])  

    # Set tolerances
    road_tolerance = 15
    main_road_tolerance = 50
    sm_road_tolerance = 7
    med_road_tolerance = 5
    vegetation_tolerance = 15
    water_tolerance = 45
    building_tolerance = 7

    # Initialize output tensor (B, num_classes, H, W)
    B, H, W, C = image.shape
    masks = np.zeros((B, 4, H, W))

    for b in range(B):
        # Create masks for each feature
        building_matrix = np.abs(image[b] - building_color) <= building_tolerance
        road_matrix = np.add(np.add(np.abs(image[b] - road_color) <= road_tolerance,
                         np.abs(image[b] - main_road_color) <= main_road_tolerance),
                         np.add(np.abs(image[b] - sm_road_color) <= sm_road_tolerance,
                                np.abs(image[b] - med_road_color) <= med_road_tolerance))
        
        building_mask = np.all(building_matrix, axis=2)
        road_mask = np.all(np.logical_and(road_matrix, np.logical_not(building_matrix)), axis=2)
        vegetation_mask = np.all(np.abs(image[b] - vegetation_color) <= vegetation_tolerance, axis=2)
        water_mask = np.all(np.abs(image[b] - water_color) <= water_tolerance, axis=2)

        # Apply denoising
        building_mask = cv2.medianBlur(building_mask.astype(np.uint8), 5).astype(bool)
        road_mask = cv2.medianBlur(road_mask.astype(np.uint8), 5).astype(bool)
        vegetation_mask = cv2.medianBlur(vegetation_mask.astype(np.uint8), 5).astype(bool)
        water_mask = cv2.medianBlur(water_mask.astype(np.uint8), 3).astype(bool)

        masks[b, 0] = road_mask
        masks[b, 1] = vegetation_mask
        masks[b, 2] = water_mask
        masks[b, 3] = building_mask

    return masks.astype(np.float32)

def check_dataset_structure():
    path = "/Users/nandizhang/.cache/kagglehub/datasets/alincijov/pix2pix-maps/versions/1"
    
    print("Checking if path exists:", os.path.exists(path))
    print("\nDirectory contents:")
    for root, dirs, files in os.walk(path):
        print(f"\nRoot: {root}")
        print("Directories:", dirs)
        print("Files:", len(files), "files")
        if files:
            print("Sample files:", files[:5])

def check_image_structure():
    # Let's check one image from the val set
    img_path = "/Users/nandizhang/.cache/kagglehub/datasets/alincijov/pix2pix-maps/versions/1/val/63.jpg"
    
    # Read and display the image
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return
        
    print("Image shape:", img.shape)
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # If this is a pix2pix dataset, images should be concatenated side by side
    # Let's split it in half and show both sides
    width = img.shape[1]
    half_width = width // 2
    
    img_A = img_rgb[:, :half_width]
    img_B = img_rgb[:, half_width:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_A)
    ax1.set_title('Left half (A)')
    ax1.axis('off')
    
    ax2.imshow(img_B)
    ax2.set_title('Right half (B)')
    ax2.axis('off')
    
    plt.show()

def verify_segmentation():
    opt = Options()
    dataset = CustomMapDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get a sample image
    data = next(iter(dataloader))
    
    # Get the original map image (B)
    map_img = data['B'][0].permute(1, 2, 0).numpy() * 0.5 + 0.5
    map_img = (map_img * 255).astype(np.uint8)
    
    # Get the generated segmentation mask and resize to 256x256
    seg_mask = data['seg_mask'][0].numpy()
    resized_seg_mask = np.zeros((5, 256, 256))
    for i in range(seg_mask.shape[0]):
        resized_seg_mask[i] = cv2.resize(seg_mask[i], (256, 256), interpolation=cv2.INTER_NEAREST)
    seg_mask = resized_seg_mask
    
    print(f"Map image shape: {map_img.shape}")
    print(f"Segmentation mask shape: {seg_mask.shape}")
    
    # Create visualization figure
    fig = plt.figure(figsize=(20, 15))
    
    # Plot original map
    plt.subplot(3, 4, 1)
    plt.imshow(map_img)
    plt.title('Original Map')
    plt.axis('off')
    
    # Plot each segmentation channel and its color detection
    mask_names = ['Road', 'Vegetation', 'Water', 'Building', 'Other']
    
    for i, name in enumerate(mask_names[:4]):  # Excluding 'Other'
        # Plot the segmentation mask
        plt.subplot(3, 4, i+2)
        plt.imshow(seg_mask[i], cmap='gray')
        plt.title(f'Mask: {name}')
        plt.axis('off')
        
        # Create color-based mask directly
        if name.lower() in dataset.colors:
            color = dataset.colors[name.lower()]
            tolerance = dataset.tolerances[name.lower()]
            
            # Create direct color mask
            direct_mask = np.all(np.abs(map_img - color) <= tolerance, axis=2)
            
            # Apply denoising
            direct_mask = cv2.medianBlur(direct_mask.astype(np.uint8), 5).astype(bool)
            
            # Plot direct color mask
            plt.subplot(3, 4, i+6)
            plt.imshow(direct_mask, cmap='gray')
            plt.title(f'Direct Color Mask: {name}')
            plt.axis('off')
            
            # Calculate and display agreement percentage
            agreement = np.mean(direct_mask == seg_mask[i]) * 100
            print(f"{name} mask agreement: {agreement:.2f}%")
    
    plt.tight_layout()
    plt.show()
    
    # Detailed color analysis
    print("\nDetailed Color Analysis:")
    for name, color in dataset.colors.items():
        print(f"\n{name.upper()}:")
        print(f"Target RGB: {color}")
        print(f"Tolerance: ±{dataset.tolerances[name]}")
        
        # Find pixels that match this feature in the map
        matching_pixels = np.all(np.abs(map_img - color) <= dataset.tolerances[name], axis=2)
        num_matching = np.sum(matching_pixels)
        
        print(f"Pixels matched: {num_matching}")
        if num_matching > 0:
            matching_colors = map_img[matching_pixels]
            avg_color = np.mean(matching_colors, axis=0)
            std_color = np.std(matching_colors, axis=0)
            print(f"Average RGB of matched pixels: {avg_color}")
            print(f"Standard deviation: {std_color}")

def analyze_pixel_distribution():
    opt = Options()
    dataset = CustomMapDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Get a sample image
    data = next(iter(dataloader))
    map_img = data['B'][0].permute(1, 2, 0).numpy() * 0.5 + 0.5
    map_img = (map_img * 255).astype(np.uint8)
    
    # Create 3D color histogram
    hist = np.histogramdd(
        map_img.reshape(-1, 3),
        bins=[8, 8, 8],
        range=[[0, 256], [0, 256], [0, 256]]
    )[0]
    
    # Find dominant colors
    flat_hist = hist.flatten()
    top_indices = np.argsort(flat_hist)[-10:][::-1]
    bin_size = 256 // 8
    
    print("\nDominant Colors Analysis:")
    for idx in top_indices:
        # Convert flat index to 3D coordinates
        r = (idx // (8 * 8)) * bin_size + bin_size // 2
        g = ((idx % (8 * 8)) // 8) * bin_size + bin_size // 2
        b = (idx % 8) * bin_size + bin_size // 2
        
        pixel_count = flat_hist[idx]
        percentage = (pixel_count / np.sum(flat_hist)) * 100
        
        print(f"RGB({r}, {g}, {b}): {percentage:.2f}% of pixels")
        
        # Check which feature this might correspond to
        for feature, color in dataset.colors.items():
            if np.all(np.abs([r, g, b] - color) <= dataset.tolerances[feature]):
                print(f"  Matches feature: {feature}")

# if __name__ == "__main__":
#     print("Verifying segmentation...")
#     verify_segmentation()
#     print("\nAnalyzing pixel distribution...")
#     analyze_pixel_distribution()

def test_determinism():
    opt = Options()
    dataset = CustomMapDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Make sure shuffle is False
    
    print("Running first segmentation...")
    data1 = next(iter(dataloader))
    seg_mask1 = data1['seg_mask'][0].numpy()
    
    print("Running second segmentation...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data2 = next(iter(dataloader))
    seg_mask2 = data2['seg_mask'][0].numpy()
    
    # Resize both masks to 256x256
    resized_seg_mask1 = np.zeros((5, 256, 256))
    resized_seg_mask2 = np.zeros((5, 256, 256))
    for i in range(seg_mask1.shape[0]):
        resized_seg_mask1[i] = cv2.resize(seg_mask1[i], (256, 256), interpolation=cv2.INTER_NEAREST)
        resized_seg_mask2[i] = cv2.resize(seg_mask2[i], (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Compare the masks
    for i, name in enumerate(['Road', 'Vegetation', 'Water', 'Building', 'Other']):
        agreement = np.mean(resized_seg_mask1[i] == resized_seg_mask2[i]) * 100
        print(f"{name} mask agreement between runs: {agreement:.2f}%")
    
    # Check if they're exactly the same
    is_identical = np.array_equal(resized_seg_mask1, resized_seg_mask2)
    print(f"\nSegmentation masks are {'identical' if is_identical else 'different'} between runs")

# if __name__ == "__main__":
#     print("Testing segmentation determinism...")
#     test_determinism()

def verify_segmentation_raw():
    opt = Options()
    opt.no_flip = True
    opt.preprocess = 'none'
    opt.load_size = 600
    opt.crop_size = 600
    
    dataset = CustomMapDataset(opt)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    data = next(iter(dataloader))
    
    # Get the map image directly from the data
    map_img = data['B'][0].permute(1, 2, 0).numpy()
    if map_img.max() <= 1.0:
        map_img = (map_img * 255).astype(np.uint8)
    
    # Convert back to tensor format as in dataset
    img_tensor = torch.from_numpy(map_img.transpose(2, 0, 1)).float() / 255.0
    
    print("Input tensor stats:")
    print(f"Shape: {img_tensor.shape}")
    print(f"Value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # Convert tensor to numpy array (CHW to HWC)
    img_np = map_img
    
    print("\nProcessed numpy array stats:")
    print(f"Shape: {img_np.shape}")
    print(f"Value range: [{img_np.min()}, {img_np.max()}]")
    
    # Get pipeline output for comparison
    pipeline_mask = dataset.color_segment(img_tensor)
    
    # Create masks exactly as in pipeline
    building_matrix = np.abs(img_np - dataset.colors['building']) <= dataset.tolerances['building']
    road_matrix = np.add(
        np.add(
            np.add(
                np.abs(img_np - dataset.colors['road']) <= dataset.tolerances['road'],
                np.abs(img_np - dataset.colors['main_road']) <= dataset.tolerances['main_road']
            ),
            np.abs(img_np - dataset.colors['sm_road']) <= dataset.tolerances['sm_road']
        ),
        np.abs(img_np - dataset.colors['med_road']) <= dataset.tolerances['med_road']
    )
    
    # Sample some pixels for debugging
    print("\nSample pixel values:")
    y, x = 300, 300  # Center pixel
    print(f"Original pixel at ({x},{y}): {img_np[y,x]}")
    print(f"Building matrix at ({x},{y}): {building_matrix[y,x]}")
    print(f"Road matrix at ({x},{y}): {road_matrix[y,x]}")
    
    building_mask = np.all(building_matrix, axis=2)
    road_mask = np.all(np.logical_and(road_matrix, np.logical_not(building_matrix)), axis=2)
    vegetation_mask = np.all(np.abs(img_np - dataset.colors['vegetation']) <= dataset.tolerances['vegetation'], axis=2)
    water_mask = np.all(np.abs(img_np - dataset.colors['water']) <= dataset.tolerances['water'], axis=2)
    
    # Apply identical denoising
    building_mask = cv2.medianBlur(building_mask.astype(np.uint8), 5).astype(bool)
    road_mask = cv2.medianBlur(road_mask.astype(np.uint8), 5).astype(bool)
    vegetation_mask = cv2.medianBlur(vegetation_mask.astype(np.uint8), 5).astype(bool)
    water_mask = cv2.medianBlur(water_mask.astype(np.uint8), 3).astype(bool)
    
    print("\nMask statistics:")
    print(f"Road positive pixels: {np.mean(road_mask) * 100:.2f}%")
    print(f"Pipeline road positive pixels: {np.mean(pipeline_mask[0].numpy()) * 100:.2f}%")
    
    print("\nAgreement with pipeline:")
    print(f"Road: {np.mean(road_mask == pipeline_mask[0].numpy()) * 100:.2f}%")
    print(f"Vegetation: {np.mean(vegetation_mask == pipeline_mask[1].numpy()) * 100:.2f}%")
    print(f"Water: {np.mean(water_mask == pipeline_mask[2].numpy()) * 100:.2f}%")
    print(f"Building: {np.mean(building_mask == pipeline_mask[3].numpy()) * 100:.2f}%")

if __name__ == "__main__":
    print("Verifying segmentation with raw images...")
    verify_segmentation_raw()