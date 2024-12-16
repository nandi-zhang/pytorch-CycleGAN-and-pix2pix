import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import cv2

class CycleGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss')
            parser.add_argument('--lambda_dice', type=float, default=10.0, help='weight for dice loss')  # Increased weight
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # Remove discriminator-related losses
        self.loss_names = ['G_cycle_A', 'G_cycle_B', 'idt_A', 'idt_B', 'dice_A']
        
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B
        
        # Remove discriminator from model names
        if self.isTrain:
            self.model_names = ['G_A', 'G_B']
        else:
            self.model_names = ['G_A', 'G_B']

        # Define generators only
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
            # Define loss functions
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # Initialize optimizer for generators only
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_dice = self.opt.lambda_dice

        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # Cycle loss
        self.loss_G_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_G_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Dice loss
        self.loss_dice_A = weighted_dice_loss(self.fake_B, self.real_B) * lambda_dice

        # Combined loss
        self.loss_G = (self.loss_G_cycle_A + self.loss_G_cycle_B + 
                      self.loss_idt_A + self.loss_idt_B +
                      self.loss_dice_A)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

@staticmethod
def weighted_dice_loss(pred_map, target_map):
    """
    Compute weighted Dice loss between generated and ground truth maps with normalized weights
    """
    # Class weights - move to GPU
    weights = {
        'roads': 2.0,      
        'buildings': 2.0,  
        'water': 1.5,      
        'vegetation': 1.0, 
        'other': 0.5      
    }
    
    # Convert weights to tensor and normalize to sum to 1
    weight_values = torch.tensor(list(weights.values()), device=pred_map.device)
    normalized_weights = weight_values / weight_values.sum()
    
    # Get segmentation masks with explicit clamping
    pred_seg = color_segment(pred_map).clamp(0, 1)    # Ensure values between 0 and 1
    target_seg = color_segment(target_map).clamp(0, 1)
    
    # Small epsilon for numerical stability
    eps = 1e-7
    
    # Compute weighted Dice loss for each class
    intersection = (pred_seg * target_seg).sum(dim=(2, 3))
    union = pred_seg.sum(dim=(2, 3)) + target_seg.sum(dim=(2, 3))
    dice_per_class = (2. * intersection + eps) / (union + eps)
    
    # Apply normalized weights and sum
    weighted_dice = (dice_per_class * normalized_weights).sum()
    return 1 - weighted_dice

def color_segment(input_image):
    """
    Segment the input image into different classes based on color thresholds
    """
    batch_size, channels, height, width = input_image.size()
    
    # Create masks tensor on same device as input
    num_classes = 5
    masks = torch.zeros(batch_size, num_classes, height, width, device=input_image.device)
    
    for b in range(batch_size):
        # Keep on GPU, no need to move between devices
        img = input_image[b].permute(1, 2, 0)
        
        # All these operations will stay on GPU
        road_mask = ((img[:,:,0] > 0.4) & (img[:,:,0] < 0.6) &
                    (img[:,:,1] > 0.4) & (img[:,:,1] < 0.6) &
                    (img[:,:,2] > 0.4) & (img[:,:,2] < 0.6))
        
        building_mask = ((img[:,:,0] > 0.6) &
                        (img[:,:,1] < 0.4) &
                        (img[:,:,2] < 0.4))
        
        water_mask = ((img[:,:,0] < 0.4) &
                     (img[:,:,1] < 0.4) &
                     (img[:,:,2] > 0.6))
        
        vegetation_mask = ((img[:,:,0] < 0.4) &
                         (img[:,:,1] > 0.6) &
                         (img[:,:,2] < 0.4))
        
        # Other (everything else)
        other_mask = ~(road_mask | building_mask | water_mask | vegetation_mask)
        
        # All masks are already on GPU, just assign them
        masks[b, 0] = road_mask.float()
        masks[b, 1] = building_mask.float()
        masks[b, 2] = water_mask.float()
        masks[b, 3] = vegetation_mask.float()
        masks[b, 4] = other_mask.float()
    
    return masks