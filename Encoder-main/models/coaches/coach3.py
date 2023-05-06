# Copyright (c) SenseTime Research. All rights reserved.

from ..coaches.base_coach import BaseCoach
from torchvision.utils import save_image

class  E4EInversion(BaseCoach):
    '''
    use_wandb
    '''
    def __init__(self, use_wandb):
        super().__init__(use_wandb)
    def get_image_inversion(self,image):
            return self.get_e4e_inversion(image)
    
