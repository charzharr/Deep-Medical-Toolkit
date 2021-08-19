

import torch
import numpy as np
from .samplers_base import RandomSampler, PatchSamplerScalar3D


class UniformSampler(RandomSampler):
    
    def get_probability_map(self, harmonizer):
        return np.ones(*harmonizer.consistent_shape)
    
    def _generate_patches(self, harmonizer, num_patches=None):
        image_size = harmonizer.consistent_shape
        lower_range = image_size - self.patch_size
        patches_left = num_patches if num_patches is not None else True
        while patches_left:
            crop_lower_indices = [torch.randint(v + 1, (1,)).item() 
                                  for v in lower_range]
            lower_indices_array = np.array(crop_lower_indices)
            yield self.crop(harmonizer, lower_indices_array, self.patch_size)
            if num_patches is not None:
                patches_left -= 1
                
                
class UniformSampler3D(UniformSampler, PatchSamplerScalar3D):
    pass