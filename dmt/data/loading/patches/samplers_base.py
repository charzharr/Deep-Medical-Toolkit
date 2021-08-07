""" Module rere/data/loading/samplers.py

Image samplers are used in the dataloading process to facilitate creation
of batch elements, especially when a single sample produces multiple examples.

Common use cases of samplers include:
 1. Patch sampling (N patches per volume) in medical image segmentation.
 2. Inference sampling where multiple transforms are applied to an image
     during inference time (samplers coupled with an aggregator).
 3. During 2D self-supervised training when a single image produces 2 large
     crops as well as additional tiny crops (e.g. in SwAV).
"""

import collections
from abc import ABC, abstractmethod
import numpy as np
import torch

from dmt.utils.parse import parse_positive_int


class PatchSampler(ABC):
    
    def __init__(self, patch_size):
        """
        Args:
            patch_size: a sequence indicating the patch_size to crop
                Can be (1) single int or sequence of length 1
                       (2) a sequence of appropriate dimensionality
        """
        self.patch_size = self._parse_patch_size(patch_size, self.ndim)

    def __call__(self, sample, num_patches=None):
        """ Main API for sampler calls. Returns a generator of crops. """
        pass
    
    @abstractmethod
    def _generate_patches(self, sample, num_patches=None):
        """ Implement for all PatchSampler inheritors. """
        pass
    
    @abstractmethod
    def _get_crop_transform(self, crop_indices, patch_size):
        """ Returns the crop transform. Differs for 2D or 3D patch samplers. """
        pass
    
    def _parse_patch_size(self, patch_size, ndim):
        msg = ('"patch_size" must be an int or a sequence of ints, given '
               f'{type(patch_size)}')
        assert isinstance(patch_size, (int, collections.Sequence)), msg
        
        patch_size_list = []
        if isinstance(patch_size, int):
            patch_size = parse_positive_int(patch_size, 'patch_size')
            patch_size_list = [patch_size] * ndim
        else:
            patch_size = list(patch_size)
            msg = ('If "patch_size" is a sequence, it must be either length '
                   f'1 or 3. Given: length {len(patch_size)}')
            assert len(patch_size) in (1, ndim), msg
            
            if len(patch_size) == 1:
                patch_size = [patch_size[0]] * ndim
                
            for size in patch_size:
                size = parse_positive_int(size, 'patch_size number')
                patch_size_list.append(size)
                
        assert len(patch_size_list) == ndim, 'Sanity gone.'
        return np.array(patch_size_list).astype(np.uint16)
    

class PatchSampler3D(PatchSampler):
    ndim = 3
    
class PatchSampler2D(PatchSampler):
    ndim = 2


    
