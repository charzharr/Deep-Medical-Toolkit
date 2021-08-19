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
import nibabel as nib

import dmt
from dmt.data import Sample
from dmt.transforms import ImageHarmonizer

from dmt.utils.parse import parse_bool, parse_positive_int


class PatchSampler(ABC):
    """ Abstract class & base functionality definition for patch samplers.
    
    Implmentation Notes:
        - Any child class of this base class must implement self.ndim for 2D/3D
        - __init__ always only takes patch_size
        - __call__ is pretty much the only API of this class. It returns a
            generator of patches in the input image format you gave. 
    """
    
    def __init__(self, patch_size, return_sample=False, copy=True):
        """
        Args:
            patch_size: a sequence indicating the patch_size to crop
                Can be (1) single int or sequence of length 1
                       (2) a sequence of appropriate dimensionality
            return_sample: flag that returns a sample if True, otherwise it
                returns an image of the same type as the one given to __call__.
            copy: flag that copies the harmonizer's sample after every get
        """
        self.patch_size = self._parse_patch_size(patch_size, self.ndim)
        self.return_sample = parse_bool(return_sample, 'return_sample')
        self.copy = parse_bool(copy, 'copy')

    def __call__(self, data, num_patches=None, include_keys=None):
        """ Main API for sampler calls. Returns a generator of crops. 
        Args:
            data: Can be a Sample, Image, Harmonizer, Torch tensor, Numpy array, 
                or SimpleITK Image.
            num_patches: number of patches the iterator will return. If it's
                None or <= 0, an infinite iterator will be returned.
            include_keys: Only used if a sample or dict is given. Only images
                with keys in include_keys will be cropped. If None, all images
                will be cropped. 
        Returns:
            Iterator of transformed image(s) in the original type given. 
        """
        
        include_keys, _ = Sample._parse_include_exclude_keys(include_keys, None)
        harmonizer = ImageHarmonizer(data, include_keys=include_keys, 
                                     copy=self.copy)

        # make sure crop size is smaller than all relevant images
        images_shape = harmonizer.consistent_shape
        if images_shape is None or np.any(self.patch_size > images_shape):
            msg = (f'Patch size {tuple(self.patch_size)} cannot be'
                   f' larger than image size {tuple(images_shape)}')
            raise RuntimeError(msg)
        return self._generate_patches(harmonizer, num_patches=num_patches)
        
    
    @abstractmethod
    def _generate_patches(self, sample, num_patches=None):
        """ Implement for all PatchSampler inheritors. 
        Returns a generator of patches to __call__(). """
        pass
    
    def crop(self, harmonizer, lower_indices, patch_size):
        upper_indices = lower_indices + patch_size
        
        # Crop Transform
        sample = harmonizer.sample  # copies data automatically
        for k in harmonizer.include_keys:
            image = sample[k]
            cropim = self.crop_dmt_image(image, lower_indices, upper_indices)
            sample[k] = cropim
            
        # Record information
        location = lower_indices.tolist() + upper_indices.tolist()
        sample.record_transform({
            'name': 'Crop',
            'p': 100,
            'include_keys': harmonizer.include_keys,
            'location': location
        })
        if self.return_sample:
            return sample
        crop_output = harmonizer.get_output(sample)
        return crop_output
    
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
    
    def __repr__(self):
        string = (
            f"{type(self).__name__}\n"
            f"  Patch Size = {self.patch_size} "
        )
        return string


class RandomSampler(PatchSampler):
    
    @abstractmethod
    def get_probability_map(self, harmonizer):
        raise NotImplementedError


##### ----- ### ------ #  Patch Dimension Base Classes   # ----- ### ----- #####

class PatchSamplerScalar3D(PatchSampler):
    ndim = 3
    
    def crop_dmt_image(self, image, lower_indices, upper_indices):
        new_origin = nib.affines.apply_affine(image.affine, lower_indices)
        new_affine = image.affine.copy()
        new_affine[:3, 3] = new_origin
        
        i0, j0, k0 = lower_indices
        i1, j1, k1 = upper_indices
        cropped_sitk_image = image.image[i0:i1, j0:j1, k0:k1]
        
        from dmt.data import ScalarImage3D
        new_image = ScalarImage3D(cropped_sitk_image)
        # print(image.origin, new_origin, '\n\n', image.affine, new_affine)
        new_image.origin = new_origin
        new_image.affine = new_affine
        return new_image
    
    
class PatchSamplerScalar2D(PatchSampler):
    ndim = 2
    
    def crop_dmt_image(self, image, lower_indices, upper_indices, affine):
        raise NotImplementedError()
    
class PatchSamplerVector2D(PatchSampler):
    ndim = 2
    
    def crop_dmt_image(self, image, lower_indices, upper_indices, affine):
        raise NotImplementedError()
    
    
##### ----- ### ------ #    Sampler Type Base Classes    # ----- ### ----- #####

    
def crop_dmt_image(image, lower_indices, upper_indices, affine):
    affine = image.affine.copy()
    new_origin = nib.affines.apply_affine(image.affine, lower_indices)
    new_affine = image.affine.copy()
    new_affine[:3, 3] = new_origin
    
    i0, j0, k0 = lower_indices
    i1, j1, k1 = upper_indices
    cropped_image = image[:, i0:i1, j0:j1, k0:k1]
    cropped_image.origin = new_origin
    cropped_image.affine = new_affine
    return cropped_image


    
