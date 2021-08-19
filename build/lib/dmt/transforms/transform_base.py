
import os
import numbers
import copy
from abc import ABC, abstractmethod
import torch

from dmt.data.samples.sample import Sample
from dmt.transforms.harmonizer import ImageHarmonizer
from dmt.utils.parse import parse_bool, parse_probability
from dmt.constants import (
    INTERPOLATIONS_2_SITK_INTERPOLATIONS as interpolations_d
)


class Transform(ABC):
    """ Abstract & base class for all image transforms.
    Implementation Structure:
        - all subclasses must define apply (opt. invertible) function
        - base class has parsing functions (input checking & processing)
    
    Transform inputs can be: 
        1. Raw Images: np.ndarray, torch.Tensor, PIL.Image, SimpleITK.Image
        2. Containers: rere.Image
    
    Args:
        p: Probability that this transform will be applied.
        copy: Make a shallow copy of the input before applying the transform.
        include_keys: Key names in samples that will be transformed if they
            are images.
        exclude_keys: Key names in samples that will be ignored. 
    """
    
    ### ------ #     Main API & Transformation Functionality      # ----- ###
    
    def __init__(self, p=1.0, copy=True, include_keys=None, exclude_keys=None):
        self.p = parse_probability(p)
        self.copy = parse_bool(copy)
        keys = Sample._parse_include_exclude_keys(include_keys, exclude_keys)
        self.include_keys, self.exclude_keys = keys
        
        self.transform_args = ()  # for reproducibility; updated in subclasses
        
    def __call__(self, data, include_keys=None, exclude_keys=None):
        """
        Handles functionality for all transforms:
            1. Determine if transform should be applied given probability.
            2. Standardize the image types given via a DataHarmonizer
            3. Copy sample if necessary.
            4. Transform the sample (rewrites new data to sample)
            5. Record the transformation arguments into new sample.
        Args:
            data: Can be a Sample, Image, Harmonizer, Torch tensor, Numpy array, 
                or SimpleITK Image.
        Returns:
            Transformed image(s) in the original type given. 
        """
        if torch.rand(1).item() > self.p:
            return data
        
        keys = include_keys, exclude_keys
        include_keys, exclude_keys = Sample._parse_include_exclude_keys(*keys)
        
        # overwrites keys specified in init if given here
        final_include_keys = self.include_keys
        if include_keys:  
            final_include_keys = include_keys 
        final_exclude_keys = self.exclude_keys
        if exclude_keys:
            final_exclude_keys = exclude_keys
        
        harmonizer = ImageHarmonizer(data)
        sample = harmonizer.get_sample()
        if self.copy:
            sample = copy.copy(sample)
            
        transformed_sample = self.apply_transform(sample)
        self._record_transform(transformed_sample)
        
        # convert to input type 
        out_data = harmonizer.get_output(transformed_sample)  
        return out_data
        
    @abstractmethod
    def apply_transform(self, sample):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def is_invertible(self):
        return hasattr(self, 'invert')
    
    ### ------ #      Transform Recording & Replication      # ----- ###
    
    def _record_transform(self, transformed_sample):
        reproducing_args = self.get_reproducing_arguments()
        transformed_sample.record_transform(reproducing_args)
    
    def get_reproducing_arguments(self):
        """
        Return a dictionary with the arguments that would be necessary to
        reproduce the transform exactly (arguments are from previous transform).
        """
        reproducing_arguments = {
            'name': self.name,
            'include_keys': self.include_keys,
            'exclude_keys': self.exclude_keys,
            'copy': self.copy,
            'p': self.p
        }
        t_args = {name: getattr(self, name) for name in self.transform_args}
        reproducing_arguments.update(t_args)
        return reproducing_arguments


