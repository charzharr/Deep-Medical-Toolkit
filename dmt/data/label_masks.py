""" Module: dmt/data/label_masks.py
Label subclasses that are all masks (usually for segmentation). 

TODO:
    - If I have a future project on VOC, add a VectorMask2D.
"""

import collections

import torch
import numpy as np
import SimpleITK as sitk

from .label_base import Label
from .images import ScalarImage3D, ScalarImage2D
from ..constants import IMTYPE_2_NP

from ..utils.io import images2d as im2d_utils, images3d as im3d_utils
from ..utils.images import visualize as viz_utils


class ScalarMask(Label):
    """ Base class for scalar masks. 
    API
        - To get raw mask data in np, tensor, or sitk form, use Image API
            e.g. mask.array, mask.tensor
        - get_one_hot_mask()
        - get_id_mask() 
    Basic mask input assumptions:
        - Masks only contain 1 channel & are non-negative
        - Each class has its own id & its id is mapped to a value via
            the 'class_values' argument.
    """
    
    def get_one_hot(self, channel_first=False, tensor=False):
        """ Outputs an array or tensor that has a separate channel for each
        class. For pixels that belong to a class, value is 1, else 0. 
            NOTE: there's always gonna be at least 2 classes. (back, fore, etc)
        Args:
            channel_first: if True, output dimensions are CxHxW(xD)
            tensor: if True, returns uint8 tensor, else uint8 array.
        """
        id_mask = self.get_ids()
        
        if channel_first:
            shape = [self.num_classes] + list(id_mask.shape)
            one_hot = np.zeros(shape, dtype=np.uint8)
            for c in range(self.num_classes):
                one_hot[c, ...][id_mask == c] = 1
        else:
            shape = list(id_mask.shape) + [self.num_classes]
            one_hot = np.zeros(shape, dtype=np.uint8)
            for c in range(self.num_classes):
                one_hot[..., c][id_mask == c] = 1
        
        if tensor:
            one_hot = torch.from_numpy(one_hot)  # takes numpy dtype, not float        
        
        return one_hot
    
    def get_ids(self, tensor=False):
        """ Outputs a mask that is the same shape as given mask, where each
        pixel has the value of its class ID. 
        Args:
            tensor: if True, returns uint32 tensor, else uint32 array.
        """
        array = self.array
        id_mask = np.zeros(array.shape, dtype=np.uint8)
        for c, val in enumerate(self.class_values):
            id_mask[array == val] = c
        
        if tensor:
            id_mask = torch.from_numpy(id_mask)  # takes numpy dtype, not float

        return id_mask
    
    ### --- Mask Attributes --- ###
    
    @property
    def class_names(self):
        return self._class_names
    
    @property
    def class_values(self):
        return self._class_values
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def class_counts(self):
        if self._class_counts is None:
            self._update_mask_attributes()
        return self._class_counts
    
    @property
    def classes_distribution(self):
        counts = self.class_counts
        return counts / sum(counts)
    
    @property
    def unique_values(self):
        if self._unique_values is None:
            self._update_mask_attributes()
        return self._unique_values
    
    def _parse_class_variables(self, class_names, class_values):
        # Verify if given class_names is valid
        cn_type = type(class_names)
        msg = f'Argument "class_names" must be a sequence. Given: {cn_type}.'
        assert isinstance(class_names, collections.Sequence), msg
        
        try:
            class_names = [str(cn) for cn in list(class_names)]
        except:
            msg = ('Argument "class_names" must be a sequence where each '
                   'element can be represented as a string.')
            raise ValueError(msg)
        
        # Verify if class_values are correct
        if class_values is None or len(class_values) == 0:
            class_values = list(range(len(class_names)))  # init default values
        
        cv_type = type(class_values)
        msg = f'Argument "class_values" must be a sequence. Given: {cv_type}.'
        assert isinstance(class_values, collections.Sequence), msg

        try:
            class_values = [int(cv) for cv in list(class_values)]
        except:
            msg = ('Argument "class_values" must be a sequence where each '
                   'element can be represented as an integer.')
            raise ValueError(msg)
        
        msg = (f'Number of class_values ({len(class_values)} does not match '
               f'the number of classnames given ({len(class_names)})')
        assert len(class_names) == len(class_values), msg
            
        return class_names, class_values
    
    def _update_mask_attributes(self, sitk_im=None):
        """ Kept separate function in-case needed more specific func later.
        Args:
            sitk_im: given in __init__ so that sitk image does not have to be 
                reread from path if permanent_load is not on.
        """
        if sitk_im is None:
            sitk_im = self.sitk_image
        np_type = IMTYPE_2_NP[self.type]
        array = self.get_array(sitk_im, np_type, isotropic=False)
        unique, counts = np.unique(array, return_counts=True)
        
        num_classes = len(unique)
        msg = (f'Given {self.num_classes} classes, but this mask has '
               f'{num_classes} unique values!')
        assert len(self._class_names) >= num_classes, msg
        
        # check if unique values match up with associated class values
        given_vals = self._given_class_values
        if given_vals is None or len(given_vals) == 0:
            pre_msg = (f'During mask init, you gave no class_values, so it was '
                       f'defaulted to {self._class_values}. ')
        else:
            pre_msg = (f'During mask init, you gave these class_values: '
                       f'{self._class_values}.')
        post_msg = (f'When loading the image, it contained values '
                    f'({unique}) that were not defined in "class_values".')
        msg = pre_msg + post_msg
        for unique_val in unique:
            assert unique_val in self._class_values, msg
            
        class_counts = np.array([0] * self.num_classes)
        for u, c in zip(unique, counts):
            cid = self.class_values.index(u)
            class_counts[cid] = c
        self._class_counts = class_counts
        
        self._unique_values = unique


class ScalarMask2D(ScalarImage2D, ScalarMask):
    
    """
    Args:
        path_or_data: str, path, np, sitk, or tensor of mask. 
        class_names: list of class names with index = class-id
        class_values: list of intensities in mask image corresponding to class
            e.g. If in mask, 0=background, 100=kidney, 200=tumor, 
                then class_values would be [0, 100, 200] assuming that backgrnd,
                kidney, and tumor have class-ids of 0, 1, and 2, resp. 
            The default class_values are set to range(C), C = # of classes
        container_type: storage type of mask (defaults to uint8)
        permanent_load: flag for mask data to remain in RAM or not
        **kwargs: additional attributes of the mask to be stored. 
    """
    
    def __init__(
            self, 
            path_or_data,                      # image object (most often path)
            class_names,                       # list of classnames
            class_values=None,
            container_type=sitk.sitkUInt8,     # container type
            permanent_load=True,               # low comp overhead for masks
            **kwargs
            ):
        
        super().__init__(   # verifies image stuff
            path_or_data, 
            container_type, 
            permanent_load=permanent_load, 
            **kwargs)
        
        self._given_class_values = class_values  # for informative error msg
        class_names, class_values = self._parse_class_variables(class_names, 
                                                                class_values)
        
        # Store attributes
        self._class_names = class_names
        self._class_values = class_values
        self._num_classes = len(class_names)
        
        self._class_counts = None
        self._unique_values = None
        if self.permanent_load:
            self._update_mask_attributes(sitk_im=self.sitk_image)
    
    
    ### ---- ### ---- \\    Mask Label Properties     // ---- ### ---- ###
        
    def get_array(self, sitk_im, out_type,
                  interpolation='nearest', **kwargs):
        msg = f'Given type "{out_type}" is not a valid numpy dtype.'
        assert out_type in IMTYPE_2_NP.values(), msg
        
        array = sitk.GetArrayFromImage(sitk_im)
        if array.dtype != out_type:
            array = array.astype(out_type)
        return array


class ScalarMask3D(ScalarImage3D, ScalarMask):
    """
    Args:
        path_or_data: str, path, np, sitk, or tensor of mask. 
        class_names: list of class names with index = class-id
        class_values: list of intensities in mask image corresponding to class
            e.g. If in mask, 0=background, 100=kidney, 200=tumor, 
                then class_values would be [0, 100, 200] assuming that backgrnd,
                kidney, and tumor have class-ids of 0, 1, and 2, resp. 
            The default class_values are set to range(C), C = # of classes
        container_type: storage type of mask (defaults to uint8)
        permanent_load: flag for mask data to remain in RAM or not
        **kwargs: additional attributes of the mask to be stored. 
    """
    
    def __init__(
            self, 
            path_or_data,                      # image object (most often path)
            class_names,                       # list of classnames
            class_values=None,
            container_type=sitk.sitkUInt8,     # container type
            permanent_load=True,               # low comp overhead for masks
            **kwargs
            ):
        
        super().__init__(   # verifies image stuff
            path_or_data, 
            container_type, 
            permanent_load=permanent_load, 
            **kwargs)
        
        self._given_class_values = class_values  # for informative error msg
        class_names, class_values = self._parse_class_variables(class_names, 
                                                                class_values)
        
        # Store attributes
        self._class_names = class_names
        self._class_values = class_values
        self._num_classes = len(class_names)
        
        self._class_counts = None
        self._unique_values = None
        if self.permanent_load:
            self._update_mask_attributes(sitk_im=self.sitk_image)
    
    ### ---- ### ---- \\    Mask Label Properties     // ---- ### ---- ###
    
    def get_array(self, sitk_im, out_type,
                  isotropic=False, interpolation='nearest', **kwargs):
        """ OVERLOAD - if you want custom sitk to numpy conversion in subclass. 
        This function overloads the get_array() from Image. 
        """
        msg = f'Given type "{out_type}" is not a valid numpy dtype.'
        assert out_type in IMTYPE_2_NP.values(), msg
        
        if isotropic:
            sitk_im = im3d_utils.resample_sitk_isotropic(
                            sitk_im, interpolation=interpolation, smooth=0)
        array = sitk.GetArrayFromImage(sitk_im)
        if array.dtype != out_type:
            array = array.astype(out_type)
        # print(self.type, np_image.dtype)
        return array
    
    
    
    
        
    
    

                
