""" Module data/samples/images.py 
Extensions of the base image class for generalized loading of most used
image formats in medical imaging research. 

Organization (class definitions in order): 
    VectorImage2D
    ScalarImage2D
    ScalarImage3D
"""

from __future__ import annotations  # allows Class annotations before definition
import sys, os
import pathlib
import logging

import torch
import PIL
import numpy as np
import SimpleITK as sitk

from ..utils.io import images2d as im2d_utils, images3d as im3d_utils
from ..utils.images import visualize as viz_utils
from .image_base import Image, VALID_INPUT_TYPES
from ..constants import (
    IMTYPE_2_SCALAR_SITK, 
    IMTYPE_2_VECTOR_SITK, 
    IMTYPE_2_NP, 
    IMTYPE_2_TORCH
)


### ---- ### ---- \\    2D Images     // ---- ### ---- ###

class VectorImage2D(Image):
    
    def __init__(
            self, 
            path_or_data,                             # image object
            container_type=sitk.sitkVectorUInt8,      # sitk container type
            permanent_load=False,
            **kwargs
            ):
        # Validate 
        self._validate_init_args(path_or_data, container_type)
        self._is_2d = True
        
        super().__init__(
            path_or_data, 
            container_type, 
            permanent_load=permanent_load, 
            **kwargs
        )
        logging.debug(f'🖼️ Image object created: {repr(self)}')
    
    def get_array(self, sitk_im, out_type, channel_first=False, **kwargs):
        msg = f'Given type "{out_type}" is not a valid numpy dtype.'
        assert out_type in IMTYPE_2_NP.values(), msg
        
        array = sitk.GetArrayFromImage(sitk_im)
        if array.dtype != out_type:
            array = array.astype(out_type)
        
        if channel_first:
            array = np.moveaxis(array, -1, 0)
        return array
    
    def plot(self, save=''):
        image = self.array
        title = self.name
        viz_utils.plot_2d_image(image, title=title, image_size=10, save=save)
    
    def _validate_init_args(self, path_or_data, container_type):
        msg = (f'Given type must be one of the following sitk vector types: '
               f'{IMTYPE_2_VECTOR_SITK.values()}')
        assert container_type in IMTYPE_2_VECTOR_SITK.values(), msg
        
        msg = (f'Input type (given: {type(path_or_data)}) is invalid. ' 
               f'Only these types are accepted: {VALID_INPUT_TYPES}')
        assert isinstance(path_or_data, VALID_INPUT_TYPES), msg
        
        if isinstance(path_or_data, (str, pathlib.Path)):
            msg = f"File doesn't exist: {path_or_data}"
            assert os.path.isfile(str(path_or_data)), msg
        elif isinstance(path_or_data, sitk.SimpleITK.Image):
            if path_or_data.GetPixelID() in IMTYPE_2_SCALAR_SITK.values():
                msg = f'If you input a sitk image, it must be a vector type.'
                raise ValueError(msg)
        elif isinstance(path_or_data, (np.ndarray, torch.Tensor)):
            shape = path_or_data.shape
            msg = f'Input array or tensor must have 3 relevant dimensions.'
            assert sum([i > 1 for i in shape]) == 3, msg
            msg = (f'Input image must be in channels-last format, and the last '
                   f'channel must have 3 dimensions (given: {shape}).')
            assert shape[-1] == 3, msg


class ScalarImage2D(Image):
    """
    Behavior
     - If a 2D RGB image is given in the form of data, then it will do its best
        to convert it to grayscale.
     - Images with any relevant dim of size 1 cannot be given as array or tensor
        e.g. 3x1 image <- dim 1 gonna be squeezed(), 4x4x1 or 1x4x4x1 OK
    """
    
    def __init__(
            self, 
            path_or_data,                       # image object
            container_type=sitk.sitkUInt8,      # sitk container type
            permanent_load=False,
            **kwargs
            ):
        # Validate 
        self._validate_init_args(path_or_data, container_type)
        path_or_data = self._process_image_input(path_or_data, container_type)
        self._is_2d = True
        
        super().__init__(
            path_or_data, 
            container_type, 
            permanent_load=permanent_load, 
            **kwargs
        )
        logging.debug(f'🖼️ Image object created: {repr(self)}')
    
    def get_array(self, sitk_im, out_type, **kwargs):
        msg = f'Given type "{out_type}" is not a valid numpy dtype.'
        assert out_type in IMTYPE_2_NP.values(), msg
        
        array = sitk.GetArrayFromImage(sitk_im)
        if array.dtype != out_type:
            array = array.astype(out_type)
        return array
    
    def plot(self, save=''):
        image = self.array
        title = self.name
        viz_utils.plot_2d_image(image, title=title, image_size=10, save=save)
    
    def _validate_init_args(self, path_or_data, container_type):
        msg = (f'Given type must be one of the following sitk vector types: '
               f'{IMTYPE_2_SCALAR_SITK.values()}')
        assert container_type in IMTYPE_2_SCALAR_SITK.values(), msg
        
        msg = (f'Input type (given: {type(path_or_data)}) is invalid. ' 
               f'Only these types are accepted: {VALID_INPUT_TYPES}')
        assert isinstance(path_or_data, VALID_INPUT_TYPES), msg
        
        if isinstance(path_or_data, (str, pathlib.Path)):
            msg = f"File doesn't exist: {path_or_data}"
            assert os.path.isfile(str(path_or_data)), msg    
        
    def _process_image_input(self, path_or_data, container_type):
        """ Turn any RGB image into a grayscale sitk if data is given. """
        if isinstance(path_or_data, (str, pathlib.Path)):
            path_or_data = str(path_or_data)
            return path_or_data
        
        # If sitk is a vector type object, convert to numpy then grayscale
        if isinstance(path_or_data, sitk.SimpleITK.Image):
            if path_or_data.GetPixelID() in IMTYPE_2_SCALAR_SITK.values():
                return path_or_data
            path_or_data = sitk.GetArrayFromImage(path_or_data)
        
        # If input is a tensor, convert it to numpy for further processing
        if isinstance(path_or_data, torch.Tensor):
            path_or_data = path_or_data.detach().cpu().numpy()
        
        # If numpy is given or has been processed, turn image into grayscale
        assert isinstance(path_or_data, np.ndarray), 'Oops, missed something'
        
        array = path_or_data.squeeze()
        if array.ndim == 2:  # gray image
            pass
        elif array.ndim == 3 and 3 in array.shape:  # assumed RGB
            assert array.shape[-1] == 3, 'Image must be in channel last format.'
            array = np.dot(array[...,:3], [0.2989, 0.5870, 0.1140])
            assert array.ndim == 2, 'Losing sanity rapidly..'
        else:
            msg = (f"Only grayscale images & RGB images can be inputted, "
               f"not image of shape {array.shape}.")
            raise ValueError(msg)

        sitk_im = sitk.GetImageFromArray(array, isVector=False)
        if sitk_im.GetPixelID() != container_type:
            sitk_im = sitk.Cast(sitk_im, container_type)
        return sitk_im
        
        
### ---- ### ---- \\    3D Images     // ---- ### ---- ###
        

class ScalarImage3D(Image):
    """ 
    Overloaded: get_array(), plot().
    Default Functionality: read_image(), get_tensor().
    """
    
    VALID_EXTS = ['.dcm', '.nii', '.nii.gz']
    
    def __init__(
            self, 
            path_or_data,                       # image object
            container_type=sitk.sitkInt16,      # sitk container type
            permanent_load=False,
            **kwargs
            ):
        # Validate 
        self._validate_init_args(path_or_data, container_type)
        self._is_2d = False
        
        super().__init__(
            path_or_data, 
            container_type, 
            permanent_load=permanent_load, 
            **kwargs
        )
        logging.debug(f'🖼️ Image object created: {repr(self)}')
    
    def get_array(self, sitk_im, out_type,
                  isotropic=False, interpolation='bspline', **kwargs):
        """
        Args:
            **kwargs: for compatibility and consistent API. 
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
        
    def plot(self, save=''):
        image = self.array
        viz_utils.plot_3d_slices(image, save=save)
    
    def _validate_init_args(self, path_or_data, container_type):
        msg = (f'Given type must be one of the following sitk scalars: '
               f'{IMTYPE_2_SCALAR_SITK.values()}')
        assert container_type in IMTYPE_2_SCALAR_SITK.values(), msg
        
        msg = (f'Input type (given: {type(path_or_data)}) is invalid. ' 
               f'Only these types are accepted: {VALID_INPUT_TYPES}')
        assert isinstance(path_or_data, VALID_INPUT_TYPES), msg
        
        if isinstance(path_or_data, (str, pathlib.Path)):
            path_or_data = str(path_or_data)
            msg = f"File doesn't exist: {path_or_data}"
            assert os.path.isfile(str(path_or_data)), msg
            
            msg = f'Given file {path_or_data} must have {self.VALID_EXTS} exts.'
            cond = sum([path_or_data[-len(s):] == s for s in self.VALID_EXTS])
            assert cond, msg
        elif isinstance(path_or_data, sitk.SimpleITK.Image):
            if path_or_data.GetPixelID() in IMTYPE_2_VECTOR_SITK.values():
                msg = f'If you input a sitk image, it must be a scalar type.'
                raise ValueError(msg)
        elif isinstance(path_or_data, (np.ndarray, torch.Tensor)):
            shape = path_or_data.shape
            msg = f'Input array or tensor must have 3 relevant dimensions.'
            assert sum([i > 1 for i in shape]) == 3, msg
                


