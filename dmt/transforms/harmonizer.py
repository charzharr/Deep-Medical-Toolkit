""" dmt/transforms/harmonizer.py 
Contains the ImageHarmonizer class which takes in various image data types & 
standardizes them to a Sample. This way, images of different types can be
processed, and returned as their original type.
"""


import warnings
import copy
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from dmt.data.image_base import Image
from dmt.data.images import VectorImage2D, ScalarImage2D, ScalarImage3D
from dmt.data.samples.sample import Sample
from dmt.utils.parse import parse_bool


class ImageHarmonizer:
    """ Handles the conversion of various Image types to be stored in a 
    universal format (i.e. the Sample) for easy processing. Also supports
    outputting of the same data-type when processing is done. This is a core
    component of transforms. """
    
    input_types = ('tensor', 'array', 'sitk', 'image', 'sample')
    
    def __init__(
            self, 
            data, 
            include_keys=None, 
            image_key='main_image',
            copy=True
            ):
        """
        Args:
            data: Can be a tensor, array, sitk image, Image object, or Sample.
            include_keys: list of keys to check size compatibility & copy
                data over if input data is a dict or Sample.
            image_key: key-name to be used in sample.
                Only used if input data is not a sample. 
            copy: Flag to copy data components when getting a new sample.
                If harmonizer.sample is called n times, n copies are returned.
        """
        
        self.data = data
        self.image_key = image_key  # key to store Image in new sample
        self.copy = parse_bool(copy, 'copy')
        
        self.is_dict = False
        self.is_tensor = False
        self.is_array = False
        self.is_sitk = False
        self.is_image = False
        self.is_sample = False
        
        self._sample = self._get_sample()
        self.include_keys = self._parse_include_keys(self._sample, include_keys)
        
    @property
    def sample(self):
        """ Gets the sample (the universal image data type). """
        if not self.copy:
            return self._sample
        return copy.copy(self._sample)
    
    @property
    def consistent_shape(self):
        """ Returns the spatial size of the images. """
        sizes = set()
        for k in self.include_keys:
            sizes.add(tuple(self._sample[k].shape))
        assert sizes  # sanity
        if len(sizes) > 1:
            return None
        return np.array(sizes.pop())
    
    def _get_sample(self):
        data = self.data
        if isinstance(data, dict) and not isinstance(data, Sample):
            sample_d = {k: self._parse_data(v) for k, v in data.items()}
            sample = Sample(sample_d)
            self.is_dict = True
        elif isinstance(data, np.ndarray):
            image = self._parse_data(data)
            sample = Sample({self.image_key: image})
            self.is_array = True
        elif isinstance(data, torch.Tensor):
            image = self._parse_data(data)
            sample = Sample({self.image_key: image})
            self.is_tensor = True
        elif isinstance(data, sitk.SimpleITK.Image):
            image = self._parse_data(data)
            sample = Sample({self.image_key: image})
            self.is_sitk = True
        elif isinstance(data, Image):
            sample = Sample({self.image_key: data})
            self.is_image = True
        elif isinstance(data, Sample):
            sample = data
            self.is_sample = True
        else:
            msg = ('Only tensors, ndarrays, sitk images, Image objects, '
                   'Samples, or dictionaries of the previous are valid. '
                   f'You gave {type(data)}') 
            raise ValueError(msg)
        
        assert sample.num_images > 0, 'You did not give a single image.'
        assert sum([self.is_dict, self.is_tensor, self.is_array, self.is_sitk,
                    self.is_image, self.is_sample]) == 1  # sanity
        return sample
    
    def get_output(self, transformed_sample):
        """ Return data in same data-type as given in init. """
        msg = '"transformed_sample" must be a Sample.'
        assert isinstance(transformed_sample, Sample), msg
        
        if self.is_dict:
            out = {}
            for k in self.data.keys():
                out[k] = transformed_sample[k]
            return out
        elif self.is_array or self.is_tensor:
            image_type = infer_image_type(self.data)
            image = transformed_sample[self.image_key]
            out_type = self.data.dtype
            cf = True if image_type.is_channel_first else False
            if self.is_array:
                return image.get_array(image.image, out_type, channel_first=cf)
            return image.get_tensor(image.image, out_type, channel_first=cf)
        elif self.is_sitk:
            image = transformed_sample[self.image_key]
            return image.image
        elif self.is_image:
            return transformed_sample[self.image_key]
        elif self.is_sample:
            return transformed_sample
        raise RuntimeError('Something is very wrong')
    
    def _parse_data(self, data):
        """ Create Image object from generic image data types. """
        inferrable_types = (np.ndarray, torch.Tensor, sitk.SimpleITK.Image)
        if isinstance(self.data, inferrable_types):
            image_type = infer_image_type(data)
            if image_type.is_2d:
                if image_type.is_vector:
                    return VectorImage2D(data)
                return ScalarImage2D(data)
            return ScalarImage3D(data)
        elif isinstance(self.data, Image):
            return Image
        else:
            return data  # dict or Sample entry is not an Image
    
    def _parse_include_keys(self, sample, include_keys):
        if self.is_dict or self.is_sample:
            image_keys = sample.get_image_keys(include_labels=True)
            if not include_keys:
                return image_keys
            final_include_keys = []
            for k in include_keys:
                if k not in image_keys:
                    msg = (f'Given key ({k}) in "include_keys" is either not '
                           f'in the input data or is not an Image')
                    warnings.warn(msg)
                    continue
                final_include_keys.append(k)
            msg = ('The keys in "include_keys" you gave are all useless. '
                   'Either none of them were in data or none were images.')
            assert final_include_keys, msg
            return final_include_keys
        else:
            return [self.image_key]


def infer_image_type(image):
    """ Through some basic assumptions, try to guess whether an image is 
    (1) 2d or 3d, (2) vector or scalar, or (3) channel first / last
    """
    from collections import namedtuple
    type_args = ['is_2d', 'is_vector', 'is_channel_first']
    ImageType = namedtuple('ImageType', type_args)
    
    if isinstance(image, (torch.Tensor, np.ndarray)):
        # using shape to infer image type
        shape = [d for d in tuple(image.shape) if d != 1]  # squeezed shape
        assert len(shape) in (2, 3), 'Only 2D or 3D images supported'
        if len(shape) == 2:
            is_2d, is_vector, is_channel_first = True, False, None
        else:
            if 3 in shape:
                if shape[0] == 3:
                    is_2d, is_vector, is_channel_first = True, True, True
                    assert shape[-1] != 3, f'Cannot infer shape {shape}'
                elif shape[-1] == 3:
                    is_2d, is_vector, is_channel_first = True, True, False
                    assert shape[0] != 3, f'Cannot infer shape {shape}'
                else:  # assume it is a 3D image with 3 as a middle dim
                    is_2d, is_vector, is_channel_first = False, False, None
            else:
                is_2d, is_vector, is_channel_first = False, False, None
    elif isinstance(image, Image):
        if isinstance(image, ScalarImage3D):
            is_2d = False
            is_vector = False
        else:
            assert isinstance(image, (ScalarImage2D, VectorImage2D))
            is_2d = True
            is_vector = True if isinstance(image, VectorImage2D) else False
        is_channel_first = None
    elif isinstance(image, sitk.SimpleITK.Image):
        dims = image.GetDimension()
        assert dims in (2, 3), 'Only 2D or 3D images supported.'
        is_2d = True if dims == 2 else False
        pixel_depth = image.GetNumberOfComponentsPerPixel()
        assert pixel_depth in (1, 3), 'Wtf kind of image is this'
        is_vector = True if pixel_depth > 1 else False
        is_channel_first = None
    elif isinstance(image, nib.Nifti1Image):
        is_2d = False
        is_vector = False
        is_channel_first = None  # N/A
    else:
        msg = ('Only tensors, ndarrays, Image objects, sitk image, nib images'
               f'types are supported for inferring. You gave {type(image)}') 
        raise ValueError(msg)
    
    image_type = ImageType(is_2d, is_vector, is_channel_first)
    return image_type