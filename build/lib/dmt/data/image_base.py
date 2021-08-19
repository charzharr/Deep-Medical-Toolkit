""" Module dmt/data/samples/image_base.py

Contains the base class for all images (2D/3D Scalar/Vector).

Design Notes
 - WARNING: treat dot notation as getting items only (primary implementation
   of kwargs is in a dict); setting items with dot will not update dict entries!
 - Two input categories
    1. file (str_path) <- serialization depends on permanent_load argument
    2. image (sitk, numpy, tensor) <- serialized by default
 - Image properties are lazy loaded (loaded if available on input, if it 
    requires reading then Image attributes are populated when accessed) 
        Properties: shape, spacing, origin, direction, affine.
- For data access, here is what you need to know for function overloading:
    1. self.read_image(): path -> sitk_im  (only if path is inputed)
    2. self.get_sitk(): sitk_im -> sitk_im (serves are processing of sitk)
    3. self.get_array/tensor(): sitk_im -> array/tensor 

"""

from __future__ import annotations
from abc import ABC, abstractmethod

import os, sys
import weakref
import pathlib
import pprint
from numpy.lib.type_check import _asfarray_dispatcher

import torch
import PIL
import numpy as np
import SimpleITK as sitk

from ..constants import (
    IMTYPE_2_SCALAR_SITK, 
    IMTYPE_2_VECTOR_SITK, 
    IMTYPE_2_NP, 
    IMTYPE_2_TORCH)
from ..utils.io import images2d as im2d_utils, images3d as im3d_utils
from ..utils.images import visualize as viz_utils


# Module-specific Constants
VALID_INPUT_TYPES = (str, pathlib.Path, sitk.SimpleITK.Image, 
                     np.ndarray, torch.Tensor)


class Image(dict):
    """ Base class for an image (2D/3D scalar/vector, including masks). 
    
    Under the hood, image backend uses SimpleITK, has near universal support. 
    
    Args:
        path_or_data: str_path, np, sitk, or torch_tensor
            Tensor assumed to be channel_last (like standard numpy).
        container_type: sitk type to store path_or_data as 
            This is a very important setting that informs whether color channels
            are needed or not during data retrieval of any type. Also, informs
            the 'is_vector' property.
        permanent_load: if true, saves sitk image in self.image for fast loading
            Only in effect if a path is given through path_or_data. 
        **kwargs: additional image properties to be added to object
    """
    
    
    # Class Variables
    _id_counter = 0  # global runtime ID generator for all Image subclasses
    _gid_to_instance = weakref.WeakValueDictionary()
    _default_attrs = ['path', 'given_type', 'container_type', 'shape',
                         'spacing', 'origin', 'direction']
    
    def __init__(
            self,
            path_or_data,  # already verified in subclass __init__
            container_type,  # sitk type to store as (in: data) or read (in: path)
            permanent_load=False,  # only applies if input is a path
            **kwargs
            ):
        
        # Global Image ID tracking
        self.gid = Image._id_counter
        Image._gid_to_instance[self.gid] = self
        Image._id_counter += 1
        
        # Load sitk image backend and default attributes
        self.given_type = path_or_data.__class__
        self.container_type = container_type
        self.permanent_load = permanent_load  # updated in _process_input
        
        # Lazy-loaded image attributes to be updated in parse_input
        self._sitk_image = None  # either None (if path given) or sitk image
        self.path = None  # this is None unless updated or given path
        self._shape = None  # lazy-loaded unless data is given as input
        self._spacing = None  # " "  Same for all attributes here & below
        self._origin = None
        self._direction = None
        self._affine = None
        self._process_input(path_or_data, container_type, permanent_load)        
        
        super().__init__(**kwargs)  # update Image attributes to dict
        self.update_attributes()  # make dot notation available
    
    
    ### ------ #  Data Access + Custom Functionality  # ----- ###
    
    def read_image(self, path, sitk_type):
        """ OVERLOAD for custom loading. This is the default implementation.
        Loads an image from file into an sitk image.
            NOTE: this is only called to get sitk image if path is given.
            If any other data was given, then self.sikt_image will be used.
        Args:
            path: str or pathlib to image file
            sitk_type: sitk type to be casted as & to fill isVector argument
        Returns:
            sitk_image
        """
        sitk_type = self._parse_sitk_type(sitk_type)
        path = self._parse_path(path)
        
        sitk_image = sitk.ReadImage(path, sitk_type)
        return sitk_image
    
    def get_array(self, sitk_im, out_type, channel_first=False):
        """ OVERLOAD this default functionality to get array from stik.
        This is probably the most important function in data retrieval. 
            - Given the sitk either from input, file, array, or tensor,
                this function is in charge of preprocessing it & getting array.
            - get_tensor() depends on this
        
        Args:
            sitk_im: SimpleITK.Image which is the default structure in which
                images are stored or read from file.
            out_type: numpy dtype to cast output of conversion to.
            channel_first: If it's a vector image, return channel first format?
        """
        msg = f'Given type "{out_type}" is not a valid numpy dtype.'
        assert out_type in IMTYPE_2_NP.values(), msg
        
        ### Preprocess sitk (do resampling, etc. here)
        if sitk_im.GetPixelID() != self.container_type:
            sitk_im = sitk.Cast(sitk_im, self.container_type)
        
        ### Get numpy array
        array = sitk.GetArrayFromImage(sitk_im)
        if array.dtype != out_type:
            array = array.astype(out_type)
        
        if self.is_vector and channel_first:
            array = np.moveaxis(array, -1, 0)
        return array
        
    def get_tensor(self, sitk_im, out_type, channel_first=True):
        """ OVERLOAD this default functionality to get tensor from np. """
        msg = f'Given type "{out_type}" is not a valid torch tensor dtype.'
        assert out_type in IMTYPE_2_TORCH.values(), msg
        
        general_type_key = self.type
        array_type = IMTYPE_2_NP[general_type_key]
        array = self.get_array(sitk_im, array_type)
        tensor = torch.from_numpy(array)
        
        # Channel-first if is vector
        if self.is_vector and channel_first:
            first_dims = tuple(range(tensor.ndim - 1))
            tensor = tensor.permute(-1, *first_dims)
        tensor = tensor.to(out_type)
        return tensor
    
    def save(self, file):
        """ OVERLOAD if you want saving functionality. 
        You could make use of self.array or self.sitk_image to get image data. 
        """
        sitk.WriteImage(self.sitk_image, file)
        
    def plot(self):
        """ OVERLOAD if you want plotting functionality. 
        You could make use of self.array to get image data. 
        """
        raise NotImplementedError()
    
    
    ### ------ #  Image Properties / Default Attributes  # ----- ###
    
    @property
    def sitk_image(self):
        """ Gets raw SimpleITK.Image object. If this object is None when a path
        is given, then an sitk object is read and returned without interally
        saving.
        """
        sitk_im = self._sitk_image
        if self._sitk_image is None:
            sitk_im = self.read_image(self.path, self.container_type)
            if self._affine is None:  # take the chance to update
                self._update_image_attributes(sitk_im=sitk_im) 
        return sitk_im
    
    @property
    def array(self):
        """ Gets numpy array from output type dervied from give storage type."""
        sitk_im = self.sitk_image
        if self.sitk_image is None:
            sitk_im = self.read_image(self.path, self.container_type)
            if self._affine is None:  # take the chance to update
                self._update_image_attributes(sitk_im=sitk_im) 
        
        # Get corresponding np dtype from sitk type
        general_type_key = self.type
        array_type = IMTYPE_2_NP[general_type_key]
        array = self.get_array(sitk_im, array_type)
        return array
    
    @property
    def tensor(self):
        sitk_im = self.sitk_image
        if self.sitk_image is None:
            sitk_im = self.read_image(self.path, self.container_type)
            if self._affine is None:  # take the chance to update
                self._update_image_attributes(sitk_im=sitk_im) 
            
        # Get corresponding tensor dtype from sitk type
        general_type_key = self.type
        tensor_type = IMTYPE_2_TORCH[general_type_key]
        tensor = self.get_tensor(sitk_im, tensor_type)
        return tensor
    
    
    ### ------ # Image Properties # ----- ###
    
    @property
    def shape(self):
        if self._shape is None:
            self._update_image_attributes()
        return self._shape
    
    @property
    def spacing(self):
        if self._spacing is None:
            self._update_image_attributes()
        return self._spacing
    
    @property
    def origin(self):
        if self._origin is None:
            self._update_image_attributes()
        return self._origin
    
    @origin.setter
    def origin(self, value):
        old_origin = self.origin

        msg = '"Image.origin" must be set to a flattened np.ndarray.'
        assert isinstance(value, np.ndarray), msg
        
        msg = (f'Given "origin" shape {value.shape} does not match with the '
               f'old "origin" shape {old_origin.shape}.')
        assert old_origin.shape == value.shape, msg
        
        self._origin = value
    
    @property
    def direction(self):
        if self._direction is None:
            self._update_image_attributes()
        return self._direction
    
    @property
    def affine(self):
        if self._affine is None:
            self._update_image_attributes()
        return self._affine
    
    @affine.setter
    def affine(self, value):
        old_affine = self.affine
        
        msg = 'Image.affine must be set to a flattened np.ndarray.'
        assert isinstance(value, np.ndarray), msg
        
        msg = (f'Given "affine" shape {value.shape} does not match with the '
               f'old affine shape {old_affine.shape}.')
        assert old_affine.shape == value.shape, msg
        
        self._affine = value
    
    @property
    def memory(self): 
        """ Get memory use in bytes. """
        tot_bytes = 0
        for o in self.__dict__.values():
            if isinstance(o, sitk.SimpleITK.Image):
                tot_bytes += Image.get_sitk_memory(o)
            else:
                tot_bytes += sys.getsizeof(o)
        return tot_bytes
    
    @property
    def type(self):
        if self.is_vector:
            sitk_type_dict = IMTYPE_2_VECTOR_SITK 
        else:
            sitk_type_dict = IMTYPE_2_SCALAR_SITK
        type_key = None
        for k, v in sitk_type_dict.items():
            if v == self.container_type:
                type_key = k
                break
        assert type_key is not None, 'Oops, check implementation.'
        return type_key
    
    @property
    def sitk_type(self):
        return sitk.GetPixelIDValueAsString(self.container_type)
    
    @property
    def is_vector(self):
        """ Returns True if given type is a vector image, False otherwise. """
        sitk_type = self.container_type
        isVector = True if sitk_type in IMTYPE_2_VECTOR_SITK.values() else False
        return isVector
    
    @property
    def is_2d(self):
        return self._is_2d
    
    @property
    def pixelid(self):
        return self.container_type
    
    @property
    def name(self):
        path = str(pathlib.Path(self.path).name) if self.path else ''
        name = f"{self.__class__.__name__}_gid{self.gid}"
        if path:
            name += f'_{path}'
        return name
    
    def _update_image_attributes(self, sitk_im=None):
        # print(f"COSTLY!! UPDATING ATTRIBUTES!")
        if sitk_im is None:
            sitk_im = self.sitk_image
        self._shape = np.array(tuple(sitk_im.GetSize()))
        self._spacing = np.array(tuple(sitk_im.GetSpacing()))
        self._origin = np.array(tuple(sitk_im.GetOrigin()))
        self._direction = np.array(tuple(sitk_im.GetDirection()))
        self._affine = Image.get_affine(sitk_im)
    
    def _process_input(self, path_or_data, container_type, permanent_load):
        r""" Processes input and sets image attributes if available. 
        Args:
            path_or_data: str_path, sitk_image, np_array, or torch_tensor
            container_type: casted to this sitk type if input is path
            permanent_load: boolean that is only considered if input is a path
        Returns: None
        """
        if isinstance(path_or_data, (str, pathlib.Path)):
            path_or_data = str(path_or_data)  # convert to str if pathlib
            
            msg = ('If data input is a path, then container_type must be '
                   'given as an sitk image type.')
            assert container_type is not None, msg
            
            container_type = self._parse_sitk_type(container_type)
            path_or_data = self._parse_path(path_or_data)
            
            self.path = path_or_data
            if permanent_load:
                self._sitk_image = self.read_image(path_or_data, container_type)
        elif isinstance(path_or_data, sitk.SimpleITK.Image):
            self._sitk_image = path_or_data  # type will be cast below
        elif isinstance(path_or_data, torch.Tensor):
            path_or_data = path_or_data.detach().cpu().numpy()  # handled later
        
        #  NOTE: PILs are error prone when converting so do it manually.
        # elif isinstance(path_or_data, PIL.Image.Image):
        #     path_or_data = np.array(path_or_data)
        
        if isinstance(path_or_data, np.ndarray):
            isVector = self.is_vector
            self._sitk_image = sitk.GetImageFromArray(path_or_data, 
                                                      isVector=isVector)
        
        # Update image attributes if self.image is defined
        if self._sitk_image is not None:
            if self._sitk_image.GetPixelID() != container_type:
                self._sitk_image = sitk.Cast(self._sitk_image, container_type)
            self.permanent_load = True
            self._update_image_attributes()
        
        # Sanity Checks
        if permanent_load:
            msg = 'Img not there'
            assert isinstance(self._sitk_image, sitk.SimpleITK.Image), msg
        if self.image is None:
            assert self.path, 'If image is None, then path must be set'
            assert os.path.isfile(self.path), 'File {self.path} does not exist.'
            
    def _parse_sitk_type(self, sitk_type):
        valid_sitk_types = list(IMTYPE_2_SCALAR_SITK.values()) + \
                           list(IMTYPE_2_VECTOR_SITK.values())
        msg = f'Given sitk type {str(sitk_type)} isn\'t valid.'
        assert sitk_type in valid_sitk_types, msg
        return sitk_type
    
    def _parse_path(self, path):
        msg = 'Path must be a string or pathlib.Path'
        assert isinstance(path, (str, pathlib.Path)), msg
        
        path = str(path)
        assert os.path.isfile(path), f'File does not exist: {path}'
        return path
    
    
    @staticmethod
    def get_affine(sitk_im):
        ndim = sitk_im.GetDimension()
        affine = np.eye(ndim + 1)

        rotation = np.array(sitk_im.GetDirection()).reshape(ndim, ndim)
        origin = np.array(sitk_im.GetOrigin())
        spacing = np.array(sitk_im.GetSpacing())
        flip_xy = np.diag([1] * ndim)  # used to switch between LPS and RAS
        # print(rotation, origin, spacing, flip_xy)

        M = np.dot(flip_xy, rotation) * spacing
        T = np.dot(flip_xy, origin)
        affine = np.eye(ndim + 1)
        affine[:ndim, :ndim] = M
        affine[:ndim, ndim] = T
        return affine
    
    @staticmethod
    def get_sitk_memory(sitk_im):
        """ Get total memory used by a sitk image object in bytes. """
        import re
        regex = r"Size: [\d]+"
        matches = re.findall(regex, str(sitk_im))[0]
        return int(matches.split()[-1])
        
    @staticmethod
    def get_image_from_id(img_id):
        assert isinstance(img_id, int), f"Image ID must be an integer!"
        if img_id < 0 or img_id >= Image._id_counter:
            return None
        return Image._gid_to_instance[img_id]
    
    def __repr__(self):
        path = '' 
        if self.path:
            path = f"w/ image: {str(pathlib.Path(self.path).name)}"
        sitk_type = sitk.GetPixelIDValueAsString(self.container_type)
        
        affine = self._affine
        if affine is not None:
            affine = affine.flatten()
        
        attrs = {k: v for k, v in self.items() if k not in self._default_attrs}
        attrs_str = pprint.pformat(attrs, indent=1, compact=True)
        
        string = (
            f"{self.__class__.__name__} (gid={self.gid}) {path}\n"
            f"  Given Format (type={self.given_type}) \n"
            f"  Type: {sitk_type}, Shape {self._shape}, Spacing {self._spacing} \n"
            f"  Origin {self._origin}, Direction {self._direction} \n"
            f"  Affine {affine} \n"
            f"  Tot Mem = {self.memory:,} bytes. \n"
            f"Other Attributes: \n {attrs_str} \n"
        )
        return string
    
    ### Functions below offers compatible/correct dict functionality
    ###   e.g. must update __dict__ after every update & override default funcs
    
    def __setattr__(self, name, value):
        """ Setting attrs in dot notation requires updating dict. """
        self[name] = value
        super().__setattr__(name, value)
    
    def __getitem__(self, item):
        # print(f'Getting item: {item}')
        return super().__getitem__(item)
    
    def __setitem__(self, key, value):
        # print(f'Setting {key} to {value}')
        super().__setitem__(key, value)
        self.update_attributes()
        
    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                msg = f"Update expected at most 1 arguments, got {len(args)}."
                raise TypeError(msg)
            for k, v in dict(args[0]).items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v
        self.update_attributes()
            
    def setdefault(self, key, value=None):
        if key not in self:
            self[key] = value
        self.update_attributes()
        return self[key]
    
    def update_attributes(self):
        # Allows attribute access through dot notation, e.g. image.spacing
        self.__dict__.update(self)
    
