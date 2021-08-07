""" Module utils/io/images3d.py 
Contains common utilities for 3D images (most functionality through sitk).
"""

import sys, os
import SimpleITK as sitk
import numpy as np

from . import files as file_utils

### Constants ###
IMAGE_EXTS = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif',
              '.nii', '.nii.gz', '.dcm']


### ======================================================================== ###
### * ### * ### * ### *          API Definitions         * ### * ### * ### * ### 
### ======================================================================== ###

__all__ = ['read_sitk3d',
           'write_sitk_gray3d', 'write_np_gray3d',
           'to_np_channel_last', 'to_np_channel_first'
          ]

### ---- ### ---- \\    Image Read/Write     // ---- ### ---- ###

def read_sitk3d(im_path, pixtype=sitk.sitkInt16):
    image = sitk.ReadImage(im_path, pixtype)
    dims = image.GetNumberOfComponentsPerPixel()
    assert dims == 1, f"Only 3D gray sitk images are supported."
    return image


def write_sitk_gray3d(sitk_im, path, compress=False):
    r""" Save 3D sitk grayscale image. 
    Args:
        sitk_im: SimpleITK.Image
        path: str or pathlib.Path
        compress: bool
            If save-type is .nii or .nii.gz, doesn't affect output.
    """
    channels = sitk_im.GetNumberOfComponentsPerPixel()
    assert channels == 1, f"Only 3D gray sitk images are supported."
    _write_sitk_image(sitk_im, path, compress=compress)


def write_np_gray3d(np_im, path, extra_channel=False, compress=False):
    r""" Save 3D grayscale image of shape 1xHxWxD, HxWxD, or HxWxDx1 """
    if extra_channel:
        np_im = np_im.squeeze(0).squeeze(-1)
    assert np_im.ndim == 3, f"Only 3D gray images are supported."
    img = sitk.GetImageFromArray(np_im, isVector=False, compress=False)
    _write_sitk_image(img, path, compress=compress)


def _write_sitk_image(img, path, compress=False):
    r""" Creates directory structure and writes using sitk's writer. 
    Arguments:
        img (sitk image)
        path (str)
    Returns:
        True / False (bool) if successful or failed.
    """ 
    file_utils.create_dirs_from_file(path)
    try:
        writer = sitk.ImageFileWriter()
        writer.SetFileName(path)
        writer.SetUseCompression(compress)
        writer.Execute(img)
        return True
    except:
        print(f"Error writing image to {path}!")
        file_utils.delete_file(path)
        return False


### ---- ### ---- \\    Common np Operations     // ---- ### ---- ###


def hu_to_np_image(sitk_np_im, clip=[-1024, 325], scale=[0, 255], 
                  outtype=np.uint8, rgb=False):
    r""" 1. Clip  2. Normalize + Scale  3. Cast Type (opt to RGB) """
    np_im = sitk_np_im
    if isinstance(sitk_np_im, sitk.Image):
        np_im = sitk.GetImageFromArray(sitk_np_im, isVector=False)
        np_im = np_im.astype(np.float32)
    assert isinstance(np_im, np.ndarray)
    if clip:
        np_im = np.clip(np_im, clip[0], clip[1])
    
    # normalize between 0 & 1
    im_min, im_max = np_im.min(), np_im.max()
    np_im = (np_im - im_min) / (im_max - im_min + 1e-6)
    if scale:
        assert scale[1] > scale[0]
        np_im = np_im * (scale[1] - scale[0]) - scale[0]
    if rgb:
        np_im = np.stack([np_im, np_im, np_im], axis=-1)
    if outtype is np.uint8:
        np_im = np.round(np_im)    
    np_im = np_im.astype(outtype)
    return np_im


### ---- ### ---- \\    Common sitk Operations     // ---- ### ---- ###


def resample_sitk_isotropic(sitk_im, spacing=None, smooth=2,
                             interpolation='nearest'):
    r"""
    Args:
        sitk_im: sitk.Image
            sitk image object to be made isotropic.
        ?spacing: None or float
            If None, then uses minimum spacing as reference, else the given one.
        ?smooth: None or False or positive float
            Applies Gaussian smoothing before resampling.
        ?interpolation: str ∈ {'nearest', 'linear'}
            Apply either linear or nearest neighbor interpolation
    """
    orig_size = sitk_im.GetSize()
    ndim = len(orig_size)
    orig_spacing = sitk_im.GetSpacing()
    if all(s == orig_spacing[0] for s in orig_spacing):
        return sitk_im
    
    ref_spacing = min(orig_spacing) if spacing is None else spacing
    new_spacing = [ref_spacing] * ndim
    new_size = [int(round(orig_size[i] * orig_spacing[i] / ref_spacing)) \
                for i in range(ndim)]
    if 'linear' in interpolation:
        interpolation = sitk.sitkLinear
    else:
        interpolation = sitk.sitkNearestNeighbor
    
    if smooth and smooth > 0:
        sitk_im = sitk.SmoothingRecursiveGaussian(sitk_im, smooth)
    direction = sitk_im.GetDirection()
    resampled_im = sitk.Resample(sitk_im, new_size, sitk.Transform(),
                                 interpolation, sitk_im.GetOrigin(),
                                 new_spacing, direction, 0, 
                                 sitk_im.GetPixelIDValue())
    return resampled_im


### ---- ### ---- \\    Image Transforms     // ---- ### ---- ###


### Image Storage Transforms

def sitk_to_np(sitk_im):
    return sitk.GetArrayFromImage(sitk_im)


def np_to_sitk(np_im, has_channels=False):
    r"""
    Args:
        has_channels: bool
            If image is rgb, has_channels=True
    """
    return sitk.GetImageFromArray(np_im, isVector=has_channels)
    

def to_np_channel_last(np_im):
    if np_im.shape[0] not in range(1, 6):
        print(f"(to_np_channel_last) WARNING: Weird image {np_im.shape}")
    return np.moveaxis(np_im, 0, -1)


def to_np_channel_first(np_im):
    if np_im.shape[-1] not in range(1, 6):
        print(f"(to_np_channel_first) WARNING: Weird image {np_im.shape}")
    return np.moveaxis(np_im, -1, 0)
        