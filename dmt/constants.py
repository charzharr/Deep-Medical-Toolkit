

import SimpleITK as sitk
import numpy as np
import torch


### ---- ### ---- \\    Keys     // ---- ### ---- ###

IMAGE = 'image'
SHAPE = 'size'

### ---- ### ---- \\    Image Constants     // ---- ### ---- ###


IMTYPE_2_NP = {
    'bool': np.bool_,
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint32': np.uint32,
    'int8': np.int8,
    'int16': np.int16,
    'int': np.int32,
    'int32': np.int32,
    'long': np.int64,
    'int64': np.int64,
    'float': np.float32,
    'float32': np.float32,
    'double': np.float64,
    'float64': np.float64
}

IMTYPE_2_TORCH = {
    'bool': torch.bool,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'int': torch.int32,
    'int32': torch.int32,
    'long': torch.int64,
    'int64': torch.int64,
    'float': torch.float32,
    'float32': torch.float32,
    'double': torch.float64,
    'float64': torch.float64
}

IMTYPE_2_SCALAR_SITK = {
    'uint8': sitk.sitkUInt8,
    'uint16': sitk.sitkUInt16,
    'uint32': sitk.sitkUInt32,
    'int8': sitk.sitkInt8,
    'int16': sitk.sitkInt16,
    'int': sitk.sitkInt32,
    'int32': sitk.sitkInt32,
    'long': sitk.sitkInt64,
    'int64': sitk.sitkInt64,
    'float': sitk.sitkFloat32,
    'float32': sitk.sitkFloat32,
    'double': sitk.sitkFloat64,
    'float64': sitk.sitkFloat64
}

IMTYPE_2_VECTOR_SITK = {
	'uint8': sitk.sitkVectorUInt8,
    'uint16': sitk.sitkVectorUInt16,
    'uint32': sitk.sitkVectorUInt32,
    'int8': sitk.sitkVectorInt8,
    'int16': sitk.sitkVectorInt16,
    'int': sitk.sitkVectorInt32,
    'int32': sitk.sitkVectorInt32,
    'long': sitk.sitkVectorInt64,
    'int64': sitk.sitkVectorInt64,
    'float': sitk.sitkVectorFloat32,
    'float32': sitk.sitkVectorFloat32,
    'double': sitk.sitkVectorFloat64,
    'float64': sitk.sitkVectorFloat64
}

IMAGE_TYPES_STR = ['bool', 'uint8', 'uint16', 'uint32', 'int8', 'int16', 
               'int', 'int32', 'long', 'int64', 
               'float', 'float32', 'double', 'float64']

IMAGE_TYPES = list(IMTYPE_2_NP.values()) + list(IMTYPE_2_TORCH.values()) + \
    list(IMTYPE_2_SCALAR_SITK.values()) + list(IMTYPE_2_VECTOR_SITK.values())
    
INTERPOLATIONS_2_SITK_INTERPOLATIONS = {
    'nearest': 'sitkNearestNeighbor',  # Copies NN position of non-int pixel position
    'linear': 'sitkLinear', # Linearly interpolates intensity at a non-integer pixel position.
    'bspline': 'sitkBSpline',  # Cubic interpolation
    'cubic': 'sitkBSpline',  # Same as bspline
    'gaussian': 'sitkGaussian',  # sigma=0.8, alpha=4
    'label_gaussian': 'sitkLabelGaussian',  # smooth interp multi-label images, σ=1, α=1
    'hamming': 'sitkHammingWindowedSinc',  # Hamming windowed sinc kernel.
    'cosine': 'sitkCosineWindowedSinc',  # Cosine windowed sinc kernel.
    'welch': 'sitkWelchWindowedSinc',  # Welch windowed sinc kernel.
    'lanczos': 'sitkLanczosWindowedSinc',  #: Lanczos windowed sinc kernel.
    'blackman': 'sitkBlackmanWindowedSinc'  #: Blackman windowed sinc kernel.
}