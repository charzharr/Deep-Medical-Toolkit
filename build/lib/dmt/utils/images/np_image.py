
import numpy as np



### ---- ### ---- \\    Getting Image Statistics     // ---- ### ---- ###

def print_image_summary(np_image):
    shape = np_image.shape
    shape_str = 'x'.join([str(s) for s in shape])
    dtype = str(np_image.dtype)
    max_val = np_image.max()
    max_str = f"{max_val}" if 'int' in str(type(max_val)) else f"{max_val:.2f}"
    min_val = np_image.min()
    min_str = f"{min_val}" if 'int' in str(type(min_val)) else f"{min_val:.2f}"
    print(f"{shape_str} {dtype} Image (max:{max_str} min:{min_str})")

def get_max_coord(np_image):
    r""" Gets the ndim coordinate for the max value in np_image. 
    Arguments:
        np_image (np.ndarray) - any dimensional image or array
    Returns:
        tuple of coordinate (only tuples exhibit proper array indexing behavior)
    """
    max_flat_index = np.argmax(np_image)
    coord = np.unravel_index(max_flat_index, np_image.shape)
    return coord


def get_min_coord(np_image):
    r""" Gets the ndim coordinate for the min value in np_image. 
    Arguments:
        np_image (np.ndarray) - any dimensional image or array
    Returns:
        tuple of coordinate (only tuples exhibit proper array indexing behavior)
    """
    max_flat_index = np.argmin(np_image)
    coord = np.unravel_index(max_flat_index, np_image.shape)
    return coord


def standardize_2dimage(standard_np_image):
    r""" Given a standard 2D numpy image, change image into a plottable form.
    A 'standard' numpy image is defined as:
     (1) Values are either {True, False} or [0, 1] or [0, 255]
     (2) Image is either gray (HxW, HxWx1, 1xHxW)
            or rgb (HxWx3, 3xHxW) or rgba (HxWx4, 4xHxW)
            
    Return:
        np_image that is [0, 255] uint8 & HxW or HxWx3 or HxWx4 based on type.
    """
    if not is_standard_2dimage(standard_np_image):
        print(f"(standardize_2dimage) WARNING: non-standard image given.")
        return standard_np_image
    
    np_type = str(standard_np_image.dtype)
    shape = standard_np_image.shape
    
    # convert to standard shape (HxW, HxWx[3,4])
    if len(shape) == 3:
        if shape[0] in (1, 3, 4):
            standard_np_image = to_channel_last(standard_np_image)
        new_shape = standard_np_image.shape
        if new_shape[-1] == 1:
            standard_np_image = standard_np_image.squeeze(-1)
    
    if 'bool' in np_type:
        return standard_np_image.astype(np.uint8) * 255
    if standard_np_image.max() <= 1:
        return (standard_np_image * 255).astype(np.uint8)
    return standard_np_image.astype(np.uint8)

    
def is_standard_2dimage(np_image):
    if np_image.ndim not in (2, 3):
        return False
    if 'bool' in str(np_image.dtype):
        return True
    if np_image.min() < 0 or np_image.max() > 255:  # separate line for optim
        return False
    shape = np_image.shape
    if np_image.ndim == 3:
        if shape[0] in (1, 3, 4) or shape[-1] in (1, 3, 4):
            return True
        return False
    return True


### ---- ### ---- \\    Image Basic Transforms     // ---- ### ---- ###

def resize(np_im, size, interp='bicubic'):
    pass

def gray_to_rgb(np_im, channel_last=True):
    if channel_last:
        return np.stack((np_im, np_im, np_im), axis=-1)
    return np.stack((np_im, np_im, np_im), axis=0)


def to_channel_last(np_im):
    return np.moveaxis(np_im, 0, -1)

def to_channel_first(np_im):
    return np.moveaxis(np_im, -1, 0)
    
    