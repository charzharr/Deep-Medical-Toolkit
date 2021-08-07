
# Deep Medical Tools (dmt)

## Implementation Details

### Similarities to [Torchio](https://github.com/fepegar/torchio)
 - Same design hierarchy where samples (dict subclass) can hold arbitrary
    attributes, and library-specific data like Images (e.g. ScalarImage3D), 
    Labels (e.g Masks)
 - Transforms take samples (i.e. subjects) as they contain all the abstractions
    and data format conversions built-in. Also their custom attributes feature
    allows for easy storage of transformation history. 

### Improvements Over [Torchio](https://github.com/fepegar/torchio)
 - Overall objects and shift in design..
    - Introduced data abstractions like samples (i.e. subject in torchio) and
    examples (elements in a batch). This distinction is important. 
    - More extensible to allow custom behavior for data structures.  
    
 - Added general data structures for 2D & 3D images, and labels. 
    - 3D: ScalarMask3D, ScalarImage3D
    - 2D: ScalarMask2D, ScalarImage2D, VectorImage2D
    - Classification: CategoricalLabel
 - Extended transformations to both 2D & 3D. Also added some 3D ones as well.
    - Added 3D transforms:
    - All 2D transforms:  
 - Improved existing data structures.
    - For labels, added categorical (both multi-class & multi-label). 
    - For Images, gives you the option to permanently load data. 
    - Extensibility is improved for almost all data structures. For example, in 
    an Image, you can overload how a file is read, what preprocessing you want,
    how to get an array/tensor from the preprocessed sitk image.

Additional Verbose Improvements 
 - 
 - Universally, numpy.ndarrays are passed around (instad of tensors like torchio)
 - Sample images (one sample = one patient) are lazy-loaded as an sitk object
if a path is given.


TODO:
- [ ] Remove printing private attributes in __repr__ for images & others.

# Deep-Medical-Toolkit-
A personal code toolkit for the purpose of facilitating medical imaging research. The main components of this library consists of a neural network model zoo, image transformations (for preprocessing &amp; augmentation), common metrics/losses, fast multiprocessed data loading, and data structures for image samples.
