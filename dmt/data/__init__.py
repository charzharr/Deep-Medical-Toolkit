
__all__ = [
    'Image',
    'ScalarImage3D',
    'ScalarImage2D',
    'VectorImage2D',
    'Label',
    'ScalarMask2D',
    'Scalarmask3D',
    'CategoricalLabel',
]

# Sample Components: images, labels, categories
from .image_base import Image
from .images import (
    ScalarImage3D,
    ScalarImage2D,
    VectorImage2D,
)

from .label_base import Label
from .label_categories import CategoricalLabel
from .label_masks import (
    ScalarMask3D,
    ScalarMask2D
)

# Samples & their collections
from .samples.sample import Sample
from .samples.sampleset import SampleSet