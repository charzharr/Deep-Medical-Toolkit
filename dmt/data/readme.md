
# Design Logic  

Say we are loading a dataset for skin lesion classification (multi-class), 
we want to create 3 SampleSets for training, validation, and testing. 
Within each SampleSet is a collection of Samples. In this case, each sample
has a VectorImage2D (colored 2D image) and a ClassLabel. If the task was 
segmentation, you would have a ScalarMask2D instead of ClassLabel.

Additionally, for each Sample, you can have arbitrary sample attributes
(e.g. if patient, you can have their age, height, etc.). The same is true for
classes that inherity Image since both are fundamentally Dict objects. 

Abstraction Hierarchy:
> SampleSet (Dataset)
    > Sample
        > Image
            > ScalarImage2D
            > VectorImage2D
            > ScalarMask2D
            > ScalarImage3D
        > ScalarMask3D
            > Category
            > ClassLabel
