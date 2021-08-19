
import torch

from .sample import Sample


class SampleSet(torch.utils.data.Dataset):
    """ Base container object for a dataset's subset (e.g. KiTS train).
    
    Intended Functionality:
      (1) Serve as a container for a data subset (training, val set)
      (2) Return a multi-processing iterator for preprocessed images.
           For 3D images, this could be resampling, for 2D, it could just
           pre-load the data in memory using a Dataloader wrapper. 
    """
    
    def __init__(
            self, 
            samples,
            transforms=None,  # intended for preprocessing of entire images
            ):
        samples = self._parse_samples(samples)
        self._samples = samples
        
    @property
    def samples(self):
        return self._samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx < 0 or idx > len(self):
            raise ValueError(f"Index {idx} is out of bounds.")
        return self.samples[idx]
        
    def _parse_samples(self, samples):
        try:
            iter(samples)
        except TypeError as e:
            msg = f'Samples must be an iterable, not {type(samples)}'
            raise TypeError(msg) from e
        
        if not samples:
            raise ValueError('Given samples list is empty')

        for sample in samples:
            if not isinstance(sample, Sample):
                msg = (
                    'Samples list must contain instances of sample.Sample,'
                    f' not "{type(sample)}"'
                )
                raise TypeError(msg)
        return samples
    
    def __repr__(self):
        string = (f'SampleSet Object: {len(self)} samples.')
        return string

