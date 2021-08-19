
import logging
import weakref

import dmt
from ..image_base import Image
from ..label_base import Label


class Sample(dict):
    """ Base class for all data samples (like subjects in TorchIO).
    
    This object contains a counter to be used for global sample identification.
    """
    _id_counter = 0
    _gid_to_instance = weakref.WeakValueDictionary()
    _reserved_attributes = ('gid', 'num_images', 'num_labels')
    
    def __init__(self, dictionary=None, **kwargs):
        """
        Args:
            dictionary: dict of attributes to be updated in sample
            excluded_images: 
        """
        # Global Sample ID tracking
        self.gid = Sample._id_counter  # global ID for Sample objects
        Sample._gid_to_instance[self.gid] = self
        Sample._id_counter += 1
        
        self.applied_transforms = []
        if dictionary:
            assert isinstance(dictionary, dict), "The type is in the name man.."
            kwargs.update(dictionary)
        super().__init__(**kwargs)
        self.update_attributes()
        logging.debug(f'📦 Sample object created: {repr(self)}')
    
    @property
    def num_images(self):
        """ Returns number of non-label images in this sample. """
        return len(self.get_image_keys(include_labels=False))
    
    @property
    def num_labels(self):
        """ Returns number of image labels (doesn't include categories). """
        return len(self.get_label_keys(only_images=True))
    
    
    ### ------ #      Functionality for Image Retrieval      # ----- ###
    
    def get_image_keys(self, include_labels=False):
        """ Returns a list of key names that are image objects. """
        image_keys = []
        for k, v in self.items():
            if isinstance(v, dmt.data.Image):
                if not include_labels and isinstance(v, dmt.data.Label):
                    continue
                image_keys.append(k)
        return image_keys
    
    def get_images_dict(self, include_labels=False):
        image_keys = self.get_image_keys(include_labels=include_labels)
        images_dict = {k: self[k] for k in image_keys}
        return images_dict
    
    def get_label_keys(self, only_images=True):
        """ Returns a list of key names that are labels. """
        label_keys = []
        for k, v in self.items():
            if isinstance(v, dmt.data.Label):
                if only_images and not isinstance(v, Image):
                    continue
                label_keys.append(k)
        return label_keys
    
    def get_labels_dict(self, only_images=True):
        label_keys = self.get_label_keys(only_images=only_images)
        labels_dict = {k: self[k] for k in label_keys}
        return labels_dict
    
    def get_target_image_keys(self, include_keys=None, exclude_keys=None):
        include_keys, exclude_keys = Sample._parse_include_exclude_keys(
            include_keys, exclude_keys)
        candidate_image_keys = set(self.get_image_keys(include_labels=True))
        if include_keys:
            candidate_image_keys = candidate_image_keys.intersection(
                                    include_keys)
        for k in exclude_keys:
            if k in candidate_image_keys:
                candidate_image_keys.remove(k)
        return list(candidate_image_keys)
    
    @staticmethod
    def _parse_include_exclude_keys(include_keys, exclude_keys):
        if exclude_keys is None:
            exclude_keys = []
        elif isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        msg = ('Keys in "exclude_keys" must be a string or seqence of strings. '
               f'Got: {exclude_keys}')
        for key in exclude_keys:
            assert isinstance(key, str), msg
        
        new_include_keys = []
        if include_keys is None:
            include_keys = []
        elif isinstance(include_keys, str):
            include_keys = [include_keys]
        msg = ('Keys in "include_keys" must be a string or seqence of strings. '
               f'Got: {include_keys}')
        for key in include_keys:
            assert isinstance(key, str), msg
            if key in exclude_keys:
                continue
            new_include_keys.append(key)
        
        return new_include_keys, exclude_keys
    
    
    ### ------ #      Transforms History & Reproducibility      # ----- ###
    
    def record_transform(self, transform_reproducing_arguments):
        item = transform_reproducing_arguments
        self.applied_transforms.append(item)
        logging.debug(f'Sample {self.gid}: Updating transforms {item}. \n'
                      f'  Transforms history: {self.applied_transforms}.')
    
    
    ### ------ #      Other Functionality      # ----- ###
    
    def __repr__(self):
        string = (
            f'{self.__class__.__name__} (gid={self.gid}) \n'
            f'{self.num_images} Image(s) \n'
        )
        for name, img in self.get_images_dict().items():
            perm = 'RAM-loaded' if img.permanent_load else 'lazy-load'
            t = img.type
            string += f'  "{name}": {img.__class__.__name__} ({perm}, {t})\n'
            if img.permanent_load:
                string += f'   {img.shape}, mem={img.memory/1e6} MB \n'
            else:
                string += f'   "{img.path}"\n'
        string += f'{self.num_labels} Label(s) \n'
        for name, lab in self.get_labels_dict().items():
            string += f'  "{name}": {lab.__class__.__name__} \n'
            if not hasattr(lab, 'permanent_load') or lab.permanent_load:
                string += f'   Has ids: {lab.unique_values} \n'
        return string
    
    def __len__(self):
        """ Returns the number of image & label items. """
        return len(self.get_images_dict()) + len(self.get_labels_dict)
    
    def __copy__(self):
        print('Sample copy called!')
        import copy
        result_dict = {}
        for key, value in self.items():
            if isinstance(value, Image):
                value = copy.copy(value)
            else:
                value = copy.deepcopy(value)
            result_dict[key] = value
        new = Sample(result_dict)
        new.applied_transforms = self.applied_transforms[:]  # shallow copy
        return new
    
    ### Functions below offers compatible/correct dict functionality
    ###   i.e. must update __dict__ after every update & override default funcs
    
    def __setattr__(self, name, value):
        """ Setting attrs in dot notation requires updating dict. """
        self[name] = value
        super().__setattr__(name, value)
    
    def __getitem__(self, item):
        item = super().__getitem__(item)
        logging.debug(f'Sample {self.gid}: Getting item: {item}')
        return item
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.update_attributes()
        logging.debug(f'Sample {self.gid}: Setting {key} to {value}')
        
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
    

        