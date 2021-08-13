
import numbers
import collections
import pprint
import torch
import numpy as np

from .label_base import Label


class CategoricalLabel(Label):
    """ Class for categorical labels. Shared functionality definition. 
    Implementation Details:
        - Internally, labels are stored as a list of class_ids with a class's
            id defined by its index in the 'class_names' argument
    """
    
    def __init__(
            self, 
            label, 
            class_names, 
            multi_label=False,
            **kwargs
            ):
        """
        Args:
            label: the categor(y/ies) of label. Can be a number (class id), a 
                string (class name), a sequence of numbers (class ids), or a 
                sequence of strings (class names). 
            class_names: a sequence of strings indicating the names of classes
            multi_label: flag for if it is a multi-label task 
            **kwargs: additional custom attributes to be added to label
        """
        
        class_names = self._parse_class_names(class_names)
        self._class_names = class_names
        
        self._multi_label = multi_label
        self._given_label = label
        self._ids = self._parse_label(label, multi_label, class_names)
        
        super().__init__(**kwargs)  # store extra custom attributes 
    
    @property
    def class_names(self):
        return self._class_names
    
    @property
    def num_classes(self):
        return len(self._class_names)
    
    @property
    def unique_values(self):
        return self._ids
    
    def get_one_hot(self, tensor=False):
        ids = self.get_ids(squeeze=False)
        oh = np.zeros(self.num_classes, dtype=np.uint8)  # uint8, max val = 1
        for i in ids:
            oh[i] = 1
        
        if tensor:
            oh = torch.tensor(oh).double()
        return oh
    
    def get_ids(self, tensor=False, squeeze=True):
        ids = self._ids
        if tensor:
            ids = torch.tensor(ids).double()
        if squeeze:
            ids = ids.squeeze()
        return ids
    
    def _parse_class_names(self, class_names):
        # Verify if given class_names is valid
        cn_type = type(class_names)
        msg = f'Argument "class_names" must be a sequence. Given: {cn_type}.'
        assert isinstance(class_names, collections.Sequence), msg
        
        try:
            class_names = [str(cn) for cn in list(class_names)]
        except:
            msg = ('Argument "class_names" must be a sequence where each '
                   'element can be represented as a string.')
            raise ValueError(msg)
            
        return class_names
    
    def _parse_label(self, label, multi_label, class_names):
        if isinstance(label, (numbers.Number, str)):
            label = [label]
        
        l_type = type(label)
        msg = (f'Argument "label" must be a number, a string, or a squence of '
               f'either. You gave: {l_type}.')
        assert isinstance(label, collections.Sequence), msg
        
        assert len(label) > 0, f'Given label sequence cannot be empty.'
        
        n_classes = len(class_names)
        ids = []
        for lab in label:
            if isinstance(lab, int):  # assumed to be the class id
                msg = (f'Given class id "{lab}" in labels "{label}" is larger '
                       f'than #classes - 1.')
                assert lab < n_classes, msg
                
                ids.append(lab)
                continue
            elif isinstance(lab, float):  # assumed to be class name
                lab = str(lab)
                
            if isinstance(lab, str):  # assumed to be class name
                msg = (f'Give class name "{lab}" in labels "{label}" is not in '
                       f'class_names ({class_names}).')
                assert lab in class_names, msg
                ids.append(class_names.index(lab))
            else:
                l_type = type(lab)
                msg = (f'Argument "label" must contain a number (class id), or '
                       f'a string (class name). You gave: {l_type}.')
                raise ValueError(msg)
                
        if not multi_label:
            msg = (f'You gave multiple categories in a multi-class task. '
                   'Set init argument "multi_label" to be true.')
            assert len(ids) <= 1, msg
        
        return np.array(sorted(ids), dtype=np.int32)
    
    def __repr__(self):
        cat_type = 'multi-label' if self._multi_label else 'multi-class'
        ids = list(self.get_ids(squeeze=False))
        lab_names = [self.class_names[i] for i in ids]
        attrs = {k: v for k, v in self.items() if k not in self._default_attrs}
        attrs_str = pprint.pformat(attrs, indent=2, compact=True)
        string = (
            f'{self.__class__.__name__} ({cat_type}, gid={self.gid})\n'
            f'  Label: cls-ids={ids}, cls-names={lab_names}\n'
            f'  {self.num_classes} classes: {self.class_names}\n'
            f'Other Attributes: \n {attrs_str} \n')
        return string
            
