

class Label(dict):
    """ Base class for all labels (e.g. masks, class ids). """
    
    # Class Variables
    _id_counter = 0  # global runtime ID generator for all Image subclasses
    # _gid_to_instance = {}
    _default_attrs = ['num_classes', 'class_names']
    
    def __init__(self, **kwargs):
        
        # Global Label class ID tracking
        self.global_id = Label._id_counter
        # Image._gid_to_instance[self.global_id] = self  # save memory
        Label._id_counter += 1
        
        super().__init__(**kwargs)  # update given attributes to dict
        self.update_attributes()  # make dot notation available
    
    @property
    def num_classes(self):
        raise NotImplementedError()  # weak enforcement
    
    @property
    def class_names(self):
        raise NotImplementedError()  # weak enforcement
    
    def get_one_hot(self):
        raise NotImplementedError()  # weak enforcement
    
    def get_ids(self):
        raise NotImplementedError()  # weak enforcement
    
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
    
    

    
