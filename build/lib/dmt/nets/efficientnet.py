""" EfficientNet Implementation from PyPi timm


EfficientNet Overview (from: https://arxiv.org/pdf/1905.11946v5.pdf)
---------------------
EfficientNet-B0 - 77.1% top1, 5.3M params, 0.39B FLOPs
EfficientNet-B1 - 79.1% top1, 7.8M params, 0.70B FLOPS
EfficientNet-B2 - 80.1% top1, 9.2M params, 1.0B  FLOPs
EfficientNet-B3 - 81.6% top1, 12M  params, 1.8B  FLOPs
EfficientNet-B4 - 82.9% top1, 19M  params, 4.2B  FLOPs
EfficientNet-B5 - 83.6% top1, 30M  params, 9.9B  FLOPs
EfficientNet-B6 - 84.0% top1, 43M  params, 19B   FLOPs
EfficientNet-B7 - 84.3% top1, 66M  params, 37B   FLOPs

Resources
----------
Pretrained Weights / Model Implementations:
    https://github.com/rwightman/pytorch-image-models/

"""

import os
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F


from .basemodel import BaseModel


model_params_path = '/afs/crc.nd.edu/user/y/yzhang46/_DLResources/Models/'
model_params = {
    'b0': os.path.join(model_params_path, 'efficientnetb0.pth'),
    'b1': os.path.join(model_params_path, 'efficientnetb1.pth'),
    'b2': os.path.join(model_params_path, 'efficientnetb2.pth'),
    'b3': os.path.join(model_params_path, 'efficientnetb3.pth'),
}


def load_state_dict(model, state_dict, load_classifier_params=False):
    if not load_classifier_params:
        del_keys = [k for k in state_dict.keys() if "classifier" in k]
        for k in del_keys:
            state_dict.pop(k, None)
    print(' (EfficientNet pretrained=True) Loading pretrained stat_dict..')
    print('\t', model.load_state_dict(state_dict, strict=False))

def get_model(name, pretrained=True, num_classes=1000, only_encoder=False):
    """
    Parameters (defaults are based on PyTorch values):
        pretrained (bool) - use ImageNet pretrained parameters
        only_encoder (bool) - only get feature extractor (full FMs)
        layer_drop_rate (probability) - drop rate after every dense layer
        final_drop_rate (probability) - drop rate before last pooling layer
        num_classes (int) - number of output classes (cut if only_encoder)
    """
    name = str(name)
    if 'b0' in name:
        model = timm.create_model('efficientnet_b0', pretrained=False,
        	num_classes=num_classes)
       	params_key = 'b0'
    elif 'b1' in name:
        model = timm.create_model('efficientnet_b1', pretrained=False,
            num_classes=num_classes)
        params_key = 'b1'
    elif 'b2' in name:
        model = timm.create_model('efficientnet_b2', pretrained=False,
            num_classes=num_classes)
        params_key = 'b2'
    elif 'b3' in name:
        model = timm.create_model('efficientnet_b3', pretrained=False,
            num_classes=num_classes)
        params_key = 'b3'
    else:
        raise ValueError(f"DenseNet name ({name}) is not supported.")

    if pretrained:
        cdiff = False if not pretrained or num_classes != 1000 else True
        state_dict = torch.load(model_params[params_key])
        load_state_dict(model, state_dict, load_classifier_params=cdiff)
    
    if only_encoder:
        raise NotImplementedError()
        model = list(model.children())[0]

    N_params = sum([p.numel() for p in model.parameters()])
    print(f" (EfficientNet get_model) {name} model (pretrained={pretrained})"
          f" loaded w/{N_params:,} params.")
    return model