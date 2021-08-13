
from .base_model import BaseModel

from .unets import unet2d, unet3d, bigunet2d, denseunet2d


__all__ = ['BaseModel']


model_names_2d = {
    'unet2d',
    'denseunet2d',
    'bigunet2d',
}

model_names_3d = {
    'unet3d', 
}


def get_model(name, **kwargs):
    
    if name in model_names_2d:
        
        if name == 'unet2d':
            model = unet2d.get_model(**kwargs)
        elif name == 'denseunet2d':
            model = denseunet2d.get_model(**kwargs)
        elif name == 'bigunet2d':
            model = bigunet2d.get_model(**kwargs)
            
        
    elif name in model_names_3d:
        
        if name == 'unet3d':
            model = unet3d.get_model(**kwargs)
        
    else:
        msg = (f'"{name}" is not a valid model name. \n2D models include: '
               f'{model_names_2d}. \n3D models include: {model_names_3d}.')
        raise ValueError(msg)
    
    return model
        
    # name = cfg['model']['name']
    # settings = cfg['model'][name]

    # ds_name = cfg['data']['name']
    # out_channels = len(cfg['data'][ds_name]['classnames'])
    # in_channels = 0
    # for entry in cfg['train']['transforms']:
    #     if entry[0] == 'normmeanstd':
    #         in_channels = len(entry[1][0])
    # assert in_channels > 0, f"Could not get input dim from normmeanstd transform"

    print(f"  > Fetching {name} model..", end='')
    if name == 'unet':
        bilinear = settings['bilinear']
        model = unet.UNet(in_channels, out_channels, bilinear=bilinear)
    elif 'bigunet' in name:
        bilinear = settings['bilinear']
        base_size = settings['base_size']
        model = bigunet.BigUNet(in_channels, out_channels, bilinear=bilinear,
                    base_size=base_size)
    elif name == 'nestedunet':
        model = nestedunet.NestedUNet(
            out_channels, 
            input_channels=in_channels,
            deep_supervision=cfg['model']['nestedunet']['deepsup']
        )
    elif name == 'r2attentionunet':
        model = r2attentionunet.R2Attention_UNet(
            in_ch=in_channels, out_ch=out_channels, t=2
        )
    elif name == 'segdensenet':
        model = segdensenet.get_model(
            str(settings['layers']), 
            num_classes=out_channels, 
            pretrained=settings['pretrained']
        )
    elif name == 'denseunet':
        model = denseunet.get_model(
            str(settings['layers']),
            pretrained=settings['pretrained'],
            num_classes=out_channels, 
            deconv=settings['deconv']
        )
    elif name == 'msradenseunet':
        model = msradenseunet.get_model(
            str(settings['layers']),
            pretrained=settings['pretrained'],
            num_classes=out_channels, 
            deconv=settings['deconv']
        )
    elif name == 'imsradenseunet':
        model = imsradenseunet.get_model(
            str(settings['layers']),
            pretrained=settings['pretrained'],
            num_classes=out_channels, 
            deconv=settings['deconv']
        )
    # elif name == 'denseunet_deepsup':
    #     model = denseunet_deepsup.get_model(
    #         str(settings['layers']),
    #         pretrained=settings['pretrained'],
    #         num_classes=out_channels, 
    #         deconv=settings['deconv']
    #     )
    else:
        

    print(f"\t{name} initialized ({model.param_counts[0]} params).")
    return model

