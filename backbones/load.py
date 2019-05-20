import torch
from collections import OrderedDict


def load_model(model, model_file, is_restore=False, show_warning=False):
    print("=> Load pretrained model...")
    if isinstance(model_file, str):
        state_dict = torch.load(model_file)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    if show_warning:
        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys

        if len(missing_keys) > 0:
            print('Missing key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in missing_keys)))

        if len(unexpected_keys) > 0:
            print('Unexpected key(s) in state_dict: {}'.format(
                ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict

    return model
