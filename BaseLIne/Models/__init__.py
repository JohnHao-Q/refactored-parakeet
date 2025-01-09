"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

from .DFuseNet import DFuseNet as DFusemodel
from .my_model import New_net
model_dict = {'DFuseNet':DFusemodel, 'New_net':New_net}

def allowed_models():
    return model_dict.keys()


def define_model(mod, **kwargs):
    if mod not in allowed_models():
        raise KeyError("The requested model: {} is not implemented".format(mod))
    else:
        return model_dict[mod](**kwargs)
