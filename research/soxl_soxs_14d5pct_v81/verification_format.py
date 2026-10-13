"""Restore only the empty mean-model coefficient matrix lost in JSON encoding."""
import numpy as np


def serialized_field(model,key):
    if model['mode']=='mean' and key=='coefficients':
        for name in ['ids','center','scale','design_center','coefficients','penalty']:
            if model[name]!=[]:raise ValueError('Mean model must have no slopes or design coordinates')
        return np.empty((0,2),float)
    return model[key]
