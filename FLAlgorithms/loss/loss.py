import dgl
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn

def unscaled_metrics(y_pred, y, name):
    mask = (y != 0).float()
    mask /= mask.mean()

    mse = ((y_pred - y) ** 2)* mask.mean()
    mae = torch.abs(y_pred - y)* mask.mean()
    mape = torch.abs((y_pred - y) / y)* mask.mean()
    
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    #loss[loss != loss] = 0
    return {
        '{}/mse'.format(name): mse.detach(),
        # '{}/rmse'.format(name): rmse.detach(),
        '{}/mae'.format(name): mae.detach(),
        '{}/mape'.format(name): mape.detach()
    }