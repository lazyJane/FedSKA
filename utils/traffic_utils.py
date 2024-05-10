
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn


def unscaled_metrics(y_pred, y, scaler, name):
    #mask = (y != 0).float().cpu()
    #mask /= mask.mean()
    y = scaler.inverse_transform(y.detach().cpu())
    y_pred = scaler.inverse_transform(y_pred.detach().cpu())
    # mse
    mse = ((y_pred - y) ** 2)
    # RMSE
    # rmse = torch.sqrt(mse)
    # MAE
    mae = torch.abs(y_pred - y)
    # MAPE
    #print(y_pred)
    #print(torch.abs((y_pred - y) / y))
    mape = torch.abs((y_pred[y!=0] - y[y!=0]) / y[y!=0]) 
    #loss[loss != loss] = 0
    #print(mape)
    #input()
    return {
        '{}/mse'.format(name): mse.mean().detach(),
        # '{}/rmse'.format(name): rmse.detach(),
        '{}/mae'.format(name): mae.mean().detach(),
        '{}/mape'.format(name): mape.mean().detach()
    }