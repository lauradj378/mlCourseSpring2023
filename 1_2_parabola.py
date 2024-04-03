#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dl
import dl.data
import dl.networks
import dl.models

params = {
    'dim': 2,
    'width': 50,
    'depth': 3,
    'lr': 0.1,
    'epochs': 25,
    'n_samples': 100,
    'batch_size': 10,
}

if __name__ == '__main__':

    dataset = dl.data.parabola_nd(params['n_samples'],params['dim'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params['batch_size'], shuffle=True)

    in_dim = dataset.data.shape[-1]
    out_dim = dataset.targets.shape[-1]
    network = dl.networks.Dense(in_dim, out_dim, width=params['width'], depth=params['depth'])
    optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
    model = dl.models.Model(network, optimizer, F.mse_loss, params['epochs'])
    _, loss = model.fit(dataloader)

    x = dataset.data
    y = dataset.targets
    pred = model.predict(x)
    plt.plot(x.detach().numpy().flat, y.detach().numpy(), color='red')
    plt.plot(x.detach().numpy().flat, pred.detach().numpy(), color='blue')
    plt.show()
    
    x_new = torch.linspace(-2, 2, 50).reshape((-1,1))
    y_new = x_new**2
    dataset2 = torch.utils.data.TensorDataset(x_new, y_new)
    dataset2.data = x_new
    dataset2.targets = y_new
    
    dataloader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=params['batch_size'], shuffle=True)
    _, loss = model.fit(dataloader2)
    pred_new = model.predict(x_new)
    plt.plot(x_new.detach().numpy().flat, y_new.detach().numpy(), color='red')
    plt.plot(x_new.detach().numpy().flat, pred_new.detach().numpy(), color='blue')
    
# =============================================================================
#     epoch 0, loss 0.10866720974445343
#     epoch 1, loss 0.049426354467868805
#     epoch 2, loss 0.06556465476751328
#     epoch 3, loss 0.11537539958953857
#     epoch 4, loss 0.06353957951068878
#     epoch 5, loss 0.03683387115597725
#     epoch 6, loss 0.06040366739034653
#     epoch 7, loss 0.019777441397309303
#     epoch 8, loss 0.039352379739284515
#     epoch 9, loss 0.015753814950585365
#     epoch 10, loss 0.014490006491541862
#     epoch 11, loss 0.007483478635549545
#     epoch 12, loss 0.0013581642415374517
#     epoch 13, loss 0.0028804235626012087
#     epoch 14, loss 0.0034211785532534122
#     epoch 15, loss 0.0009159590117633343
#     epoch 16, loss 0.0010492380242794752
#     epoch 17, loss 0.0005283114151097834
#     epoch 18, loss 0.0011514584766700864
#     epoch 19, loss 0.000449048267910257
#     epoch 20, loss 0.000210005440749228
#     epoch 21, loss 0.0005093336803838611
#     epoch 22, loss 0.0003246806445531547
#     epoch 23, loss 0.00020465065608732402
#     epoch 24, loss 0.0006762717384845018
#     epoch 0, loss 4.169510364532471
#     epoch 1, loss 2.7814602851867676
#     epoch 2, loss 1.7783952951431274
#     epoch 3, loss 1.7632513046264648
#     epoch 4, loss 1.967017412185669
#     epoch 5, loss 1.0062204599380493
#     epoch 6, loss 1.1667784452438354
#     epoch 7, loss 0.8154374361038208
#     epoch 8, loss 1.0364855527877808
#     epoch 9, loss 1.3638694286346436
#     epoch 10, loss 1.8569986820220947
#     epoch 11, loss 1.4755535125732422
#     epoch 12, loss 0.8878347277641296
#     epoch 13, loss 0.47776132822036743
#     epoch 14, loss 0.3447931110858917
#     epoch 15, loss 0.027661476284265518
#     epoch 16, loss 0.24468760192394257
#     epoch 17, loss 0.6901770234107971
#     epoch 18, loss 0.2955207824707031
#     epoch 19, loss 0.45428818464279175
#     epoch 20, loss 0.21936288475990295
#     epoch 21, loss 0.050032854080200195
#     epoch 22, loss 0.10353536903858185
#     epoch 23, loss 0.0696919709444046
#     epoch 24, loss 0.025126466527581215
# =============================================================================
    