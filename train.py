#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typing
import torch
import torch.utils.data


def train_loop(
        network: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], None],
        callback_fn: typing.Callable[[dict], None]
    ) -> None:
    
    network.train()
    for batch, (x,y) in enumerate(dataloader):
        pred = network(x)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        callback_fn(**{'x': x, 'y': y, 'loss': loss, 'pred': pred})
        
def test_loop(
        network: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], None],
        callback_fn: typing.Callable[[dict], None]
    ) -> None:
    network.eval()
    with torch.no_grad():
        for batch, (x,y) in enumerate(dataloader):
            pred = network(x)
            loss = loss_fn(pred, y)
            callback_fn(**{'x': x, 'y': y, 'loss': loss, 'pred': pred})