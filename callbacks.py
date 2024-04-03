#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import typing
import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import os
#from tensorboard.plugins.hparams import summary_v2 as hp_summary_v2

class Callback:
    def set_model(self,model):
        self.model = model
        
    def set_data(self,train_dataloader,test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
    def on_train_batch_end(self,x,y,pred,loss):
        pass
    
    def on_train_epoch_end(self,epoch):
        pass
    
    def on_train_epoch_start(self,epoch):
        pass
    
    def on_test_epoch_start(self,epoch):
        pass 
    
    def on_test_batch_end(self,x,y,pred,loss):
        pass
    
    def on_test_epoch_end(self,epoch):
        pass
    
    def on_train_end(self):
        pass
    

def callback_fn(
    callbacks: typing.List[Callback]
) -> typing.Callable[[str], typing.Callable[..., None]]:
    
    def _callback(method):
        def _callback_fn(*args,**kwargs):
            for c in callbacks:
                getattr(c,method)(*args,**kwargs)
        return _callback_fn
    return _callback

class MeanMetricCallback(Callback):
    """
    Computes the mean of a metric across all batches in one epoch.
    """
    
    def __init__(self):
        
        self.train_metric = 0
        self.test_metric = 0
        
    def metric(self,x,y,pred,loss):
        
        return 0
    
    def set_data(self,*args,**kwargs):
        super().set_data(*args,**kwargs)
        
        self.num_train_batches = len(self.train_dataloader)
        self.num_test_batches = len(self.test_dataloader)
        
    def on_train_batch_end(self,x,y,pred,loss):
        
        self.train_metric += self.metric(x,y,pred,loss)
        #metric_result, _ = self.metric(x, y, pred, loss)
        #print(metric_result)
        #self.train_metric += metric_result
        
    def on_test_batch_end(self,x,y,pred,loss):
        
        self.test_metric += self.metric(x,y,pred,loss)        
        
    def on_train_epoch_start(self,epoch):
        
        self.train_metric = 0

    def on_test_epoch_start(self,epoch):
        
        self.test_metric = 0
        
    def mean_train_metric(self):
        
        return self.train_metric / self.num_train_batches
    
    def mean_test_metric(self):
        
        return self.test_metric / self.num_test_batches
    
class PrintMeanTestLoss(MeanMetricCallback):
    
    def metric(self,x,y,pred,loss):
        
        return loss.item()
    
    def on_test_epoch_end(self, epoch):
        
        print(f"Avg Test Loss: {self.mean_test_metric():>8f}")
        
class PrintTestCorrect(MeanMetricCallback):
    
    def metric(self,x,y,pred,loss):
        
        batch_size = x.shape[0]
        return (pred.argmax(1) == y).type(torch.float).sum().item() / batch_size
    
    def on_test_epoch_end(self, epoch):
        
        print(f"Test Correct: {(100*self.mean_test_metric()):>0.1f}%")
        

class PrintLoss(Callback):
    
    def set_data(self, *args, **kwargs):
        super().set_data(*args,**kwargs)
        
        self.num_batches = len(self.train_dataloader)
        self.size = len(self.train_dataloader.dataset)
        self.batch_idx = 0
        
    def on_train_batch_end(self, x, loss, **kwargs):
        if self.batch_idx % max(1,(self.num_batches // 5)) == 0:
            loss, n_x_done = loss.item(), self.batch_idx * len(x)
            print(f"loss: {loss:>7f} [{n_x_done:>5d}/{self.size:>5d}")
            
        self.batch_idx += 0
        
    def on_train_epoch_start(self,epoch):
        self.batch_idx = 0
        print(f'\n--- Epoch {epoch} ---')
        
class TBLoss(MeanMetricCallback):

    def __init__(self,log_dir,normalize,dropout):
        #print(f"Creating TBLoss instance for normalize={normalize}, dropout={dropout}")
        self.log_dir = log_dir
        self.normalize = normalize
        self.dropout = dropout
        self.writer = SummaryWriter(log_dir)
        #print(log_dir)
        self.min_test_loss = np.inf
        self.hparams = {'normalize': self.normalize, 'dropout': self.dropout}
        self.global_min_test_loss = np.inf  # new variable
    def metric(self,x,y,pred,loss):
        return loss.item()
    
    def on_test_epoch_end(self, epoch):
        test_loss = self.mean_test_metric()
        if test_loss < self.min_test_loss:
            self.min_test_loss = test_loss
        if test_loss < self.global_min_test_loss:  # update global_min_test_loss
            self.global_min_test_loss = test_loss
        self.writer.add_scalar('test/loss', test_loss, epoch)
        #self.writer.add_scalar('test/min_loss', self.min_test_loss, epoch)
        self.writer.add_scalar('test/normalize', int(self.normalize == 1) * 2 - 1, epoch)
        self.writer.add_scalar('test/dropout', self.dropout, epoch)
        self.writer.add_hparams(self.hparams, {'test/global_min_loss': self.global_min_test_loss})
        #print(self.min_test_loss)
    def on_train_end(self):
        #self.writer.add_scalar('test/min_loss', self.min_test_loss)
        self.writer.add_hparams(self.hparams, {'test/global_min_loss': self.global_min_test_loss})

class GDLoss(MeanMetricCallback):
        
    def __init__(self,log_dir,optimizer_val):
        self.optimizer_name = optimizer_val.__name__
        self.log_dir = log_dir
        self.optimizer_names = {torch.optim.SGD: 'SGD', torch.optim.Adagrad: 'Adagrad', torch.optim.Adam: 'Adam'}
        self.writer = SummaryWriter(log_dir)
        self.hparams = {'optimizer': self.optimizer_name}
        self.step = 0

    def metric(self,x,y,pred,loss):
        return loss.item()
    
    def on_test_epoch_end(self, epoch):
        self.test_loss = self.mean_test_metric()
        self.writer.add_hparams({'optimizer': self.optimizer_name}, {'test/loss': self.test_loss})

    def on_train_batch_end(self, x, y, pred, loss):
        self.writer.add_scalar('train/loss_vs_step', loss.item(), self.step)
        self.step += 1