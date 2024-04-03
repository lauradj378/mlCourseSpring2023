#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/Users/laurahull/Documents/UCF/Spring_23/Machine_Learning_Course/dl")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Model

logdir = '/Users/laurahull/Documents/UCF/Spring_23/Machine_Learning_Course/dl/assignments/HW_4_2'

#Generate 5 samples of >30 length
generator = pipeline("text-generation", model='gpt2')
results = generator("My favorite country I have traveled to is", max_length = 50, num_return_sequences = 5)
for result in results:
    print(result['generated_text'])

#Verify the model is a subclass of torch.nn.Module
if isinstance(generator.model, nn.Module):
    print("Model is an instance of torch.nn.Module")
else:
    print("Model is not an instance of torch.nn.Module")

#Compute total number of weights in the network
total_weights = sum(weights.numel() for weights in generator.model.parameters())
print(f"Total number of weights in the network: {total_weights}")

def generate(network, idx, length, block_size):
    for _ in range(length):
         idx_tail = idx[:, -block_size:]
         y = network(idx_tail).logits # B, T, C
         last = y[:, -1, :] # B, C
         prob = F.softmax(last, dim=-1)
         next = torch.multinomial(prob, num_samples=1) # B, 1
         idx = torch.cat((idx, next), dim=1) # B, T+1
    return idx

#Generate text from neural network
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input = "I cannot stop listening to"
input_tokens = tokenizer.encode(input, return_tensors="pt")
generated_tokens = generate(generator.model, input_tokens, length=50, block_size=10)
generated_text = tokenizer.decode(generated_tokens[0])
print(generated_text)

class GPT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        output = self.model(idx)
        logits = output.last_hidden_state
        return logits

gpt2_model = GPT2Model.from_pretrained("gpt2")
gpt2 = GPT(gpt2_model)

#Visualize NN in Tensorboard
input_ids = torch.randint(low=0, high=50257, size=(1, 1024)) #dummy input tensor
writer = SummaryWriter(logdir)
writer.add_graph(gpt2, input_ids)
writer.close()