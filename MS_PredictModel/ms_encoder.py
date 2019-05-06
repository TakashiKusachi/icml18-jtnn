#!/bin/bash

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ms_peak_encoder(nn.Module):
    def __init__(self,input_size,output_size=56):
        super(ms_peak_encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.W1 = nn.Linear(input_size*2, output_size*2)
        self.W2 = nn.Linear(output_size*2,output_size*2)
        self.W3 = nn.Linear(output_size*2, output_size)
        
    def forward(self,x,y):
        inp = torch.cat((x,y),dim=1)
        inp = Variable(inp).cuda()
        h = F.relu(self.W1(inp))
        h = F.relu(self.W2(h))
        h = self.W3(h)
        return h

class ms_peak_encoder_lstm(nn.Module):
    def __init__(self,input_size,output_size=56,hidden_size=100,max_mpz=1000,embedding_size=10):
        super(ms_peak_encoder_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn = nn.GRU(input_size=embedding_size+1,hidden_size=hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size,output_size)
        self.embedding = nn.Embedding(max_mpz,embedding_size)
    
    def forward(self,x,y):
        batch_size = x.size()[0]
        number_peak = x.size()[1]
        x = x.long()
        inp = self.embedding(x)
        y = y.float()
        y = y.view(batch_size,number_peak,1)
        inp = torch.cat((inp,y),2)
        h,_ = self.rnn(inp)
        h = self.out(h[:,-1,:])
        return h
    
if __name__=="__main__":
    print("What?")