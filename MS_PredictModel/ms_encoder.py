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
    def __init__(self,input_size,output_size=56,hidden_size=100):
        super(ms_peak_encoder_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.GRU(input_size=2,hidden_size=hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size,output_size)
    
    def forward(self,x,y):
        batch_size = x.size()[0]
        inp = torch.stack((x,y),2)
        #indc = inp[:,:,1].argsort(1,descending=True)
        #inp = torch.stack([torch.index_select(inp[batch,:,:],0,indc[batch,:]) for batch in range(batch_size)],0)
        #print(inp.size)
        inp = Variable(inp).cuda()
        h,_ = self.rnn(inp)
        h = self.out(h[:,-1,:])
        return h
if __name__=="__main__":
    print("What?")