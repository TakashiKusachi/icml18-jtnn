#!/bin/bash

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
        
if __name__=="__main__":
    print("What?")