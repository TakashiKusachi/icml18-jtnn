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
    def __init__(self,input_size,output_size=56,hidden_size=100,\
                 max_mpz=1000,embedding_size=10,\
                 num_rnn_layers=1,bidirectional=False,\
                 dropout_rate=0.2):
        super(ms_peak_encoder_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.embedding = nn.Embedding(max_mpz,embedding_size)
        self.rnn = nn.GRU(input_size=embedding_size+1,hidden_size=hidden_size,batch_first=True,num_layers=num_rnn_layers,bidirectional=bidirectional)
        if self.bidirectional:
            self.T_mean = nn.Linear(hidden_size*2, output_size/2)
            self.T_var = nn.Linear(hidden_size*2, output_size/2)
            self.G_mean = nn.Linear(hidden_size*2, output_size/2)
            self.G_var = nn.Linear(hidden_size*2, output_size/2)
        else:
            self.T_mean = nn.Linear(hidden_size, output_size/2)
            self.T_var = nn.Linear(hidden_size, output_size/2)
            self.G_mean = nn.Linear(hidden_size, output_size/2)
            self.G_var = nn.Linear(hidden_size, output_size/2)
    
    def forward(self,x,y,sample=False,training=True):
        batch_size = x.size()[0]
        number_peak = x.size()[1]
        x = x.long()
        inp = self.embedding(x)
        y = y.float()
        y = y.view(batch_size,number_peak,1)
        inp = torch.cat((inp,y),2)
        h,_ = self.rnn(inp)
        if self.bidirectional:
            h = h[:,-1,:]+h[:,0,:]
        else:
            h = h[:,-1,:]
        h = F.relu(h)
        
        h = F.dropout(h,p=self.dropout_rate,training=training)
        if sample:
            t_vecs,t_kl_loss = self.rsample(h,self.T_mean,self.T_var)
            g_vecs,g_kl_loss = self.rsample(h,self.G_mean,self.G_var)
            h = torch.cat((t_vecs,g_vecs),1)
            kl_loss = t_kl_loss + g_kl_loss
            return h,kl_loss
        else:
            t_vecs = self.T_mean(h)
            g_vecs = self.G_mean(h)
            h = torch.cat((t_vecs,g_vecs),1)
            return h,torch.zeros((1)).cuda(h.device)
    
    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

class ms_peak_encoder_cnn(nn.Module):
    def __init__(self,input_size,max_mpz=1000,embedding_size=10,\
                 conv1_channel=30,kernel1_width=3,\
                 conv2_channel=50,kernel2_width=3,\
                 conv_output_channel=100,conv_output_width=3,
                 hidden_size=100,num_rnn_layers=2,bidirectional=False,\
                 output_size=56,dropout_rate=0.5):
        
        super(ms_peak_encoder_cnn, self).__init__()
        self.embedding_size = embedding_size
        self.max_mpz = max_mpz
        self.bidirectional=bidirectional
        
        self.embedding = nn.Embedding(max_mpz,embedding_size)
        
        self.convSequential=nn.Sequential(\
                nn.Conv2d(            1,conv1_channel, (kernel1_width, embedding_size+1), stride=1, padding=((kernel1_width-1)/2,0)),nn.ReLU(),\
                nn.Conv2d(conv1_channel,conv1_channel, (kernel1_width, 1), stride=1, padding=((kernel1_width-1)/2,0)),nn.ReLU(),\
                #nn.MaxPool2d((3,1)),\
                nn.Conv2d(conv1_channel,conv2_channel, (kernel2_width, 1), stride=1, padding=((kernel2_width-1)/2,0)),nn.ReLU(),\
                nn.Conv2d(conv2_channel,conv2_channel, (kernel2_width, 1), stride=1, padding=((kernel2_width-1)/2,0)),nn.ReLU(),\
                #nn.MaxPool2d((3,1)),\
                nn.Conv2d(conv2_channel, conv_output_channel, (conv_output_width, 1), stride=1, padding=((conv_output_width-1)/2, 0)),nn.ReLU(),\
                nn.Conv2d(conv_output_channel, conv_output_channel, (conv_output_width, 1), stride=1, padding=((conv_output_width-1)/2, 0)),nn.ReLU(),\
                #nn.MaxPool2d((3,1))\
                )
        
        self.rnn = nn.GRU(input_size=conv_output_channel,hidden_size=hidden_size,batch_first=True,num_layers=num_rnn_layers,bidirectional=bidirectional)
        
        if self.bidirectional:
            self.T_mean = nn.Linear(hidden_size*2, output_size/2)
            self.T_var = nn.Linear(hidden_size*2, output_size/2)
            self.G_mean = nn.Linear(hidden_size*2, output_size/2)
            self.G_var = nn.Linear(hidden_size*2, output_size/2)
        else:
            self.T_mean = nn.Linear(hidden_size, output_size/2)
            self.T_var = nn.Linear(hidden_size, output_size/2)
            self.G_mean = nn.Linear(hidden_size, output_size/2)
            self.G_var = nn.Linear(hidden_size, output_size/2)
            
    def forward(self,x,y,sample=False,training=True):
        batch_size = x.size()[0]
        number_peak = x.size()[1]
        x = x.long()
        y = y.float()
        
        inp = self.embedding(x) # inp.size = (batch_size,number_peak,embedding_size)
        y = y.view(batch_size,number_peak,1) # y.size = (batch_size,number_peak,1)
        inp = torch.cat((inp,y),2) # inp.size = (batch_size,number_peak,embedding_size+1)
        
        inp = inp.view(batch_size,1,number_peak,self.embedding_size+1)
        h = self.convSequential(inp).squeeze().transpose(1,2) # h.size = (batch_size,number_peak,conv_output_channel)
        
        h,_ = self.rnn(h) 
        if self.bidirectional:
            h = h[:,-1,:]+h[:,0,:]
        else:
            h = h[:,-1,:]
        h = F.relu(h)
        
        if sample:
            t_vecs,t_kl_loss = self.rsample(h,self.T_mean,self.T_var)
            g_vecs,g_kl_loss = self.rsample(h,self.G_mean,self.G_var)
            h = torch.cat((t_vecs,g_vecs),1)
            kl_loss = t_kl_loss + g_kl_loss
            return h,kl_loss
        else:
            t_vecs = self.T_mean(h)
            g_vecs = self.G_mean(h)
            h = torch.cat((t_vecs,g_vecs),1)
            return h,torch.zeros((1)).cuda(h.device)
        
    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss
if __name__=="__main__":
    print("What?")