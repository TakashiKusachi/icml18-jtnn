#!/bin/bash

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random
from tqdm import tqdm

from fast_jtnn.mol_tree import MolTree
from fast_jtnn.datautils import tensorize,set_batch_nodeID

import mysql
from mysql import connector
from getpass import getpass,getuser

import warnings
from multiprocessing import Pool

NUMBET_OF_MP_PROCESSES = 3 # Number of processes use to parse and expand data.

def analyze(i,one):
    ret = get_spectrum(*one)
    return ret
class MS_Dataset(object):
    
    QUERY= """select smiles,file_path from massbank where ms_type="MS" and instrument_type="EI-B"; """
    """
    """
    def __init__(self,vocab,host,database,batch_size,user=None,passwd=None,port=3306):
        terget_list = []
        try:
            if not isinstance(user,str):
                user = raw_input("user")
            if not isinstance(passwd,str):
                passwd = getpass()
            connect = connector.connect(host=host,user=user,password=passwd,port=port,database=database)
            cursor = connect.cursor()
            cursor.execute(MS_Dataset.QUERY)
            succes = 0
            fault = 0
            data_list = cursor.fetchall()
        except mysql.connector.Error as e:
            print("Something went wrong: {}".format(e))
            sys.exit(1)
        finally:
            if connect: connect.close()
            if cursor: cursor.close()
        
        succes = 0
        fault = 0
        max_spectrum_size = 0
        for one in tqdm(data_list):
            ret = get_spectrum(*one)
            if ret is not None:
                max_spectrum_size = max(len(ret[0]),max_spectrum_size)
                terget_list.append(ret)
                succes+=1
            else:
                fault += 1
        print("success {},fault {}".format(succes,fault))
        self.max_spectrum_size = max_spectrum_size
        self.vocab = vocab
        self.dataset = terget_list
        self.batch_size = batch_size
        self.shuffle = True
    
    def __len__(self):
        return len(self.dataset)
        
    def __iter__(self):
        if self.shuffle: 
            random.shuffle(self.dataset) #shuffle data before batch
            
        batches = [zip(*self.dataset[i : i + self.batch_size]) for i in xrange(0, len(self.dataset), self.batch_size)]
        if len(batches[-1]) < self.batch_size:
            batches.pop()
        dataset = MS_subDataset(batches,self.vocab,self.max_spectrum_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x:x[0])
        for b in dataloader:
            yield b
        del batches, dataset, dataloader

class MS_subDataset(Dataset):
    def __init__(self,datasets,vocab,max_spectrum_size):
        self.datasets = datasets
        self.vocab = vocab
        self.max_spectrum_size=max_spectrum_size
        
    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self,idx):
        spec_x = [np.pad(one,(0,self.max_spectrum_size-len(one)),"constant",constant_values=-1) for one in self.datasets[idx][0]]
        spec_y = [np.pad(one,(0,self.max_spectrum_size-len(one)),"constant",constant_values=-1) for one in self.datasets[idx][1]]
        return tensorize(self.datasets[idx][2], self.vocab, assm=True)+(torch.tensor(spec_x),)+(torch.tensor(spec_y),)
# 
def get_spectrum(smiles,path):
    x_list=[]
    y_list=[]
    try:
        mol = MolTree(smiles)
        mol.recover()
        mol.assemble()
        for node in mol.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
        del mol.mol
        for node in mol.nodes:
            node.mol
            
    except AttributeError as e:
        warnings.warn("Entered An SMILES that does not meet the rules")
        return None
    with open(path,"r") as f:
        lines = f.read().split("\n")
        num = [i for i,one in enumerate(lines) if one.split(": ")[0] == "PK$PEAK"][0] # Perhaps it is faster to use for.
        for one in lines[num+1:-2]:
            x,y,y2 = one.split(" ")[2:]
            x_list.append(float(x))
            y_list.append(float(y))
    return np.asarray(x_list,dtype=np.float32),np.asarray(y_list,dtype=np.float32),mol
if __name__=="__main__":
    print("test")