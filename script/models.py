# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 00:27:03 2024

@author: 18811
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,SAGPooling,GATv2Conv
from torch_geometric.nn import global_mean_pool as gmp
from torch.autograd import Variable
import math


#%%
#测试模型，仅使用最基础的MLP，已检测数据质量

class test_model(nn.Module):
    
    def __init__(self,d_model=512,dropout=0.2):
        super(test_model,self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc0 = nn.Linear(d_model,d_model)
        self.fc1 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024,512)
        self.out = nn.Linear(512,1)
    
    def forward(self,probe_emd,prot_emd):
        prot_emd = self.dropout(F.relu(self.fc0(prot_emd)))
        prot_emd = prot_emd.mean(dim=1, keepdim=True)
        prot_emd = prot_emd.squeeze(1)
        probe_emd = probe_emd.squeeze(1)
        
        X = torch.cat([prot_emd,probe_emd],dim=1)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.dropout(F.relu(self.fc2(X)))
        out = F.sigmoid(self.out(X))
        
        return out

#%%
class dizco_CNN(nn.Module):
    
    def __init__(self,str_dim,pos_dim,dropout=0.2,kernel_size=(1,3),stride=1,padding=0):
        super(dizco_CNN,self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(125)
        self.bn3 = nn.BatchNorm1d(175)
        self.bn4 = nn.BatchNorm1d(50)
        
        self.w1 = nn.Linear(str_dim, 100)
        self.w2 = nn.Linear(100, 100)
        self.w3 = nn.Linear(100, 125)
        self.w4 = nn.Linear(125, 125)
        self.w5 = nn.Linear(125, 175)
        self.w6 = nn.Linear(175, 175)
        self.w7 = nn.Linear(175, 125)
        self.w8 = nn.Linear(125, 100)
        self.w9 = nn.Linear(100, 50)
        self.w10 = nn.Linear(50, 50)
        self.w11 = nn.Linear(50, 30)
        
        h_cal = lambda x:(x-kernel_size[0]+2*padding)/stride+1
        w_cal = lambda x:(x-kernel_size[1]+2*padding)/stride+1
        self.b,self.h,self.w = pos_dim[0],pos_dim[1],pos_dim[2]
        for _ in range(4):
            self.h = h_cal(self.h)
            self.w = w_cal(self.w)
        
        self.pos_conv_model = nn.Sequential(
            
            nn.Conv2d(1,10,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.ReLU(),
            nn.Conv2d(10,20,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20,25,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(25),
            nn.Conv2d(25,50,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.ReLU(),
            #nn.BatchNorm2d(50),
            nn.Flatten(1,-1),
            nn.Linear(int(self.w*50*self.h), 20),
            nn.ReLU(),
            nn.Linear(20, 50),
            nn.ReLU()
            
            )
        
        self.forward_ff_model = nn.Sequential(
            
            nn.Linear(80, 25),
            nn.ReLU(),
            self.dropout,
            nn.Linear(25, 30),
            nn.ReLU(),
            nn.BatchNorm1d(30),
            nn.Linear(30, 30),
            nn.ReLU(),
            self.dropout,
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20)
            
            )
        
        self.out = nn.Linear(20, 1)
        
        
    def forward(self,str_values,pos_values):
        pos_values = pos_values.unsqueeze(1)
        str_X = dizco_CNN.str_ff_model(self,str_values)
        pos_X = self.pos_conv_model(pos_values)
        pos_X = F.relu(self.w11(self.bn4(pos_X)))
        X = torch.cat([str_X,pos_X],dim=1)
        X = self.forward_ff_model(X)
        prob = F.sigmoid(self.out(X))
        
        return prob
        
    
    def str_ff_model(self,str_values):
        R0 = F.relu(self.w1(str_values))
        X = self.dropout(R0)
        X = F.relu(self.w2(X))
        X = torch.add(R0, X)
        RR1 = X
        X = F.relu(self.w3(self.dropout(self.bn1(X))))
        R1 = X
        X = F.relu(self.w4(self.dropout(X)))
        X = torch.add(R1, X)
        X = F.relu(self.w5(self.dropout(self.bn2(X))))
        R2 = X
        X = F.relu(self.w6(self.dropout(X)))
        X = torch.add(R2, X)
        X = F.relu(self.w7(self.dropout(self.bn3(X))))
        R3 = X
        X = F.relu(self.w4(self.dropout(X)))
        X = torch.add(R3, X)
        X = F.relu(self.w8(self.dropout(self.bn2(X))))
        X = torch.add(RR1, X)
        R4 = X
        X = F.relu(self.w2(self.dropout(X)))
        X = torch.add(R4, X)
        X = F.relu(self.w9(self.dropout(self.bn1(X))))
        R5 = X
        X = F.relu(self.w10(self.dropout(X)))
        X = torch.add(R5, X)
        
        return X

#%%
class FGragh(nn.Module):
    
    def __init__(self,graph_feat,probe_feat,dropout=0.2):
        super(FGragh, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.sage = GraphSAGENet(graph_feat,128,32)
        #self.str_ff = StructDNN(struct_feat,64,32,self.dropout)
        self.probe_ff = ProbeDNN(probe_feat,128,32,self.dropout)
        
        self.fc1 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        
    def forward(self,graph,probe):
        graph_emd = self.sage(graph.x,graph.edge_index,graph.batch)
        #struct_emd = self.str_ff(struct,graph.batch)
        probe_emd = self.probe_ff(probe)
        
        X = torch.cat([graph_emd,probe_emd],dim=1)
        X = self.dropout(F.relu(self.fc1(X)))
        out = F.sigmoid(self.out(X))
        
        return out

class GraphSAGENet(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,num_layers=2):
        super(GraphSAGENet, self).__init__()

        self.conv1 = GATv2Conv(in_dim, hid_dim,heads=4,concat=False,dropout=0.2)
        self.conv2 = GATv2Conv(hid_dim, hid_dim,heads=4,concat=False,dropout=0.2)
        self.conv3 = GATv2Conv(hid_dim, out_dim,heads=4,concat=False,dropout=0.2)
        self.pool = SAGPooling(hid_dim,ratio=0.6,GNN=SAGEConv)
        self.fc = nn.Linear(hid_dim,out_dim)
        
    def forward(self,x_feat,edge_index,batch):
        x = F.relu(self.conv1(x_feat,edge_index))
        x1 = gmp(x, batch)
        x, edge_index, _, batch, _, _ = self.pool(x,edge_index,None,batch)
        x2 = gmp(x, batch)
        x = F.relu(self.conv2(x,edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x,edge_index,None,batch)
        x3 = gmp(x, batch)
        x = x1+x2+x3
        x = F.relu(self.fc(x))
   
        return x

class StructDNN(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,dropout):
        super(StructDNN, self).__init__()
        
        self.dropout = dropout
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
    
    def forward(self,x_feat,batch):
        x = F.relu(self.fc1(x_feat))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        
        return gmp(x,batch)

class ProbeDNN(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,dropout):
        super(ProbeDNN, self).__init__()
        
        self.dropout = dropout
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
    
    def forward(self,x_feat):
        x = F.relu(self.fc1(x_feat))
        return F.relu(self.fc2(self.dropout(x)))

#%%
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding,self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*
                             -(math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self,x):
        x = x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)
        


class FFFTrans(nn.Module):
    def __init__(self,d_model=512,nhead=8,num_layers=1,dropout=0.2):
        super(FFFTrans, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoder = PositionalEncoding(d_model,dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024,512)
        self.out = nn.Linear(512,1)
        
    def forward(self,probe_emd,prot_emd,att_mask):
        prot_emd = self.pos_encoder(prot_emd)
        prot_emd = self.encoder(prot_emd, src_key_padding_mask=att_mask.bool())
        prot_emd = prot_emd.mean(dim=1, keepdim=True)
        prot_emd = prot_emd.squeeze(1)
        probe_emd = probe_emd.squeeze(1)
        
        X = torch.cat([prot_emd,probe_emd],dim=1)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.dropout(F.relu(self.fc2(X)))
        out = F.sigmoid(self.out(X))
        
        return out









