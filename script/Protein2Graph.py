# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:34:36 2024

@author: 18811
"""

import sys
sys.path.append('E:/Proteomics/PhD_script/1. Dizco/')
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from ExtractCoordinate import gain_prot_loc
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import pickle


class prot2graph():
    
    def __init__(self,file_path,uniprot_infor):
        self.file_path = file_path
        
        self.res_id_trans = {
            'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'GLU': 'E', 'GLN': 'Q',
            'ASP': 'D', 'ASN': 'N', 'HIS': 'H', 'TRP': 'W', 'PHE': 'F',
            'TYR': 'Y', 'ARG': 'R', 'LYS': 'K', 'SER': 'S', 'THR': 'T',
            'MET': 'M', 'ALA': 'A', 'GLY': 'G', 'PRO': 'P', 'CYS': 'C'
            }
        
        self.res_symbol = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
            'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
            'U': 21, 'O':22
            }
        
        self.blosum80_feature = {
            'A': [7, -3, -3, -3, -1, -2, -2, 0, -3, -3, -3, -1, -2, -4, -1, 2, 0, -5, -4, -1, -3, -2, -1],
            'R': [-3, 9, -1, -3, -6, 1, -1, -4, 0, -5, -4, 3, -3, -5, -3, -2, -2, -5, -4, -4, -2, 0, -2],
            'N': [-3, -1, 9, 2, -5, 0, -1, -1, 1, -6, -6, 0, -4, -6, -4, 1, 0, -7, -4, -5, 5, -1, -2],
            'D': [-3, -3, 2, 10, -7, -1, 2, -3, -2, -7, -7, -2, -6, -6, -3, -1, -2, -8, -6, -6, 6, 1, -3],
            'C': [-1, -6, -5, -7, 13, -5, -7, -6, -7, -2, -3, -6, -3, -4, -6, -2, -2, -5, -5, -2, -6, -7, -4],
            'Q': [-2, 1, 0, -1, -5, 9, 3, -4, 1, -5, -4, 2, -1, -5, -3, -1, -1, -4, -3, -4, -1, 5, -2],
            'E': [-2, -1, -1, 2, -7, 3, 8, -4, 0, -6, -6, 1, -4, -6, -2, -1, -2, -6, -5, -4, 1, 6, -2],
            'G': [0, -4, -1, -3, -6, -4, -4, 9, -4, -7, -7, -3, -5, -6, -5, -1, -3, -6, -6, -6, -2, -4, -3],
            'H': [-3, 0, 1, -2, -7, 1, 0, -4, 12, -6, -5, -1, -4, -2, -4, -2, -3, -4, 3, -5, -1, 0, -2],
            'I': [-3, -5, -6, -7, -2, -5, -6, -7, -6, 7, 2, -5, 2, -1, -5, -4, -2, -5, -3, 4, -6, -6, -2],
            'L': [-3, -4, -6, -7, -3, -4, -6, -7, -5, 2, 6, -4, 3, 0, -5, -4, -3, -4, -2, 1, -7, -5, -2],
            'K': [-1, 3, 0, -2, -6, 2, 1, -3, -1, -5, -4, 8, -3, -5, -2, -1, -1, -6, -4, -4, -1, 1, -2],
            'M': [-2, -3, -4, -6, -3, -1, -4, -5, -4, 2, 3, -3, 9, 0, -4, -3, -1, -3, -3, 1, -5, -3, -2],
            'F': [-4, -5, -6, -6, -4, -5, -6, -6, -2, -1, 0, -5, 0, 10, -6, -4, -4, 0, 4, -2, -6, -6, -3],
            'P': [-1, -3, -4, -3, -6, -3, -2, -5, -4, -5, -5, -2, -4, -6, 12, -2, -3, -7, -6, -4, -4, -2, -3],
            'S': [2, -2, 1, -1, -2, -1, -1, -1, -2, -4, -4, -1, -3, -4, -2, 7, 2, -6, -3, -3, 0, -1, -1],
            'T': [0, -2, 0, -2, -2, -1, -2, -3, -3, -2, -3, -1, -1, -4, -3, 2, 8, -5, -3, 0, -1, -2, -1],
            'W': [-5, -5, -7, -8, -5, -4, -6, -6, -4, -5, -4, -6, -3, 0, -7, -6, -5, 16, 3, -5, -8, -5, -5],
            'Y': [-4, -4, -4, -6, -5, -3, -5, -6, 3, -3, -2, -4, -3, 4, -6, -3, -3, 3, 11, -3, -5, -4, -3],
            'V': [-1, -4, -5, -6, -2, -4, -4, -6, -5, 4, 1, -4, 1, -2, -4, -3, 0, -5, -3, 7, -6, -4, -2],
            'B': [-3, -2, 5, 6, -6, -1, 1, -2, -1, -6, -7, -1, -5, -6, -4, 0, -1, -8, -5, -6, 6, 0, -3],
            'Z': [-2, 0, -1, 1, -7, 5, 6, -4, 0, -6, -5, 1, -3, -6, -2, -1, -2, -5, -4, -4, 0, 6, -1],
            'X': [-1, -2, -2, -3, -4, -2, -2, -3, -2, -2, -2, -2, -2, -3, -3, -1, -1, -5, -3, -2, -3, -1, -2],
            }
        
        self.str_infor_dic = pickle.load(open('D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/pkl_files/structure_infor_summary.pkl','rb'))
        self.prot_infor = uniprot_infor
    
    def CreateCoordGraph(self,pdbFile,protName='pdbFormat',graph_format='distance',Return=False,max_len=1400,thre=8.0):
        
        if protName == 'pdbFormat':
            prot = pdbFile.split('-')[1]
            coord_table = gain_prot_loc(pdbFile)[prot]
            if len(coord_table) > max_len: coord_table = coord_table.iloc[:max_len,:]
            seq = ''.join([self.res_id_trans[res] for res in coord_table['Residue'].values])
        elif protName == 'idFormat':
            seq = self.uniprot_infor[self.uniprot_infor['Entry']==pdbFile].iloc[0,-1]
            
        seq_emd = prot2graph.SeqEmbedding(self,seq)
        #blosum_emd = torch.tensor(np.array([self.blosum80_feature[res] for res in seq]),dtype=torch.float32)
        #coord_emd = torch.tensor(coord_table[['x','y','z']].values,dtype=torch.float32)
        # str_table = self.str_infor_dic[prot].copy()
        # if len(str_table) != len(seq): return None
        # str_table = StandardScaler().fit_transform(str_table.iloc[:,3:])
        # str_table = np.asarray(str_table).astype(np.float32)
        # str_emd = torch.tensor(str_table)
        
        #X = torch.cat([coord_emd,str_emd,seq_emd,blosum_emd], dim=1)
        X = seq_emd
        
        Y = torch.tensor(np.zeros(len(seq),dtype=np.int64()),dtype=torch.long)
        
        if graph_format == 'distance':
            dist_matrix = squareform(pdist(coord_table[['x','y','z']].values))
            src, dst = np.where((dist_matrix<thre)&(dist_matrix>0))
        elif graph_format == 'sequential':
            src = list(coord_table.index)[:-1]
            dst = list(coord_table.index)[1:]
        edge_index = torch.tensor(np.array([src, dst]),dtype=torch.long)
        
        data = Data(x=X,edge_index=edge_index)
        
        if Return:
            return data
        else: torch.save(data,f'{self.file_path}{prot}.pt')
        
    @staticmethod
    def SeqEmbedding(self,seq,emd_dim=64):
        seq_symbol = [self.res_symbol[res] for res in seq]
        prot_embed = nn.Embedding(num_embeddings=32, embedding_dim=emd_dim, padding_idx=0)
        return prot_embed(torch.tensor(seq_symbol))
        