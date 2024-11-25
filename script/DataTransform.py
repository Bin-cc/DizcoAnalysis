# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 23:07:37 2024

@author: 18811
"""
#这是为了构建transformer模型下蛋白和探针的输入
#在氨基酸层面

import sys
sys.path.append('D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/Ligand Discovery/fragment-embedding/')
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from fragmentembedding import FragmentEmbedder
from rdkit import RDLogger


class ParseSeqEmd():
    
    def __init__(self):
        self.res_symbol = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
            'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
            'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,
            'U': 21, 'O':22
            }
        
    def SeqEmd(self, seq, max_len, emd_dim=512):
        seq_symbol = [self.res_symbol[res] for res in seq]
        if len(seq_symbol) < max_len:
            seq_symbol = F.pad(torch.tensor(seq_symbol), (0, max_len-len(seq_symbol)), value=0)
        else: seq_symbol = torch.tensor(seq_symbol[:max_len])
        att_mask = (seq_symbol == 0).long()
        prot_embed = nn.Embedding(num_embeddings=42, embedding_dim=emd_dim, padding_idx=0)
        return prot_embed(seq_symbol), att_mask

class ParseProbeEmd():
    
    def __init__(self):
        self.probe_smile = {
            'AJ5': 'C#CCCC1(N=N1)CCC(NC)=O',
            'AJ8': 'C#CCCC1(N=N1)CCC(NC[C@H]2[C@@H](C)CCCN2CC3=CC=C(OC)C=C3)=O',
            'AJ12': 'C#CCCC1(N=N1)CCNC(/C(C2=CC=CC=C2)=C/C3=CC=CC=C3)=O',
            'AJ14': 'C#CCCC1(N=N1)CCC(NC(C2(C[C@H](C3)C4)C[C@H]4C[C@H]3C2)C)=O',
            'AJ22': 'C#CCCC1(N=N1)CCNC(/C(CC)=C/C2=CC=CC([N+]([O-])=O)=C2)=O',
            'AJ32': 'C#CCCC1(N=N1)CCNC(C2(CC2)C3=CC(OC(F)(F)O4)=C4C=C3)=O',
            'AJ39': 'C#CCCC1(N=N1)CCNC(CCC2=NC(C3=CC=CC=C3)=C(C4=CC=CC=C4)O2)=O',
            'CP78': 'C#CCCC1(N=N1)CCC(N2C(CC3=CC=CC=C3)CCCC2)=O'
            }
        
        self.fe = FragmentEmbedder()
    
    def ProbeEmd(self, probe, prot_len, max_len):
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        probe_smiles = self.probe_smile[probe]
        probe_emd = self.fe.transform([probe_smiles])
        probe_emd = probe_emd.repeat(prot_len, axis=0)
        if prot_len < max_len:
            probe_emd = F.pad(torch.tensor(probe_emd), (0, 0, 0, max_len-prot_len),value=0)
        else: probe_emd = torch.tensor(probe_emd[:max_len,:])
        
        return probe_emd

class CustomDataset(Dataset):
    def __init__(self, pairwise, max_length=1200):
        self.data = pairwise
        self.max_len = max_length
        self.pse = ParseSeqEmd()
        self.ppe = ParseProbeEmd()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        probe = self.data[idx][0]
        prot_seq = self.data[idx][1]
        res_label = self.data[idx][2]
        
        probe_emd = self.ppe.ProbeEmd(probe, len(prot_seq), self.max_len)
        prot_emd, att_mask = self.pse.SeqEmd(prot_seq, self.max_len)
        if len(prot_seq) < self.max_len:
            res_label = F.pad(torch.tensor(res_label), (0, self.max_len-len(prot_seq)), value=0)
        else: res_label = torch.tensor(res_label[:self.max_len])
        
        return probe_emd, prot_emd, res_label, att_mask
        
        
        