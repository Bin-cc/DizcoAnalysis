# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:23:23 2024

@author: 18811
"""

from os import listdir
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyteomics import mass,parser


class MSPiepline():
    
    def __init__(self,ms_path,outputName):
        self.ms_path = ms_path
        self.outputName = outputName
        self.h = 1.00728
        try: self.summary = pd.read_csv(f'{self.ms_path}{self.outputName}.csv')
        except: self.summary = MSPiepline.generate_ms2(self.ms_path,self.outputName)
        self.summary_ms1_norm = MSPiepline.generate_ms1(self.summary)
        
    
    #将.mgf文件中的MS1数据提取并整理为dataframe
    @staticmethod
    def generate_ms2(ms_path,outputName):
        summary = pd.DataFrame()
        for file in listdir((ms_path)):
            with open(ms_path+file, 'r') as file:
                lines = file.readlines()
            
            result = pd.DataFrame()
            for line in tqdm(lines):
                if 'BEGIN IONS' in line: values = []
                elif 'TITLE' in line:
                    fileName = line.split('"')[1]
                    scan = line.split('scan=')[1].split('"')[0]
                elif 'RTINSECONDS' in line: rt = line.split('=')[1]
                elif 'PEPMASS' in line:
                    try: ms1,intensity = line.split('=')[1].split(' ')
                    except: ms1,intensity = line.split('=')[1].split('\n')[0],np.nan
                elif 'CHARGE' in line:charge = line.split('+')[0].split('=')[1]
                elif 'END IONS' not in line:
                    mz,inten = line.split('\n')[0].split(' ')
                    values.append(tuple((fileName,scan,rt,ms1,intensity,charge,np.array(mz),np.array(inten))))
                if 'END IONS' in line:
                    df_values = pd.DataFrame(values,columns=['fileName','scan','RT','M.W','Seq_Intensity','Charge','Fragment m/z','Fragment Intensity'])
                    result = pd.concat([result,df_values])
            summary = pd.concat([summary,result])
        summary.to_csv(f'{ms_path}{outputName}.csv',index=False)
        return summary
    
    @staticmethod
    def generate_ms1(summary):
        summary_ms1 = summary.iloc[:,:6].drop_duplicates().reset_index(drop=True)
        summary_ms1_norm = summary_ms1.round({'M.W':1}).copy()
        return summary_ms1_norm

    def generate_mod(self,seq,mod_mass,min_len=6,max_len=144,remain=1):
        pep_lt = parser.cleave(seq, parser.expasy_rules['trypsin'], 0)
        pep_lt = [pep for pep in pep_lt if len(pep)>=min_len and len(pep)<=max_len]
        pep_df = pd.DataFrame(pep_lt,columns=['Peptide'])
        pep_df['pep_mass'] = [mass.calculate_mass(sequence=pep) for pep in pep_lt]
        pep_df['pep_mass_mod'] = pep_df['pep_mass'].values+mod_mass
        
        result,result_mod = pd.DataFrame(),pd.DataFrame()
        for (file,ms,charge),table in self.summary_ms1_norm.groupby(by=['fileName','M.W','Charge']):
            pep_df_copy = pep_df.copy()
            pep_df_copy['non_mod_norm'] = (pep_df_copy['pep_mass'].values+self.h*int(charge))/int(charge)
            pep_df_copy['mod_norm'] = (pep_df_copy['pep_mass_mod'].values+self.h*int(charge))/int(charge)
            pep_df_copy['non_mod_norm'] = pep_df_copy['non_mod_norm'].round(remain)
            pep_df_copy['mod_norm'] = pep_df_copy['mod_norm'].round(remain)
            
            if ms in pep_df_copy['non_mod_norm'].values:
                table_copy = table.copy()
                table_copy = MSPiepline.data_match(seq,ms,table_copy,pep_df_copy,'non_mod_norm')
                result = pd.concat([result,table_copy],axis=0)
            
            if ms in pep_df_copy['mod_norm'].values:
                table_copy = table.copy()
                table_copy = MSPiepline.data_match(seq,ms,table_copy,pep_df_copy,'mod_norm')
                result_mod = pd.concat([result_mod,table_copy],axis=0)
        
        return result.reset_index(drop=True),result_mod.reset_index(drop=True)
    
    @staticmethod
    def data_match(seq,ms,table_copy,pep_df_copy,mod_type):
        match_pep = pep_df_copy[pep_df_copy[mod_type]==ms].iloc[0,0]
        table_copy['match_pep'] = match_pep
        start_site = seq.find(match_pep)+1
        table_copy['start_site'] = start_site
        table_copy['end_site'] = start_site+len(match_pep)
        return table_copy
        
class ParseMS2():
    
    def __init__(self, ion_type=['b','y'], max_charge=4):
        self.ion_type = ion_type
        self.max_charge = max_charge
        
    def theoMS2(self,seq):
        summary_ms2 = {}
        for charge in range(1,self.max_charge+1):
            ms2 = []
            for i in range(1,len(seq)):
                data = []
                for ion in self.ion_type:
                    if ion == 'b':
                        s = seq[:i]
                        index = i
                    elif ion == 'y':
                        s = seq[i:]
                        index = len(seq)-i
                    frag_mass = mass.calculate_mass(sequence=s, ion_type=ion, charge=charge)
                    data.append(tuple((s,f'{ion}{index}^{charge}',frag_mass)))
                data = pd.DataFrame(data,columns=['fragment','cleavage_site','frag_mass'])
                data_trans = tuple((i,))+tuple(data['fragment'].values)+tuple(data['cleavage_site'].values)+tuple(data['frag_mass'].values)
                ms2.append(data_trans)
            ms2 = pd.DataFrame(ms2,columns=['Index','fragment (b)','fragment (y)','site (b)','site (y)','frag_mass (b)','frag_mass (y)'])
            summary_ms2.setdefault(charge,ms2)
        return summary_ms2
            
            