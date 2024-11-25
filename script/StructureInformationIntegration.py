# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 21:43:04 2024

@author: 18811
"""

import pandas as pd
import os
import subprocess
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.HSExposure import HSExposureCB
from Bio.PDB.ResidueDepth import ResidueDepth
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import pickle
import warnings
warnings.filterwarnings('ignore')


def dssp_analysis(model,pdbFile,prot_loc):
    #没有TCO、Kappa和Alpha等值
    dssp_path = 'D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/Structural Information Calculation/DSSP/dssp-3.0.0-win32.exe'
    dssp = DSSP(model, path+pdbFile,dssp=dssp_path)

    dssp_result = [tuple((dssp[res])) for res in list(dssp.keys())]
    columns = ['DSSP index','Residue','Secondary structure','Relative ASA',
               'Phi','Psi','nho1_relidx','nho1_energy','onh1_relidx','onh1_energy',
               'nho2_relidx','nho2_energy','onh2_relidx','onh2_energy']
    dssp_result = pd.DataFrame(dssp_result,columns=columns)
    return dssp_result

def extract_dssp_infor(output_path):
    with open(output_path, 'r') as file:
        lines = file.readlines()
    
    columns = ['DSSP index', 'Residue', 'AA', 'Secondary structure', 'Relative ASA', 
               'Phi', 'Psi', 'TCO', 'Kappa', 'Alpha', 
               'nho1_relidx', 'nho1_energy', 'onh1_relidx', 'onh1_energy',
               'nho2_relidx', 'nho2_energy', 'onh2_relidx', 'onh2_energy']
    
    dssp_result = []
    for line in lines[28:]:
        if line[13] == '!': continue
        
        dssp_index = int(line[0:5].strip())  # DSSP index
        residue = int(line[5:10].strip())    # Residue number
        aa = line[13]                        # Amino acid one-letter code
        sec_structure = line[16]             # Secondary structure
        rel_asa = float(line[35:38].strip()) # Relative ASA
        phi = float(line[103:109].strip())   # Phi angle
        psi = float(line[109:115].strip())   # Psi angle
        tco = float(line[85:91].strip())     # TCO value
        kappa = float(line[91:97].strip())   # Kappa value
        alpha = float(line[97:103].strip())  # Alpha value
        # Extract hydrogen bonding information
        nho1_relidx = int(line[39:45].strip())
        nho1_energy = float(line[46:50].strip())
        onh1_relidx = int(line[50:56].strip())
        onh1_energy = float(line[57:61].strip())
        nho2_relidx = int(line[61:67].strip())
        nho2_energy = float(line[68:72].strip())
        onh2_relidx = int(line[72:78].strip())
        onh2_energy = float(line[79:83].strip())
    
        row = [dssp_index, residue, aa, sec_structure, rel_asa, phi, psi, 
               tco, kappa, alpha, 
               nho1_relidx, nho1_energy, onh1_relidx, onh1_energy,
               nho2_relidx, nho2_energy, onh2_relidx, onh2_energy]
        
        dssp_result.append(row)
    
    dssp_result = pd.DataFrame(dssp_result, columns=columns)
    return dssp_result
    
def retrieve_dssp_from_file(pdbFile):
    dssp_path = 'D:/All_for_paper/DSSP/dssp-3.0.0-win32.exe'
    exe_path = 'D:/All_for_paper/DSSP/'
    input_path = path+pdbFile
    output_path = exe_path+'output_{:}.dssp'.format(pdbFile.split('-')[1])
    cmd = '{:} -i "{:}" -o {:}'.format(dssp_path,input_path,output_path)
    subprocess.call(cmd, cwd=exe_path, shell=True)
    
    dssp_result = extract_dssp_infor(output_path)
    os.remove(output_path)
    return dssp_result

def sasa_cal(structure):
    sr = ShrakeRupley(n_points=600)
    sr.compute(structure, level="R")
    
    sasa_result = []
    for res in structure.get_residues():
        sasa_result.append(tuple((res.get_id()[1],res.get_resname(),res.sasa)))
    sasa_result = pd.DataFrame(sasa_result,columns=['Res_id','Residue','sasa'])
    return sasa_result

def hse_cal(model,prot_loc):
    hse = HSExposureCB(model)
    hse_result = [hse[key] for key in hse.keys()]
    hse_result = pd.DataFrame(hse_result,columns=['HSE-up','HSE-down','angle'])
    hse_result.insert(0, 'Res_id', [key[1][1] for key in hse.keys()])
    hse_result = pd.merge(hse_result, prot_loc[['Residue','Res_id']],how='left')
    hse_result = hse_result[['Residue','Res_id','HSE-up','HSE-down','angle']]
    return hse_result

def depth_cal(model,prot_loc):
    rd = ResidueDepth(model)
    rd_result = [tuple((key[1][1],rd[key][0],rd[key][1])) for key in rd.keys()]
    rd_result = pd.DataFrame(rd_result,columns=['Res_id','depth','surface_exposure'])
    rd_result = pd.merge(rd_result, prot_loc[['Residue','Res_id']],how='left')
    rd_result = rd_result[['Residue','Res_id','depth','surface_exposure']]
    return rd_result

def coord_cal(pdbFile,thre=5.0):    
    u = mda.Universe(path+pdbFile)
    ca_atoms = u.select_atoms('name CA')
    dist_matrix = distance_array(ca_atoms.positions, ca_atoms.positions,)
    coord_num = (dist_matrix < thre).sum(axis=1) - 1
    coord_num = [tuple((atom.resname,atom.resid,cn)) for atom, cn in zip(ca_atoms, coord_num)]
    coord_num = pd.DataFrame(coord_num,columns=['Residue','Res_id','Coordination Number'])
    return coord_num

def data_integration(pdbFile):
    prot = pdbFile.split('-')[1]
    pdb = PDBParser(PERMISSIVE=1)
    structure = pdb.get_structure(prot, path+pdbFile)
    model = structure[0]
    prot_loc = prot_loc_dic[prot]
    
    dssp_result_bio = dssp_analysis(model,pdbFile,prot_loc)
    dssp_result_exe = retrieve_dssp_from_file(pdbFile)
    sasa_result = sasa_cal(structure)
    sasa_result['RSA'] = dssp_result_bio.loc[:,'Relative ASA']
    hse_result = hse_cal(model,prot_loc)
    rd_result = depth_cal(model,prot_loc)
    coord_num = coord_cal(pdbFile,thre=12)
    
    use_col = ['DSSP index','Residue','Secondary structure','Phi','Psi',
               'nho1_energy','onh1_energy','nho2_energy','onh2_energy']
    data_infor = dssp_result_bio[use_col].copy()
    data_infor = pd.merge(data_infor, dssp_result_exe[['DSSP index','TCO', 'Kappa', 'Alpha']],on='DSSP index')
    data_infor = data_infor.rename(columns={'DSSP index':'Res_id'})
    data_infor = pd.merge(data_infor, sasa_result.drop('Residue',axis=1),on='Res_id')
    data_infor = pd.merge(data_infor, hse_result.drop(['Residue','angle'],axis=1),on='Res_id')
    data_infor = pd.merge(data_infor, rd_result.drop('Residue',axis=1),on='Res_id')
    data_infor = pd.merge(data_infor, coord_num.drop('Residue',axis=1),on='Res_id')
    
    return {prot:data_infor}
    

path = 'D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/AlphaFold_pdbFiles/'
prot_loc_dic = pickle.load(open('D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/pkl_files/Coordinates of residues of proteins.pkl','rb'))

res_id_trans = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP',
                'C':'CYS','E':'GLU','Q':'GLN','G':'GLY',
                'H':'HIS','I':'ILE','L':'LEU','K':'LYS',
                'M':'MET','F':'PHE','P':'PRO','S':'SER',
                'T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    