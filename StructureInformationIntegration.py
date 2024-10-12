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
from Bio.PDB.HSExposure import HSExposureCB,ExposureCN
from Bio.PDB.ResidueDepth import ResidueDepth
import pickle
import warnings
warnings.filterwarnings('ignore')


#使用Bio.PDB.DSSP包进行分析
#没有TCO、Kappa和Alpha等值
#输入包括蛋白模型model，pdb文件名以及蛋白氨基酸位点的位置信息
def dssp_analysis(model,pdbFile,prot_loc):
    
    #设置指向dssp.exe文件的路径并对模型数据分析
    dssp_path = 'D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/Structural Information Calculation/DSSP/dssp-3.0.0-win32.exe'
    dssp = DSSP(model, path+pdbFile,dssp=dssp_path)
    
    #遍历并提取dssp结果中每一个氨基酸的特征，不包括TCO、Kappa和Alpha
    dssp_result = [tuple((dssp[res])) for res in list(dssp.keys())]
    columns = ['DSSP index','Residue','Secondary structure','Relative ASA',
               'Phi','Psi','nho1_relidx','nho1_energy','onh1_relidx','onh1_energy',
               'nho2_relidx','nho2_energy','onh2_relidx','onh2_energy']
    dssp_result = pd.DataFrame(dssp_result,columns=columns)
    return dssp_result

#从DSSP命令行输出结果文件中提取每个氨基酸的信息
#输入为命令行输出结果文件路径
def extract_dssp_infor(output_path):
    
    #按行读取.dssp文件，存在list中
    with open(output_path, 'r') as file:
        lines = file.readlines()
    
    #选择要读取和保留的信息
    columns = ['DSSP index', 'Residue', 'AA', 'Secondary structure', 'Relative ASA', 
               'Phi', 'Psi', 'TCO', 'Kappa', 'Alpha', 
               'nho1_relidx', 'nho1_energy', 'onh1_relidx', 'onh1_energy',
               'nho2_relidx', 'nho2_energy', 'onh2_relidx', 'onh2_energy']
    
    #从每一行氨基酸的DSSP字符串中，从相应位置提取信息
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
    
    #将提取结果转为DataFrame并输出
    dssp_result = pd.DataFrame(dssp_result, columns=columns)
    return dssp_result

#使用DSSP的命令行执行相关分析
#输出TCO、Kappa和Alpha等值
#输入为蛋白pdb文件名
def retrieve_dssp_from_file(pdbFile):
    
    #设置DSSP.exe文件路径，命令行执行环境路径以及输入与输出路径
    dssp_path = 'D:/DSSP/DSSP/dssp-3.0.0-win32.exe'
    exe_path = 'D:/DSSP/DSSP/'
    input_path = path+pdbFile
    output_path = exe_path+'output_{:}.dssp'.format(pdbFile.split('-')[1])
    
    #按照DSSP包的要求编写命令并用subprocess执行
    cmd = '{:} -i "{:}" -o {:}'.format(dssp_path,input_path,output_path)
    subprocess.call(cmd, cwd=exe_path, shell=True)
    
    #使用extract_dssp_infor函数读取输出文件中的DSSP信息，处理完后删除输出文件
    dssp_result = extract_dssp_infor(output_path)
    os.remove(output_path)
    return dssp_result

#计算SASA
#输入为蛋白的结构structure
def sasa_cal(structure):
    
    #调用Bio.PDB.SASA包计算SASA，计算层面为Residue
    sr = ShrakeRupley(n_points=600)
    sr.compute(structure, level="R")
    
    #将计算结果转为DataFrame格式并输出
    sasa_result = []
    for res in structure.get_residues():
        sasa_result.append(tuple((res.get_id()[1],res.get_resname(),res.sasa)))
    sasa_result = pd.DataFrame(sasa_result,columns=['Res_id','Residue','sasa'])
    return sasa_result

#计算Half-sphere exposure (HSE)
#输入为蛋白模型model和蛋白氨基酸位点的位置信息
def hse_cal(model,prot_loc):
    
    #调用HSExposureCB包计算HSE
    hse = HSExposureCB(model)
    hse_result = [hse[key] for key in hse.keys()]
    
    #将计算结果转为DataFrame格式并输出
    hse_result = pd.DataFrame(hse_result,columns=['HSE-up','HSE-down','angle'])
    hse_result.insert(0, 'Res_id', [key[1][1] for key in hse.keys()])
    hse_result = pd.merge(hse_result, prot_loc[['Residue','Res_id']],how='left')
    hse_result = hse_result[['Residue','Res_id','HSE-up','HSE-down','angle']]
    return hse_result

#计算蛋白氨基酸的深度depth
#输入为蛋白模型model和蛋白氨基酸位点的位置信息
def depth_cal(model,prot_loc):
    
    #调用ResidueDepth包计算depth
    rd = ResidueDepth(model)
    rd_result = [tuple((key[1][1],rd[key][0],rd[key][1])) for key in rd.keys()]
    
    #将计算结果转为DataFrame格式并输出
    rd_result = pd.DataFrame(rd_result,columns=['Res_id','depth','surface_exposure'])
    rd_result = pd.merge(rd_result, prot_loc[['Residue','Res_id']],how='left')
    rd_result = rd_result[['Residue','Res_id','depth','surface_exposure']]
    return rd_result

#计算氨基酸的配体数
#输入为蛋白模型model
def coord_cal(model,radius=5):    
    
    ##调用ExposureCN包计算coord number
    coord_num = ExposureCN(model,radius=radius)
    coord_result = [coord_num[key] for key in coord_num.keys()]
    
    #将计算结果转为DataFrame格式并输出
    coord_result = pd.DataFrame(coord_result,columns=['Coordination Number'])
    coord_result.insert(0, 'Res_id', [key[1][1] for key in coord_num.keys()])
    return coord_result

#对给定的蛋白，计算当前所有的结构信息，并整合汇总
#输入为pdb文件名
def data_integration(pdbFile):
    
    #根据输入的pdb文件，计算蛋白的结构和模型
    prot = pdbFile.split('-')[1]
    pdb = PDBParser(PERMISSIVE=1)
    structure = pdb.get_structure(prot, path+pdbFile)
    model = structure[0]
    prot_loc = prot_loc_dic[prot]
    
    #依次计算蛋白的不同结构特征
    dssp_result_bio = dssp_analysis(model,pdbFile,prot_loc)
    dssp_result_exe = retrieve_dssp_from_file(pdbFile)
    sasa_result = sasa_cal(structure)
    sasa_result['RSA'] = dssp_result_bio.loc[:,'Relative ASA']
    hse_result = hse_cal(model,prot_loc)
    rd_result = depth_cal(model,prot_loc)
    coord_result = coord_cal(model,radius=12)
    
    #将输出的结构特征汇总整合并输出
    use_col = ['DSSP index','Residue','Secondary structure','Phi','Psi',
               'nho1_energy','onh1_energy','nho2_energy','onh2_energy']
    data_infor = dssp_result_bio[use_col].copy()
    data_infor = pd.merge(data_infor, dssp_result_exe[['DSSP index','TCO', 'Kappa', 'Alpha']],on='DSSP index')
    data_infor = data_infor.rename(columns={'DSSP index':'Res_id'})
    data_infor = pd.merge(data_infor, sasa_result.drop('Residue',axis=1),on='Res_id')
    data_infor = pd.merge(data_infor, hse_result.drop(['Residue','angle'],axis=1),on='Res_id')
    data_infor = pd.merge(data_infor, rd_result.drop('Residue',axis=1),on='Res_id')
    data_infor = pd.merge(data_infor, coord_result,on='Res_id')
    
    return {prot:data_infor}
    

path = 'D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/AlphaFold_pdbFiles/'
prot_loc_dic = pickle.load(open('D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/pkl_files/Coordinates of residues of proteins.pkl','rb'))

res_id_trans = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP',
                'C':'CYS','E':'GLU','Q':'GLN','G':'GLY',
                'H':'HIS','I':'ILE','L':'LEU','K':'LYS',
                'M':'MET','F':'PHE','P':'PRO','S':'SER',
                'T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    