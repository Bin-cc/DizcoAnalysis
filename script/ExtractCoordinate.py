# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:42:06 2024

@author: 18811
"""

import pandas as pd
from Bio.PDB.PDBParser import PDBParser

def gain_atom_loc(structure):
    atom_result = []
    for residue in structure.get_residues():
        for atom in residue:
            res_name = residue.resname
            res_id = residue.id[1]
            atom_name,bfactor,element = atom.get_name(),atom.get_bfactor(),atom.element
            x,y,z = atom.get_vector()
            atom_result.append(tuple((res_name,res_id,atom_name,x,y,z,bfactor,element)))
    atom_result = pd.DataFrame(atom_result,columns=['Residue','Res_id','Atom_name','x','y','z','bfactor','element'])
    return atom_result

def gain_prot_loc(pdbFile):
    des_path = 'D:/All_for_paper/1. PhD Work Program/3. Research project/1. Dizco/AlphaFold_pdbFiles/'
    prot = pdbFile.split('-')[1]
    pdb = PDBParser(PERMISSIVE=1)
    structure = pdb.get_structure(prot, des_path+pdbFile)
    atom_result = gain_atom_loc(structure)
    res_result = atom_result[atom_result['Atom_name']=='CA'].reset_index(drop=True)
    return {prot:res_result}
