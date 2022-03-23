import re
from tkinter import W
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import mol_graph_cycloadd
from scaling import min_max_normalize, min_max_normalize_reaction,reaction_to_reactants,scale_targets


elem_list = ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S',
             'Si', 'B', 'I', 'K', 'Na', 'P', 'Mg', 'Li', 'Al', 'H']


def convert_str_to_array(string):
    string = string.strip('"[]\n')
    string_list = string.split()

    try:
        string_list = list(map(float, string_list))
    except:
        string_list = np.array([])

    return np.array(string_list)


def get_reactive_sites(reactive_core):
    return list(map(int, reactive_core.strip('[]').split(',')))


def get_atom_list(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    atom_list = []
    for atom in mol.GetAtoms():
        atom_list.append(atom.GetSymbol())

    return atom_list


def get_aromatic_list(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    arom_list = []
    for atom in mol.GetAtoms():
        arom_list.append(atom.GetIsAromatic())

    return arom_list

targets = pd.read_csv('output_attempt2_mapped.csv')[['smiles','DG_TS']]
descriptors = pd.read_pickle('atom_desc_cycloadd_wln.pkl')
reactants = reaction_to_reactants(descriptors['smiles'].tolist())
descriptors,_ = min_max_normalize(descriptors.copy(), train_smiles=reactants)
descriptors.to_csv('atom_desc_scaled.csv')
mol_descs = pd.read_csv('reaction_desc_cycloadd_wln.csv')
#mol_descs,_ = min_max_normalize_reaction(mol_descs.copy(), train_smiles=targets['smiles'])
#mol_descs.to_csv('reaction_desc_scaled.csv')

reactions = pd.merge(mol_descs, targets, on='smiles')
reactions['reactants'] = reactions['smiles'].apply(lambda x: x.split('>')[0])
reactions['products'] = reactions['smiles'].apply(lambda x: x.split('>')[-1])

#reactions = reactions[['reactants', 'products']]
#descriptors = descriptors.set_index('smiles')
reactions['desc_core'] = reactions.apply(lambda x: mol_graph_cycloadd.smiles2graph_pr(descriptors, x['reactants'], x['products']), axis=1)

reactions = reactions[reactions['desc_core'] != 'lol']

reaction_feature_list = []

tmp_list = reactions['desc_core'].values.tolist()
tmp_list_Er = reactions['E_r'].values.tolist()
tmp_list_G = reactions['G'].values.tolist()
tmp_list_G_alt1 = reactions['G_alt1'].values.tolist()
tmp_list_G_alt2 = reactions['G_alt2'].values.tolist()
tmp_list_target = reactions['DG_TS'].values.tolist() 

for idx, reaction in enumerate(tmp_list):
    rxn_desc_array = np.array([tmp_list_Er[idx], tmp_list_G[idx], tmp_list_G_alt1[idx], tmp_list_G_alt2[idx], tmp_list_target[idx]])
    reaction_feature_list.append(np.concatenate((np.concatenate([arr[:-1] for arr in reaction[0]]), rxn_desc_array)))
    
columns_list = []

for i in range(1,6):
    for desc_name in ['partial_charge', 'fukui_elec', 'fukui_neu', 'nmr', 'spin_dens_triplet', 'sasa', 'pint']:
        columns_list.append(f'{desc_name}_{i}') 

columns_list += ['E_r', 'G', 'G_alt1', 'G_alt2', 'DG_TS']

print(len(reaction_feature_list[1]), len(columns_list))

df = pd.DataFrame(reaction_feature_list, columns=columns_list)

print(df.head())

df.to_pickle('input_random_forest.pkl')

#for i in range(len(tmp_list)):
#    for j in range(len(tmp_list[i][0])):
#        data_point_list.append(tmp_list[i][0][j])

#df = pd.DataFrame(data_point_list, columns=['partial_charge', 'fukui_elec', 'fukui_neu', 'nmr', 'spin_dens', 'spin_dens_triplet',
#                                            'sasa','pint','atom_type'])
#df['atom_type'] = df['atom_type'].apply(lambda x: elem_list[int(x)])

#df.to_csv('atom_desc_reactive_sites_cycloadd.csv')
#df.to_pickle('atom_desc_reactive_sites_cycloadd.pkl')

