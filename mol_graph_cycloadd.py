import rdkit
import rdkit.Chem as Chem
import numpy as np
import pandas as pd
import os

# import tensorflow as tf

elem_list = ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S',
             'Si', 'B', 'I', 'K', 'Na', 'P', 'Mg', 'Li', 'Al', 'H']

atom_fdim_geo = len(elem_list) + 6 + 6 + 6 + 1

bond_fdim_geo = 6
bond_fdim_qm = 25 + 40
max_nb = 10

qm_descriptors = None


def initialize_qm_descriptors(df=None, path=None):
    global qm_descriptors
    if path is not None:
        qm_descriptors = pd.read_pickle(path).set_index('smiles')
    elif df is not None:
        qm_descriptors = df


def get_atom_classes():
    atom_classes = {}
    token = 0
    for e in elem_list:     #element
        for d in [0, 1, 2, 3, 4, 5]:    #degree
            for ev in [1, 2, 3, 4, 5, 6]:   #explicit valence
                for iv in [0, 1, 2, 3, 4, 5]:  #inexplicit valence
                    atom_classes[str((e, d, ev, iv))] = token
                    token += 1
    return atom_classes


def rbf_expansion(expanded, mu=0, delta=0.01, kmax=8):
    k = np.arange(0, kmax)
    return np.exp(-(expanded - (mu + delta * k))**2 / delta)


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE,
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def _mol2graph(qm_descriptors, rs, selected_descriptors, core=[]):
    atom_fdim_qm = len(selected_descriptors) + 1

    mol_rs = Chem.MolFromSmiles(rs)
    if not mol_rs:
        raise ValueError("Could not parse smiles string:", smiles)

    fatom_index = {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in mol_rs.GetAtoms()}

    n_atoms = mol_rs.GetNumAtoms()
    fatoms_qm = np.zeros((n_atoms, atom_fdim_qm))
    core_mask = np.zeros((n_atoms,), dtype=np.int32)

    for mol_id, smiles in enumerate(rs.split('.')):

        mol = Chem.MolFromSmiles(smiles)
        fatom_index_mol = {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in mol.GetAtoms()}

        qm_series = qm_descriptors.loc[smiles]

        partial_charge = qm_series['partial_charge'].reshape(-1, 1)

        fukui_elec = qm_series['fukui_elec'].reshape(-1, 1)

        fukui_neu = qm_series['fukui_neu'].reshape(-1, 1)

        nmr = qm_series['NMR'].reshape(-1, 1)

        spin_dens = qm_series['spin_dens'].reshape(-1, 1)

        spin_dens_triplet = qm_series['spin_dens_triplet'].reshape(-1, 1)

        sasa = qm_series['sasa'].reshape(-1,1)

        pint = qm_series['pint'].reshape(-1,1)

        selected_descriptors = set(selected_descriptors)

        atom_qm_descriptor = None

        # start from partial charge or fukui_elec or orbitals
        if "partial_charge" in selected_descriptors:
            atom_qm_descriptor = partial_charge

        if "fukui_elec" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, fukui_elec], axis=-1)
            else:
                atom_qm_descriptor = fukui_elec

        if "fukui_neu" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, fukui_neu], axis=-1)
            else:
                atom_qm_descriptor = fukui_neu

        if "nmr" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, nmr], axis=-1)
            else:
                atom_qm_descriptor = nmr

        if "spin_dens" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, spin_dens], axis=-1)
            else:
                atom_qm_descriptor = spin_dens

        if "spin_dens_triplet" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, spin_dens_triplet], axis=-1)
            else:
                atom_qm_descriptor = spin_dens_triplet

        if "sasa" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, sasa], axis=-1)
            else:
                atom_qm_descriptor = sasa

        if "pint" in selected_descriptors:
            if atom_qm_descriptor is not None:
                atom_qm_descriptor = np.concatenate([atom_qm_descriptor, pint], axis=-1)
            else:
                atom_qm_descriptor = sasa

        atom_qm_descriptor = np.concatenate([atom_qm_descriptor, np.array([[elem_list.index(a.GetSymbol())] for a in Chem.AddHs(mol).GetAtoms()])], axis=-1)

        for map_idx in fatom_index_mol:
            fatoms_qm[fatom_index[map_idx], :] = atom_qm_descriptor[fatom_index_mol[map_idx], :]
            #fatoms_qm[fatom_index[map_idx]].append(amap_idx)
            if fatom_index[map_idx] in core:
                core_mask[fatom_index[map_idx]] = 1

    reactive_core_descriptors = [0]*5
    for i in range(len(core_mask)):
        if core_mask[i] != 0:
            if mol_rs.GetAtoms()[i].GetFormalCharge() == 1:
                reactive_core_descriptors[0] = fatoms_qm[i]
            elif mol_rs.GetAtoms()[i].GetFormalCharge() == -1:
                reactive_core_descriptors[1] = fatoms_qm[i]
            elif any([neighbor.GetFormalCharge() == 1 for neighbor in mol_rs.GetAtoms()[i].GetNeighbors()]):
                reactive_core_descriptors[2] = fatoms_qm[i]
        else:
            if isinstance(reactive_core_descriptors[3], int):
                reactive_core_descriptors[3] = fatoms_qm[i]
            else:
                reactive_core_descriptors[4] = fatoms_qm[i]       

    for atom_descs in reactive_core_descriptors:
        if isinstance(atom_descs, int):
            print(rs)
            return None

    return reactive_core_descriptors


def smiles2graph_pr(qm_descriptors, r_smiles, p_smiles, selected_descriptors=["partial_charge", "fukui_elec",
                                                                            "fukui_neu", "nmr", #"spin_dens",
                                                                            "spin_dens_triplet","sasa","pint"], core_buffer=0):
    rs, rs_core, p_core = _get_reacting_core(r_smiles, p_smiles, core_buffer)
    rs_features = _mol2graph(qm_descriptors, r_smiles, selected_descriptors, core=rs_core)
    
    if rs_features != None:
        return rs_features, r_smiles
    else:
        return 'lol'


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    '''
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp('molAtomMapNumber'): a for a in r_mols.GetAtoms()}
    p_dict = {a.GetIntProp('molAtomMapNumber'): a for a in p_mol.GetAtoms()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp('molAtomMapNumber') in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        # FIXME chiral change
        # if str(p_dict[a_map].GetChiralTag()) != str(rs_dict[a_map].GetChiralTag()):
        #    core_mapnum.add(a_map)

        a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = \
        {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()}

    core_rs = [fatom_index[x] for x in core_rs]
    core_p = [fatom_index[x] for x in core_p]

    return rs_reactants, core_rs, core_p


def _get_reacting_core(rs, p, buffer):
    '''
    use molAtomMapNumber of molecules
    buffer: neighbor to be considered as reacting center
    return: atomidx of reacting core
    '''
    r_mols = Chem.MolFromSmiles(rs)
    p_mol = Chem.MolFromSmiles(p)

    rs_dict = {a.GetIntProp('molAtomMapNumber'): a for a in r_mols.GetAtoms()}
    p_dict = {a.GetIntProp('molAtomMapNumber'): a for a in p_mol.GetAtoms()}

    rs_reactants = []
    for r_smiles in rs.split('.'):
        for a in Chem.MolFromSmiles(r_smiles).GetAtoms():
            if a.GetIntProp('molAtomMapNumber') in p_dict:
                rs_reactants.append(r_smiles)
                break
    rs_reactants = '.'.join(rs_reactants)

    core_mapnum = set()
    for a_map in p_dict:
        # FIXME chiral change
        # if str(p_dict[a_map].GetChiralTag()) != str(rs_dict[a_map].GetChiralTag()):
        #    core_mapnum.add(a_map)

        a_neighbor_in_p = set([a.GetIntProp('molAtomMapNumber') for a in p_dict[a_map].GetNeighbors()])
        a_neighbor_in_rs = set([a.GetIntProp('molAtomMapNumber') for a in rs_dict[a_map].GetNeighbors()])
        if a_neighbor_in_p != a_neighbor_in_rs:
            core_mapnum.add(a_map)
        else:
            for a_neighbor in a_neighbor_in_p:
                b_in_p = p_mol.GetBondBetweenAtoms(p_dict[a_neighbor].GetIdx(), p_dict[a_map].GetIdx())
                b_in_r = r_mols.GetBondBetweenAtoms(rs_dict[a_neighbor].GetIdx(), rs_dict[a_map].GetIdx())
                if b_in_p.GetBondType() != b_in_r.GetBondType():
                    core_mapnum.add(a_map)

    core_rs = _get_buffer(r_mols, [rs_dict[a].GetIdx() for a in core_mapnum], buffer)
    core_p = _get_buffer(p_mol, [p_dict[a].GetIdx() for a in core_mapnum], buffer)

    fatom_index = \
        {a.GetIntProp('molAtomMapNumber') - 1: a.GetIdx() for a in Chem.MolFromSmiles(rs_reactants).GetAtoms()}

    core_rs = [fatom_index[x] for x in core_rs]
    core_p = [fatom_index[x] for x in core_p]

    return rs_reactants, core_rs, core_p


def _get_buffer(m, cores, buffer):
    neighbors = set(cores)

    for i in range(buffer):
        neighbors_temp = list(neighbors)
        for c in neighbors_temp:
            neighbors.update([n.GetIdx() for n in m.GetAtomWithIdx(c).GetNeighbors()])

    neighbors = [m.GetAtomWithIdx(x).GetIntProp('molAtomMapNumber') - 1 for x in neighbors]

    return neighbors


def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i, 0:n, 0:m] = arr
    return a


def pack2D_withidx(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i, 0:n, 0:m, 0] = i
        a[i, 0:n, 0:m, 1] = arr
    return a


def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i, 0:n] = arr
    return a


def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in range(arr.shape[0]):
            a[i][j] = 1
    return a


def smiles2graph_list(smiles_list, idxfunc=lambda x: x.GetIdx()):
    res = list(map(lambda x: smiles2graph(x, idxfunc), smiles_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(
        nb_list), get_mask(fatom_list)


def get_bond_edits(reactant_smi, product_smi):
    reactants = Chem.MolFromSmiles(reactant_smi)
    products = Chem.MolFromSmiles(product_smi)
    conserved_maps = [a.GetAtomMapNum() for a in reactants.GetAtoms() if a.GetAtomMapNum()]
    bond_changes = set()

    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()])
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()])
        if (nums[0] not in conserved_maps) or (nums[1] not in conserved_maps): continue
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))  # lost bond
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # changed bond
    for bond in bonds_new:
        if bond not in bonds_prev:
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))  # new bond

    return bond_changes


if __name__ == "__main__":
    # np.set_printoptions(threshold='nan')
    graph = _mol2graph("c1cccnc1")
    #print(smiles2graph_pr("[Br:1][c:5]1[c:4]([OH:3])[c:9]([F:10])[cH:8][cH:7][cH:6]1",
    #                      "[Br:1][Br:2].[OH:3][c:4]1[cH:5][cH:6][cH:7][cH:8][c:9]1[F:10]", core_buffer=0))
