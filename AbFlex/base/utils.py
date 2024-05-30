import numpy as np
from pathlib import Path
from biopandas.pdb import PandasPdb

from AbFlex.base.atom_types.atom_types import Typer


standard_aa_map = {'ALA': 0,
                   'ARG': 1,
                   'ASN': 2,
                   'ASP': 3,
                   'CYS': 4,
                   'GLU': 5,
                   'GLN': 6,
                   'GLY': 7,
                   'HIS': 8,
                   'ILE': 9,
                   'LEU': 10,
                   'LYS': 11,
                   'MET': 12,
                   'PHE': 13,
                   'PRO': 14,
                   'SER': 15,
                   'THR': 16,
                   'TRP': 17,
                   'TYR': 18,
                   'VAL': 19,
                   'XXX': 20,
                   }


extended_aa_map = {'ALA': 0,
                   'ARG': 1,
                   'ASN': 2,
                   'ASP': 3,
                   'IAS': 3,  # non standard AA
                   'CYS': 4,
                   'CSO': 4,  # non standard AA
                   'GLU': 5,
                   'GLN': 6,
                   'GLY': 7,
                   'HIS': 8,
                   'HIC': 8,  # non standard AA
                   'ILE': 9,
                   'LEU': 10,
                   'LYS': 11,
                   'MLY': 11,  # non standard AA
                   'MET': 12,
                   'MSE': 12,  # non standard AA
                   'PHE': 13,
                   'PRO': 14,
                   'SER': 15,
                   'SEP': 15,  # non standard AA
                   'THR': 16,
                   'TRP': 17,
                   'TYR': 18,
                   'TPQ': 18,  # non standard AA
                   'TYQ': 18,  # non standard AA
                   'VAL': 19,
                   'XXX': 20,
                   }


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_type_map(types: list = None):
    t = Typer()

    if types is None:
        types = [
            ['AliphaticCarbonXSHydrophobe'],
            ['AliphaticCarbonXSNonHydrophobe'],
            ['AromaticCarbonXSHydrophobe'],
            ['AromaticCarbonXSNonHydrophobe'],
            ['Nitrogen', 'NitrogenXSAcceptor'],
            ['NitrogenXSDonor', 'NitrogenXSDonorAcceptor'],
            ['Oxygen', 'OxygenXSAcceptor'],
            ['OxygenXSDonor', 'OxygenXSDonorAcceptor'],
            ['Sulfur', 'SulfurAcceptor'],
            ['Phosphorus']
        ]
    out_dict = {}
    generic = []
    for i, element_name in enumerate(t.atom_types):
        for types_list in types:
            if element_name in types_list:
                out_dict[i] = types.index(types_list)
                break
        if i not in out_dict.keys():
            generic.append(i)

    generic_type = len(types)
    for other_type in generic:
        out_dict[other_type] = generic_type
    return out_dict


def parse_pdb_to_parquet(pdb_file: Path,
                         parquet_path: Path,
                         lmg_typed: bool = True,
                         ca: bool = False):
    """Parses a pdb file to smaller, faster parquet df format.
    Returns the resulting df.

    Args:
        pdb_file (Path): Path to the pdb file
        parquet_path (Path): Output filename
        lmg_typed (bool, optional): Use typer functionality to generate lmg
            types for each atom. Defaults to True.

    Returns:
        pd.DataFrame: The pdb df.
    """
    pdb_df = PandasPdb().read_pdb(str(pdb_file)).df["ATOM"]

    # remove individually resolved hydrogens
    bool_sel = pdb_df['atom_name'].apply(lambda x: x.strip() not in ['H'])
    pdb_df = pdb_df[bool_sel]

    # store lmg typings - types and occupancies
    if lmg_typed:
        typer = Typer()
        types, occupancies = typer.run(pdb_file)
        pdb_df['lmg_types'] = types
        pdb_df['occ'] = occupancies

        # drop columns that are not needed downstream
        pdb_df = pdb_df[['atom_number',
                         'atom_name',
                         'chain_id',
                         'residue_number',
                         'insertion',
                         'x_coord',
                         'y_coord',
                         'z_coord',
                         'occ',
                         'lmg_types',
                         'residue_name']]
    else:
        pdb_df = pdb_df[['atom_number',
                         'atom_name',
                         'chain_id',
                         'residue_number',
                         'insertion',
                         'x_coord',
                         'y_coord',
                         'z_coord',
                         'residue_name']]

    if ca:
        pdb_df = pdb_df[pdb_df.atom_name == 'CA']
    pdb_df.to_parquet(parquet_path)
    return pdb_df
