import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

SYNTHON_LABELS = ['U','Np','Pu','Am']


def add_charge(mol, charge_tertiary=True, charge_imidazole=True):
    # Define SMARTS queries for different nitrogen-containing substructures
    Chem.SanitizeMol(mol)
    substructure_queries = {
        "amine": '[$([NX3]);!$(N~[!C]);!$(NC~[!#6]);!$(N[#6][#6]([F,Cl,Br])[F,Cl,Br]);!$(N(C=*)@C=*);!$(N[#6]=[#6][#6]=[*])]',

        "imidazole": "[$([nX2]);$([nr5]:[cr5]:[nr5]:[cr5]:[cr5])]",

        "guanidine": "[$([NX2]);$([#7]=[#6]([#7])[#7])]",

        "amidine": "[$([NX2]);$([#7]=[#6]([#7])[C])]",

        "tetrazole": "[$([nX3;H1]);$([nr5]:[nr5]:[cr5]:[nr5]:[nr5]),$([nr5]:[cr5]:[nr5]:[nr5]:[nr5])]"
    }

    # Loop through each subgroup and modify the charges for matching substructures
    for subgroup, smarts in substructure_queries.items():
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)
        
        # Apply charge to nitrogen atoms in the matching substructures
        for match in matches:
            for atom_idx in match:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 0:  # Check if it's a nitrogen atom with no charge
                    if subgroup == "tetrazole":
                        atom.SetFormalCharge(-1)  # Set negative charge for tetrazole
                        atom.SetNumExplicitHs(0)  # Remove explicit hydrogen to maintain valence
                    else:
                        atom.SetFormalCharge(+1)  # Set positive charge for other subgroups

    return mol 

def add_mols(current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column to the DataFrame with RDKit molecule objects generated from the SMILES column
    """
    synthon_df = current_df.copy()
    synthon_df['mol'] = synthon_df['SMILES'].apply(Chem.MolFromSmiles)
    return synthon_df

def calculate_and_filter_on_parameters(mol: Chem.Mol, filter_config: dict) -> dict | None:
    """
    Calculates key properties, and filters based on the provided filter configuration.
    Works with filter_config where ranges are specified as lists [min, max].
    Args:
        mol (Chem.Mol): RDKit molecule object.
        filter_config (dict): Dictionary with property filter ranges, e.g., {"MW": [100, 500], ...}

    Returns:
        dict or None: Dictionary of properties (stringified) if passes filters, otherwise None.
    """

    # Calculate molecular properties
    properties = {        
        'HAC': mol.GetNumHeavyAtoms(),
        'MW': round(Descriptors.MolWt(mol), 2),
        'HBA': round(Descriptors.NumHAcceptors(mol), 2),
        'HBD': round(Descriptors.NumHDonors(mol), 2),
        'logP': round(Descriptors.MolLogP(mol), 2),
        'RotB': round(Descriptors.NumRotatableBonds(mol), 2),
    }

    # Apply filtering
    for prop_name, (min_val, max_val) in filter_config.items():
        if prop_name not in properties:
            continue  # Skip unknown properties
        if not (min_val <= properties[prop_name] <= max_val):
            return None

    # Return stringified properties
    properties_str = {k: str(v) for k, v in properties.items()}
    return properties_str


def calculate_parameters(mol: Chem.Mol) -> dict | None:
    """
    Calculates key properties
    Args:
        mol (Chem.Mol): RDKit molecule object.
    Returns:
        dict or None: Dictionary of properties (stringified) if passes filters, otherwise None.
    """
    # Calculate molecular properties
    properties = {        
        'HAC': mol.GetNumHeavyAtoms(),
        'MW': round(Descriptors.MolWt(mol), 2),
        'HBA': round(Descriptors.NumHAcceptors(mol), 2),
        'HBD': round(Descriptors.NumHDonors(mol), 2),
        'logP': round(Descriptors.MolLogP(mol), 2),
        'RotB': round(Descriptors.NumRotatableBonds(mol), 2),
    }
    # Return stringified properties
    properties_str = {k: str(v) for k, v in properties.items()}
    return properties_str

def sanitize_calculate_and_filter_on_parameters(mol: Chem.Mol, filter_config: dict) -> dict | None:
    """
    Sanitizes the molecule, calculates key properties, and filters based on the provided filter configuration.
    Works with filter_config where ranges are specified as lists [min, max].
    Args:
        mol (Chem.Mol): RDKit molecule object.
        filter_config (dict): Dictionary with property filter ranges, e.g., {"MW": [100, 500], ...}

    Returns:
        dict or None: Dictionary of properties (stringified) if passes filters, otherwise None.
    """
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return None  # Molecule invalid

    # Calculate molecular properties
    properties = {        
        'HAC': mol.GetNumHeavyAtoms(),
        'MW': round(Descriptors.MolWt(mol), 2),
        'HBA': round(Descriptors.NumHAcceptors(mol), 2),
        'HBD': round(Descriptors.NumHDonors(mol), 2),
        'logP': round(Descriptors.MolLogP(mol), 2),
        'RotB': round(Descriptors.NumRotatableBonds(mol), 2),
    }

    # Apply filtering
    for prop_name, (min_val, max_val) in filter_config.items():
        if prop_name not in properties:
            continue  # Skip unknown properties
        if not (min_val <= properties[prop_name] <= max_val):
            return None

    # Return stringified properties
    properties_str = {k: str(v) for k, v in properties.items()}
    return properties_str

def sanitize_calculate_and_filter_on_parameters_old(mol: Chem.Mol, filter_config: dict) -> tuple:
    """
    calculates key properties, and filters based on the provided filter configuration.
    If the molecule passes the filter, it will return the calculated properties; otherwise, it returns None.
    """
    Chem.SanitizeMol(mol)

    mw = round(Descriptors.MolWt(mol), 2)
    HAC_value = round(mol.GetNumHeavyAtoms())
    hba = round(Descriptors.NumHAcceptors(mol), 2)
    hbd = round(Descriptors.NumHDonors(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    rotb = round(Descriptors.NumRotatableBonds(mol), 2)

    # Apply filters based on the filter_config ranges
    if not (filter_config["MW_filter"][0] <= mw <= filter_config["MW_filter"][1]) or \
       not (filter_config["HBA_filter"][0] <= hba <= filter_config["HBA_filter"][1]) or \
       not (filter_config["HBD_filter"][0] <= hbd <= filter_config["HBD_filter"][1]) or \
       not (filter_config["logP_filter"][0] <= logp <= filter_config["logP_filter"][1]) or \
       not (filter_config["RotB_filter"][0] <= rotb <= filter_config["RotB_filter"][1]):
        return None

    properties_str = {
        'HAC': str(HAC_value),
        'MW': str(mw),
        'HBA': str(hba),
        'HBD': str(hbd),
        'logP': str(logp),
        'RotB': str(rotb),
    }

    return properties_str

def sanitize_calculate_and_filter_on_parameters_old(mol: Chem.Mol, filter_config: dict) -> tuple:
    """
    calculates key properties, and filters based on the provided filter configuration.
    If the molecule passes the filter, it will return the calculated properties; otherwise, it returns None.
    Returns:
        tuple: A tuple containing:
            - A dictionary with the calculated properties of the molecule (in string format).
            - An integer representing the heavy atom count (HAC).
            - A boolean indicating whether the molecule was processed (True if processed, False if filtered out).
            
    """
    Chem.SanitizeMol(mol)

    mw = round(Descriptors.MolWt(mol), 2)
    HAC_value = round(mol.GetNumHeavyAtoms())
    hba = round(Descriptors.NumHAcceptors(mol), 2)
    hbd = round(Descriptors.NumHDonors(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    rotb = round(Descriptors.NumRotatableBonds(mol), 2)

    # Apply filters based on the filter_config ranges
    if not (filter_config["MW_filter"][0] <= mw <= filter_config["MW_filter"][1]) or \
       not (filter_config["HBA_filter"][0] <= hba <= filter_config["HBA_filter"][1]) or \
       not (filter_config["HBD_filter"][0] <= hbd <= filter_config["HBD_filter"][1]) or \
       not (filter_config["logP_filter"][0] <= logp <= filter_config["logP_filter"][1]) or \
       not (filter_config["RotB_filter"][0] <= rotb <= filter_config["RotB_filter"][1]):
        return None

    properties_str = {
        'HAC': str(HAC_value),
        'MW': str(mw),
        'HBA': str(hba),
        'HBD': str(hbd),
        'logP': str(logp),
        'RotB': str(rotb),
    }

    return properties_str

def process_external_minimal_synthons(external_synthons: pd.DataFrame) -> dict[int:pd.DataFrame]:
    """
    Processes DataFrame of external minimal synthons (created manually or from the previous runs)
    and generates a dictionary of minimal synthons indexed by synthon positions.

    Args:
        external_synthons (pd.DataFrame): A DataFrame containing external synthons with columns 'synton#' and 'SMILES'.
                                          'synton#' corresponds to the synthon position, and 'SMILES' contains the SMILES strings.

    Returns:
        dict[int, pd.DataFrame]: A dictionary where the keys are synthon positions (int), and the values are DataFrames
                                 containing the processed synthons at each position, with added 'mol' column for RDKit molecules.
    """

    minimal_synthons = {}

    # Collecting the external minimal synthons into dictionary to use in the enumeration
    for synthon_posisiton in external_synthons['synton#'].unique():
        synthon_at_position = external_synthons[external_synthons['synton#'] == synthon_posisiton].copy()
        synthon_at_position['mol'] = synthon_at_position.SMILES.apply(Chem.MolFromSmiles)
        minimal_synthons[int(synthon_posisiton)] = synthon_at_position

    return minimal_synthons

def get_minimal_synthon(synthon_table: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the synthon with the lowest molecular weight and applies isotope labeling.

    This function calculates the exact molecular weight for each molecule in the input DataFrame,
    selects the smallest synthon (in case of ties, the first occurrence is chosen), and applies 
    isotope labeling to the selected molecule.

    Args:
        synthon_table (pd.DataFrame): A DataFrame with synthons containing a column 'mol' with RDKit Mol objects.

    Returns:
        pd.DataFrame: A DataFrame containing the smallest synthon with isotope labeling applied.
    """

    # calculating molecular weight for all input synthons
    synthon_table['mw'] = synthon_table['mol'].apply(Descriptors.ExactMolWt)

    # Select the synthon(s) with the minimum molecular weight (picking the first in case of ties)
    smallest_synthon = synthon_table[synthon_table.mw == synthon_table.mw.min()].iloc[[0]]

    # Apply isotope labeling to the selected synthon
    smallest_synthon_isotope = change_mol_to_isotope_labelled(smallest_synthon, SYNTHON_LABELS)

    return smallest_synthon_isotope

def change_mol_to_isotope_labelled(synthon_df: pd.DataFrame, 
                                   label_list: list[str]) -> pd.DataFrame:
    """
    Modifies a molecular structure by adding isotope labels to atoms that are not in the specified label list.

    This function retrieves a molecule from the first row of the input DataFrame, 
    clones it to avoid unintended modifications, and updates the isotope labels 
    for atoms that do not belong to the provided label list.

    Args:
        synthon_df (pd.DataFrame): A 1-row DataFrame that contains the potential minimal synthon for labelling.
                                    Must contain 'mol' column with RDKit Mol.
        label_list (list[str]): A list of atoms that are synthon labels and should not be relabelled.

    Returns:
        pd.DataFrame: The updated DataFrame with the isotope-labeled molecule.
    """

    # getting the mol from the first row of df, dataframe contains 1 row total
    mol = synthon_df.iloc[0]['mol']

    # making a copy for editing so that it does not alter all dataframes
    mol = Chem.Mol(mol)

    # Iterate over atoms and apply isotope labeling to those not in label_list
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in label_list:
            atomic_num = atom.GetMass()

            isotope = int(atomic_num) + 1
            atom.SetIsotope(isotope)
    synthon_df.loc[synthon_df.index[0], 'mol'] = mol

    return synthon_df

def check_if_bridge_reaction(minimal_synthons: dict,
                             label_list: list[str]) -> bool:
    """
    Determines whether a reaction follows a 'bridge' reaction pattern based on the number of components
    and atom label counts in each synthon.

    The function analyzes the label distribution in synthons and classifies reactions based on the following patterns:
        - Bridge reaction: [1,1,2]
        - Three-component reactions: [1,2,3] or [2,2,2]
        - Two-component reactions: [1,1] or [2,2]

    Args:
        minimal_synthons (dict): A dictionary where keys indicate synthon position (1,2,3) and
                                values contain DataFrames with a column 'mol' with RDKit Mol objects.
        label_list (list[str]): A list of atoms that are synthon labels.

    Returns:
        tuple: (bool, int or bool)
            - If a bridge reaction is detected, returns (True, bridge_position).
            - Otherwise, returns (False, False).
    """
    # 3 labels per molecule means 3 component reaction
    label_counts = []

    # Iterate through synthons and count occurrences of labeled atoms
    for synthon_pos, synthon_table in minimal_synthons.items():
        mol = synthon_table.iloc[0]['mol']

        label_counter = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in label_list:
                label_counter += 1

        label_counts.append(label_counter)

    # Classify the reaction based on label distribution
    if max(label_counts) == 3 or all(item == 2 for item in label_counts):  # 3 component
        return False, False
   
    else:  # bridge has [1,1,2] pattern
        bridge_position = label_counts.index(2)  # to know which synthon is the bridge
        return True, bridge_position + 1

def check_if_bridge_reaction_external(minimal_synthons: dict, 
                                      label_list: list[str]) -> tuple:
    """
    Determines whether a reaction follows a 'bridge' reaction pattern based on the atom label counts
    in each synthon.

    Args:
        minimal_synthons (dict): A dictionary where keys indicate synthon position (1, 2, 3) and
                                  values contain DataFrames with a column 'mol' with RDKit Mol objects.
        label_list (list[str]): A list of atoms that are synthon labels.

    Returns:
        tuple: (bool, int or bool)
            - If a bridge reaction is detected, returns (True, bridge_position).
            - Otherwise, returns (False, False).
    """

    label_counts = {1: 0, 2: 0, 3: 0}  # Initialize dictionary with 3 keys, 1, 2, 3

    # Iterate through synthons and count occurrences of labeled atoms
    for synthon_pos, synthon_table in minimal_synthons.items():
        mol = synthon_table.iloc[0]['mol']

        label_counter = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in label_list:
                label_counter += 1

        label_counts[synthon_pos] = label_counter  # Store the label count in the dictionary

       

    # Check for 3-component reaction
    if all(count > 0 for count in label_counts.values()):
        return False, False  # No missing synthon, it's a 3-component reaction
    
    for key, count in label_counts.items():
        if count == 0 :
            return True, key
