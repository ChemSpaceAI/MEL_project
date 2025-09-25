import os, re, json
import pandas as pd
from rdkit import Chem


def clean_and_copy_csvs_with_v_inside(base_path, output_folder, cols_to_drop=None, print_flag=False):
    """
    Cleans CSV files inside a directory tree by dropping specified columns 
    and copying them to a new output folder.

    Parameters:
        base_path (str): Root directory containing subfolders with CSV files.
        output_folder (str): Destination folder for cleaned CSV files.
        cols_to_drop (list, optional): Columns to remove if they exist in the file.
    """
    os.makedirs(output_folder, exist_ok=True)
    cols_to_drop = cols_to_drop or []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.startswith('v') and filename.endswith('.csv'):
                    file_path = os.path.join(folder_path, filename)

                    df = pd.read_csv(file_path)

                    # Drop only existing columns from cols_to_drop
                    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

                    output_path = os.path.join(output_folder, filename)
                    df.to_csv(output_path, index=False)

                    if print_flag:
                        print(f"Processed: {filename}")


def combine_cleaned_csvs(folder_path, print_flag=False):
    """
    Combines CSV files from a given folder into a single DataFrame,
    extracting reaction_id_type and synton# from filenames.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing cleaned CSV files.

    Returns
    -------
    pandas.DataFrame
    """
    combined_df = pd.DataFrame()

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv') and filename.startswith('v'):
            file_path = os.path.join(folder_path, filename)
            
            if print_flag:
                print(f"Processing file: {filename}")

            # Read the file
            df = pd.read_csv(file_path)
            
            # Extract reaction_id_type and synton# using regex
            name_parts = filename.replace('.csv', '').split('_')
            if len(name_parts) == 2 and name_parts[0].startswith('v'):
                reaction_id_type = name_parts[0][1:]  # drop the leading 'v'
                synton_num = int(name_parts[1])
            else:
                reaction_id_type = None
                synton_num = None

            df['reaction_id_type'] = reaction_id_type
            df['synton#'] = synton_num


            # Append to combined DataFrame
            combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True, sort=False)

    return combined_df

def neutralize_radicals(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES input.")

    for atom in mol.GetAtoms():
        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    mol.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)

def neutralize_synthon_radicals(smiles: str) -> str:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES input.")

    skip_atoms = {'U', 'Am', 'Np', 'Pu'}

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in skip_atoms:
            continue  # Skip actinides

        num_radicals = atom.GetNumRadicalElectrons()
        if num_radicals:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radicals)
            atom.SetNumRadicalElectrons(0)

    mol.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)


def extract_reaction_ids(df, output_file='ids.txt', print_flag=False):
    """
    Extracts reaction IDs from a DataFrame and saves them as a tab-separated file.
    """

    reaction_id_map = {}
    all_ids = []

    # Build dictionary mapping reaction types to unique IDs
    for _, row in df.iterrows():
        rxn_ids = row['rxn_list'].split(';')
        key = row['reaction_id_type']

        if key not in reaction_id_map:
            reaction_id_map[key] = set()

        reaction_id_map[key].update(rxn_ids)

    reaction_id_map = {k: sorted(list(v)) for k, v in reaction_id_map.items()}
    
    if print_flag:
        for ids in reaction_id_map.values():
            print(ids)

    for val in reaction_id_map.values():
        if isinstance(val, list):
            all_ids.extend(val)
        else:
            all_ids.append(val)

    # Create DataFrame and save
    reaction_id_df = pd.DataFrame({'reaction_id': all_ids})
    reaction_id_df.sort_values(by='reaction_id', inplace=True)
    reaction_id_df.to_csv(output_file, index=False, sep='\t')

    print(f"✅ Saved {len(reaction_id_df)} reaction IDs to {output_file}")
    return reaction_id_map



def make_new_folder_in_the_same_directory(input_dir: str, name: str) -> str:
    """Creates a new folder in the same directory as the input directory.
    Args:
        input_dir (str): The path of the input directory.
        name (str): The name of the new folder to be created.
    Returns:
        str: The path of the newly created folder.
    """

    parent_dir = os.path.dirname(input_dir)
    new_path = os.path.join(parent_dir, name)
    os.makedirs(new_path, exist_ok=True)
    return new_path




def export_mel_reaction_caps(un_caps_df, reaction_id_map, output_folder, mel_type="3_component"):
    """
    Export reaction-specific SMILES caps files for MEL generation.

    Parameters
    ----------
    un_caps_df : pd.DataFrame
        DataFrame containing reaction data with columns:
        'SMILES', 'rxn_list', 'synton#', 'reaction_id_type'.
    reaction_id_map : dict
        Mapping of reaction_id_type to a list of reaction_ids.
    output_folder : str
        Path where output text files will be saved.
    mel_type : str, optional
        MEL type label to add to output files. Default is "3_component".

    Returns
    -------
    int
        Number of files created.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_count = 0

    for reaction_id_type, reaction_list in reaction_id_map.items():
        type_sub_df = un_caps_df[un_caps_df['reaction_id_type'] == reaction_id_type]

        for reaction_id in reaction_list:
            file_path = os.path.join(output_folder, f'{reaction_id}.txt')

            filtered_df = type_sub_df[type_sub_df['rxn_list'].str.contains(reaction_id)].copy()

            filtered_df['mel_type'] = mel_type
            filtered_df['reaction_id'] = reaction_id
            filtered_df['synton_id'] = filtered_df['synton#'].apply(lambda x: f's{x}')

            filtered_df.drop(columns=['rxn_list'], inplace=True, errors='ignore')

            filtered_df = filtered_df[
                ['SMILES', 'synton_id', 'synton#', 'reaction_id_type', 'mel_type', 'reaction_id']
            ]

            filtered_df.to_csv(file_path, index=False, sep='\t')
            file_count += 1

    return file_count


def count_unique_reaction_ids_in_df(un_caps_df, rxn_list_col="rxn_list"):
    """
    Count unique reaction IDs across the entire DataFrame by splitting rxn_list values.

    Parameters
    ----------
    un_caps_df : pd.DataFrame
        DataFrame containing a column with reaction lists separated by ';'.
    rxn_list_col : str, optional
        Name of the column containing reaction lists. Default is 'rxn_list'.

    Returns
    -------
    int
        Number of unique reaction IDs.
    set
        Set of unique reaction IDs.
    """
    all_ids = set()

    for value in un_caps_df[rxn_list_col].dropna():
        ids = map(str.strip, value.split(';'))
        all_ids.update(ids)

    return len(all_ids), all_ids

def filter_and_save_reaction_file(
    reaction_id_map, 
    input_file, 
    output_file, 
    mel_type=None, 
    bridge_json_path=None
):
    """
    Filter the REACTION_file_for_MEL.tsv based on reaction_id_map keys 
    and save the filtered file with additional columns. Optionally assign 
    bridge_position from a JSON file if provided.

    Parameters
    ----------
    reaction_id_map : dict
        Dictionary mapping reaction_id_type to lists of reaction_ids.
    input_file : str
        Path to the REACTION_file_for_MEL.tsv file.
    output_file : str
        Path where the filtered file will be saved.
    mel_type : str, optional
        MEL type to assign. Default is None.
    bridge_json_path : str, optional
        Path to JSON file containing bridge_position info per reaction_id.
        If None, bridge_position will be set to None for all rows.
    """
    # Load bridge position data if provided
    bridge_data = {}
    if bridge_json_path:
        with open(bridge_json_path, "r") as f:
            bridge_data = json.load(f)


    reaction_ids = reaction_id_map.keys()

    # Read the reaction file
    reaction_df = pd.read_csv(input_file, sep='\t')

    # Filter rows by reaction_id_type
    filtered_df = reaction_df[reaction_df['reaction_id_type'].isin(reaction_ids)].copy()

    filtered_df['mel_type'] = mel_type

    if bridge_data:
        filtered_df['bridge_position'] = filtered_df['reaction_id'].apply(lambda rid: bridge_data.get(rid, {}).get('bridge_position', None))
    
    filtered_df.to_csv(output_file, sep='\t', index=False)

    print(f"✅ Saved filtered reaction file with {len(filtered_df)} rows to {output_file}")
    return filtered_df
