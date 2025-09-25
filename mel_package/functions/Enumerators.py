import os
import sys
import logging
from typing import Dict, List
import pandas as pd

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Suppress RDKit warnings

# Import custom functions from your project
from mel_package.functions.Chemistry_logic import (
    add_mols,
    add_charge,
    calculate_and_filter_on_parameters,
    calculate_parameters
)


def enumerator_engine_for_3_component(
    synthons_1: pd.DataFrame, 
    synthons_2: pd.DataFrame, 
    synthons_3: pd.DataFrame, 
    rxn_mol,
    reaction_id: str, 
    CSM_id: str,
    filters_config: dict, 
    logger: logging.Logger,
    cap_position: int 
    ) -> Dict[str, List]:


    """
    Enumerates products from a 3-component reaction given three sets of synthons.
    Handles possible cap replacement at a specific position.

    Returns a dictionary where keys are SMILES strings and values are lists containing:
    [result_synthon_id, charged_smiles, synthon_1_SMILES, synthon_2_SMILES, synthon_3_SMILES, molecular_properties_list]
    """

    result_dict = {}

    # Prepare molecules
    synthons_1 = add_mols(synthons_1)
    synthons_2 = add_mols(synthons_2)
    synthons_3 = add_mols(synthons_3)
    
    for i, row1 in synthons_1.iterrows():
        for j, row2 in synthons_2.iterrows():
            for k, row3 in synthons_3.iterrows():
                try:
                    # Run the reaction
                    product_tuples = rxn_mol.RunReactants((row1["mol"], row2["mol"], row3["mol"]))
                    product = product_tuples[0][0]

                    product_copy = Chem.Mol(product)
                    Chem.SanitizeMol(product_copy)
                    charged_product = add_charge(product_copy)

                    charged_smile = Chem.MolToSmiles(charged_product)
                    result_smile = Chem.MolToSmiles(product)

                    # Filter based on molecular properties
                    if filters_config:
                        mol_properties = calculate_and_filter_on_parameters(charged_product, filters_config)
                        if mol_properties is None:
                            continue  # Molecule didn't pass the filters
                    else:
                        mol_properties = calculate_parameters(charged_product)
                    # Initialize synthon_id parts
                    synthon_parts = [row1["synton_id"], row2["synton_id"], row3["synton_id"]]

                    # Replace the correct synthon ID with 's1', 's2', or 's3' if a cap is specified
                    if cap_position in {1, 2, 3}:
                        synthon_parts[cap_position - 1] = f's{cap_position}'


                    # Format the synthon_id string
                    result_synthon_id = f'{reaction_id}-{"_".join(map(str, synthon_parts))}'

                    # Create result entry
                    result_dict[result_smile] = [
                        result_synthon_id,
                        CSM_id,
                        charged_smile,
                        row1['SMILES'],
                        row2['SMILES'],
                        row3['SMILES'],
                        *mol_properties.values()
                    ]

                except Exception as e:
                    logger.info(f"Error processing: {row1['SMILES']} + {row2['SMILES']} + {row3['SMILES']}. Reaction_ID {reaction_id}. Cap pos {cap_position} Exception: {str(e)}")
    
    return result_dict


def enumerator_engine_for_2_component(
    synthons_1: pd.DataFrame, 
    synthons_2: pd.DataFrame, 
    rxn_mol,
    reaction_id: str, 
    CSM_id: str,
    filters_config: dict, 
    logger: logging.Logger
) -> Dict[str, List]:
    """
    Enumerates products from a 2-component reaction given two sets of synthons.
    Returns a dictionary where keys are SMILES strings and values are lists 

    Args:
        synthons_1 (pd.DataFrame): DataFrame containing the first set of synthons.
        synthons_2 (pd.DataFrame): DataFrame containing the second set of synthons.
        rxn_mol (Chem.Reaction): Reaction object for the 2-component reaction.
        reaction_id (str): Unique identifier for the reaction.
        filters_config (dict): Configuration for molecular property filtering.
        logger (logging.Logger): Logger object for logging information and errors.
        
    Returns:
        Dict[str, List]: Dictionary where keys are result SMILES, and values are lists containing:
                         - result_synthon_id
                         - charged SMILES
                         - synthon 1 SMILES
                         - synthon 2 SMILES
                         - '' for synthon 3 
                         - molecular property values
    """
    result_dict = {}

    # Add molecules to the synthons (assumed to be defined in add_mols function)
    synthons_1 = add_mols(synthons_1)
    synthons_2 = add_mols(synthons_2)

    # Enumerate over all combinations of synthons
    for i, row1 in synthons_1.iterrows():
        for j, row2 in synthons_2.iterrows():
            try:
                # Run reaction on the pair of synthons
                product_tuples = rxn_mol.RunReactants([row1["mol"], row2["mol"]])
                product = product_tuples[0][0]

                # Sanitize and charge the product
                product_copy = Chem.Mol(product)
                Chem.SanitizeMol(product_copy)
                charged_product = add_charge(product_copy)

                charged_smile = Chem.MolToSmiles(charged_product)
                result_smile = Chem.MolToSmiles(product)

                # Filter based on molecular properties
                if filters_config:
                    mol_properties = calculate_and_filter_on_parameters(charged_product, filters_config)
                    if mol_properties is None:
                        continue  # Molecule didn't pass the filters
                else:
                    mol_properties = calculate_parameters(charged_product)

                # Create a unique result synthon ID
                result_synthon_id = f'{reaction_id}-{row1["synton_id"]}_{row2["synton_id"]}'
                
                # Store the result
                result_dict[result_smile] = [
                    result_synthon_id,
                    CSM_id,
                    charged_smile,
                    row1['SMILES'],
                    row2['SMILES'],
                    '',
                    *mol_properties.values()  # Convert properties to list
                ]

            except Exception as e:
                # Log errors for debugging purposes
                logger.error(f"Error processing pair: {row1['SMILES']} + {row2['SMILES']}. Reaction_ID {reaction_id}. Exception: {str(e)}")
                continue  # Safely continue on errors
    
    return result_dict


def enumerator_2_component(synthons_1, synthons_2, rxn_mol, synthon_position):
    """
    Performs enumeration of two-component reactions by generating the product of reaction.
    Enumeration is performed for one of the synthon dataframes (defined by synthon_position), 
    the other are passed to the function as minimal caps.

    Args:
        synthons_1 (pd.DataFrame): DataFrame containing the information about synthon 1.
        synthons_2 (pd.DataFrame): DataFrame containing the information about synthon 2.
        rxn_mol (rdkit.Chem.rdChemReactions.ChemicalReaction): An RDKit reaction object for transformation.
        synthon_position (int): Indicates for which synthon the enumeration is going to be performed.
        All other synthons will be passed into the function in the form of minimal caps.

    Returns:
        pd.DataFrame: A DataFrame containing the generated molecular structures with assigned IDs.
    """

    mel_dataframe = pd.DataFrame()

    # Perform combinatorial reaction enumeration
    for i, row1 in synthons_1.iterrows():
        for j, row2 in synthons_2.iterrows():
            product_tuples = rxn_mol.RunReactants([row1["mol"], row2["mol"]])
            product = product_tuples[0][0]
            
            # checking which one to write
            row_to_write = row1 if synthon_position == 1 else row2

            row_to_write['MEL_smiles'] = Chem.MolToSmiles(product)
            mel_dataframe = pd.concat([mel_dataframe, row_to_write.to_frame().T], ignore_index=True)

    # Assign unique IDs for each enumerated molecule based on reaction and synthon information
    reaction_id = synthons_1.iloc[0]['reaction_id']

    if synthon_position == 1:
        mel_dataframe['mel_synthon_id'] = mel_dataframe.synton_id.map(lambda x: f"{reaction_id}-{x}_s2_0")
    else:
        mel_dataframe['mel_synthon_id'] = mel_dataframe.synton_id.map(lambda x: f"{reaction_id}-s1_{x}_0")

    return mel_dataframe


def enumerator_3_component(synthons_1, synthons_2, synthons_3, rxn_mol, synthon_position):
    """
    Performs enumeration of three-component and bridge reactions by generating the product of reaction.
    Enumeration is performed for one of the synthon dataframes (defined by synthon_position), 
    the other are passed to the function as minimal caps.

    Args:
        synthons_1 (pd.DataFrame): DataFrame containing the information about synthon 1.
        synthons_2 (pd.DataFrame): DataFrame containing the information about synthon 2.
        synthons_3 (pd.DataFrame): DataFrame containing the information about synthon 3.
        rxn_mol (rdkit.Chem.rdChemReactions.ChemicalReaction): An RDKit reaction object for transformation.
        synthon_position (int): Indicates for which synthon the enumeration is going to be performed.
        All other synthons will be passed into the function in the form of minimal caps.

    Returns:
        pd.DataFrame: A DataFrame containing the generated molecular structures with assigned IDs.
    """

    mel_dataframe = pd.DataFrame()

    # Perform combinatorial reaction enumeration
    for i, row1 in synthons_1.iterrows():
        for j, row2 in synthons_2.iterrows():
            for k, row3 in synthons_3.iterrows():
                product_tuples = rxn_mol.RunReactants((row1["mol"], row2["mol"], row3["mol"]))
                product = product_tuples[0][0]

                # checking which one is the mel
                if synthon_position == 1:
                    row_to_write = row1
                elif synthon_position == 2:
                    row_to_write = row2
                else:
                    row_to_write = row3

                row_to_write['MEL_smiles'] = Chem.MolToSmiles(product)
                mel_dataframe = pd.concat([mel_dataframe, row_to_write.to_frame().T], ignore_index=True)

    # generating id for the full dataframe
    reaction_id = synthons_1.iloc[0]['reaction_id']

    if synthon_position == 1:
        mel_dataframe['mel_synthon_id'] = mel_dataframe.synton_id.map(lambda x: f"{reaction_id}-{x}_s2_s3")
    elif synthon_position == 2:
        mel_dataframe['mel_synthon_id'] = mel_dataframe.synton_id.map(lambda x: f"{reaction_id}-s1_{x}_s3")
    else:
        mel_dataframe['mel_synthon_id'] = mel_dataframe.synton_id.map(lambda x: f"{reaction_id}-s1_s2_{x}")

    return mel_dataframe


def enumerate_bridge_with_replaced_SMILES(reaction_id, synthons_table, minimal_synthons_table, bridge_position):
    """
    Performs enumeratio bridge reactions by generating the product of reaction.
   
    Args:
        
    Returns:
        pd.DataFrame: A DataFrame containing the generated molecular structures with assigned IDs.
    """
    try:
        mel_dataframe = pd.DataFrame()

        ultimate_reaction_smarts = '[U]-[*:1].[U]-[*:2]>>[*:1]-[*:2]'
        rxn_mol = AllChem.ReactionFromSmarts(ultimate_reaction_smarts)
            
        synthons_table = add_modified_mols(synthons_table)

        synthons_1 =synthons_table.loc[synthons_table['synton#'] == 1]
        synthons_2 =synthons_table.loc[synthons_table['synton#'] == 2]
        synthons_3 =synthons_table.loc[synthons_table['synton#'] == 3]

        minimal_synthons_table = add_modified_mols(minimal_synthons_table)

        min_s1 = minimal_synthons_table.loc[minimal_synthons_table['synton#'] == 1]
        min_s2 = minimal_synthons_table.loc[minimal_synthons_table['synton#'] == 2]
        min_s3 = minimal_synthons_table.loc[minimal_synthons_table['synton#'] == 3]
    
    except Exception as e:
        raise ValueError(f"Error during bridge information: {str(e)}")


    rows_to_add = []
    
    if bridge_position == 1:

        # bridge_s2_s3 => 'min_cap + synton_ID_2 + s3'  and  'min_cap + s2 + synton_ID_3 ==> 's1_9999_s3' and s1_s2_9999'
        # bridge synthons is broken on min s1 and min s2 

        # STEP 1. SubEnumration of minimalcap s2 with synthon s2
        try: 
            current_minimal_cap = min_s2.iloc[0]
        except Exception as e:
            raise ValueError(f"Error accessing min_s2 for bridge position 1")
        sub_enumeration(
            current_syntons=synthons_2,
            current_minimal_cap=current_minimal_cap,
            synthon_position=2,
            reaction_id=reaction_id,
            rows_to_add=rows_to_add,
            rxn_mol=rxn_mol
        )
        # STEP 2. SubEnumration of minimalcap s3 with synthon s3
        try:
            current_minimal_cap = min_s3.iloc[0]
        except Exception as e:
            raise ValueError(f"Error accessing min_s3 for bridge position 1")

        sub_enumeration(
            current_syntons=synthons_3,
            current_minimal_cap=current_minimal_cap,
            synthon_position=3,
            reaction_id=reaction_id,
            rows_to_add=rows_to_add,
            rxn_mol=rxn_mol
        )

    elif bridge_position == 2:

        # STEP 1. SubEnumration of minimalcap s1 with synthon s1
        try:
            current_minimal_cap = min_s1.iloc[0]
        except Exception as e:
            raise ValueError(f"Error accessing min_s1 for bridge position 2")

        sub_enumeration(
            current_syntons=synthons_1,
            current_minimal_cap=current_minimal_cap,
            synthon_position=1,
            reaction_id=reaction_id,
            rows_to_add=rows_to_add,
            rxn_mol=rxn_mol
        )
        # STEP 2. SubEnumration of minimalcap s3 with synthon s3
        try:
            current_minimal_cap = min_s3.iloc[0]
        except Exception as e:
            raise ValueError(f"Error accessing min_s3 for bridge position 2")

        sub_enumeration(
            current_syntons=synthons_3,
            current_minimal_cap=current_minimal_cap,
            synthon_position=3,
            reaction_id=reaction_id,
            rows_to_add=rows_to_add,
            rxn_mol=rxn_mol
        )

    elif bridge_position == 3:
        # STEP 1. SubEnumration of minimalcap s1 with synthon s1
        try:
            current_minimal_cap = min_s1.iloc[0]
        except Exception as e:
            raise ValueError(f"Error accessing min_s1 for bridge position 3")

        sub_enumeration(
            current_syntons=synthons_1,
            current_minimal_cap=current_minimal_cap,
            synthon_position=1,
            reaction_id=reaction_id,
            rows_to_add=rows_to_add,
            rxn_mol=rxn_mol
        )
        # STEP 2. SubEnumration of minimalcap s2 with synthon s2
        try:
            current_minimal_cap = min_s2.iloc[0]
        except Exception as e:
            raise ValueError(f"Error accessing min_s2 for bridge position 3")

        sub_enumeration(
            current_syntons=synthons_2,
            current_minimal_cap=current_minimal_cap,
            synthon_position=2,
            reaction_id=reaction_id,
            rows_to_add=rows_to_add,
            rxn_mol=rxn_mol
        )

    else:
        return None

    mel_dataframe = pd.concat([mel_dataframe, pd.DataFrame(rows_to_add)], ignore_index=True)
    return mel_dataframe


def generate_mel_synthon_id(reaction_id: str, synthon_position: int, x: str) -> str:
    """
    Generate MEL synthon ID based on reaction ID, synthon position, and x value.

    Args:
        reaction_id (str): The reaction ID.
        synthon_position (int): The position of the synthon (1, 2, or 3).
        x (str): The dynamic part of the ID.

    Returns:
        str: The generated MEL synthon ID.
    """
    if synthon_position == 1:
        return f"{reaction_id}-{x}_s2_s3"
    elif synthon_position == 2:
        return f"{reaction_id}-s1_{x}_s3"
    elif synthon_position == 3:
        return f"{reaction_id}-s1_s2_{x}"
    else:
        raise ValueError(f"Unsupported synthon position: {synthon_position}")


def add_modified_mols(current_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a new column to the DataFrame with RDKit molecule objects generated from the SMILES column
        """
        def replace_labels(smiles):
            SYNTHON_LABELS = ['U', 'Np', 'Pu', 'Am']
            if smiles is not None:
                for label in SYNTHON_LABELS:
                    smiles = smiles.replace(label, 'U')
            return smiles

        synthon_df = current_df.copy()
        synthon_df['modified_SMILES'] = synthon_df['SMILES'].apply(replace_labels)
        synthon_df['mol'] = synthon_df['modified_SMILES'].apply(Chem.MolFromSmiles)
        return synthon_df


def sub_enumeration(current_syntons, current_minimal_cap, synthon_position, reaction_id, rows_to_add, rxn_mol):

        for _, synthons_row in current_syntons.iterrows():
            product_tuples = rxn_mol.RunReactants([current_minimal_cap["mol"], synthons_row["mol"]])

            product = product_tuples[0][0]
            product_smi = Chem.MolToSmiles(product)

            row_to_write = {
                    'SMILES': synthons_row['SMILES'],
                    'synton_id': synthons_row['synton_id'],
                    'synton#': synthons_row['synton#'],
                    'reaction_id': synthons_row['reaction_id'],

                    'MEL_smiles': product_smi,
                    'mel_synthon_id': generate_mel_synthon_id(
                        reaction_id=reaction_id,
                        synthon_position=synthon_position,
                        x=synthons_row['synton_id']
                    ),
                    'mel_type': 'bridge'
                }

            rows_to_add.append(row_to_write)

