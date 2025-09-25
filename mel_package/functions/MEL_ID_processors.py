# Import necessary libraries
import os
import pandas as pd
import logging
from typing import Dict, List

# Import custom functions from your project
from mel_package.functions.Chemistry_logic import add_mols, add_charge, calculate_and_filter_on_parameters
from mel_package.functions.Enumerators import enumerator_engine_for_2_component, enumerator_engine_for_3_component
# RDKit imports
from rdkit import RDLogger
from rdkit import Chem
# Disable RDKit logging to suppress warnings
RDLogger.DisableLog('rdApp.')
 
def mel_ID_enumeration_for_2_level_iteration(
    mel_ID: str, 
    CSM_id: str,
    config: dict, 
    logger: logging.Logger, 
    file_of_reactions: pd.DataFrame, 
    filters_config: dict
) -> dict:
    """
    Processes one MEL ID by loading synthons, enumerating products, and applying filters.

    Args:
        mel_ID (str): The MEL ID representing a specific reaction and its synthon pair.
        config (dict): The configuration dictionary containing paths and settings.
        logger (logging.Logger): Logger instance for logging status and errors.
        file_of_reactions (pd.DataFrame): DataFrame containing reaction information.
        filters_config (dict): Filter configurations to apply during enumeration.

    Returns:
        dict: A dictionary where the keys are SMILES strings of the products, and values are 
              lists containing [result_synthon_id, charged_smiles, synthon_1_SMILES, synthon_2_SMILES, synthon_3_SMILES, molecular_properties].
    """
    # Split the MEL ID to extract reaction and synthon formula IDs
    reaction_id, synthons_formula_ID = mel_ID.split('-')

    # Load synthon data for the given reaction
    synthon_file_path = os.path.join(config['syntons_folder_path'], f'{reaction_id}.txt')
    try:
        synthons_df = pd.read_csv(synthon_file_path, sep='\t')
    except FileNotFoundError:
        logger.error(f"Synthon file not found: {synthon_file_path}")
        return None  # Return None if the synthon file is missing
    
    # Extract the reaction SMARTS and other details from the reaction file
    reaction_row = file_of_reactions[file_of_reactions['reaction_id'] == reaction_id]
    if reaction_row.empty:
        logger.warning(f"Reaction ID {reaction_id} not found in reactions file.")
        return None  # Return None if the reaction ID doesn't exist in the reaction file
    rxn_mol = reaction_row.iloc[0]["Reaction_MOL"]


    # Parse synthon IDs Example: 9173816_0_3465810
    s1_ID, s2_ID, s3_ID = synthons_formula_ID.split('_')

    if s3_ID == 's3':

        #Enumeration 1: synthon_1 (anchor)  + synthon_2 (anchor) + synthon_3 (variable_full)

        anchor_synthon_ID_1 = int(s1_ID)
        anchor_synthon_row_1 = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID_1] 

        anchor_synthon_ID_2 = int(s2_ID)
        anchor_synthon_row_2 = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID_2] 

        variable_synthons = synthons_df[synthons_df['synton#'] == 3]

        result = enumerator_engine_for_3_component(
            synthons_1= anchor_synthon_row_1,
            synthons_2= anchor_synthon_row_2,
            synthons_3= variable_synthons,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id,
            CSM_id=CSM_id,
            cap_position=0
        )
 
    elif s2_ID == 's2':

        #Enumeration 1: synthon_1 (anchor)  + synthon_2 (variable_full)  + synthon_3  (anchor)

        anchor_synthon_ID_1 = int(s1_ID)
        anchor_synthon_row_1 = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID_1] 

        anchor_synthon_ID_2 = int(s3_ID)
        anchor_synthon_row_2 = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID_2] 

        variable_synthons = synthons_df[synthons_df['synton#'] == 2]
        result = enumerator_engine_for_3_component(
            synthons_1= anchor_synthon_row_1,
            synthons_2= variable_synthons,
            synthons_3= anchor_synthon_row_2,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id,
            CSM_id=CSM_id,
            cap_position=0
        )

    elif s1_ID == 's1':

        #Enumeration 1: synthon_1 (variable_full)  + synthon_2  (anchor)  + synthon_3  (anchor)

        anchor_synthon_ID_1 = int(s2_ID)
        anchor_synthon_row_1 = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID_1] 

        anchor_synthon_ID_2 = int(s3_ID)
        anchor_synthon_row_2 = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID_2] 

        variable_synthons = synthons_df[synthons_df['synton#'] == 1]

        result = enumerator_engine_for_3_component(
            synthons_1= variable_synthons,
            synthons_2= anchor_synthon_row_1,
            synthons_3= anchor_synthon_row_2,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id,
            CSM_id=CSM_id,
            cap_position=0
        )
       
    # Handle cases where the synthon IDs are not as expected
    else:
        logger.warning(f"Unexpected synthon IDs format: {mel_ID}")
        return None  # Return None for unhandled synthon formats

    return result

def mel_ID_enumeration_for_bridge(
    mel_ID: str, 
    CSM_id: str,
    config: dict, 
    logger: logging.Logger, 
    file_of_reactions: pd.DataFrame, 
    filters_config: dict
) -> dict:
    """
    Processes one MEL ID by loading synthons, enumerating products, and applying filters.

    Args:
        mel_ID (str): The MEL ID representing a specific reaction and its synthon pair.
        config (dict): The configuration dictionary containing paths and settings.
        logger (logging.Logger): Logger instance for logging status and errors.
        file_of_reactions (pd.DataFrame): DataFrame containing reaction information.
        filters_config (dict): Filter configurations to apply during enumeration.

    Returns:
        dict: A dictionary where the keys are SMILES strings of the products, and values are 
              lists containing [result_synthon_id, charged_smiles, synthon_1_SMILES, synthon_2_SMILES, synthon_3_SMILES, molecular_properties].
    """
    reaction_id, synthons_formula_ID = mel_ID.split('-')

    # Load synthon data for the given reaction
    synthon_file_path = os.path.join(config['syntons_folder_path'], f'{reaction_id}.txt')
    try:
        synthons_df = pd.read_csv(synthon_file_path, sep='\t')
    except FileNotFoundError:
        logger.error(f"Synthon file not found: {synthon_file_path}")
        return None  # Return None if the synthon file is missing
    
  
    # Load minimal_caps data for the given reaction
    minimal_caps_file_path= os.path.join(config[ "minimal_caps_folder_path"], f'{reaction_id}.txt')
    try:
        minimal_caps_df = pd.read_csv(minimal_caps_file_path, sep='\t')
    except FileNotFoundError:
        logger.error(f"Minimal Caps file not found: {synthon_file_path}")
        return None
    
    
    # Extract reaction SMARTS and other details
    reaction_row = file_of_reactions[file_of_reactions['reaction_id'] == reaction_id]
    if reaction_row.empty:
        logger.warning(f"Reaction ID {reaction_id} not found in reactions file.")
        return None
    rxn_mol = reaction_row.iloc[0]["Reaction_MOL"]

    try:
        bridge_position = int(reaction_row.iloc[0]["bridge_position"])
    except KeyError:
        logger.error("Smth is wrong with brdige position", reaction_id)
        return None
    
    # Parse synthon IDs
    s1_ID, s2_ID, s3_ID = synthons_formula_ID.split('_')

    #s1_s2_26213476 s1_23795974_s3 24703580_s2_s3
    
    if ("s2" == s2_ID) and ('s3' == s3_ID):

        #Enumeration 1: synthon_1 (anchor)  + synthon_2 + synthon_3

        anchor_synthon_ID = int(s1_ID)

        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID] 

        if bridge_position == 2:

            variable_synthons = synthons_df[synthons_df['synton#'] == 2]
            cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 3]
            
            result = enumerator_engine_for_3_component(
                synthons_1= anchor_synthon_row,
                synthons_2= variable_synthons,
                synthons_3= cap_synthons,
                rxn_mol=rxn_mol,
                filters_config=filters_config,
                logger=logger,
                reaction_id=reaction_id,
                CSM_id=CSM_id,
                cap_position=3
            )
            
        elif bridge_position == 3: 

            variable_synthons = synthons_df[synthons_df['synton#'] == 3]
            cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 2]
            
            result = enumerator_engine_for_3_component(
                synthons_1= anchor_synthon_row,
                synthons_2= cap_synthons,
                synthons_3= variable_synthons,
                rxn_mol=rxn_mol,
                filters_config=filters_config,
                logger=logger,
                reaction_id=reaction_id,
                CSM_id=CSM_id,
                cap_position=2
            )

    elif ("s1" == s1_ID) and ('s2' == s2_ID):

        #Enumeration 2:  synthon_1 + synthon_2 + synthon_3 (anchor)
        
        anchor_synthon_ID = int(s3_ID)

        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID] 

        if bridge_position == 1:

            variable_synthons = synthons_df[synthons_df['synton#'] == 1]
            cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 2]
            
            result = enumerator_engine_for_3_component(
                synthons_1= variable_synthons,
                synthons_2= cap_synthons,
                synthons_3= anchor_synthon_row,
                rxn_mol=rxn_mol,
                filters_config=filters_config,
                logger=logger,
                reaction_id=reaction_id,
                CSM_id=CSM_id,
                cap_position=2
            )
            
            
        elif bridge_position == 2: 

            variable_synthons = synthons_df[synthons_df['synton#'] == 2]
            cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 1]
            
            result = enumerator_engine_for_3_component(
                synthons_1= cap_synthons,
                synthons_2= variable_synthons,
                synthons_3= anchor_synthon_row,
                rxn_mol=rxn_mol,
                filters_config=filters_config,
                logger=logger,
                reaction_id=reaction_id,
                CSM_id=CSM_id,
                cap_position=1
            )

    elif ("s1" == s1_ID) and ('s3' == s3_ID):
        #Enumeration 3: synthon_1 + synthon_2 (anchor) + synthon_3 

        anchor_synthon_ID = int(s2_ID)

        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID] 

        if bridge_position == 1:

            variable_synthons = synthons_df[synthons_df['synton#'] == 1]
            cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 3]
            
            result = enumerator_engine_for_3_component(
                synthons_1= variable_synthons,
                synthons_2= anchor_synthon_row,
                synthons_3= cap_synthons,
                rxn_mol=rxn_mol,
                filters_config=filters_config,
                logger=logger,
                reaction_id=reaction_id,
                CSM_id=CSM_id,
                cap_position=3
            )
           

        elif bridge_position == 3: 

            variable_synthons = synthons_df[synthons_df['synton#'] == 3]
            cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 1]
            
            result = enumerator_engine_for_3_component(
                synthons_1= cap_synthons,
                synthons_2= anchor_synthon_row,
                synthons_3= variable_synthons,
                rxn_mol=rxn_mol,
                filters_config=filters_config,
                logger=logger,
                reaction_id=reaction_id, 
                CSM_id=CSM_id,
                cap_position=1
            )
            

      
    # Handle cases where the synthon IDs are not as expected
    else:
        logger.warning(f"Unexpected synthon IDs format: {mel_ID}")
        return None  # Return None for unhandled synthon formats
    return result

def mel_ID_enumeration_for_3_component_iteration_1(
    mel_ID: str, 
    CSM_id: str,
    config: dict, 
    logger: logging.Logger, 
    file_of_reactions: pd.DataFrame, 
    filters_config: dict
) -> dict:
    """
    Processes one MEL ID by loading synthons, enumerating products, and applying filters.

    Args:
        mel_ID (str): The MEL ID representing a specific reaction and its synthon pair.
        config (dict): The configuration dictionary containing paths and settings.
        logger (logging.Logger): Logger instance for logging status and errors.
        file_of_reactions (pd.DataFrame): DataFrame containing reaction information.
        filters_config (dict): Filter configurations to apply during enumeration.

    Returns:
        dict: A dictionary where the keys are SMILES strings of the products, and values are 
              lists containing [result_synthon_id, charged_smiles, synthon_1_SMILES, synthon_2_SMILES, synthon_3_SMILES, molecular_properties].
    """
    
    reaction_id, synthons_formula_ID = mel_ID.split('-')

    # Load synthon data for the given reaction
    synthon_file_path = os.path.join(config['syntons_folder_path'], f'{reaction_id}.txt')
    try:
        synthons_df = pd.read_csv(synthon_file_path, sep='\t')
    except FileNotFoundError:
        logger.error(f"Synthon file not found: {synthon_file_path}")
        return None  # Return None if the synthon file is missing
    
  
    # Load minimal_caps data for the given reaction
    minimal_caps_file_path= os.path.join(config[ "minimal_caps_folder_path"], f'{reaction_id}.txt')
    try:
        minimal_caps_df = pd.read_csv(minimal_caps_file_path, sep='\t')
    except FileNotFoundError:
        logger.error(f"Minimal Caps file not found: {synthon_file_path}")
        return None
    
    
    # Extract reaction SMARTS and other details
    reaction_row = file_of_reactions[file_of_reactions['reaction_id'] == reaction_id]
    if reaction_row.empty:
        logger.warning(f"Reaction ID {reaction_id} not found in reactions file.")
        return None
    rxn_mol = reaction_row.iloc[0]["Reaction_MOL"]

    # Parse synthon IDs Example: s1_s2_26213476, s1_23795974_s3, 24703580_s2_s3.
    s1_ID, s2_ID, s3_ID = synthons_formula_ID.split('_')

    if ("s2" == s2_ID) and ('s3' == s3_ID):

        #Enumeration 1: synthon_1 (anchor)  + synthon_2 + synthon_3
        anchor_synthon_ID = int(s1_ID)
        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID] 

        #Combinatorial Case 1:  synthon_1 (anchor) + synthon_2 (variable_full) + synthon_3 (cap)
        variable_synthons = synthons_df[synthons_df['synton#'] == 2]
        cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 3]

        result = enumerator_engine_for_3_component(
            synthons_1= anchor_synthon_row,
            synthons_2= variable_synthons,
            synthons_3= cap_synthons,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id, 
            CSM_id=CSM_id,
            cap_position=3
        )

        #Combinatorial Case 2:  synthon_1 (anchor) + synthon_2 (cap) + synthon_3 (variable_full) 
        variable_synthons = synthons_df[synthons_df['synton#'] == 3]
        cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 2]

        result = enumerator_engine_for_3_component(
            synthons_1= anchor_synthon_row,
            synthons_2= cap_synthons,
            synthons_3= variable_synthons,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id, 
            CSM_id=CSM_id,
            cap_position = 2
        )

    elif ("s1" == s1_ID) and ('s2' == s2_ID):

        #Enumeration 2:  synthon_1 + synthon_2 + synthon_3 (anchor)
        anchor_synthon_ID = int(s3_ID)
        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID] 

        #Combinatorial Case 1:  synthon_1 (cap) + synthon_2 (variable_full) + synthon_3 (anchor) 
        variable_synthons = synthons_df[synthons_df['synton#'] == 2]
        cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 1]

        result = enumerator_engine_for_3_component(
            synthons_1= cap_synthons, 
            synthons_2= variable_synthons,
            synthons_3= anchor_synthon_row,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id, 
            CSM_id=CSM_id,
            cap_position= 1
        )

        #Combinatorial Case 2:  synthon_1 (variable_full)  + synthon_2 (cap) + synthon_3 (anchor) 
        variable_synthons = synthons_df[synthons_df['synton#'] == 1]
        cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 2]

        result = enumerator_engine_for_3_component(
            synthons_1= variable_synthons,
            synthons_2= cap_synthons, 
            synthons_3= anchor_synthon_row,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id, 
            CSM_id=CSM_id,
            cap_position = 2
        )


    elif ("s1" == s1_ID) and ('s3' == s3_ID):

        #Enumeration 3: synthon_1 + synthon_2 (anchor) + synthon_3 
        anchor_synthon_ID = int(s2_ID)
        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID] 

        #Combinatorial Case 1:  synthon_1 (cap) + synthon_2 (anchor) + synthon_3 (variable_full)
        variable_synthons = synthons_df[synthons_df['synton#'] == 3]
        cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 1]

        result = enumerator_engine_for_3_component(
            synthons_1= cap_synthons, 
            synthons_2= anchor_synthon_row, 
            synthons_3= variable_synthons,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id, 
            CSM_id=CSM_id,
            cap_position=1
        )

        #Combinatorial Case 2:  synthon_1 (variable_full) + synthon_2 (anchor) + synthon_3 (cap)
        variable_synthons = synthons_df[synthons_df['synton#'] == 1]
        cap_synthons = minimal_caps_df[minimal_caps_df['synton#'] == 3]

        result = enumerator_engine_for_3_component(
            synthons_1= variable_synthons,
            synthons_2= anchor_synthon_row, 
            synthons_3= cap_synthons,
            rxn_mol=rxn_mol,
            filters_config=filters_config,
            logger=logger,
            reaction_id=reaction_id, 
            CSM_id=CSM_id,
            cap_position=3
        )

    
    else:
        logger.warning(f"Unexpected synthon IDs format: {mel_ID}")
        return {}
    
    return result 

def mel_ID_enumeration_for_2_component(
    mel_ID: str, 
    CSM_id: str,
    config: dict, 
    logger: logging.Logger, 
    file_of_reactions: pd.DataFrame, 
    filters_config: dict
) -> dict:
    """
    Processes one MEL ID by loading synthons, enumerating products, and applying filters.

    Args:
        mel_ID (str): The MEL ID representing a specific reaction and its synthon pair.
        config (dict): The configuration dictionary containing paths and settings.
        logger (logging.Logger): Logger instance for logging status and errors.
        file_of_reactions (pd.DataFrame): DataFrame containing reaction information.
        filters_config (dict): Filter configurations to apply during enumeration.

    Returns:
        dict: A dictionary where the keys are SMILES strings of the products, and values are 
              lists containing [result_synthon_id, charged_smiles, synthon_1_SMILES, synthon_2_SMILES, '', molecular_properties].
    """
    
    # Split the MEL ID to extract reaction and synthon formula IDs
    reaction_id, synthons_formula_ID = mel_ID.split('-')

    # Load synthon data for the given reaction
    synthon_file_path = os.path.join(config['syntons_folder_path'], f'{reaction_id}.txt')
    try:
        synthons_df = pd.read_csv(synthon_file_path, sep='\t', dtype={'synton_id': str})
    except FileNotFoundError:
        logger.error(f"Synthon file not found: {synthon_file_path}")
        return None  # Return None if the synthon file is missing
    
    # Extract the reaction SMARTS and other details from the reaction file
    reaction_row = file_of_reactions[file_of_reactions['reaction_id'] == reaction_id]
    if reaction_row.empty:
        logger.warning(f"Reaction ID {reaction_id} not found in reactions file.")
        return None  # Return None if the reaction ID doesn't exist in the reaction file
    rxn_mol = reaction_row.iloc[0]["Reaction_MOL"]

    # Parse synthon IDs from the formula
    s1_ID, s2_ID, s3_ID = synthons_formula_ID.split('_')

    # If it's a 3-component reaction, we don't handle it in this function
    if s3_ID != '0':
        logger.warning(f"Unexpected 3rd synthon ID ({s3_ID}) for {mel_ID}. Skipping.")
        return None  # Skip 3-component reactions

    # Case where s1 is the variable synthon
    if s1_ID == "s1":
        anchor_synthon_ID = s2_ID
        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID]

        if anchor_synthon_row.empty:
            logger.warning(f"Anchor synthon ID {anchor_synthon_ID} not found. Skipping {mel_ID}.")
            return None  # Skip if anchor synthon not found

        variable_synthons = synthons_df[synthons_df['synton#'] == 1]

        result = enumerator_engine_for_2_component(
            synthons_1=variable_synthons,
            synthons_2=anchor_synthon_row,
            rxn_mol=rxn_mol,
            reaction_id=reaction_id,
            CSM_id=CSM_id,
            filters_config=filters_config,
            logger=logger
        )
       
    # Case where s2 is the variable synthon
    elif s2_ID == "s2":
        anchor_synthon_ID = s1_ID
        anchor_synthon_row = synthons_df[synthons_df['synton_id'] == anchor_synthon_ID]

        if anchor_synthon_row.empty:
            logger.warning(f"Anchor synthon ID {anchor_synthon_ID} not found. Skipping {mel_ID}.")
            return None  # Skip if anchor synthon not found

        variable_synthons = synthons_df[synthons_df['synton#'] == 2]

        result = enumerator_engine_for_2_component(
            synthons_1=anchor_synthon_row,
            synthons_2=variable_synthons,
            rxn_mol=rxn_mol,
            reaction_id=reaction_id,
            CSM_id=CSM_id,
            filters_config=filters_config,
            logger=logger
        )
      
    # Handle cases where the synthon IDs are not as expected
    else:
        logger.warning(f"Unexpected synthon IDs format: {mel_ID}")
        return None  # Return None for unhandled synthon formats

    return result  # Return the generated molecules as a dictionary

