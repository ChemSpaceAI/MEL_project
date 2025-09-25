import os
import json
import logging
import argparse
from datetime import datetime
import time
import re
import pandas as pd
import numpy as np
import os
import sys
import platform
from collections import defaultdict


def setup_logger(file_name: str, log_folder: str) -> logging.Logger:

    # Ensure the log folder exists
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    # Create a logger instance with a unique name to prevent conflicts
    logger = logging.getLogger(f'logger_{file_name}')
    logger.setLevel(logging.INFO)
    
    # Define the full path for the log file
    log_file_path = os.path.join(log_folder, f'{file_name}.log')

    # Create a file handler to write logs to the specified file in append mode
    handler = logging.FileHandler(log_file_path, mode='w')  
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # Set the log format
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    return logger

def read_json_config(json_file_path: str) -> dict:

    with open(json_file_path, 'r') as file:
        config = json.load(file)

    return config

def get_reaction_ids_to_process_old(reaction_id_list: str, 
                                synthons_folder: str) -> list[str]:
  
    
    all_reaction_ids = [
        os.path.basename(x).replace('.txt','') for x in os.listdir(synthons_folder)
    ]
    
    if reaction_id_list == -1: # processing all reactions
        return all_reaction_ids
    elif isinstance(reaction_id_list, list) and len(reaction_id_list) > 0:
        if set(reaction_id_list).issubset(set(all_reaction_ids)):
            return reaction_id_list
        else:
            raise ValueError("Reaction ids are not present in the input")
    else:
        raise ValueError("The reaction_id has to be a list of ids or -1.")

def get_reaction_ids_to_process(reaction_id_list_path: str, synthons_folder: str) -> list[str]:
   
    if reaction_id_list_path: 

        # Read reaction IDs from CSV
        reaction_id_df = pd.read_csv(reaction_id_list_path)
        
        # Check if the expected column exists
        if 'reaction_id' not in reaction_id_df.columns:
            raise ValueError("CSV file must contain a 'reaction_id' column.")
        
        reaction_id_list = reaction_id_df['reaction_id'].astype(str).tolist()

        # Get all reaction IDs from the synthons folder
        all_reaction_ids = {
            os.path.basename(x).replace('.txt', '') for x in os.listdir(synthons_folder)
        }

        # Return only reaction IDs that are present in the synthons folder
        return [rid for rid in reaction_id_list if rid in all_reaction_ids]

    else:
    # Get all reaction IDs from the synthons folder
            all_reaction_ids = {
                os.path.basename(x).replace('.txt', '') for x in os.listdir(synthons_folder)
            }
            return list(all_reaction_ids)
    
def log_script_info(logger, config):
    """
    Logs system and Python information, experiment configuration, and launch time.
    """
    launch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    logger.info('-'*150)
    logger.info('Input INFO')
    logger.info(f"Script name: {sys.argv[0]}")
    logger.info(f"Script started at {launch_time}")
    logger.info(f"Command-line arguments: {sys.argv}")  

    logger.info(f"Configuration: {json.dumps(config, indent=4)}")
    if config.get('filter_config_path', False):
        logger.info(f"Filters Configuration: {json.dumps(read_json_config(config['filter_config_path']), indent=4)}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"System version: {platform.platform()}")
    logger.info(f"Machine architecture: {platform.machine()}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Number of CPUs: {config['N_cores']}/{os.cpu_count()}")
    logger.info('-'*150)
    #logger.info(f"Environment variables: {json.dumps(dict(os.environ), indent=4)}")

def calculate_full_space_size(synthons_df: pd.DataFrame, column_name='synton#') -> int:

    space_size = (
        synthons_df.groupby('reaction_id')[column_name]
        .apply(lambda x: x.value_counts().prod()).sort_values(ascending=False)
    )
    return space_size

def setup_report_logger(output_folder='DEFAULt', file_name='Report',  mode: str = 'w') -> logging.Logger:
    """
    Sets up a logger to log script execution details with time format only.
    """
    # Ensure the log folder exists
    if not os.path.exists(output_folder):
         os.makedirs(output_folder, exist_ok=True)
    
    logger = logging.getLogger(f'{file_name}')
    logger.setLevel(logging.INFO)
    
    log_file_path = os.path.join(output_folder, f'{file_name}.log')
    handler = logging.FileHandler(log_file_path, mode=mode)  
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    formatter = logging.Formatter('- %(message)s')
    handler.setFormatter(formatter)
    
    return logger

def mel_id_generator(tuple_input):
    for full_synthon_ID, CSM_id in tuple_input:
        mel_IDs = full_synthon_ID.split('|')
        for mel_ID in mel_IDs:
            yield (mel_ID, CSM_id)

def mel_id_generator_with_logger(tuple_input, logger=None):
    for idx, (full_synthon_ID, CSM_id) in enumerate(tuple_input, start=1):
        # Логуємо номер ітерації та CSM_id, якщо передано logger
        if logger:
            logger.info(f"{idx}\t{CSM_id}\t{full_synthon_ID}")

        mel_IDs = full_synthon_ID.split('|')
        for mel_ID in mel_IDs:
            yield (mel_ID, CSM_id)

def mel_id_generator_old(full_synthons_IDs_list):
    for full_synthon_ID in full_synthons_IDs_list:
        mel_IDs = full_synthon_ID.split('|')
        for mel_ID in mel_IDs:
            yield mel_ID


def mel_id_generator_new(full_synthons_IDs_list):
    for full_synthon_ID in full_synthons_IDs_list:
        # Handle old format (with '|') or new format (no delimiter)
        if '|' in full_synthon_ID:
            mel_IDs = full_synthon_ID.split('|')
        else:
            mel_IDs = re.split(r'(?=m_)', full_synthon_ID)
        
        # Yield non-empty IDs
        for mel_ID in mel_IDs:
            if mel_ID.strip():  # skip empty strings
                print(mel_ID)
                yield mel_ID


def get_reaction_category(filename: str, categories: list) -> dict:
    """
    Determine the reaction category based on the filename.
    """
    filename = filename.lower()
    for category in categories:
        if category in filename:
            return category
    return None


def predict_the_amount_of_fragments_by_for_MEL_SMILES(full_mel_synthon_id, mel_type, synthons_dict_counts ):
 
    mel_ids =full_mel_synthon_id.split('|')
    general_count_of_fragments = 0

    for mel_id in mel_ids:

        try:
            reaction_id, synthons_formula_ID = mel_id.split('-')
            s1_ID, s2_ID, s3_ID = synthons_formula_ID.split('_')
            reaction_id_dict_count = synthons_dict_counts[reaction_id]
        except Exception:
            print(mel_id, Exception)
            continue  # Skip malformed or unknown entries

        if mel_type == '3_component':
            
            if ("s2" == s2_ID) and ('s3' == s3_ID):
                fragments_amount= reaction_id_dict_count['s2'] + reaction_id_dict_count['s3']

            elif ("s1" == s1_ID) and ('s2' == s2_ID):
                fragments_amount= reaction_id_dict_count['s1'] + reaction_id_dict_count['s2']

            elif ("s1" == s1_ID) and ('s3' == s3_ID):
                fragments_amount= reaction_id_dict_count['s1'] + reaction_id_dict_count['s3']
            else:
                print(mel_id, 'why NOLE')
                fragments_amount= 0

        if mel_type == 'bridge': 
            bridge_position = reaction_id_dict_count['bridge_position']

            if ("s2" == s2_ID) and ('s3' == s3_ID):

                if bridge_position == 2:
                    fragments_amount=  reaction_id_dict_count['s2']

                elif bridge_position == 3: 
                    fragments_amount = reaction_id_dict_count['s3']

            elif ("s1" == s1_ID) and ('s2' == s2_ID):

                if bridge_position == 1:
                    fragments_amount =  reaction_id_dict_count['s1']
                    
                elif bridge_position == 2: 
                    fragments_amount = reaction_id_dict_count['s2']
    
            elif ("s1" == s1_ID) and ('s3' == s3_ID):
                if bridge_position == 1:
                    fragments_amount =  reaction_id_dict_count['s1']
                    
                elif bridge_position == 3: 
                    fragments_amount = reaction_id_dict_count['s3']

            else:
                print(mel_id, 'why NOLE')
                fragments_amount= 0

        if mel_type  == '2_component': 
            if "s1" in synthons_formula_ID :
                fragments_amount = reaction_id_dict_count['s1']
            
            elif "s2" in synthons_formula_ID:
                fragments_amount = reaction_id_dict_count['s2']
            else:
                print(mel_id, 'why NOLE')
                fragments_amount= 0

        general_count_of_fragments += fragments_amount

    return general_count_of_fragments


def generate_csm_id(row, full_dict):
    """
    Generate a Chemspace MEL ID in the format:
    CS_<tX>_f<fragments>_l<lengths>_o<order>_<suffix>

    tX: MEL type short code (t2 = two-component, t3 = three-component, tB = bridge)
    f : number of fragments (predicted)
    l : number of components in 'mel_synthon_id'
    o : row order (1-based index)
    suffix: always 'e0' for MEL records
    """

    full_mel_synthon_id = row['mel_synthon_id']
    mel_type = row['mel_type']
    
    
    # t mel type
    mel_type_map = {
        '2_component': 't2',
        '3_component': 't3',
        'bridge': 'tB'
    }
    t_val = mel_type_map.get(mel_type, 'Error')
    
    # f - calculated fragments
    f_val = predict_the_amount_of_fragments_by_for_MEL_SMILES(full_mel_synthon_id, mel_type, full_dict)
    
    # l - number of components in synthon ID
    l_str = str(len(full_mel_synthon_id.split('|')))
    
    # o - order from row index (starting at 1)
    o_val = row.name + 1  # row.name is the index
    
    # e0 - static suffix for MEL
    e_suffix = "e0"
    
    # Example: CS_t2_f244_l14_o123_e0
    CSM_id = f"CS_{t_val}_f{f_val}_l{l_str}_o{o_val}_{e_suffix}"

    return CSM_id



def predict_the_amount_of_fragments_for_MEL_SMILES_enumeration(full_mel_synthon_id, mel_type, iteration_level, synthons_dict_counts ):

    if mel_type  == '2_component':
            return 0
    
    if iteration_level == 2:
        return 0
     
    mel_ids =full_mel_synthon_id.split('|')
    general_count_of_fragments = 0
    fragments_amount = 0 
    for mel_id in mel_ids:

        try:
            reaction_id, synthons_formula_ID = mel_id.split('-')
            s1_ID, s2_ID, s3_ID = synthons_formula_ID.split('_')
            reaction_id_dict_count = synthons_dict_counts[reaction_id]
        except Exception:
            print(mel_id, Exception)
            continue  # Skip malformed or unknown entries

        if mel_type == '3_component' or mel_type == 'bridge':
            
            if 's1' in synthons_formula_ID:
                fragments_amount+=reaction_id_dict_count['s1']

            if 's2' in synthons_formula_ID:
                fragments_amount+=reaction_id_dict_count['s2']

            if 's3' in synthons_formula_ID:
                fragments_amount+=reaction_id_dict_count['s3']

        general_count_of_fragments += fragments_amount

    return general_count_of_fragments





def log_processing_report(results, reaction_id_map_path, logger):
    """
    Логування підсумкового звіту по обробці реакцій:
    - на початку успішні реакції, згруповані по reaction_id_type
    - потім помилки, згруповані по reaction_id_type і по тексту помилки
    """

    # Завантажуємо мапу reaction_id_type -> reaction_ids
    with open(reaction_id_map_path, 'r') as f:
        reaction_map = json.load(f)

    # Створюємо зворотну мапу: reaction_id -> reaction_id_type
    reaction_to_type = {}
    for rtype, rlist in reaction_map.items():
        for rid in rlist:
            reaction_to_type[rid] = rtype

    total = len(results)
    successes = [r for r in results if r['status'] == 'success']
    fails = [r for r in results if r['status'] == 'fail']

    logger.info(f"Success: {len(successes)}/{total}")
    logger.info(f"Failed: {len(fails)}")

    # ==== 1. Вивід успішних реакцій ====
    if successes:
        grouped_success_by_type = defaultdict(list)
        for succ in successes:
            r_id = succ['reaction_id']
            r_type = reaction_to_type.get(r_id, "UNKNOWN")
            grouped_success_by_type[r_type].append(r_id)

        logger.info("\n=== Success===")
        for r_type in sorted(grouped_success_by_type.keys(), key=lambda x: (x != "UNKNOWN", x)):
            ids_sorted = sorted(grouped_success_by_type[r_type])
            logger.info(f"Reaction ID Type: {r_type}")
            logger.info(f"  IDs: {', '.join(ids_sorted)}")

    # ==== 2. Вивід помилок ====
    if not fails:
        logger.info("No Mistakes found.")
        return

    logger.info("\n=== Failed ===")
    grouped_by_type = defaultdict(list)
    for fail in fails:
        r_id = fail['reaction_id']
        r_type = reaction_to_type.get(r_id, "UNKNOWN")
        grouped_by_type[r_type].append(fail)

    for r_type in sorted(grouped_by_type.keys(), key=lambda x: (x != "UNKNOWN", x)):
        errors = grouped_by_type[r_type]
        logger.info(f"=== Reaction ID Type: {r_type} ===")

        # Групування по тексту помилки
        errors_by_message = defaultdict(list)
        for err in errors:
            errors_by_message[err['error']].append(err['reaction_id'])

        for error_msg, reaction_ids in errors_by_message.items():
            joined_ids = ", ".join(sorted(reaction_ids))
            logger.info(f"--- IDs   : {joined_ids}")
            logger.info(f"--- Error : {error_msg}")

    # ==== 3. Підрахунок успішних reaction_id_type ====
    successful_types = set()
    for succ in successes:
        r_id = succ['reaction_id']
        r_type = reaction_to_type.get(r_id, "UNKNOWN")
        successful_types.add(r_type)

    logger.info(f"\n✅ Successful reaction_id_types: {len(successful_types)}")
    logger.info(f"   Types: {', '.join(sorted(successful_types))}")



def extract_mel_synthon_ids_from_sdf(sdf_path: str) -> list:
    mel_ids = []
    try:
        with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.strip() == "> <mel_synthon_id>":
                if i + 1 < len(lines):
                    value = lines[i + 1].strip()
                    if value:
                        mel_ids.append(value)

        print(f"Extracted {len(mel_ids)} mel_synthon_id values from {sdf_path}")

    except Exception as e:
        print(f"Error reading SDF file: {e}")
        return []


def extract_mel_synthon_ids_from_sdf(sdf_path: str) -> list:
    mel_ids = []
    try:
        with open(sdf_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.strip() == "> <mel_synthon_id>":
                if i + 1 < len(lines):
                    value = lines[i + 1].strip()
                    if value:
                        mel_ids.append(value)

        print(f"Extracted {len(mel_ids)} mel_synthon_id values from {sdf_path}")

    except Exception as e:
        print(f"Error reading SDF file: {e}")
        return []



    return mel_ids