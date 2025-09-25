import os, glob, argparse, json
from collections import defaultdict
import pandas as pd

from mel_package.functions.Utils import (
    setup_logger,
    get_reaction_category,
    read_json_config, 
    generate_csm_id,
)


def main(input_dir: str, output_dir: str, chunk_size: int, mel_types: list, dict_with_counts_path):

    os.makedirs(output_dir, exist_ok=True)
    logger =  setup_logger(file_name='_', log_folder=output_dir)

    # Categorize files
    files_by_category = defaultdict(list)
    
    for file in glob.glob(os.path.join(input_dir, "*.csv")):
        category = get_reaction_category(file, mel_types)
        if category:
            files_by_category[category].append(file)

    for category in files_by_category:
        logger.info(f"Category: {category}, Number of files: {len(files_by_category[category])}")
        
    # Merge files
    for mel_type, files_by_mel_type in files_by_category.items(): 
        
        total_lines = 0
        merged_file_path=os.path.join(output_dir, f'merged_{mel_type}.csv')

        logger.info(f"{mel_type}: {len(files_by_category[mel_type])} files")

        with open(merged_file_path, "w") as merged_file:

            header_written = False

            for file in files_by_mel_type:

                with open(file, "r") as infile:
                    lines = infile.readlines()

                    # Write header from the first file only
                    if not header_written:
                        merged_file.writelines(lines[:1]) 
                        header_written = True

                    # Write the remaining lines (excluding the header)
                    merged_file.writelines(lines[1:])
                    total_lines += len(lines) - 1 

        logger.info(f"Total lines in merged file {mel_type}: {total_lines}")

        # Group Duplicates 
        current_df = pd.read_csv(merged_file_path)

        df_grouped = current_df.groupby([ "MEL_smiles", "mel_type"]).agg({
            "mel_synthon_id": lambda x: "|".join(map(str, set(x))),
            "SMILES": lambda x: ".".join(map(str, set(x)))
        }).reset_index()

        # Create CSM_ID 
        with open(dict_with_counts_path, 'r') as f:
            dict_with_counts = json.load(f)
        
        df_grouped['CSM_id'] = df_grouped.apply(lambda row: generate_csm_id(row, dict_with_counts), axis=1)
        df_grouped = df_grouped[['SMILES', 'MEL_smiles', 'CSM_id', 'mel_type', 'mel_synthon_id', ]]

        df_grouped = df_grouped[df_grouped['SMILES'].astype(str).str.contains('.')]

        unique_smiles_count = len(df_grouped)
        logger.info(f"{mel_type}: {unique_smiles_count} unique SMILES")

        mel_type_folder = os.path.join(output_dir, mel_type)
        os.makedirs(mel_type_folder, exist_ok=True)

        # Split dataframe into chunks 
        for i, chunk in enumerate(range(0, unique_smiles_count, chunk_size)):

            chunk_df = df_grouped.iloc[chunk:chunk + chunk_size]
            chunk_file = os.path.join(mel_type_folder, f"{mel_type}_chunk_{i+1}.csv")
            chunk_df.to_csv(chunk_file, index=False)

            logger.info(f"Len of {os.path.basename(chunk_file)} : {len(chunk_df)}")

def make_processed_dir(input_dir: str, suffix: str = "processed_") -> str:

    parent_dir = os.path.dirname(input_dir)
    base_name = os.path.basename(input_dir)
    return os.path.join(parent_dir, suffix+ base_name )


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config_test.json', help='Path to the configuration file')
    args = parser.parse_args()

    config = read_json_config(args.config)

    input_dir = config["input_dir"]
    output_dir = make_processed_dir(input_dir)
    chunk_size = config.get("chunk_size", 40000)
    mel_types = config.get("mel_types", ["3_component", "2_component", "bridge"])
    dict_with_counts = config.get('synthon_counts_with_bridge_position', False)

    main(input_dir, output_dir, chunk_size, mel_types, dict_with_counts)


    #python 2_Process_MEL_generation.py --input_dir Modified_generation_MEL/enumerated_MEL --output_dir Modified_generation_MEL/processed_enumerated_MEL

    '''df_grouped = current_df.groupby("SMILES").agg({
            "synton_id": lambda x: "|".join(map(str, set(x))),  
            "reaction_id": lambda x: "|".join(map(str, set(x))),  
            "MEL_smiles": "first", 
            "mel_synthon_id": lambda x: "|".join(map(str, set(x))), 
            "mel_type": "first"  
        }).reset_index() 
    df_grouped = df_grouped[['SMILES', 'MEL_smiles', 'CSM_id', 'reaction_id', 'mel_synthon_id', 'mel_type']]'''