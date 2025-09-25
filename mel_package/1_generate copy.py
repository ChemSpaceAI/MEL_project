import os, json, time, argparse
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

from mel_package.functions.Utils import (
    setup_logger,
    read_json_config,
    get_reaction_ids_to_process
)
from mel_package.functions.Chemistry_logic import (
    change_mol_to_isotope_labelled,
    get_minimal_synthon,
    check_if_bridge_reaction,
    check_if_bridge_reaction_external,
    process_external_minimal_synthons
)
from mel_package.functions.Enumerators import (
    enumerator_2_component,
    enumerator_3_component,
    enumerate_bridge_with_replaced_SMILES
)

SYNTHON_LABELS = ['U','Np','Pu','Am']


def process_reaction_worker(worker_args: tuple) -> None:
    try:
        reaction_id, reaction_table, config, logger = worker_args
        logger.info(f'Enumerating reaction {reaction_id}')
 
        print('reaction_id', reaction_id)
        # check if 2 or 3 component and type
        n_components = int(reaction_table['components'].iloc[0])
        if n_components in (2,3) == False:
            raise ValueError("The script currently supports 2 and 3 component reactions")
        
        # Get rdkit mols
        synthons_table = pd.read_csv(os.path.join(config['synthons_folder_path'], f'{reaction_id}.txt'), sep='\t')
        synthons_table['mol'] = synthons_table.SMILES.apply(Chem.MolFromSmiles)

        # checking for external minimal synthons and processing them
        minimal_synthons = {}
        external_synthons_flag = False
        
        if config.get('external_minimal_synthons_path', False):
            minimal_synthons_table = pd.read_csv(os.path.join(config[ "external_minimal_synthons_path"], f'{reaction_id}.txt'), sep='\t')
            minimal_synthons = process_external_minimal_synthons(minimal_synthons_table)
            external_synthons_flag = True


        # if no external minimal synthons - generating minimal synthons + extracting the data on synthons
        synthon_tables_by_position = {}

        for synthon_position in range(1, n_components + 1):
            current_synthon_table = synthons_table[synthons_table['synton#'] == synthon_position].copy()
            synthon_tables_by_position[synthon_position] = current_synthon_table
            if not external_synthons_flag:
                minimal_synthons[synthon_position] = get_minimal_synthon(current_synthon_table)
        
     
        # Perform Enumeration. Calculate the MEL smiles and assign new ids

        rxn_smarts = reaction_table.iloc[0]["Reaction"]
        rxn_mol = AllChem.ReactionFromSmarts(rxn_smarts)
        result_dict = {}

        # Processing 2-component reactions
        if n_components == 2:

            result_dict[1] = enumerator_2_component(synthon_tables_by_position.get(1), minimal_synthons.get(2), 
                                                    rxn_mol, synthon_position = 1)
            result_dict[2] = enumerator_2_component(minimal_synthons.get(1), synthon_tables_by_position.get(2), 
                                                    rxn_mol, synthon_position = 2)
            output_type = '2_component'

        # Processing 3-component and bridge reactions
        else:
        
            if config.get('external_minimal_synthons_path', False):

                mel_type = reaction_table.iloc[0]['mel_type']
    
                if mel_type == 'bridge':
                    
                    
                    output_type = 'bridge'
                    bridge_position = int(reaction_table.iloc[0]['bridge_position'])

                    result_dict[1] = enumerate_bridge_with_replaced_SMILES(reaction_id, synthons_table, minimal_synthons_table, bridge_position)
                
                elif mel_type == '3_component':
                    output_type = '3_component'

                    result_dict[1] = enumerator_3_component(synthon_tables_by_position.get(1), minimal_synthons.get(2), 
                                                            minimal_synthons.get(3), rxn_mol, synthon_position = 1)
                    result_dict[2] = enumerator_3_component(minimal_synthons.get(1), synthon_tables_by_position.get(2), 
                                                            minimal_synthons.get(3), rxn_mol, synthon_position = 2)
                    result_dict[3] = enumerator_3_component(minimal_synthons.get(1), minimal_synthons.get(2), 
                                                            synthon_tables_by_position.get(3), rxn_mol, synthon_position = 3)

            else:
                bridge_reaction_flag, bridge_position = check_if_bridge_reaction(minimal_synthons, SYNTHON_LABELS)

                result_dict[1] = enumerator_3_component(synthon_tables_by_position.get(1), minimal_synthons.get(2), 
                                                        minimal_synthons.get(3), rxn_mol, synthon_position = 1)
                result_dict[2] = enumerator_3_component(minimal_synthons.get(1), synthon_tables_by_position.get(2), 
                                                        minimal_synthons.get(3), rxn_mol, synthon_position = 2)
                result_dict[3] = enumerator_3_component(minimal_synthons.get(1), minimal_synthons.get(2), 
                                                        synthon_tables_by_position.get(3), rxn_mol, synthon_position = 3)
                output_type = '3_component'

                # For bridge reactions the bridge synthon enumerated with minimal synthons is not needed
                # Discarding information for this synthon
                if bridge_reaction_flag:
                    del result_dict[bridge_position]
                    output_type = 'bridge'

        enumerated_mel_table = pd.concat(result_dict.values(), ignore_index=True)

        # exporting MEL
        if 'mol' in enumerated_mel_table.columns:
            enumerated_mel_table.drop('mol', axis=1, inplace=True)
            enumerated_mel_table['mel_type'] = output_type

        enumerated_mel_table.to_csv(os.path.join(config["output_folder_path"], 'enumerated_MEL',f'{reaction_id}_{output_type}.csv'), index=False)

        # processing minimal synthons and exporting
        minimal_synthons_df = pd.concat(minimal_synthons.values(), ignore_index=True)
        minimal_synthons_df.SMILES = minimal_synthons_df.mol.apply(Chem.MolToSmiles)
        
        if 'mol' in  minimal_synthons_df.columns:
            minimal_synthons_df.drop('mol', axis=1, inplace=True)

        minimal_synthons_df.to_csv(os.path.join(config["output_folder_path"], 'minimal_synthons',f'{reaction_id}.csv'), index=False)

    except Exception as e:
        with open('error_log.txt','a') as error_file:
            error_file.write(f"{reaction_id}, {e} \n")

def main(config):
    # Setting up the logger
    with open('error_log.txt','w') as error_file:
            error_file.write('\n')
            
    logger =  setup_logger(file_name='log', log_folder=config['output_folder_path'])
    
    logger.info('Starting enumeration')
    start = time.time()

    # reading the input tables
    file_with_reactions = pd.read_csv(config['reaction_file_path'], sep='\t', dtype={'reaction_id_number': str} )

    # getting the reaction ids to process from config or extracting all ids
    reaction_ids_to_process = sorted(get_reaction_ids_to_process(config['custom_reaction_id'], config['synthons_folder_path']))
    #reaction_ids_to_process = get_reaction_ids_to_process(config['reaction_id'], config['synthons_folder_path'])

    print(len(reaction_ids_to_process))

    # creating the output folders
    os.makedirs(os.path.join(config["output_folder_path"], 'enumerated_MEL'), exist_ok=True)
    os.makedirs(os.path.join(config["output_folder_path"], 'minimal_synthons'), exist_ok=True)
    '''
    for reaction_id in reaction_ids_to_process:
        reaction_table = file_with_reactions[file_with_reactions['reaction_id'] == reaction_id].copy()
        process_reaction_worker((reaction_id, reaction_table, config, logger))
    '''
    # creating tasks for processing
    multiprocessing_arguments = [(reaction_id, 
                                file_with_reactions[file_with_reactions['reaction_id']==reaction_id].copy(), 
                                config,
                                logger) 
                                for reaction_id in reaction_ids_to_process]

    # Parallelizing the task execution to workers
    with Pool(processes=config['n_processes']) as pool:
        pool.map(process_reaction_worker, multiprocessing_arguments)

    logger.info(f'Enumeration done in {(time.time()-start) / 60:.2f} minutes')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='config_test.json', help='Path to the configuration file')
    args = parser.parse_args()

    config = read_json_config(args.config)
    main(config)