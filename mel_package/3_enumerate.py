# === Standard Library Imports ===
import os, sys, time, json, argparse, csv
import sqlite3 as sql
from pathlib import Path
from multiprocessing import Process, Queue, Event
from multiprocessing.queues import Empty

# === Third-Party Library Imports ===
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, PandasTools
from rdkit import RDLogger

# === Suppress RDKit Warnings ===
RDLogger.DisableLog('rdApp.')

# === Project Specific Imports (Custom Functions) ===
from mel_package.functions.Utils import (
    setup_logger,
    read_json_config,
    log_script_info,
    setup_report_logger,
    mel_id_generator,
    mel_id_generator_with_logger, 
    predict_the_amount_of_fragments_for_MEL_SMILES_enumeration
)

from mel_package.functions.MEL_ID_processors import (
    mel_ID_enumeration_for_2_component,
    mel_ID_enumeration_for_3_component_iteration_1,
    mel_ID_enumeration_for_bridge, 
    mel_ID_enumeration_for_2_level_iteration
)


def main(config: dict) -> None:
    """
    Main function for preprocessing, selecting the appropriate enumeration function,
    and running the MEL enumeration workflow.

    Args:
        config (dict): Configuration dictionary loaded from JSON.
    """

    start_of_script = time.time()

    # 1. Setup output folders
    output_folder = Path(config['output_folder_path'])
    output_folder.mkdir(parents=True, exist_ok=True)

    # 2. Setup logging
    logger = setup_report_logger(output_folder=config['output_folder_path'])
    status_logger = setup_logger('Status', config['output_folder_path'])
    fragments_logger = setup_logger('Fragments', config['output_folder_path'])
    log_script_info(logger, config)

    logger.info('Starting preprocessing...')

    # 3. Load Reactions File
    try:
        file_of_reactions = pd.read_csv(config['reactions_file_path'], sep='\t')
        if not {'reaction_id', 'Reaction'}.issubset(file_of_reactions.columns):
            raise ValueError("Reactions file must contain 'reaction_id' and 'Reaction' columns.")

        file_of_reactions['Reaction_MOL'] = file_of_reactions['Reaction'].apply(AllChem.ReactionFromSmarts)
        file_of_reactions = file_of_reactions[['reaction_id', 'Reaction', 'Reaction_MOL', 'bridge_position']]
        logger.info(f"Loaded {len(file_of_reactions)} reactions.")
    
    except Exception as e:
        logger.error(f"Failed to load or process reactions file: {e}")
        sys.exit(1)

    # 4. Load Tuple MEL Synthon IDs and CSM_ID from CSV
    try:
        csv_file = config['path_to_input_csv']
        input_df = pd.read_csv(csv_file)

        # Список списків [mel_synthon_id, CSM_id]
        tuple_input = input_df[['mel_synthon_id', 'CSM_id']].values.tolist()

        logger.info(f"Loaded {len(tuple_input)} synthons from {csv_file}")

    except Exception as e:
        logger.error(f"Failed to load synthons from CSV: {e}")
        sys.exit(1)


    # 5. Load Filters Config 
    filters_config: dict = {}
    if config.get('filter_config_file_path'):
        try:
            filters_config = read_json_config(config['filter_config_file_path'])
            logger.info("Loaded filter configuration.")
        except Exception as e:
            logger.warning(f"Could not load filters config: {e}")
            filters_config = {}

    # 6. Select and Execute the Appropriate Enumeration Function
    mel_type = config.get("mel_type")
    iteration_level = config.get("iteration_level")

    # Mapping (mel_type, iteration_level) to enumeration function
    function_selector = {
        ('2_component', 1): mel_ID_enumeration_for_2_component,
        ('3_component', 1): mel_ID_enumeration_for_3_component_iteration_1,
        ('3_component', 2):  mel_ID_enumeration_for_2_level_iteration,
        ('bridge', 1): mel_ID_enumeration_for_bridge, 
        ('bridge', 2):  mel_ID_enumeration_for_2_level_iteration,
    }

    enumerate_function = function_selector.get((mel_type, iteration_level))

    if enumerate_function is None:
        logger.error(f"Unsupported MEL type {mel_type} and iteration level {iteration_level}.")
        sys.exit(1)

    logger.info(f"Selected MEL Type: {mel_type}, Iteration Level: {iteration_level}")

    # Execute Enumeration
    results = run_parallelized_enumerations(
        config=config,
        logger=status_logger,
        fragments_logger=fragments_logger,
        file_of_reactions=file_of_reactions,
        tuple_input=tuple_input,
        filters_config=filters_config,
        enumeration_function=enumerate_function,
    )

    # Final Reporting
    logger.info(f"{'-' * 100}")
    logger.info(f"Total number of lines written: {results:,}")
    expected = config['N_compounds_to_Enumerate']
    if results < expected:
        logger.warning(f"Only {results:,} compounds were enumerated, which is less than the target of {expected:,}!!!")
    else:
        logger.info(f"Enumeration goal reached: {results:,} compounds (target was {expected:,})")

    logger.info(f"Script completed in {(time.time() - start_of_script) / 60:.2f} minutes.")
    logger.info(f"{'-' * 100}")
    print('Done')


def run_parallelized_enumerations(config: dict, logger, fragments_logger, file_of_reactions, tuple_input: list, filters_config: dict, enumeration_function) -> int:
    """
    Handles parallelized enumeration with multiprocessing.

    Args:
        config (dict): Configuration dictionary.
        logger: Main logger instance.
        file_of_reactions (DataFrame): DataFrame with reaction data.
        tuple_input (list): List of tuples (mel_synthon_id, CSM_id).
        filters_config (dict): Filter configuration dictionary.
        enumeration_function (function): Function to perform enumeration.

    Returns:
        int: Total number of compounds written.
    """
    logger.info("Starting parallelized enumeration...")

    n_jobs = config.get('N_cores')
    if not n_jobs:
        sys.exit("Error: No number of cores specified.")

    result_queue = Queue(maxsize=1000)
    input_queue = Queue(maxsize=1000)
    count_queue = Queue(maxsize=10)
    stop_event = Event()

    N_compounds_to_Enumerate = config['N_compounds_to_Enumerate']
    output_path = os.path.join(config['output_folder_path'], 'enumeration_results.csv')

    logger.info(f"Generating {N_compounds_to_Enumerate} compounds using {n_jobs} cores.")

    mel_id_gen = mel_id_generator_with_logger(tuple_input, fragments_logger)

    # Prefill input queue
    logger.info(f"Prefilling input queue with {10 * n_jobs} mel_ids...")
    try:
        for _ in range(10 * n_jobs):
            input_queue.put(next(mel_id_gen))
        logger.info(f"Prefilled {10 * n_jobs} MEL IDs.")
    except StopIteration:
        logger.warning("Ran out of MEL IDs during prefill.")

    # Start collector
    collector_proc = Process(target=collector, args=(result_queue, stop_event, N_compounds_to_Enumerate, output_path, logger, count_queue,config))
    collector_proc.start()
    logger.info(f"Collector started. PID: {collector_proc.pid}")

    # Start workers
    workers = []
    for idx in range(n_jobs):
        p = Process(
            target=worker,
            args=(input_queue, result_queue, stop_event, logger, config, filters_config, file_of_reactions, enumeration_function)
        )
        p.start()
        workers.append(p)
        logger.info(f"Worker {idx+1}/{n_jobs} started. PID: {p.pid}")

    # Refill input queue dynamically
    logger.info("Dynamically refilling input queue...")
    while not stop_event.is_set():
        if input_queue.empty():
            try:
                input_queue.put(next(mel_id_gen))
            except StopIteration:
                logger.warning("No more MEL IDs available to enqueue. All MEL IDs have been enqueued.")
                break
    
    # Wait for collector
    logger.info("Waiting for collector to finish...")
    stop_event.set()
    collector_proc.join()
    logger.info(f"Collector finished. Exit code: {collector_proc.exitcode}")

    # Drain and close input queue
    while not input_queue.empty():
        try:
            input_queue.get_nowait()
        except Empty:
            break
    input_queue.close()
    logger.info("Input queue drained and closed.")

    # Drain and close result queue
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Empty:
            break
    result_queue.close()
    logger.info("Result queue drained and closed.")

    # Join or terminate workers
    logger.info("Waiting for workers to be properly closed...")
    for idx, p in enumerate(workers):
        p.join(timeout=0.1)
        if p.is_alive():
            #logger.warning(f"Worker PID {p.pid} still alive after timeout. Terminating...")
            p.terminate()
            p.join()
            #logger.info(f"Worker {idx+1}: PID={p.pid}, still alive after timeout. Terminated. Exit Code={p.exitcode}")
        else:
            logger.info(f"Worker {idx+1}: PID={p.pid}, already closed. Finished. Exit Code={p.exitcode} ")

    logger.info("All worker processes joined successfully.")


    final_count = count_queue.get()
    count_queue.close()
    logger.info(f"Final number of enumerated compounds collected: {final_count:,}")

    return final_count

def worker(
    input_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    logger,
    config: dict,
    filters_config: dict,
    file_of_reactions: pd.DataFrame,
    enumeration_function
) -> None:
    """
    Worker process that pulls MEL IDs from the input queue, enumerates molecules, and puts results into the result queue.

    Args:
        input_queue (Queue): Queue with MEL IDs to process.
        result_queue (Queue): Queue to collect enumeration results.
        stop_event (Event): Event to signal workers to stop.
        logger: Logger instance for status updates.
        config (dict): Configuration parameters.
        filters_config (dict): Filters applied during enumeration.
        file_of_reactions (pd.DataFrame): Reactions to use in enumeration.
        enumeration_function (function): Function performing the enumeration.
    """
    #logger.info(f"Worker PID {os.getpid()} started.  ")

    while not stop_event.is_set():
        try:
            tuple_input = input_queue.get(timeout=0.1)
            mel_id, CSM_id = tuple_input
        except Empty:
            continue  # Keep checking stop_event

        molecules_dict = enumeration_function(
            mel_id, CSM_id, config, logger, file_of_reactions, filters_config
        )
        result_queue.put(molecules_dict)

    #logger.info(f"Worker PID {os.getpid()} received stop signal.  ")
    result_queue.close()

def collector(
    result_queue: Queue,
    stop_event: Event,
    N_compounds_to_Enumerate: int,
    output_path: str,
    logger,
    count_queue,
    config,
) -> None:
    """
    Collects results from the result queue, merges them, and saves to a CSV.

    Args:
        result_queue (Queue): Queue containing the enumerated results.
        stop_event (Event): Event to signal the collection process to stop.
        N_compounds_to_Enumerate (int): The number of compounds to collect before stopping.
        output_path (str): Path to save the final results CSV.
        logger: Logger instance to log status updates.
    """
    results = {}
    next_log_threshold = 100_000

    while len(results) < N_compounds_to_Enumerate:
        try:
            molecules_dict = result_queue.get(timeout=0.1)
            if molecules_dict is None:
                continue
            if not molecules_dict:  # Check if the dictionary is empty
                continue
        except Empty:
            if stop_event.is_set():
                break  # Stop if stop_event is triggered
            continue

        # Merge results and update synthon IDs
        for result_smile, value_list in molecules_dict.items():

            if result_smile not in results:
                # Initialize synthon ID as a set
                synthon_id_set = {value_list[0]}
                CSM_id_set = {value_list[1]}

                results[result_smile] = [synthon_id_set, CSM_id_set] + value_list[2:]
            else:
                # Update the synthon ID set
                results[result_smile][0].add(value_list[0])
                results[result_smile][1].add(value_list[1])

        current_N_enum = len(results)
        if current_N_enum >= next_log_threshold:
            logger.info(f"Collected {current_N_enum:,} molecules so far...  ")
            next_log_threshold += 100_000

        if current_N_enum >= N_compounds_to_Enumerate:
            stop_event.set()  # Stop once the required compounds are collected
            break

    # Save results to CSV
    logger.info(f"Saving {current_N_enum:,} molecules to {output_path}...  ")
    save_results(results, output_path, logger, config)
    logger.info(f"Collector finished. Total molecules collected: {current_N_enum:,}  ")

    count_queue.put(len(results))  # Send final number of collected molecules

def save_results(results: dict, output_path: str, logger, config) -> None:
    """
    Saves the results dictionary to a CSV file.

    Args:
        results (dict): Dictionary containing SMILES strings as keys and a list of values for each key.
        output_path (str): The path where the CSV file should be saved.

    Writes a CSV with the following columns: ['Smiles', 'mel_synthon_id', "CSM_ID", 'Charged_Smiles', 'synthon_1',
    'synthon_2', 'synthon_3', 'HAC', 'MW', 'HBD', 'HBA', 'LogP', 'RotB'].
    """

    header = ['Smiles', 'mel_synthon_id', "CSM_id", 'Charged_Smiles', 'synthon_1',
              'synthon_2', 'synthon_3', 'HAC', 'MW', 'HBD', 'HBA', 'LogP', 'RotB']
    

    synthons_counts_with_bridge_position=config.get("synthon_counts_with_bridge_position")
    logger.info("Modifying CSM_id to include counts with bridge position...")
    try:
        with open(synthons_counts_with_bridge_position, 'r') as f:
            synthon_counts_with_bridge_position = json.load(f)
    except Exception as e:
            logger.error(f"Failed to load synthon counts with bridge position: {e}")
            return
        

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write the header row

        # Write each row of data
        order_counter = 0
        for smile, values in results.items():

            order_counter += 1

            synthon_id_set = values[0]  # This should be a set
            full_synthon_id_str = "|".join(sorted(synthon_id_set))

        
            # Generate new CSM_id 
            CSM_id_set = values[1]  # This should be a set
            CSM_id_new_set = set()
            len_synthon_id_set = len(synthon_id_set)

            fragments_amount = predict_the_amount_of_fragments_for_MEL_SMILES_enumeration(full_synthon_id_str, config.get("mel_type"), config.get("iteration_level",1), synthon_counts_with_bridge_position)

            for CSM_id_first_part in CSM_id_set:

                    CSM_id_str = f"{CSM_id_first_part}_f{fragments_amount}_l{len_synthon_id_set}_o{order_counter}_e{config.get('iteration_level',1)}"
                    CSM_id_new_set.add(CSM_id_str)

            full_CSM_id_str = "|".join(sorted(CSM_id_new_set))


            # Write the row
            row = [smile, full_synthon_id_str, full_CSM_id_str] + values[2:]
            writer.writerow(row)


    # Log the size of the file for monitoring
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    logger.info(f"Results saved to {output_path}. File size: {file_size:.2f} MB")


    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for MEL enumeration.')
    
    parser.add_argument(
        '--config',
        type=str,
        default='Config_For_Enumeration_3_comp_iter_2.json',
        help='Path to the configuration file (default: Config_For_Enumeration.json)'
    )
    
    args = parser.parse_args()
    config: dict = read_json_config(args.config)
    main(config)
