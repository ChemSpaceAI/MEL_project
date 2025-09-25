# MEL Enumeration Pipeline

This project provides a multi-step pipeline for generating, processing, and enumerating MEL data from custom synthons and reactions

## 🧩 Project Structure
```
MEL_Project/
├── mel_package/                  
│   ├── __init__.py
│   ├── 1_generate.py              
│   ├── 2_process_generation.py              
│   ├── 3_enumerate.py             
│   └── functions/              
│       └── ...                  
│
├── configs/                   
│   └── 1_Generation.json
|   └── 2_Process_Generation.json
|   └── 3_Enumeration.json
│
├── MAIN_data/                       
│   ├── FINAL_data/
│   ├── Generated_MEL_on_FINAL_data/
│   ├── Test_Input_1/
│   └── Test_Output_1/
│
├── scripts/                  
│   └── generate.sh
│   └── rocess_generation.sh
│   └── enumeration.sh
│
├── notebooks/                 
│   └── preprocess.ipynb
│
├── functional_jsons/                   
│   └── Filter_Config.json
|   └── reaction_id_type_map.json
|   └── reaction_id_with_counts.json
│
├── requirements.txt
├── README.md

```
---

## 🧪 1. Generate MEL

Takes a reaction table and synthon libraries to generate MELs.

### Comments 

custom_reaction_id: set path with file that have only reaction id column,
                         If set to false, filtering is skipped.

### 🧾 JSON Config Example (1_Generation.json)
```json
{
    "synthons_folder_path": "MAIN_data/SYNTHONS_by_reaction_id_modified_for_MEL",   // path to Folder with prepared synthons
    "reaction_file_path": "MAIN_data/REACTIONS_for_MEL.tsv",                   // File with reaction SMARTS
    "external_minimal_synthons_path": "MAIN_data/MIN_caps_for_generation",          // path tp Folder with CAPS

    "output_folder_path": "Generated_MEL",                                          

    "n_processes": 55,                                                             
    "custom_reaction_id": false,                                                    

    "consequent_processing": false,                                                 // (⚠️ usually do not change)
    "id_mapper_path": "functional_jsons/reaction_id_type_map.json"                  // (⚠️ usually do not change)
}

```
### Run
```bash
bash scripts/generation.sh 
```
---
## 🧹 2. Process MELs

Processes generated MEL files into manageable chunks by MEL type.


### Comments 

### 🧾 JSON Config Example (2_Process_Generation.json)

```json
{
    "input_dir": "Generated_MEL/enumerated_MEL",
    "chunk_size": 50000,
    "mel_types": ["2_component", "3_component", "bridge"],
    "synthon_counts_with_bridge_position":"functional_jsons/reaction_id_with_counts.json"
}
```
### Run
```bash
bash scripts/process_generation.sh 
```
---
## 🔄 3. Enumerate MEL

Fully enumerates final molecules from MEL structures.

### Comments 

mel_type: Type of MEL structure. Possible values:
  - "2_component"
  - "3_component"
  - "bridge"

iteration_level: Enumeration depth. Possible values:
  - 1 or 2

path_to_input_csv: The input csv must contain a mel_synthon_id column.

filter_config_file_path: If set to a valid config file, filtering will be applied.
                         If set to false, filtering is skipped.

### 🧾 JSON Config Example (3_Enumeration.json)
```json
{    
    "syntons_folder_path": "main_data/SYNTHONS_by_reaction_id_modified_for_MEL",
    
    "reactions_file_path": "MAIN_data/REACTIONS_for_MEL.tsv",

    "minimal_caps_folder_path": "combined_all/MIN_caps_for_enumeration",

    "filter_config_file_path": "main_data/Filter_Config.json",

    "output_folder_path": "test_2/Test_3_comp_12M",
    "path_to_input_csv": "test_2/test_all_3comp_selected.csv",

    "mel_type":"3_component",
    "iteration_level": 2,

    "N_cores":30,
    "N_compounds_to_Enumerate": 1000000,

    "synthon_counts_with_bridge_position":"functional_jsons/Reaction_id_with_counts.json"   // (⚠️ usually do not change)
}

```

### Run
```bash
bash scripts/enumeration.sh
```

### CSM_id generation: 
`CS_t<mel_type>_f<fragments>_l<lengths>_o<order>_e0`

*Example*: CS_t2_f244_l14_o123_e0

CS – fixed prefix for Chemspace.

`t<mel_type>` – MEL type from the dataset:  
- `2` – two-component MEL  
- `3` – three-component MEL  
- `B` – bridge-type MEL  

`f<fragments>` – number of fragments, predicted from the MEL SMILES using  `predict_the_amount_of_fragments_by_for_MEL_SMILES()`.  

`l<lengths>` – number of ids in `full_mel_synthon_id`.  

`o<order>` – row position (1-based index) in the processed dataset.  

`e0` – stands for MEL generation 


### CSM_id iteration 1: 
`CS_t<mel_type>_f<fragments>_l<lengths>_o<order>_e0` + `f<fragments>_l<lengths>_o<order>_e1`

`f<fragments>` – number of fragments, predicted from the MEL SMILES using  `predict_the_amount_of_fragments_by_for_MEL_SMILES()`. (0 for 2 components)

`l<lengths>` – number of ids in `full_mel_synthon_id`.  

`o<order>` – row position (1-based index) in the processed dataset.  

`e1` – stands for enumeration iteration 1.

### CSM_id iteration 2 : 
`CS_t<mel_type>_f<fragments>_l<lengths>_o<order>_e0_f<fragments>_l<lengths>_o<order>_e1` + `l<lengths>_o<order>_e2`

`l<lengths>` – number of ids in `full_mel_synthon_id`.  

`o<order>` – row position (1-based index) in the processed dataset.  

`e2` for enumeration iteration 2.


