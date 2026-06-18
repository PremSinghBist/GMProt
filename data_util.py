import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter

import seqs_prott5 as prott5
from extract_structural_features import get_contact_map
import physiochem_feature_extractor as PFE
from blosum62 import load_blosum62_features
import position_encoding_extractor as PEE 
import position_aware_features as PAF


EMB_DIM = 1024
#E.coli files
CONTACT_MAP_FILE = "data/ecoli_contact_map.csv"

EMBEDDING_FILE_ECOLI_PROTT5 = "data/embed/prott5_ecoli_residue_level.npz"
EMBEDDING_FILE_ECOLI_PROTBERT = "data/embed/protbert_ecoli_residue_level.npz"
EMBEDDING_FILE_ECOLI_ESM2 = "data/embed/esm2_ecoli_residue_level.npz"
EMBEDDING_FILE_P_AERUGINOSA_PROTT5 = "./data/embed/prott5_p_aeruginosa_residue_level.npz"

PHYSIO_CHEM_FILE_ECOLI = 'data/ecoli_phyiochem.csv'
SINUSOIDAL_ENCODING_FILE_ECOLI = './data/ecoli_sinusoidal_encoding.csv'
BLOSUM62_FILE_ECOLI  = "data/ecoli_blosum62_features.csv"
DATASET_CSV_ECOLI = "./data/ecoli_dataset.csv"

PHYSIO_CHEM_FILE_P_AERUGINOSA = 'data/p_aeruginosa_phyiochem.csv'
SINUSOIDAL_ENCODING_FILE_P_AERUGINOSA = './data/p_aeruginosa_sinusoidal_encoding.csv'
BLOSUM62_FILE_P_AERUGINOSA  = "data/p_aeruginosa_blosum62_features.csv"


DATASET_PATH_ECOLI_PROTT5 = "data/five_fold_ecoli/ecoli_prott5_based_datasets.pkl"
DATASET_PATH_ECOLI_PROTBERT = "./data/five_fold_ecoli/ecoli_protbert_based_datasets.pkl"
DATASET_PATH_ECOLI_ESM2 = "data/five_fold_ecoli/ecoli_esm2_based_datasets.pkl"

DATASET_PATH_P_AERUGINOSA_PROTT5_70_15 = "data/five_fold_p_aeruginosa/p_aeruginosa_prott5_based_datasets_70_15_15.pkl"
DATASET_PATH_P_AERUGINOSA_PROTT5_80_10 = "data/five_fold_p_aeruginosa/p_aeruginosa_prott5_based_datasets_80_10_10.pkl"
DATASET_CSV_P_AERUGINOSA = "./data/p_aeruginosa_dataset.csv"

DATASET_PATH_SAUREUS_PROTT5_80_10 = "data/five_fold_s_aureus/s_aureus_prott5_based_datasets_80_10_10.pkl"
DATASET_PATH_SAUREUS_PROTT5_70_15 = "data/five_fold_s_aureus/s_aureus_prott5_based_datasets_70_15_15.pkl"

# DATSET_SIZE = [3259, 815, 719] ##Ecoli train, val, test | total 4793 
# DATSET_SIZE = [4469, 248, 249]  #Original P. Aeruginosa size 4966 total ,  80:10:10 
#DATSET_SIZE = [3478, 745, 743] #Original P. Aeruginosa  4966 with 70:15:15 split
# DATSET_SIZE = [2178, 272, 272] #2722 Cleaned P. Aeruginosa with 80:10:10 split | 
# DATSET_SIZE = [1905, 409, 408] #2722 Cleaned P. Aeruginosa with 70:15:15 split
# DATSET_SIZE = [3067, 657, 657] #train, val, test | total 4381(70:15:15) S_aureus
DATSET_SIZE = [3505, 438, 438] #train, val, test | total 4381(80:10:10) S_aureus

POSITION_AWARE_FILE_ECOLI = "./data/position_aware_features.csv" #Currently not used due to performance issues


#Staphylococcus Aureus files
SINUSOIDAL_ENCODING_FILE_SAUREUS = './data/s_aureus_sinusoidal_encoding.csv'
PHYSIO_CHEM_FILE_SAUREUS = 'data/s_aureus_phyiochem.csv'
EMBEDDING_FILE_SAUREUS_PROTT5 = "data/embed/prott5_s_aureus_residue_level.npz"
BLOSUM62_FILE_SAUREUS = "data/s_aureus_blosum62_features.csv"
DATASET_CSV_SAUREUS = "./data/s_aureus_dataset.csv"




def stratified_train_val_test_splits(
    features,
    train_size=DATSET_SIZE[0],
    val_size=DATSET_SIZE[1],
    test_size=DATSET_SIZE[2],
    seed=42,
    n_bins=5,
    n_datasets=5,
    max_attempts=100
):  
    '''
    Features tuple :  emb, , physio_feature, blosum_feature, sinu_feature, seq, mic
    '''
    assert train_size + val_size + test_size == len(features)

    mic_values = np.array([item[5] for item in features])
    indices = np.arange(len(features))

    kbd = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile"
    )

    mic_binned = (
        kbd.fit_transform(mic_values.reshape(-1, 1))
        .astype(int)
        .ravel()
    )

    datasets = []
    attempts = 0
    split_seed = seed

    while len(datasets) < n_datasets and attempts < max_attempts:
        attempts += 1

        sss_1 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            test_size=val_size + test_size,
            random_state=split_seed
        )

        train_idx, temp_idx = next(
            sss_1.split(indices, mic_binned)
        )

        # ---- Validate train bins ----
        if min(Counter(mic_binned[train_idx]).values()) < 2:
            split_seed += 1
            continue

        # ---- Validate temp bins ----
        temp_bins = mic_binned[temp_idx]
        if min(Counter(temp_bins).values()) < 2:
            split_seed += 1
            continue

        sss_2 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_size,
            test_size=test_size,
            random_state=split_seed + 10_000
        )

        val_sub_idx, test_sub_idx = next(
            sss_2.split(temp_idx, temp_bins)
        )

        datasets.append((
            [features[i] for i in train_idx],
            [features[i] for i in temp_idx[val_sub_idx]],
            [features[i] for i in temp_idx[test_sub_idx]]
        ))

        split_seed += 1

    if len(datasets) < n_datasets:
        raise RuntimeError(
            f"Only generated {len(datasets)} valid splits "
            f"after {attempts} attempts."
        )

    return datasets

def normalize(features, mean=None, std=None, eps=1e-8):
    """
    Z-score normalization for BLOSUM features.

    features: np.ndarray of shape (N, [?])
    mean, std: computed on training set and reused for val/test

    Preserves realative scales between different feature dimensions.
    Returns:
        features_norm: normalized features
        mean: mean used for normalization (per feature dimension). feautres dim is 20, there will be 20 means.
        std: std used for normalization (per feature dimension)
    """
    if mean is None:
        mean = features.mean(axis=0)   
    if std is None:
        std = features.std(axis=0)    

    features_norm = (features - mean) / (std + eps)
    return features_norm, mean, std

def load_embeddings(filepath):
    data = np.load(filepath, allow_pickle=True)
    original_seqs = data["original_sequences"]
    processed_seqs = data["processed_sequences"]
    embeddings = data["embeddings"]

    print(f"Loaded {len(original_seqs)} sequences")
    print(f"Embeddings array shape: {embeddings.shape}")

    return original_seqs, processed_seqs, embeddings

# ============================================================
# DATA
# ============================================================
def load_features(normalize_features=True, embedding_file=None, blosum_csv='', physio_csv='', sinusoidal_csv='', dataset_csv=''):
    '''
    Returns:
    features: List[
        emb,
        physio_feature,
        blosum_feature,
        sinu_feature,
        seq,
        mic
    ]
        
        Note CONTACT_MAP_FILE is obsolete and not used (Structural features do not improve performance)
    '''
    # df = pd.read_csv(CONTACT_MAP_FILE)
    df = pd.read_csv(dataset_csv) #eg: p_aeruginosa_dataset.csv
    mic_dict = dict(zip(df['sequence'], df['value']))
    
    
    seqs, _, embs = load_embeddings(embedding_file) #ProtT5/ProtBert/ESM2 embeddings
    print(f"Embeeding 1st shape: {np.array(embs[0]).shape}")

    blosum_dict = load_blosum62_features(csv_path=blosum_csv) #20 features
    physio_dict = PFE.load_physio_features_as_numpy_all(physio_csv) #32 features


    # Load sinusoidal positional encodings | dict[Seq1: np.ndarray (32,), ...]
    sinusoidal_encoding_dict = PEE.load_sinusoidal_encoding(sinusoidal_csv)

    #postion aware features | Performance adverse
    # position_aware_dict = PAF.load_feature_csv_as_dict(POSITION_AWARE_FILE)

    features = []
    physio_list = []
    blosum_list = []
    sinusoidal_encoding_list = []
    # position_aware_list = []

    # -------------------------------
    # Load raw features
    # -------------------------------
    for seq, emb in zip(seqs, embs):
        # cm, mic = get_contact_map(seq, df)#mic values are normalized between 0 and 1
        
        if seq not in mic_dict:
            raise ValueError(f"{seq} missing in dataset")
        
        mic = mic_dict[seq]  # 'mic' value exist with name 'value' in dataset csv file
        

        emb = np.asarray(emb, np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(-1, EMB_DIM)

        if seq not in physio_dict or seq not in blosum_dict or seq not in sinusoidal_encoding_dict:
            raise ValueError(f"Sequence {seq} missing in physio or blosum or sinusoidal or position aware features.")

        physio_feature = physio_dict[seq].astype(np.float32)
        blosum_feature = blosum_dict[seq].astype(np.float32)
        sinu_feature = sinusoidal_encoding_dict[seq].astype(np.float32)
        # position_aware_feature = position_aware_dict[seq].astype(np.float32)

        # features.append([emb, cm, physio_feature, blosum_feature, sinu_feature, seq, mic])
        features.append([emb, physio_feature, blosum_feature, sinu_feature, seq, mic])
        physio_list.append(physio_feature)
        blosum_list.append(blosum_feature)
        sinusoidal_encoding_list.append(sinusoidal_encoding_dict[seq])
        # position_aware_list.append(position_aware_feature)  

    # -------------------------------
    # Normalize (replace in features)
    # -------------------------------
    if normalize_features:
        physio_arr = np.stack(physio_list)   # (N, Dp)
        blosum_arr = np.stack(blosum_list)   # (N, 20)
        sino_arr  = np.stack(sinusoidal_encoding_list)  # (N, 32)
        # position_aware_arr = np.stack(position_aware_list)  # (N, 60)

        physio_norm, physio_mean, physio_std = normalize(physio_arr)
        blosum_norm, blosum_mean, blosum_std = normalize(blosum_arr)
        sino_norm, sino_mean, sino_std = normalize(sino_arr) 
        # position_aware_norm, position_aware_mean, position_aware_std = normalize(position_aware_arr) 
        
        # Replace raw values with normalized ones
        for i in range(len(features)):
            features[i][1] = physio_norm[i] #Physio index 1
            features[i][2] = blosum_norm[i] #Blosum index 2
            features[i][3] = sino_norm[i]   # Sinusoidal index 3
            # features[i][4] = position_aware_norm[i] # Position aware index 4
    
    # Logging
    # -------------------------------
    sample = features[0]
    print(f"Loaded features for {len(features)} sequences.")
    print("****Sample feature shapes:****")
    print(f"  Embedding: {sample[0].shape}")
    # print(f"  Contact Map: {sample[1].shape}") #Contact map is not used
    print(f"  Physio-Chemical (normalized): {sample[1].shape}")
    print(f"  BLOSUM62 (normalized): {sample[2].shape}")
    print(f"  Sinusoidal PE (normalized) shape: {sample[3].shape}")
    print(f"  Sequence: {sample[4]}")
    print(f"  MIC: {sample[5]}")


    return features


def save_datasets(save_path=None, embedding_file=None, blosum_csv='', physio_csv='', sinusoidal_csv='', dataset_csv=''):
    """
    datasets: List of (train_set, val_set, test_set)
    filepath: str, e.g. 'datasets.npz'
    """
    features = load_features(embedding_file=embedding_file, blosum_csv=blosum_csv, physio_csv=physio_csv, sinusoidal_csv=sinusoidal_csv, dataset_csv=dataset_csv)
    datasets = stratified_train_val_test_splits(
        features,
        seed=42,
        n_bins=5,
        n_datasets=5
    )
    
    with open(save_path, "wb") as f:
        pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Datasets saved to {save_path}.")

def load_datasets(datasets_index=[0, 1, 2, 3, 4], dataset_path=''):
    """
    datasets_index: Which Dataset to load 
    0: First dataset 1: Second dataset and so on.
    Returns: List [(train_set, val_set, test_set), (...)]
    """
    with open(dataset_path, "rb") as f:
        datasets = pickle.load(f)


    max_idx = len(datasets) - 1
    for i in datasets_index:
        if i < 0 or i > max_idx:
            raise ValueError(
                f"Invalid dataset index {i}. Valid range: [0, {max_idx}]"
            )

    print(f"Datasets loaded from {dataset_path}.")
    return [datasets[i] for i in datasets_index]

def save_results_table(results, filename="metrics_results.csv"):
    """
    Save evaluation metrics to a tabular CSV file.

    Args:
        results (list of dict): List of metric dictionaries
        filename (str): Output CSV filename
    """
    df = pd.DataFrame(results)

    # Optional: add run index
    df.insert(0, "Run", range(1, len(df) + 1))

    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")

    return df

    

if __name__ == "__main__":
    # save_datasets(DATASET_PATH_ECOLI_PROTT5, embedding_file=EMBEDDING_FILE_PROTT5)
    # save_datasets(DATASET_PATH_ECOLI_PROTBERT, embedding_file=EMBEDDING_FILE_PROTBERT)
    # save_datasets(save_path=DATASET_PATH_ECOLI_ESM2, embedding_file=EMBEDDING_FILE_ESM2)
    
    '''save_datasets(save_path=DATASET_PATH_P_AERUGINOSA_PROTT5,
                    embedding_file=EMBEDDING_FILE_P_AERUGINOSA_PROTT5,
                    blosum_csv=BLOSUM62_FILE_P_AERUGINOSA,
                    physio_csv=PHYSIO_CHEM_FILE_P_AERUGINOSA,
                    sinusoidal_csv=SINUSOIDAL_ENCODING_FILE_P_AERUGINOSA,
                    dataset_csv=DATASET_CSV_P_AERUGINOSA
                  )'''
    
    '''save_datasets(save_path=DATASET_PATH_ECOLI_PROTT5,
                    embedding_file=EMBEDDING_FILE_ECOLI_PROTT5,
                    blosum_csv=BLOSUM62_FILE_ECOLI,
                    physio_csv=PHYSIO_CHEM_FILE_ECOLI,
                    sinusoidal_csv=SINUSOIDAL_ENCODING_FILE_ECOLI,
                    dataset_csv=DATASET_CSV_ECOLI
                  )'''
    
    '''save_datasets(save_path= DATASET_PATH_ECOLI_PROTBERT,
                    embedding_file=EMBEDDING_FILE_ECOLI_PROTBERT,
                    blosum_csv=BLOSUM62_FILE_ECOLI,
                    physio_csv=PHYSIO_CHEM_FILE_ECOLI,
                    sinusoidal_csv=SINUSOIDAL_ENCODING_FILE_ECOLI,
                    dataset_csv=DATASET_CSV_ECOLI
                  )'''
        
    '''save_datasets(save_path= DATASET_PATH_ECOLI_ESM2,
                embedding_file=EMBEDDING_FILE_ECOLI_ESM2,
                blosum_csv=BLOSUM62_FILE_ECOLI,
                physio_csv=PHYSIO_CHEM_FILE_ECOLI,
                sinusoidal_csv=SINUSOIDAL_ENCODING_FILE_ECOLI,
                dataset_csv=DATASET_CSV_ECOLI
                )'''
    
    
   
    
    save_datasets(save_path= DATASET_PATH_SAUREUS_PROTT5_80_10,
            embedding_file=EMBEDDING_FILE_SAUREUS_PROTT5,
            blosum_csv=BLOSUM62_FILE_SAUREUS,
            physio_csv=PHYSIO_CHEM_FILE_SAUREUS,
            sinusoidal_csv=SINUSOIDAL_ENCODING_FILE_SAUREUS,
            dataset_csv=DATASET_CSV_SAUREUS
            )
    
    
    
    datasets = load_datasets(dataset_path=DATASET_PATH_SAUREUS_PROTT5_80_10)
    
    '''load_features(embedding_file=EMBEDDING_FILE_P_AERUGINOSA_PROTT5,
                  blosum_csv=BLOSUM62_FILE_P_AERUGINOSA, 
                  physio_csv=PHYSIO_CHEM_FILE_P_AERUGINOSA,
                  sinusoidal_csv=SINUSOIDAL_ENCODING_FILE_P_AERUGINOSA, 
                  dataset_csv=DATASET_CSV_P_AERUGINOSA)'''


