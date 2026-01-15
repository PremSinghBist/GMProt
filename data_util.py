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
CONTACT_MAP_FILE = "data/contact_map.csv"
EMBEDDING_FILE = "data/prott5/prot_t5_xl_uniref50/prott5_residue_level.npz"
BLOSUM62_FILE = "data/blosum62_features.csv"
SINUSOIDAL_ENCODING_FILE = './data/sinusoidal_encoding.csv'
POSITION_AWARE_FILE = "./data/position_aware_features.csv"

DATASET_PATH = "data/five_fold/datasets.pkl"

def stratified_train_val_test_splits(
    features,
    train_size=3259,
    val_size=815,
    test_size=719,
    seed=42,
    n_bins=5,
    n_datasets=5,
    max_attempts=100
):  
    '''
    Features tuple :  emb, cm, physio_feature, blosum_feature, sinu_feature, seq, mic
    '''
    assert train_size + val_size + test_size == len(features)

    mic_values = np.array([item[6] for item in features])
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

# ============================================================
# DATA
# ============================================================
def load_features(normalize_features=True):
    '''
    Returns:
        features: List [emb, cm, physio_feature, blosum_feature, sinu_feature, seq, mic]
        stats: dict with normalization statistics
    '''
    df = pd.read_csv(CONTACT_MAP_FILE)
    seqs, _, embs = prott5.load_embeddings(EMBEDDING_FILE)

    blosum_dict = load_blosum62_features(csv_path=BLOSUM62_FILE) #20 features
    physio_dict = PFE.load_physio_features_as_numpy_all() # #32+9 extra n/c bias=41 features


    # Load sinusoidal positional encodings | dict[Seq1: np.ndarray (32,), ...]
    sinusoidal_encoding_dict = PEE.load_sinusoidal_encoding(SINUSOIDAL_ENCODING_FILE)

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
        cm, mic = get_contact_map(seq, df)#mic values are normalized between 0 and 1

        emb = np.asarray(emb, np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(-1, EMB_DIM)

        if seq not in physio_dict or seq not in blosum_dict or seq not in sinusoidal_encoding_dict:
            raise ValueError(f"Sequence {seq} missing in physio or blosum or sinusoidal or position aware features.")

        physio_feature = physio_dict[seq].astype(np.float32)
        blosum_feature = blosum_dict[seq].astype(np.float32)
        sinu_feature = sinusoidal_encoding_dict[seq].astype(np.float32)
        # position_aware_feature = position_aware_dict[seq].astype(np.float32)

        features.append([emb, cm, physio_feature, blosum_feature, sinu_feature, seq, mic])
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
            features[i][2] = physio_norm[i] #Physio index 2
            features[i][3] = blosum_norm[i] #Blosum index 3
            features[i][4] = sino_norm[i]   # Sinusoidal index 4
            # features[i][5] = position_aware_norm[i] # Position aware index 5
    
    # Logging
    # -------------------------------
    sample = features[0]
    print(f"Loaded features for {len(features)} sequences.")
    print("****Sample feature shapes:****")
    print(f"  Embedding: {sample[0].shape}")
    print(f"  Contact Map: {sample[1].shape}")
    print(f"  Physio-Chemical (normalized): {sample[2].shape}")
    print(f"  BLOSUM62 (normalized): {sample[3].shape}")
    print(f"  Sinusoidal PE (normalized) shape: {sample[4].shape}")
    print(f"  MIC: {sample[5]}")


    return features


def save_datasets(save_path):
    """
    datasets: List of (train_set, val_set, test_set)
    filepath: str, e.g. 'datasets.npz'
    """
    features = load_features()
    datasets = stratified_train_val_test_splits(
        features,
        seed=42,
        n_bins=5,
        n_datasets=5
    )
    
    with open(save_path, "wb") as f:
        pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Datasets saved to {save_path}.")

def load_datasets(datasets_index=[0, 1, 2, 3, 4]):
    """
    datasets_index: Which Dataset to load 
    0: First dataset 1: Second dataset and so on.
    Returns: List [(train_set, val_set, test_set), (...)]
    """
    with open(DATASET_PATH, "rb") as f:
        datasets = pickle.load(f)


    max_idx = len(datasets) - 1
    for i in datasets_index:
        if i < 0 or i > max_idx:
            raise ValueError(
                f"Invalid dataset index {i}. Valid range: [0, {max_idx}]"
            )

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
    save_datasets(DATASET_PATH)
    # datasets = load_datasets()


