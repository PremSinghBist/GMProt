#!/usr/bin/env python3
"""
extract_structure_features.py

Extract structural features from AlphaFold rank_001 PDB + corresponding scores JSON.

Saves one compressed .npz per input pair containing:
 - sequence (object/string)
 - residue_ids (N, ) tuples (chain_id, resseq)
 - ca_coords (N, 3)   # C-alpha coordinates
 - dist_matrix (N, N)
 - contact_map (N, N)  # binary
 - plddt (N,)
 - plddt_mask (N,)     # bool, True = keep (>= threshold)
 - dssp (N,)           # single-letter: H (helix), E (strand), C (coil)
 - hydrophobicity (N,) # Kyte-Doolittle
 - charge (N,)         # -1, 0, +1
"""

from pathlib import Path
import os
import json
import warnings
import subprocess
import glob

import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, is_aa
from Bio.PDB.DSSP import DSSP
from Bio.SeqUtils import seq1





from dataclasses import dataclass
from pathlib import Path


# Kyte-Doolittle hydrophobicity scale
KYTE_DOOLIT = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0   # unknown
}

# Simple residue charge at pH ~7 (side-chain contribution)
RES_CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0
}



@dataclass
class FeatureExtractionConfig:
    # Path to folder containing rank_001 AlphaFold PDB + JSON
    input_folder: Path

    # Output folder for extracted feature files
    out_dir: Path = Path("./features")

    # Parameters
    contact_thresh: float = 10.0      # Å threshold for contact map
    plddt_threshold: float = 70.0    # threshold for reliability mask

    def validate(self):
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {self.input_folder}")
        self.out_dir.mkdir(parents=True, exist_ok=True)



def find_rank001_files(folder: Path):
    """
    Search inside `folder`, which contains multiple subfolders,
    each with one AlphaFold prediction result.

    Return list of tuples:
        (pdb_path, scores_json_path)
    for all rank_001 models found.
    """

    results = []

    # Iterate through all subfolders
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue  # ignore files in top-level folder

        # find PDB file(s) in this subfolder
        pdbs = sorted(sub.glob("*rank_001*.pdb"))
        if not pdbs:
            continue  # no rank_001 here, skip

        # find scores JSON with rank_001
        score_jsons = sorted(sub.glob("*scores*rank_001*.json"))

        # pair logic
        for pdb in pdbs:
            if score_jsons:
                scores = score_jsons[0]  # assume one per folder
            else:
                warnings.warn(
                    f"No scores JSON found in {sub} for {pdb.name}. "
                    "Returning None for score path."
                )
                scores = None

            results.append((pdb, scores))

    return results



def load_plddt_from_scores(scores_json_path: Path):
    """Load per-residue pLDDT and mean plddt from the scores json produced by AlphaFold."""
    if scores_json_path is None:
        return None
    with open(scores_json_path, 'r') as fh:
        d = json.load(fh)
    # various pipelines put keys differently; try common keys
    per_res = d.get('per_residue_plddt') or d.get('plddt') or d.get('plddt_per_residue') or None
    mean_plddt = d.get('mean_plddt') or d.get('mean_pLDDT') or None
    return per_res, mean_plddt


def extract_ca_coords_and_sequence(pdb_path: Path):
    """
    Parse PDB and return:
      - sequence as str (one-letter)| eg of 1 letter sequence: AGKV"
      - residue_ids: list of (chain_id, residue_number)
      - ca_coords: (N,3) numpy array
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    ca_coords = []
    residue_ids = []
    sequence = []

    # iterate only first model
    model = next(struct.get_models())
    for chain in model:
        for res in chain:
            # skip hetero and non-amino acids
            if not is_aa(res, standard=True):
                continue
            # some residues lack CA (rare); skip those with warning
            if 'CA' not in res:
                warnings.warn(f"Residue {res.get_resname()} {res.id} in chain {chain.id} has no CA atom; skipping.")
                continue
            ca = res['CA'].get_coord() #Get coord of Alpha carbon
            ca_coords.append(ca)
            residue_ids.append((chain.id, res.id[1]))
            try:
                aa1 = seq1(res.get_resname()) ## convert 3-letter residue to 1-letter
            except KeyError:
                aa1 = 'X'

            sequence.append(aa1)
    if len(ca_coords) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    ca_coords = np.vstack(ca_coords) #converts the list of coordinates to shape (N, 3)
    sequence = "".join(sequence) #Convert sequnce list to str
    return sequence, residue_ids, ca_coords


def compute_distance_matrix(ca_coords: np.ndarray):
    """Return NxN Euclidean distance matrix from CA coords (N,3)."""
    #None: adds a new dimension of size 1
    #(N, 1, 3) - (1, N, 3) -># (N, N, 3)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]   # (N, N, 3)
    dist = np.linalg.norm(diff, axis=-1) #Operte along last axis: [dx, dy, dz]
    return dist


def compute_contact_map(dist_matrix: np.ndarray, threshold: float = 8.0):
    """Return binary contact map using given Å threshold."""
    return (dist_matrix <= threshold).astype(np.uint8)


def compute_dssp(pdb_path: Path):
    """
    Compute DSSP secondary structure per residue. (Dictinary of Secondary Structure Protein)
    Returns list of single-letter Secondary structure (H, E, C) aligned to CA-order(Alpha carbon) returned by extract_ca_coords_and_sequence.
    If DSSP is unavailable, returns list of 'C'.

    Returns: List of single letters(H, E, C) representing helix , strand(betasheet), Coil/loop.
    """
    # DSSP requires the DSSP binary (mkdssp). Try to run DSSP via Bio.PDB.DSSP
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        model = next(structure.get_models())
        
        # Bio.PDB.DSSP will call mkdssp; ensure it exists, else it will raise OSError
        dssp = DSSP(model, str(pdb_path))
        # dssp keys: (chain_id, resseq) #resseq means residue number
        ss_map = {}
        for key in dssp.keys():
            ss_letter = dssp[key][2]  # DSSP secondary structure code
            # map DSSP codes to H (helix), E (strand), else C
            if ss_letter in ('H', 'G', 'I'):  # helices
                ss = 'H'
            elif ss_letter in ('E', 'B'):     # strands / beta
                ss = 'E'
            else:
                ss = 'C' #coils or loops
            ss_map[key] = ss

        # Now produce ss list aligned to CA-order (Alpha Carbon)
        seq, residue_ids, _ = extract_ca_coords_and_sequence(pdb_path)
        ss_list = []
        for (chain_id, resseq) in residue_ids:
            key = (chain_id, resseq) #key = (chain_id, resseq means residue id)
            ss_list.append(ss_map.get(key, 'C')) #Return Secondary structure label H, E or default C 
        return np.array(ss_list, dtype='U1')
    except Exception as e:
        warnings.warn(f"DSSP failed or not available ({e}); filling secondary structure with 'C' (coil).")
        # fallback: return coil for all residues
        seq, residue_ids, ca_coords = extract_ca_coords_and_sequence(pdb_path)
        return np.array(['C'] * len(residue_ids), dtype='U1')


def compute_hydrophobicity_and_charge(sequence: str):
    """Return arrays hydrophobicity (float) and charge (int) for a sequence string."""
    hydro = np.array([KYTE_DOOLIT.get(aa, 0.0) for aa in sequence], dtype=float)
    charge = np.array([RES_CHARGE.get(aa, 0) for aa in sequence], dtype=int)
    return hydro, charge


def save_features_npz(outpath: Path, *, sequence, residue_ids, ca_coords,
                      dist_matrix, contact_map, plddt, plddt_mask,
                      dssp, hydrophobicity, charge):
    """Save the feature dict to a compressed .npz file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(outpath),
        sequence=np.array(sequence, dtype=object),
        residue_ids=np.array(residue_ids, dtype=object),
        ca_coords=ca_coords,
        dist_matrix=dist_matrix,
        contact_map=contact_map,
        plddt=plddt,
        plddt_mask=plddt_mask,
        dssp=dssp,
        hydrophobicity=hydrophobicity,
        charge=charge
    )
    print(f"Saved features to {outpath}")

def load_feature_file(path: str | Path):
    """
    Load structural feature data saved with np.savez_compressed.
    
    Returns a dict containing:
        - sequence (object array of characters or strings)
        - residue_ids (object array of (chain_id, residue_num))
        - ca_coords (N,3 float array)
        - dist_matrix (N,N float array)
        - contact_map (N,N int/bool array)
        - plddt (N float array)
        - plddt_mask (N bool array)
        - dssp (N array of H/E/C)
        - hydrophobicity (N float array)
        - charge (N int array)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Feature file created from PDB not found: {path}")

    data = np.load(path, allow_pickle=True)

    features = {
        "sequence":             data["sequence"],
        "residue_ids":          data["residue_ids"],
        "ca_coords":            data["ca_coords"],
        "dist_matrix":          data["dist_matrix"],
        "contact_map":          data["contact_map"],
        "plddt":                data["plddt"],
        "plddt_mask":           data["plddt_mask"],
        "dssp":                 data["dssp"],
        "hydrophobicity":       data["hydrophobicity"],
        "charge":               data["charge"],
    }

    return features


def process_one(pdb_path: Path, scores_json: Path = None,
                out_dir: Path = Path("./features"),
                contact_thresh: float = 8.0, plddt_threshold: float = 70.0):
    """
    Process a single rank_001 PDB + scores JSON and save features.
    The features are saved to .npy file.
    Saved features keys are are:
        [sequence
        residue_ids
        ca_coords
        dist_matrix
        contact_map
        plddt:per_res_plddt obtained from Alphafold
        plddt_mask: Binary mask representing pldt(Predicted local difference test) confidence threshold
        dssp= dictionary of secondary structure of protein,
        hydrophobicity
        charge]

    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    seq, residue_ids, ca_coords = extract_ca_coords_and_sequence(pdb_path)
    n = len(seq)

    per_res_plddt, mean_plddt = load_plddt_from_scores(scores_json) if scores_json is not None else (None, None)
    if per_res_plddt is None:
        warnings.warn(f"No pLDDT found for {pdb_path}; plddt array will be zeros and mask all True.")
        per_res_plddt = np.ones(n) * 100.0  # optimistic default
    else:
        per_res_plddt = np.array(per_res_plddt, dtype=float)

    # sanity checks
    if len(per_res_plddt) != n:
        raise ValueError(f"Length mismatch: {pdb_path} has {n} residues (CA) but pLDDT length is {len(per_res_plddt)}")

    # distance and contact
    dist = compute_distance_matrix(ca_coords) #output: N*N (residue dist to other residue)
    contact = compute_contact_map(dist, threshold=contact_thresh)

    # dssp
    dssp = compute_dssp(pdb_path)

    # hydro & charge
    hydro, charge = compute_hydrophobicity_and_charge(seq)

    # plddt mask
    plddt_mask = (per_res_plddt >= plddt_threshold)

    # build output path
    outname = pdb_path.stem
    outpath = out_dir / f"{outname}_features.npz"

    save_features_npz(
        outpath,
        sequence=seq,
        residue_ids=residue_ids,
        ca_coords=ca_coords,
        dist_matrix=dist,
        contact_map=contact,
        plddt=per_res_plddt,
        plddt_mask=plddt_mask,
        dssp=dssp,
        hydrophobicity=hydro,
        charge=charge
    )
    return outpath


def process_folder(folder: Path, out_dir: Path = Path("./features"), contact_thresh=8.0, plddt_threshold=70.0):
    """
    Process all rank_001 PDB files in folder.
    Returns list of outpaths written.
    """
    folder = Path(folder)
    pairs = find_rank001_files(folder)
    written = []
    for pdb_path, scores_json in pairs:
        try:
            outp = process_one(pdb_path, scores_json, out_dir=out_dir, contact_thresh=contact_thresh, plddt_threshold=plddt_threshold)
            written.append(outp)
        except Exception as e:
            warnings.warn(f"Failed to process {pdb_path}: {e}")
    return written

def extract_pdb_features(input_folder, out_dir):
    ''' 
        Wrapper method to compute/save features from pdb file, etc. obtained from Alphafold 2.
        args:
            input_folder : Parent folder path inside which multiple subfolders exist. Each  
            subfolder contains the result for peptide sequence (obtained from Alphafold2)

            out_dir: Save path for the computed features. Each sequence is saved to its own representaive 
            feature(.npy) file. 
    '''
    # ---- Set config here instead of using argparse ----
    config = FeatureExtractionConfig(
        input_folder=Path(input_folder),
        out_dir=Path(out_dir),
        contact_thresh=10.0, #vs 8.0
        plddt_threshold=70.0
    )
    config.validate()

    outs = process_folder(
        folder=config.input_folder,
        out_dir=config.out_dir,
        contact_thresh=config.contact_thresh,
        plddt_threshold=config.plddt_threshold
    )
    print("Wrote files:", outs)

def explain_npy_feature(npy_path: str):
    """
    Explain detailed properties available in a PDB-output-based .npy feature file.
    """
    features = load_feature_file(npy_path)

    print("Key attributes:", list(features.keys()))

    seq = features["sequence"]
    residue_ids = features["residue_ids"]
    ca_coords = features["ca_coords"]
    dssp = features["dssp"]
    hydrophobic = features["hydrophobicity"]
    charge = features["charge"]
    plddt = features["plddt"]
    plddt_mask = features["plddt_mask"]
    dist_matrix = features["dist_matrix"]
    contact_map = features["contact_map"]

    msg = f"""
            Sequence: {seq}

            Residue IDs (len={len(residue_ids)} [China, Id]): 
            {residue_ids}

            CA coordinates shape: {ca_coords.shape}

            DSSP length: {len(dssp)} 
            First 3 DSSP codes: {dssp[:3]}

            Hydrophobicity (first 2): {hydrophobic[:2]}
            Charge (first 2): {charge[:2]}

            pLDDT shape: {plddt.shape}
            {plddt}

            pLDDT mask shape: {plddt_mask.shape}
            {plddt_mask}

            Distance matrix shape: {dist_matrix.shape}
            Sample (3×3):
            {dist_matrix[:3, :3]}

            Contact map shape: {contact_map.shape}
            Sample (3×3):
            {contact_map[:3, :3]}
            """

    print(msg)



def generate_contact_map_csv(npz_folder_path, input_csv_path, save_csv_path):
    """
    Processes multiple NPZ feature files and generates one CSV.

    Each NPZ file becomes one row containing:
    - filename
    - A: full contact map as JSON list-of-lists
    - seq: sequence
    - label: normalized MIC value (from input_csv_path)
    """

    all_rows = []

    # --------------------------
    # Load sequences + MIC values
    # --------------------------
    mic_df = pd.read_csv(input_csv_path)

    #taking raw mic 'value' instead of 'normalized_value' column
    if "sequence" not in mic_df.columns or "value" not in mic_df.columns:
        raise ValueError("CSV must contain 'sequence' and 'value' columns.")

    # Normalize sequence field
    mic_df["sequence_clean"] = mic_df["sequence"].str.strip().str.upper()

    # -----------------------------------------------------
    # Process each NPZ file and append one row per NPZ file
    # -----------------------------------------------------
    list_of_npy_files = glob.glob(os.path.join(npz_folder_path, "*.npz"))
    for input_npy in list_of_npy_files:

        print(f"Processing: {input_npy}")

        data = np.load(input_npy, allow_pickle=True)
        seq_data = data["sequence"]

        
        sequence = str(seq_data).strip()

        sequence_clean = sequence.upper()

        # Contact map
        contact_map = data["contact_map"]

        # --------------------------
        # Match MIC label
        # --------------------------
        match = mic_df[mic_df["sequence_clean"] == sequence_clean]

        if match.empty:
            raise ValueError(f"Sequence not found in MIC CSV: {sequence}")

        label = match["value"].iloc[0]  # cleaner than .values[0]

        # --------------------------
        # Convert contact map → JSON string
        # --------------------------
        contact_map_str = json.dumps(contact_map.astype(float).tolist())

        filename_value = os.path.basename(input_npy).replace(".npz", "")

        # Append a row
        all_rows.append({
            "filename": filename_value,
            "A": contact_map_str,
            "seq": sequence,
            "label": label
        })

    # --------------------------
    # Create final CSV
    # --------------------------
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(save_csv_path, index=False)

    print(f"\n Saved combined CSV → {save_csv_path}")





def get_contact_map(seq, df):
    """
    Given a sequence, retrieve:
      - contact_map (column 'A', stored as JSON list-of-lists)
      - label (column 'label')
    
    df(pandas dataframe) must contain columns: ['seq', 'A', 'label'].
    The sequence match is exact (case-insensitive).
    """

    # Case-insensitive matching
    match = df[df["seq"].str.strip().str.upper() == seq.strip().upper()]

    if match.empty:
        raise ValueError(f"Sequence not found in DataFrame: {seq}")

    # Extract values
    contact_map_json = match["A"].values[0]   # JSON string
    label = match["label"].values[0]

    # Convert JSON string → Python list
    contact_map = json.loads(contact_map_json)

    return np.array(contact_map), label

def compute_distance_weighted_contact_map(dist_matrix: np.ndarray, threshold: float = 10.0, weight_type: str = "inverse") -> np.ndarray:
    """
    Compute distance-weighted contact map.

    Args:
        dist_matrix: (N, N) Euclidean distance matrix
        threshold: Å cutoff for considering contact
        weight_type: type of weighting
            - 'inverse': weight = 1/d
            - 'inverse_sq': weight = 1/d^2
            - 'exp': weight = exp(-d)
            - 'linear': weight = (threshold - d)/threshold clipped at 0

    Returns:
        (N, N) weighted contact map
    """
    weighted_map = np.zeros_like(dist_matrix, dtype=float)
    mask = dist_matrix <= threshold

    if weight_type == "inverse":
        weighted_map[mask] = 1.0 / np.maximum(dist_matrix[mask], 1e-6)
    elif weight_type == "inverse_sq":
        weighted_map[mask] = 1.0 / np.maximum(dist_matrix[mask]**2, 1e-6)
    elif weight_type == "exp":
        weighted_map[mask] = np.exp(-dist_matrix[mask])
    elif weight_type == "linear":
        weighted_map[mask] = (threshold - dist_matrix[mask]) / threshold
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Set diagonal to 0
    np.fill_diagonal(weighted_map, 0.0)
    return weighted_map


def generate_distance_weighted_csv(npz_folder_path, input_csv_path, save_csv_path, threshold=10.0, weight_type="inverse"):
    """
    Generate CSV with distance-weighted contact maps for all NPZ files.
    Each row will include:
        - filename
        - weighted_contact_map as JSON
        - sequence
        - label (from MIC CSV)
    """
    all_rows = []

    mic_df = pd.read_csv(input_csv_path)
    if "sequence" not in mic_df.columns or "normalized_value" not in mic_df.columns:
        raise ValueError("CSV must contain 'sequence' and 'normalized_value' columns.")
    mic_df["sequence_clean"] = mic_df["sequence"].str.strip().str.upper()

    npz_files = glob.glob(os.path.join(npz_folder_path, "*.npz"))
    for npz_file in npz_files:
        print(f"Processing: {npz_file}")
        data = np.load(npz_file, allow_pickle=True)
        sequence = str(data["sequence"]).strip()
        sequence_clean = sequence.upper()

        # Find MIC label
        match = mic_df[mic_df["sequence_clean"] == sequence_clean]
        if match.empty:
            raise ValueError(f"Sequence not found in MIC CSV: {sequence}")
        label = match["normalized_value"].iloc[0]

        dist_matrix = data["dist_matrix"]
        weighted_map = compute_distance_weighted_contact_map(dist_matrix, threshold=threshold, weight_type=weight_type)
        weighted_map_str = json.dumps(weighted_map.tolist())

        filename_value = os.path.basename(npz_file).replace(".npz", "")
        all_rows.append({
            "filename": filename_value,
            "A": weighted_map_str,
            "seq": sequence,
            "label": label
        })

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(save_csv_path, index=False)
    print(f"Saved distance-weighted contact map CSV → {save_csv_path}")



if __name__ == "__main__":
    #Ecoli
    '''input_folder="/data/prem001/PGAT-ABPp/code/data/alphafold_pdb/raw_pdb"
    out_dir= "/data/prem001/PGAT-ABPp/code/data/alphafold_pdb/pdb_features"'''

    #Staph
    input_folder="./data/alphafold_pdb_stap/result/raw_pdb"
    out_dir= "./data/alphafold_pdb_stap/result/pdb_features"
    
    
    # extract_pdb_features(input_folder, out_dir) #1. 
    explain_npy_feature('/data/prem001/PGAT-ABPp/code/data/alphafold_pdb_ecoli/pdb_features/ecoli_group_1_5_1_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000_features.npz')
    import sys; sys.exit(0)

    #Ecoli
    '''generate_contact_map_csv(
        npz_folder_path='/data/prem001/PGAT-ABPp/code/data/alphafold_pdb/pdb_features',
        input_csv_path='/data/prem001/PGAT-ABPp/code/data/ecoli_mic_normalized.csv',
        save_csv_path='/data/prem001/PGAT-ABPp/code/data/contact_map.csv'
    )'''
    #Staph
    generate_contact_map_csv(
        npz_folder_path='./data/alphafold_pdb_stap/result/pdb_features',
        input_csv_path='./data/s_aureus_cleaned.csv',
        save_csv_path='./data/s_aureus_contact_map.csv'
    )

    '''generate_distance_weighted_csv(
    npz_folder_path='/data/prem001/PGAT-ABPp/code/data/alphafold_pdb/pdb_features',
    input_csv_path='/data/prem001/PGAT-ABPp/code/data/ecoli_normalized.csv',
    save_csv_path='/data/prem001/PGAT-ABPp/code/data/distance_weighted_contact_map.csv',
    threshold=10.0,
    weight_type='inverse'  # options: 'inverse', 'inverse_sq', 'exp', 'linear'
    )'''




   
