import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
import re
import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
BATCH_SIZE = 64  # adjust based on GPU memory


# ------------------------
# Device selection
# ------------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


# ------------------------
# Load model + tokenizer
# ------------------------
def get_model_and_tokenizer(device):
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, tokenizer


# ------------------------
# Preprocess sequences
# ------------------------
def preprocess_peptides(sequence_examples):
    """
    Replace rare amino acids (U, Z, O, B) with 'X' and insert spaces.
    """
    rare_pattern = r"[UZOB]"
    processed = []
    rare_total = 0

    for seq in sequence_examples:
        rare_count = len(re.findall(rare_pattern, seq))
        rare_total += rare_count
        seq_proc = " ".join(re.sub(rare_pattern, "X", seq))
        processed.append(seq_proc)

    print(f"Total rare AA replaced: {rare_total}")
    return processed, rare_total


# ------------------------
# Read sequences from CSV
# ------------------------
def read_sequences(csv_file):
    df = pd.read_csv(csv_file)
    seqs = df["sequence"].astype(str).tolist()
    print(f"Loaded {len(seqs)} sequences from CSV.")
    return seqs


# ------------------------
# Tokenize batch
# ------------------------
def tokenize_batch(seqs, tokenizer, device):
    encoding = tokenizer(
        seqs,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt"
    )
    return encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


# ------------------------
# Compute residue-level embeddings
# ------------------------
def compute_residue_embeddings(input_ids, attention_mask, model):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state  # [B, L, 1024]

    # remove <cls> and <eos>
    emb = emb[:, 1:-1, :]
    att = attention_mask[:, 1:-1]

    return emb.detach().cpu().numpy(), att.detach().cpu().numpy()


# ------------------------
# Save embeddings
# ------------------------
def save_embeddings(original_seqs, processed_seqs, embeddings, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        filepath,
        original_sequences=np.array(original_seqs, dtype=object),
        processed_sequences=np.array(processed_seqs, dtype=object),
        embeddings=np.array(embeddings, dtype=object)
    )
    print(f"Saved embeddings to {filepath}")





# ------------------------
# Main pipeline
# ------------------------
def compute_save_embeddings(csv_path, save_path):
    device = get_device()
    model, tokenizer = get_model_and_tokenizer(device)

    original_sequences = read_sequences(csv_path)
    processed_sequences, rare_count = preprocess_peptides(original_sequences)
    print("Rare AA count:", rare_count)

    all_embeddings = []

    for batch_start in tqdm(
        range(0, len(processed_sequences), BATCH_SIZE),
        desc="Embedding batches"
    ):
        batch_seqs = processed_sequences[batch_start:batch_start + BATCH_SIZE]
        input_ids, attention_mask = tokenize_batch(batch_seqs, tokenizer, device)

        batch_emb, batch_att = compute_residue_embeddings(
            input_ids, attention_mask, model
        )

        for j in range(batch_emb.shape[0]):
            seq_len = int(batch_att[j].sum())
            all_embeddings.append(batch_emb[j, :seq_len, :])

        del input_ids, attention_mask, batch_emb, batch_att
        torch.cuda.empty_cache()

    print(f"Total sequences embedded: {len(all_embeddings)}")
    print(f"First embedding shape: {all_embeddings[0].shape}")

    save_embeddings(original_sequences, processed_sequences, all_embeddings, save_path)


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    #Ecoli data
    # seq_input_path = "/data/prem001/PGAT-ABPp/code/data/ecoli_normalized.csv"
    # save_path = f"/data/prem001/PGAT-ABPp/code/prott5/{Path(MODEL_NAME).name}/prott5_residue_level.npz"

    #Staphylococcus aureus data
    # seq_input_path = "/data/prem001/PGAT-ABPp/code/data/s_aureus_cleaned.csv"
    # save_path = f"/data/prem001/PGAT-ABPp/code/prott5/{Path(MODEL_NAME).name}/prott5_s_aureus_residue_level.npz"

    #Pseudomonas aeruginosa data
    seq_input_path = "./data/p_aeruginosa_dataset.csv"
    save_path = f"./data/prott5_p_aeruginosa_residue_level.npz"

    compute_save_embeddings(seq_input_path, save_path)


    #Load embeddings
    from data_util import load_embeddings
    original_seqs, processed_seqs, embeddings = load_embeddings(save_path)
    print(
        f"First sequence length: {len(original_seqs[0])}, "
        f"embedding shape: {embeddings[0].shape}"
    )
