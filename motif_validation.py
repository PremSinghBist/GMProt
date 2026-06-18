import random
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def merge_motif_files(motifs_csv_path, ranked_csv_path, output_csv_path="merged_motifs.csv"):
    """
    Merge motifs.csv and ranked_motifs.csv on motif_id,
    return merged dataframe and save to CSV.
    """

    # Load data
    motifs_df = pd.read_csv(motifs_csv_path)
    motifs_df = motifs_df.drop_duplicates(subset=["motif_id"], keep="first")
    
    ranked_df = pd.read_csv(ranked_csv_path)

    # Ensure motif_id is same type (important for safe merge)
    motifs_df["motif_id"] = motifs_df["motif_id"].astype(int)
    ranked_df["motif_id"] = ranked_df["motif_id"].astype(int)

    # Merge (inner join keeps only matching motif_id)
    merged_df = pd.merge(
        motifs_df,
        ranked_df,
        on="motif_id",
        how="inner",
        suffixes=("_detail", "_rank")
    )

    # Save result
    merged_df.to_csv(output_csv_path, index=False)

    return merged_df








# =========================================================
# 1. Load motif CSV
# =========================================================
def load_motif_csv(path):
    return pd.read_csv(path)


# =========================================================
# 2. Extract real motifs
# =========================================================
def extract_real_motifs(df):
    return df["seqlet"].dropna().astype(str).tolist()


# =========================================================
# 3. Generate random subsequences (null model)
# =========================================================
def random_subsequence(peptide, k):
    if len(peptide) <= k:
        return peptide

    start = random.randint(0, len(peptide) - k)
    return peptide[start:start + k]


def generate_random_motifs(df, motif_lengths, n_samples=2000):
    """
    Generate random subsequences matching the observed
    motif length distribution.
    """
    peptides = df["peptide"].dropna().astype(str).unique().tolist()

    random_motifs = []

    for _ in range(n_samples):
        pep = random.choice(peptides)
        k = random.choice(motif_lengths)
        random_motifs.append(random_subsequence(pep, k))

    return random_motifs


# =========================================================
# 4. Amino acid enrichment
# =========================================================
def amino_acid_enrichment(real_motifs, random_motifs):

    real_counts = Counter("".join(real_motifs))
    rand_counts = Counter("".join(random_motifs))

    real_total = sum(real_counts.values())
    rand_total = sum(rand_counts.values())

    enrichment = []

    for aa in "ACDEFGHIKLMNPQRSTVWY":

        real_freq = real_counts[aa] / real_total if real_total else 0
        rand_freq = rand_counts[aa] / rand_total if rand_total else 0

        fold_enrichment = (
            real_freq / (rand_freq + 1e-9)
            if rand_freq > 0
            else float("inf")
        )

        enrichment.append({
            "amino_acid": aa,
            "real_count": real_counts[aa],
            "random_count": rand_counts[aa],
            "real_frequency": real_freq,
            "random_frequency": rand_freq,
            "fold_enrichment": fold_enrichment
        })

    enrichment_df = pd.DataFrame(enrichment)

    return enrichment_df.sort_values(
        "fold_enrichment",
        ascending=False
    )


# =========================================================
# 5. Full enrichment pipeline
# =========================================================
def run_amino_acid_enrichment(csv_path, n_random=2000):

    df = load_motif_csv(csv_path)

    # Real motifs
    real_motifs = extract_real_motifs(df)

    # Preserve observed motif length distribution
    motif_lengths = [len(m) for m in real_motifs]

    # Random baseline
    random_motifs = generate_random_motifs(
        df,
        motif_lengths,
        n_samples=n_random
    )

    # Enrichment analysis
    enrichment_df = amino_acid_enrichment(
        real_motifs,
        random_motifs
    )

    return enrichment_df

def plot_amino_acid_enrichment_log2(
    enrichment_df,
    top_n=12,
    save_path="amino_acid_enrichment_log2.png"
):

    df = enrichment_df.copy()
    df["log2_enrichment"] = np.log2(df["fold_enrichment"] + 1e-9)

    df = df.sort_values("log2_enrichment", ascending=True).tail(top_n)

    n = len(df)

    # -----------------------------
    # Match ranked_motifs height logic exactly
    # -----------------------------
    fig_width = 3.6
    fig_height = max(2.8, min(0.35 * n + 1.2, 4.5))

    plt.figure(figsize=(fig_width, fig_height))

    plt.barh(
        df["amino_acid"],
        df["log2_enrichment"],
        color="black",
        alpha=0.85,
        height=0.65
    )

    plt.axvline(0, linestyle="--", color="gray", linewidth=0.8)

    # -----------------------------
    # Journal styling
    # -----------------------------
    plt.xlabel("log2 fold enrichment", fontsize=9)
    plt.ylabel("")

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.title("Amino Acid Enrichment (log2)", fontsize=10, pad=6)

    plt.tight_layout(pad=0.6)

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    
def load_amino_acid_enrichment(csv_path):
    """
    Load amino acid enrichment results from CSV.

    Expected columns:
        - amino_acid
        - fold_enrichment

    Returns:
        pd.DataFrame sorted by fold_enrichment (descending)
    """

    df = pd.read_csv(csv_path)

    # Basic validation
    required_cols = {"amino_acid", "fold_enrichment"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure correct types
    df["amino_acid"] = df["amino_acid"].astype(str)
    df["fold_enrichment"] = df["fold_enrichment"].astype(float)

    # Sort for consistency with plotting
    df = df.sort_values("fold_enrichment", ascending=False).reset_index(drop=True)

    return df

if __name__ == "__main__":
    base_path = "result/motif_discovery/"
    #merge_motif_files(f"{base_path}/motifs.csv", f"{base_path}/ranked_motifs_A0.9_S0.7.csv",
    #                output_csv_path=f"{base_path}/ranked_motifs_merged.csv")


    #enrichment_df = run_amino_acid_enrichment("result/motif_discovery/ranked_motifs_merged.csv", n_random=2000)
    # enrichment_df = enrichment_df[
    #     enrichment_df["real_count"] >= 2
    # ] #Filter out amino acids that appear less than 2 times in real motifs for more reliable enrichment

    enrichment_df = load_amino_acid_enrichment("result/motif_discovery/amino_acid_enrichment.csv")
    plot_amino_acid_enrichment_log2(enrichment_df, save_path="result/motif_discovery/amino_acid_enrichment_log2_new.png")

    # enrichment_df.to_csv(
    #     "result/motif_discovery/amino_acid_enrichment.csv",
    #     index=False
    # )


