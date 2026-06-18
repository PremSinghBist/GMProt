import pandas as pd
from scipy.stats import gmean

NON_STANDARD_AA = set("BJOUX")  # Non-standard amino acids


def cleanse(filtered_csv="data/p_aeruginosa_original.csv",
                    cleaned_csv="data/p_aeruginosa_cleaned.csv"):
    """
    Clean and deduplicate P. aeruginosa dataset.

    Operations:
    1. Remove sequences containing non-standard amino acids (B,J,O,U,X)
    2. Remove negative MIC values
    3. Deduplicate sequences by computing geometric mean of MIC ('value') for duplicates
    4. Save cleaned dataset

    Parameters
    ----------
    filtered_csv : str
        Path to the filtered P. aeruginosa CSV file (input)
    cleaned_csv : str
        Path to save the final cleaned dataset (output)
    """
    # Load filtered P. aeruginosa dataset
    df = pd.read_csv(filtered_csv)
    print(f"Original filtered P. aeruginosa records: {len(df)}")

    # Remove sequences with non-standard amino acids
    def has_non_standard(seq):
        return any(aa in NON_STANDARD_AA for aa in seq.upper())

    mask_non_standard = df['sequence'].apply(has_non_standard)
    df = df[~mask_non_standard].copy()
    print(f"Removed {mask_non_standard.sum()} sequences with non-standard amino acids")
    print(f"Remaining records: {len(df)}")

    # Remove negative MIC values
    mask_negative = df['value'] <= 0
    df = df[~mask_negative].copy()
    print(f"Removed {mask_negative.sum()} records with negative MIC values")
    print(f"Remaining records after negative MIC removal: {len(df)}")

    # Deduplicate sequences (geometric mean of MIC)
    df['sequence_norm'] = df['sequence'].str.strip().str.upper()
    grouped = df.groupby('sequence_norm', as_index=False)

    # Compute geometric mean of 'value' for duplicates
    df_dedup = grouped.apply(lambda x: x.assign(value=gmean(x['value']))).reset_index(drop=True)

    # Keep first occurrence per sequence and drop helper column
    df_final = df_dedup.drop_duplicates(subset=['sequence_norm']).drop(columns=['sequence_norm'])

    print(f"Final deduplicated dataset length: {len(df_final)}")

    # Save cleaned dataset
    df_final.to_csv(cleaned_csv, index=False)
    print(f"Cleaned P. aeruginosa dataset saved to: {cleaned_csv}")

    return df_final

def extract_p_aeruginosa(grampa_csv="data/grampa.csv", save_csv="data/p_aeruginosa_original.csv"):
    """
    Extract all rows with bacterium = 'P. aeruginosa' from the GRAMPA dataset
    and save to CSV.

    Parameters
    ----------
    grampa_csv : str
        Path to the input GRAMPA CSV file.
    save_csv : str
        Path to save the filtered P. aeruginosa dataset.
    """
    # Load the dataset
    df = pd.read_csv(grampa_csv)
    print(f"Original GRAMPA dataset length: {len(df)}")

    # Clean up bacterium column
    df['bacterium'] = df['bacterium'].str.strip()

    # Filter for P. aeruginosa
    df_p_aeruginosa = df[df['bacterium'] == 'P. aeruginosa']
    print(f"Filtered P. aeruginosa dataset length: {len(df_p_aeruginosa)}")

    # Save to CSV
    df_p_aeruginosa.to_csv(save_csv, index=False)
    print(f"P. aeruginosa dataset saved to: {save_csv}")

    return df_p_aeruginosa

def print_records_by_database(cleaned_csv="data/p_aeruginosa_cleaned.csv"):
    """
    Print number of P. aeruginosa records per source database.

    Parameters
    ----------
    cleaned_csv : str
        Path to cleaned P. aeruginosa CSV file
    """
    df = pd.read_csv(cleaned_csv)

    if "database" not in df.columns:
        raise ValueError("Column 'database' not found in the dataset.")

    counts = df["database"].value_counts()

    print("\nP. aeruginosa records by database:")
    for db, count in counts.items():
        print(f"{db}: {count}")


if __name__ == "__main__":
    # extract_p_aeruginosa(grampa_csv="data/grampa.csv", save_csv="data/p_aeruginosa_original.csv")

    cleanse(filtered_csv="data/p_aeruginosa_original.csv", cleaned_csv="data/p_aeruginosa_cleaned.csv")
    print_records_by_database(cleaned_csv="data/p_aeruginosa_cleaned.csv")
