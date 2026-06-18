# =========================================================
# INTEGRATED GRADIENTS FOR CNN MOTIF DISCOVERY
# =========================================================
import os
from pathlib import Path
from collections import defaultdict, Counter
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform

from experimental_config import ExperimentConfig

cfg = ExperimentConfig()

EMB_DIM = 1024
PHYSIO_DIM = cfg.physio_feature_dim + cfg.blosum_feature_dim + cfg.sinusoidal_feature_dim


MODEL_NAME = "ecoli_ProtT5"
MODEL_INDEX = 0


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = 0
UNK_TOKEN = 21


idx_to_aa = {i + 1: aa for i, aa in enumerate(AA_ALPHABET)}

aa_to_idx = {aa: i + 1 for i, aa in enumerate(AA_ALPHABET)}

idx_to_aa[UNK_TOKEN] = "X"
aa_to_idx["X"] = UNK_TOKEN

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

POSITIVE = {'K', 'R', 'H'}
NEGATIVE = {'D', 'E'}


LOW_COMPLEXITY_MOTIFS = {
    "AAAA", "GGGG", "CCCC", "SSSS", "DDDD", "EEEE", "NNNN"
}

def is_low_complexity_motif(motif):
    # repetitive or near-repetitive
    if motif in LOW_COMPLEXITY_MOTIFS:
        return True
    
    if len(set(motif)) <= 2:  # e.g., NLA-like low diversity
        return True
    
    # glycine/alanine dominated noise
    if (motif.count("G") + motif.count("A")) / len(motif) > 0.8:
        return True
    
    return False

# -----------------------------
# 1. Net charge (motif-only)
# -----------------------------
def motif_net_charge(motif):
    return sum(
        1 if aa in POSITIVE else -1 if aa in NEGATIVE else 0
        for aa in motif
    )


# -----------------------------
# 2. Mean hydrophobicity
# -----------------------------
def motif_hydrophobicity(motif):
    vals = [HYDROPHOBICITY.get(a, 0.0) for a in motif]
    return np.mean(vals)


# -----------------------------
# 3. Hydrophobic moment (motif-level approximation)
# -----------------------------
def motif_hydrophobic_moment(motif, angle=100):
    """
    Even for short motifs, we treat them as an ideal helix segment.
    """
    theta = np.deg2rad(angle)

    sin_sum, cos_sum = 0.0, 0.0

    for i, aa in enumerate(motif):
        h = HYDROPHOBICITY.get(aa, 0.0)
        sin_sum += h * np.sin(i * theta)
        cos_sum += h * np.cos(i * theta)

    return np.sqrt(sin_sum**2 + cos_sum**2) / len(motif)


# -----------------------------
# 4. Amphipathicity proxy (motif-only)
# -----------------------------
def motif_amphipathicity(motif):
    charge = abs(motif_net_charge(motif))
    hydro = motif_hydrophobicity(motif)

    # normalize hydrophobicity into AMP range (~ -1 to +2 typical useful range)
    hydro_norm = np.tanh(hydro)

    return charge * hydro_norm

def amp_validity_score(motif, net_charge, hydro, hmoment, amphipathicity):
    
    L = len(motif)

    # --------------------------
    # Charge requirement (AMP hallmark)
    # --------------------------
    charge_score = max(0, net_charge) / max(L, 1)

    # --------------------------
    # Hydrophobic balance (avoid extremes)
    # --------------------------
    hydro_score = 1.0 - abs(hydro) / 5.0  # Kyte scale normalization

    # --------------------------
    # Structural potential (helix-like AMP signal)
    # --------------------------
    structure_score = np.tanh(hmoment)

    # --------------------------
    # Amphipathicity signal
    # --------------------------
    amph_score = np.tanh(amphipathicity)

    return (
        2.0 * charge_score +
        1.5 * hydro_score +
        1.5 * structure_score +
        2.0 * amph_score
    )


# -----------------------------
# 5. Full pipeline on ranked motifs
# -----------------------------
def analyze_motif_file(csv_path):
    df = pd.read_csv(csv_path)

    results = []

    for _, row in df.iterrows():
        motif = row["consensus"]

        # -------------------------
        # FILTER ARTIFACTS FIRST
        # -------------------------
        if is_low_complexity_motif(motif):
            continue

        net_charge = motif_net_charge(motif)
        hydro = motif_hydrophobicity(motif)
        hmoment = motif_hydrophobic_moment(motif)
        amph = motif_amphipathicity(motif)

        amp_score = amp_validity_score(
            motif,
            net_charge,
            hydro,
            hmoment,
            amph
        )

        results.append({
            "motif": motif,
            "length": len(motif),

            "net_charge": net_charge,
            "mean_hydrophobicity": hydro,
            "hydrophobic_moment": hmoment,
            "amphipathicity_score": amph,

            "amp_validity_score": amp_score,

            "support": row.get("n_seqlets", None),
            "ig_score": row.get("score", None),
        })

    return pd.DataFrame(results).sort_values(
        by=["amp_validity_score", "net_charge"],
        ascending=False
    )

# -----------------------------
# Save Motif analysis results to CSV
# -----------------------------
def save_motif_analysis(csv_input_path, output_csv_path):
    df = analyze_motif_file(csv_input_path)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved motif analysis to: {output_csv_path}")
    return df

def decode_seq_ids(seq_ids):
    peptide = ""
    for idx in seq_ids:
        idx = int(idx)
        if idx == PAD_TOKEN:
            continue

        peptide += idx_to_aa.get(idx, "X")

    return peptide


# ---------------------------------------------------------
# 1. Load Model
# ---------------------------------------------------------


def load_model(model_name, model_index):
    """
    Load the saved dual-branch model
    """

    model_dir = Path("model") / model_name / f"model_{model_index}"

    from train import (
        ImprovedDualBranchGNN_AttentionFusion,
        GraphAttentionNetwork,
        TransformerEncoderReadout,
        SequenceCNN,
        CrossAttentionFusion,
    )

    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={
            "ImprovedDualBranchGNN_AttentionFusion": ImprovedDualBranchGNN_AttentionFusion,
            "GraphAttentionNetwork": GraphAttentionNetwork,
            "TransformerEncoderReadout": TransformerEncoderReadout,
            "SequenceCNN": SequenceCNN,
            "CrossAttentionFusion": CrossAttentionFusion,
        },
        compile=False,
    )

    print(f"✓ Model loaded: {model.name}")

    return model


# ---------------------------------------------------------
# Load model instance
# ---------------------------------------------------------

model = load_model(MODEL_NAME, MODEL_INDEX)

# --------------------------------------------------------- # Configuration # --------------------------------------------------------- MODEL_NAME = "ecoli_seed_42" MODEL_INDEX = 0 AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# =========================================================
# PEPTIDE ENCODING
# =========================================================
def encode_peptide_to_ids(peptide_seq, max_len):

    seq_ids = [aa_to_idx.get(aa, UNK_TOKEN) for aa in peptide_seq]

    if len(seq_ids) < max_len:
        seq_ids += [PAD_TOKEN] * (max_len - len(seq_ids))

    return np.array(seq_ids, dtype=np.int32)


# =========================================================
# CREATE FIXED INPUTS
# =========================================================


def create_fixed_inputs(background_inputs):
    fixed_inputs = {
        "atom_features": background_inputs["atom_features"],
        "molecule_indicator": background_inputs["molecule_indicator"],
        "physio_features": background_inputs["physio_features"],
    }
    return fixed_inputs


# =========================================================
# CNN EMBEDDING IG
# =========================================================
def cnn_forward(model, seq_ids):
    x = model.seq_cnn.embed(seq_ids)

    mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)
    mask = tf.expand_dims(mask, -1)

    x = x * mask

    def masked_pool(conv_out):
        large_neg = -1e9
        conv_out = conv_out + (1.0 - mask) * large_neg
        return tf.reduce_max(conv_out, axis=1)

    c3 = masked_pool(model.seq_cnn.conv3(x))
    c5 = masked_pool(model.seq_cnn.conv5(x))
    c7 = masked_pool(model.seq_cnn.conv7(x))
    c11 = masked_pool(model.seq_cnn.conv11(x))

    x = tf.concat([c3, c5, c7, c11], axis=-1)

    return model.seq_cnn.proj(x)


# =========================================================
# INTEGRATED GRADIENTS FOR CNN MOTIF DISCOVERY (FIXED)
# =========================================================
def forward_pass(seq_cnn_feat, fixed_inputs, model):
    nodes = fixed_inputs["atom_features"]
    prot_ids = fixed_inputs["molecule_indicator"]
    physio = fixed_inputs["physio_features"]

    mean_pool = tf.math.unsorted_segment_mean(
        nodes, prot_ids, tf.reduce_max(prot_ids) + 1
    )

    seq = model.seq_proj(mean_pool)
    seq = model.seq_transformer(seq, training=False)
    seq = model.seq_out(seq)
    seq_feat = model.seq_bottleneck(seq)

    physio_feat = model.physio_proj(physio)
    physio_feat = model.physio_bottleneck(physio_feat)

    seq_feat = seq_feat * (1 + model.cfg.cnn_gating_threshold * tf.tanh(seq_cnn_feat))

    fused = seq_feat + physio_feat
    return model.out(fused)


def model_forward(model, seq_cnn_feat, fixed_inputs):

    nodes = fixed_inputs["atom_features"]
    prot_ids = fixed_inputs["molecule_indicator"]
    physio = fixed_inputs["physio_features"]

    mean_pool = tf.math.unsorted_segment_mean(
        nodes, prot_ids, tf.reduce_max(prot_ids) + 1
    )

    seq = model.seq_proj(mean_pool)
    seq = model.seq_transformer(seq, training=False)
    seq = model.seq_out(seq)
    seq_feat = model.seq_bottleneck(seq)

    physio_feat = model.physio_proj(physio)
    physio_feat = model.physio_bottleneck(physio_feat)

    seq_feat = seq_feat * (1 + model.cfg.cnn_gating_threshold * tf.tanh(seq_cnn_feat))

    fused = seq_feat + physio_feat

    return model.out(fused)


def cnn_features(model, x, mask):
    def pool(y):
        # Preserve your masked max pooling logic
        y = y + (1.0 - mask) * (-1e9)
        return tf.reduce_max(y, axis=1)

    # Call the layers directly to inherit activations and biases
    c3 = pool(model.seq_cnn.conv3(x))
    c5 = pool(model.seq_cnn.conv5(x))
    c7 = pool(model.seq_cnn.conv7(x))
    c11 = pool(model.seq_cnn.conv11(x))

    cnn_feat = tf.concat([c3, c5, c7, c11], axis=-1)

    # Use the projection layer directly
    return model.seq_cnn.proj(cnn_feat)


def compute_cnn_integrated_gradients(model, seq_ids, fixed_inputs, m_steps=64):
    """
    Returns importance scores for each token in the input sequence based on integrated gradients.
    Returns score, and full IG values for deeper analysis if needed.
    """

    seq_ids = tf.cast(seq_ids, tf.int32)

    emb_layer = model.seq_cnn.embed
    embed_matrix = emb_layer.embeddings  # IMPORTANT

    seq_onehot = tf.one_hot(seq_ids, depth=embed_matrix.shape[0])
    # blve:Batch(B), Len(L), vocab_size(V) D(Feature dim)
    embedded = tf.einsum("blv,vd->bld", seq_onehot, embed_matrix)

    
    # Extract the actual trained embedding vector for your neutral/padding token
    pad_embedding = embed_matrix[PAD_TOKEN] # Shape: (D,)

    # Broadcast the neutral token embedding to match your sequence dimensions
    baseline = tf.broadcast_to(pad_embedding, tf.shape(embedded))

    total_grads = tf.zeros_like(embedded)

    mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)
    mask = tf.expand_dims(mask, -1)

    alphas = tf.linspace(0.0, 1.0, m_steps)

    for alpha in alphas:

        interp_emb = baseline + alpha * (embedded - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interp_emb)

            x = interp_emb * mask

            seq_cnn_feat = cnn_features(model, x, mask)

            pred = model_forward(model, seq_cnn_feat, fixed_inputs)

        grads = tape.gradient(pred, interp_emb)

        total_grads += grads

    avg_grads = total_grads / tf.cast(m_steps, tf.float32)

    ig = (embedded - baseline) * avg_grads

    token_scores = tf.reduce_sum(ig, axis=-1)

    mask = tf.cast(tf.not_equal(seq_ids[0], PAD_TOKEN), tf.float32)

    token_scores = token_scores * mask  # remove padding influence

    valid_len = tf.reduce_sum(mask)

    return token_scores[: tf.cast(valid_len, tf.int32)].numpy(), ig.numpy()


# =========================================================
# PROTT5 RESIDUE IG
# =========================================================


def compute_prott5_integrated_gradients(model, inputs, m_steps=64):

    atom_features = tf.cast(inputs["atom_features"], tf.float32)

    baseline = tf.zeros_like(atom_features)

    accumulated_grads = tf.zeros_like(atom_features)

    alphas = tf.linspace(0.0, 1.0, m_steps)

    for alpha in alphas:

        interpolated = baseline + alpha * (atom_features - baseline)

        with tf.GradientTape() as tape:

            tape.watch(interpolated)

            preds = model({**inputs, "atom_features": interpolated}, training=False)

        grads = tape.gradient(preds, interpolated)

        accumulated_grads += grads

    avg_grads = accumulated_grads / tf.cast(m_steps, tf.float32)

    ig = (atom_features - baseline) * avg_grads

    residue_scores = tf.reduce_sum(tf.abs(ig), axis=-1)

    return residue_scores.numpy()


# =========================================================
# VISUALIZATION
# =========================================================


def plot_residue_importance(peptide, scores, title, save_path=None):

    plt.figure(figsize=(14, 4))

    positions = np.arange(len(peptide))

    colors = ["#d62728" if s < 0 else "#2ca02c" for s in scores]

    plt.bar(positions, scores, color=colors, edgecolor="black")

    plt.xticks(positions, list(peptide), fontsize=12, fontweight="bold")

    plt.xlabel("Residue Position")
    plt.ylabel("Integrated Gradient")

    plt.title(title)

    plt.axhline(y=0, color="black", linestyle="--")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# =========================================================
# MAIN ANALYSIS
# =========================================================
import data_util


def generator_old(X):
    """Yield each sample for tf.data"""
    for emb, cm, p_feat, blosum_feat, sinu_feat, seq, y in X:
        rows, cols = np.where(
            cm > 0
        )  # return row array, col array | Each (rows[i], cols[i]) is an edge from node row[i] to node cols[i]

        # convert two 1D array into edge list [[0, 1],[1, 3] ] ...| shape num_edges x 2 | needed for GAT Input
        edges = (
            np.stack([rows, cols], axis=1).astype(np.int64)
            if rows.size
            else np.zeros((0, 2), dtype=np.int64)
        )
        weights = (
            np.ones(len(rows), dtype=np.float32)
            if rows.size
            else np.zeros((0,), dtype=np.float32)
        )
        p_feat_updated = np.concatenate(
            [p_feat, blosum_feat, sinu_feat], axis=0
        ).astype(np.float32)
        yield emb, edges, weights, p_feat_updated, seq, np.array([y], dtype=np.float32)

def generator(X):
    """Yield each sample for tf.data"""
    for emb, p_feat, blosum_feat, sinu_feat, seq, y in X:
        # rows, cols = np.where(cm > 0) #return row array, col array | Each (rows[i], cols[i]) is an edge from node row[i] to node cols[i]
        
        #convert two 1D array into edge list [[0, 1],[1, 3] ] ...| shape num_edges x 2 | needed for GAT Input
        # edges = np.stack([rows, cols], axis=1).astype(np.int64) if rows.size else np.zeros((0, 2), dtype=np.int64)
        # weights = np.ones(len(rows), dtype=np.float32) if rows.size else np.zeros((0,), dtype=np.float32)
        p_feat_updated = np.concatenate([p_feat, blosum_feat, sinu_feat], axis=0).astype(np.float32) 
        # yield emb, edges, weights, p_feat_updated, seq, np.array([y], dtype=np.float32)
        yield emb, p_feat_updated, seq, np.array([y], dtype=np.float32)


def make_dataset_old(X, shuffle=False):
    emb_spec = tf.TensorSpec(shape=[None, 1024], dtype=tf.float32)
    edges_spec = tf.TensorSpec(shape=[None, 2], dtype=tf.int64)
    weights_spec = tf.TensorSpec(shape=[None], dtype=tf.float32)
    physio_spec = tf.TensorSpec(shape=[84], dtype=tf.float32)

    label_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)
    seq_spec = tf.TensorSpec(shape=(), dtype=tf.string)
    ds = tf.data.Dataset.from_generator(
        lambda: generator(X),
        output_signature=(
            emb_spec,
            edges_spec,
            weights_spec,
            physio_spec,
            seq_spec,
            label_spec,
        ),
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)

    ds = ds.ragged_batch(cfg.batch_size)
    ds = ds.map(
        lambda emb, edges, weights, physio, seq, lbl: prepare_batch(
            emb, edges, weights, physio, seq, lbl
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
    return ds


def make_dataset(X, shuffle=False):
    emb_spec = tf.TensorSpec(shape=[None, EMB_DIM], dtype=tf.float32)
    # edges_spec = tf.TensorSpec(shape=[None, 2], dtype=tf.int64)
    # weights_spec = tf.TensorSpec(shape=[None], dtype=tf.float32)
    physio_spec = tf.TensorSpec(shape=[PHYSIO_DIM], dtype=tf.float32)

    label_spec = tf.TensorSpec(shape=(1,), dtype=tf.float32)
    seq_spec = tf.TensorSpec(shape=(), dtype=tf.string)
    ds = tf.data.Dataset.from_generator(
        lambda: generator(X),
        # output_signature=(emb_spec, edges_spec, weights_spec, physio_spec, seq_spec, label_spec)
        output_signature=(emb_spec, physio_spec, seq_spec, label_spec)
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=cfg.shuffle_buffer_size, seed=cfg.seed)

    ds = ds.ragged_batch(cfg.batch_size)
    ds = ds.map(lambda emb,  physio, seq, lbl: prepare_batch(emb, physio, seq, lbl),
                num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds


def prepare_batch_old(
    batched_emb,
    batched_edges,
    batched_weights,
    batched_physio,
    batched_seq,
    batched_labels,
):
    """
    You have a batch of graphs represented as ragged tensors(variable lenghts).
    Each graph has its own set of nodes and edges. Here Node features are ProtT5 embeddings.
    Edges are derived from contact maps.

    It will prepare batched inputs for the batched graph neural network model.
    Adjust edge indices based on node offsets in the batch.
    Prot_ids indicate which peptide each node belongs to in the batch.

    So, instead of processing each graph individually, we can process the entire batch together.


    """
    # === RAW SEQUENCE HANDLING ===
    # batched_seq is RaggedTensor of strings (B,)
    # convert to padded tensor of AA indices

    seq_strings = tf.strings.strip(batched_seq)
    # convert to list of characters | CNN needs 2D input so
    chars = tf.strings.unicode_split(
        seq_strings, "UTF-8"
    )  # (B, L) #eg: ["KLK"] ->[['K', ''L]] (Again ragged)
    # lookup table for AA → index #0 reserved for padding so start from 1
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list("ACDEFGHIKLMNPQRSTVWY")),
            values=tf.constant(list(range(1, 21)), dtype=tf.int32),
        ),
        default_value=UNK_TOKEN,
    )
    seq_ids = table.lookup(chars)  # (B, L)  convert to integer tensor
    # pad sequences to same length
    # to_tensor: automatically pads all sequences in the batch to the max length in that batch.
    seq_ids_padded = seq_ids.to_tensor(
        default_value=PAD_TOKEN
    )  # UNK_TOKEN = padding token

    labels = tf.cast(batched_labels, tf.float32)
    if len(labels.shape) == 1:
        labels = tf.expand_dims(labels, axis=-1)

    num_nodes = (
        batched_emb.row_lengths()
    )  # tf.RaggedTensor |  Shape: [batch_size, None, 1024] | (None: number of seqs in this batch)

    num_edges = (
        batched_edges.row_lengths()
    )  # [[1, 2], [0, 2, 5], [2,3]] -> [2, 3, 2] number of edges per graph in batch
    nodes_flat = batched_emb.merge_dims(
        0, 1
    )  # [batch, nodes, features] -> [total_nodes, features]
    edges_flat = batched_edges.merge_dims(0, 1)
    weights_flat = batched_weights.merge_dims(0, 1)

    # Added 0 ,  cumulative sum of num_nodes excluding last element. eg: [0, 2, 4, 5] ->[0, 2, 6, 11]
    offsets = tf.concat(
        [[0], tf.cumsum(num_nodes)[:-1]], axis=0
    )  # offsets to adjust edge indices for batching

    edge_rowids = (
        batched_edges.value_rowids()
    )  # Get which graph, each edge belongs to in the batch

    # “Pick rows from a tensor based on an index list. |  gather(input, indices)”
    edge_offsets = tf.gather(
        offsets, edge_rowids
    )  # get offset for each edge based on which graph it belongs to
    edges_flat = edges_flat + tf.cast(
        tf.expand_dims(edge_offsets, axis=-1), edges_flat.dtype
    )  # adjust edge indices based on node offsets in the batch

    prot_ids = tf.repeat(
        tf.range(tf.shape(num_nodes)[0]), num_nodes
    )  # indicator for which node belongs to which graph in the batch
    prot_ids = tf.cast(prot_ids, tf.int32)
    physio_flat = tf.cast(batched_physio, tf.float32)

    inputs = {
        "atom_features": nodes_flat,
        "pair_indices": edges_flat,
        "edge_weights": weights_flat,
        "molecule_indicator": prot_ids,
        "physio_features": physio_flat,
        "seq_ids": seq_ids_padded,
    }
    return inputs, labels

def prepare_batch(batched_emb, batched_physio,batched_seq, batched_labels):
    '''
       -Documentation outdated , no more edges and weights 
       
        You have a batch of graphs represented as ragged tensors(variable lenghts).
        Each graph has its own set of nodes and edges. Here Node features are ProtT5 embeddings.
        Edges are derived from contact maps.
    
        It will prepare batched inputs for the batched graph neural network model.
        Adjust edge indices based on node offsets in the batch.
        Prot_ids indicate which peptide each node belongs to in the batch.

        So, instead of processing each graph individually, we can process the entire batch together.


    '''
    # === RAW SEQUENCE HANDLING ===
    # batched_seq is RaggedTensor of strings (B,)
    # convert to padded tensor of AA indices

    seq_strings = tf.strings.strip(batched_seq)
    # convert to list of characters | CNN needs 2D input so
    chars = tf.strings.unicode_split(seq_strings, "UTF-8")   # (B, L) #eg: ["KLK"] ->[['K', ''L]] (Again ragged)
    # lookup table for AA → index #0 reserved for padding so start from 1
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list("ACDEFGHIKLMNPQRSTVWY")),
            values=tf.constant(list(range(1,21)), dtype=tf.int32)
        ),
        default_value=0
    )
    seq_ids = table.lookup(chars)    # (B, L)  convert to integer tensor
    #pad sequences to same length 
    #to_tensor: automatically pads all sequences in the batch to the max length in that batch.
    seq_ids_padded = seq_ids.to_tensor(default_value=0)  # 0 = padding token

    labels = tf.cast(batched_labels, tf.float32)
    if len(labels.shape) == 1:
        labels = tf.expand_dims(labels, axis=-1)

    num_nodes = batched_emb.row_lengths() #tf.RaggedTensor |  Shape: [batch_size, None, 1024] | (None: number of seqs in this batch)

    # num_edges = batched_edges.row_lengths() #[[1, 2], [0, 2, 5], [2,3]] -> [2, 3, 2] number of edges per graph in batch
    nodes_flat = batched_emb.merge_dims(0, 1) #[batch, nodes, features] -> [total_nodes, features]
    # edges_flat = batched_edges.merge_dims(0, 1)
    # weights_flat = batched_weights.merge_dims(0, 1)

    #Added 0 ,  cumulative sum of num_nodes excluding last element. eg: [0, 2, 4, 5] ->[0, 2, 6, 11] 
    offsets = tf.concat([[0], tf.cumsum(num_nodes)[:-1]], axis=0) #offsets to adjust edge indices for batching
    
    # edge_rowids = batched_edges.value_rowids() #Get which graph, each edge belongs to in the batch
   
    #“Pick rows from a tensor based on an index list. |  gather(input, indices)”
    # edge_offsets = tf.gather(offsets, edge_rowids) #get offset for each edge based on which graph it belongs to
    # edges_flat = edges_flat + tf.cast(tf.expand_dims(edge_offsets, axis=-1), edges_flat.dtype) #adjust edge indices based on node offsets in the batch
   
    prot_ids = tf.repeat(tf.range(tf.shape(num_nodes)[0]), num_nodes) #indicator for which node belongs to which graph in the batch
    prot_ids = tf.cast(prot_ids, tf.int32)
    physio_flat = tf.cast(batched_physio, tf.float32)

    inputs = {
        'atom_features': nodes_flat,
        # 'pair_indices': edges_flat,
        # 'edge_weights': weights_flat,
        'molecule_indicator': prot_ids,
        'physio_features': physio_flat,
        'seq_ids': seq_ids_padded
    }
    return inputs, labels

def load_input_for_integrated_gradients(datasets_index=[0], dataset_path=''):

    datasets = data_util.load_datasets(datasets_index=datasets_index, dataset_path=dataset_path)
    for i, (train_f, val_f, test_f) in enumerate(datasets):
        test_ds = make_dataset(test_f)
        for inputs, labels in test_ds:
            return inputs, labels


def create_arbitrary_background_from_sample(inputs):
    return {
        "atom_features": tf.zeros_like(inputs["atom_features"]),
        "molecule_indicator": tf.zeros_like(inputs["molecule_indicator"]),
        "physio_features": tf.zeros_like(inputs["physio_features"]),
        "seq_ids": tf.zeros_like(inputs["seq_ids"]),
    }


def analyze_peptide_motifs(
    model, peptide_seq, background_inputs, save_dir="./ig_results"
):

    os.makedirs(save_dir, exist_ok=True)

    max_len = cfg.seq_max_len  # 128

    seq_ids = encode_peptide_to_ids(peptide_seq, max_len)
    seq_ids = tf.convert_to_tensor(seq_ids[None, :], dtype=tf.int32)

    fixed_inputs = create_fixed_inputs(background_inputs)

    print("\nComputing CNN motif IG ...")

    cnn_scores, cnn_full_ig = compute_cnn_integrated_gradients(
        model=model, seq_ids=seq_ids, fixed_inputs=fixed_inputs, m_steps=64
    )

    cnn_scores = cnn_scores[: len(peptide_seq)]

    """plot_residue_importance(
        peptide_seq,
        cnn_scores,
        title=f"CNN IG Motif: {peptide_seq}",
        save_path=os.path.join(save_dir, f"{peptide_seq}_cnn_ig.png")
    )"""

    return {"peptide": peptide_seq, "scores": cnn_scores, "full": cnn_full_ig}


def save_ig_results(results, save_dir, file_name):
    """
    Save IG results using numpy pickle serialization.
    """
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, file_name), results, allow_pickle=True)
    print(f"✓ Saved IG results to: {os.path.join(save_dir, file_name)}")


def load_ig_results(save_dir, file_name):
    """
    Load IG results saved with np.save(..., allow_pickle=True).
    Returns a list of dictionaries containing peptide sequences and their corresponding CNN IG scores.
    eg: List[{
        "peptide": peptide_seq,
        "scores": cnn_scores, #List of IG scores for each residue in the peptide
        "full": cnn_full_ig  }. {}, {}, ...{} # List of full IG values for deeper analysis if needed
        ]

    """

    path = os.path.join(save_dir, file_name)
    results = np.load(path, allow_pickle=True).tolist()

    print(f"✓ Loaded {len(results)} IG results from: {path}")
    # print(f"Example entry: {results[0]}")

    # print("\nSample IG results for first 3 peptides:")
    # count = 0
    # for result in results:
    #     print(f"Peptide Length: {len(result['peptide'])}")
    #     print(f"CNN IG Scores: {result['scores']}")
    #     print(f"Peptide: {result['peptide']}")
    #     print(f"Full IG Shape: {result['full'].shape} \n")
    #     count += 1
    #     if count >= 3:  # Print details for first 3 entries only
    #         break

    return results


# =========================================================
# RUN
# =========================================================


def compute_ig(model_name="", model_index=0, dataset_path=''):
    database_index_0 = 0  
    inputs, labels = load_input_for_integrated_gradients([database_index_0], dataset_path=dataset_path)
    model = load_model(model_name, model_index)

    results = []
    seq_batch = inputs["seq_ids"].numpy()

    for i in range(seq_batch.shape[0]):
        peptide = decode_seq_ids(seq_batch[i])
        print(f"Analyzing peptide: {peptide}")

        # Isolate the exact structural nodes belonging to peptide i
        idx_mask = tf.equal(inputs["molecule_indicator"], i)
        sample_atoms = tf.boolean_mask(inputs["atom_features"], idx_mask)
        
        # Reset sample indicator to 0 for a batch size of 1
        sample_indicator = tf.zeros((tf.shape(sample_atoms)[0],), dtype=tf.int32)
        sample_physio = tf.expand_dims(inputs["physio_features"][i], axis=0)

        # Reconstruct a matching context dictionary for the current sequence
        current_context = {
            "atom_features": sample_atoms,
            "molecule_indicator": sample_indicator,
            "physio_features": sample_physio
        }

        result = analyze_peptide_motifs(
            model=model, peptide_seq=peptide, background_inputs=current_context
        )
        results.append(result)

    return results

def extract_seqlets_fixed_window(
    results, threshold_percentile=90, window_size=5, min_len=3
):
    """
    Extract seqlets from IG scores.

    Returns:
        List of dicts:
        {
            "peptide": str,
            "seqlets": [
                {"seq": str, "start": int, "end": int, "score": float}
            ]
        }
    """

    all_seqlets = []

    for item in results:
        peptide = item["peptide"]
        scores = np.array(item["scores"])

        if len(scores) == 0:
            continue

        # --- threshold based on percentile ---
        thresh = np.percentile(scores, threshold_percentile)

        peaks = np.where(scores >= thresh)[0]

        seqlets = []
        visited = set()

        for p in peaks:
            start = max(0, p - window_size)
            end = min(len(scores), p + window_size + 1)

            if any(i in visited for i in range(start, end)):
                continue

            visited.update(range(start, end))

            subseq = peptide[start:end]

            if len(subseq) >= min_len:
                seqlets.append(
                    {
                        "seq": subseq,
                        "start": start,
                        "end": end,
                        "score": float(np.mean(scores[start:end])),
                    }
                )

        all_seqlets.append({"peptide": peptide, "seqlets": seqlets})

    ##Print first 3 peptides and their extracted seqlets for verification
    print("\nSample extracted seqlets for first 3 peptides:")
    for item in all_seqlets[:3]:
        print(f"Peptide: {item['peptide']}")
        print("Extracted Seqlets:")
        for seqlet in item["seqlets"]:
            print(
                f"  Seqlet: {seqlet['seq']}, Start: {seqlet['start']}, End: {seqlet['end']}, Avg Score: {seqlet['score']}"
            )
        print("\n")

    return all_seqlets


def extract_seqlets_adaptive(
    results,
    peak_percentile=90,
    expand_percentile=60,
    min_len=3,
    max_len=25,
    merge_gap=1,
):
    """
    Adaptive seqlet extraction from IG scores.

    Instead of fixed windows, seqlets expand dynamically
    from high-scoring peaks until signal decays.

    Parameters
    ----------
    results : list
        List of dictionaries:
        {
            "peptide": str,
            "scores": list[float]
        }

    peak_percentile : int
        Percentile for selecting strong peaks.

    expand_percentile : int
        Percentile for expansion threshold.
        Lower than peak threshold.

    min_len : int
        Minimum allowed seqlet length.

    max_len : int
        Maximum allowed seqlet length.

    merge_gap : int
        Merge nearby seqlets if gap <= merge_gap.

    Returns
    -------
    all_seqlets : list
    """

    all_seqlets = []

    for item in results:

        peptide = item["peptide"]
        scores = np.array(item["scores"], dtype=float).squeeze()

        if len(scores) == 0:
            continue

        # ---------------------------------------------------
        # Thresholds
        # ---------------------------------------------------

        peak_thresh = np.percentile(scores, peak_percentile)
        expand_thresh = np.percentile(scores, expand_percentile)

        # Strong peak positions
        peaks = np.where(scores >= peak_thresh)[0]

        seqlets = []
        visited = set()

        # ---------------------------------------------------
        # Adaptive expansion from each peak
        # ---------------------------------------------------

        for p in peaks:

            if p in visited:
                continue

            # -------------------------
            # Expand LEFT
            # -------------------------

            left = p

            while left > 0:

                # stop if signal decays
                if scores[left - 1] < expand_thresh:
                    break

                # stop if seqlet too large (Restrict moving left beyond max_len from peak)
                if (p - (left - 1)) > max_len:
                    break

                left -= 1

            # -------------------------
            # Expand RIGHT
            # -------------------------

            right = p

            while right < len(scores) - 1:

                # stop if signal decays
                if scores[right + 1] < expand_thresh:
                    break

                # stop if seqlet too large (Restrict moving right beyond max_len from peak)
                if ((right + 1) - left) > max_len:
                    break

                right += 1

            # python slicing end-exclusive
            start = left
            end = right + 1

            # -------------------------
            # Skip overlaps
            # -------------------------

            if any(i in visited for i in range(start, end)):
                continue

            visited.update(range(start, end))

            subseq = peptide[start:end]

            # -------------------------
            # Length filtering
            # -------------------------

            if len(subseq) >= min_len:

                seqlets.append(
                    {
                        "seq": subseq,
                        "start": start,
                        "end": end,
                        "score": float(np.mean(scores[start:end])),
                        "peak_pos": int(p),
                        "peak_score": float(scores[p]),
                    }
                )

        # ---------------------------------------------------
        # Merge nearby seqlets
        # ---------------------------------------------------

        seqlets = sorted(seqlets, key=lambda x: x["start"])# Sort by start position (Arrange from left to right)

        merged_seqlets = []

        for s in seqlets:

            if not merged_seqlets:
                merged_seqlets.append(s)
                continue

            prev = merged_seqlets[-1] #last merged seqlet

            gap = s["start"] - prev["end"] #start and end are indexes

            if gap <= merge_gap:

                new_start = prev["start"]
                new_end = s["end"]

                merged_seq = peptide[new_start:new_end]

                merged_score = float(np.mean(scores[new_start:new_end]))

                merged_seqlets[-1] = {
                    "seq": merged_seq,
                    "start": new_start,
                    "end": new_end,
                    "score": merged_score,
                    "peak_pos": prev["peak_pos"],
                    "peak_score": max(prev["peak_score"], s["peak_score"]),
                }

            else:
                merged_seqlets.append(s)

        all_seqlets.append({"peptide": peptide, "seqlets": merged_seqlets})

    # ---------------------------------------------------
    # Print samples
    # ---------------------------------------------------

    print("\nSample extracted adaptive seqlets:\n")

    for item in all_seqlets[:3]:

        print(f"Peptide: {item['peptide']}")
        print("Extracted Seqlets:")

        for seqlet in item["seqlets"]:

            print(
                f"  Seqlet: {seqlet['seq']}, "
                f"Start: {seqlet['start']}, "
                f"End: {seqlet['end']}, "
                f"PeakPos: {seqlet['peak_pos']}, "
                f"PeakScore: {seqlet['peak_score']:.4f}, "
                f"AvgScore: {seqlet['score']:.4f}"
            )

        print("\n")

    return all_seqlets

def save_seqlets_to_csv(seqlets, save_path="ig_results/seqlets.csv"):
    
    rows = []
    for item in seqlets:
        peptide = item["peptide"]
        
        for s in item["seqlets"]:
            rows.append({
                "peptide": peptide,
                "seq": s["seq"],
                "start": s["start"],
                "end": s["end"],
                "score": s["score"]
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✓ Saved extracted seqlets to: {save_path}")

def flatten_seqlets(all_seqlets):
    flat = []

    for item in all_seqlets:
        peptide = item["peptide"]

        for s in item["seqlets"]:
            flat.append({
                "seq": s["seq"],
                "score": s["score"],
                "peptide": peptide
            })

    return flat

def compute_kmer_similarity(seqs, k=3):
    
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k)) #(k=min_chunk, max_chunk) slice exact k-mers, eg: data -> tri-mers: dat, ata
    X = vectorizer.fit_transform(seqs) #used in train | learns the vocabulary and builds the initial matrix configuration.
    return cosine_similarity(X)

def compute_ig_similarity_approach_1(scores):
    '''
    Approach 1: Similarity based on score differences.
    Problem: Sensitive to scale and outliers. 
    Large score differences can dominate similarity, while small differences may be ignored.
    Not robust for comparing seqlets with varying score distributions.
    
    If all seqlets have similar score range:
    sim.max() ≈ sim.min() → matrix collapses to ~0 or ~1 noise
    makes clustering unstable and often degenerates into singleton clusters.
    
    
    '''
    
    scores = np.array(scores, dtype=np.float32)

    # normalize
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    #similarity -> 1.0/(1.0 + distance) | Adding 1.0 to the denominator (Saftey, avoid division by zero) | smoothing
    sim = 1.0 / (1.0 + np.abs(scores[:, None] - scores[None, :])) #Pairwise similarity based on score differences | shape (num_seqs, num_seqs) | smaller difference → higher similarity

    # normalize to [0,1]
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

    return sim

def compute_ig_similarity(scores):
    ''' 
    Approach 2: Similarity based on exponential decay of score differences.
    sim = exp(-|s_i - s_j|)
    Benefits:
    More robust to scale and outliers. Exponential decay naturally compresses large differences and amplifies small differences,
    creating a more meaningful similarity landscape.
    Seqlets with similar scores will have high similarity,
    while those with different scores will have rapidly decreasing similarity, 
    improving clustering stability and quality.
    '''
    
    scores = np.array(scores, dtype=np.float32)

    # normalize per-seqlet
    scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    diff = np.abs(scores[:, None] - scores[None, :])

    sim = np.exp(-diff)   # MUCH more stable than 1/(1+diff)

    return sim

def combine_similarity(kmer_sim, ig_sim, alpha=0.90):
    '''Combines k-mer similarity and IG similarity using a weighted average.
    alpha: weight for k-mer similarity (0 to 1)
    (1 - alpha): weight for IG similarity
    '''
    return alpha * kmer_sim + (1 - alpha) * ig_sim

def to_distance(similarity):
    return 1 - similarity

def cluster_distance_matrix(dist_matrix, similarity_threshold=0.6):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - similarity_threshold #less than distance threshold → merge (same cluster) 
    )

    return clustering.fit_predict(dist_matrix) #returns 1D array cluster label/cluster assignments id for each sample

def build_clusters(labels, flat_seqlets):
    '''
        Groups seqlets by their assigned cluster labels.
        
        labels: cluster labels for each seqlet
       flat_seqlets: list of seqlet dicts with keys "seq", "score", "peptide"
    '''
    clusters = {}

    for label, seqlet in zip(labels, flat_seqlets):
        clusters.setdefault(label, []).append(seqlet)
        
    
    # filter noise clusters
    clusters = {
        k: v for k, v in clusters.items()
        if len(v) >= 2
    }

    return clusters



def consensus_motif(seqlets):
    '''
    Given a list of seqlets (dicts with "seq" key), 
    computes a consensus motif by taking the most common amino acid at each position.
    
    '''
    
    max_len = max(len(s["seq"]) for s in seqlets)
    motif = ""

    for i in range(max_len):# Iterate through each position in the longest seqlet
        column = [s["seq"][i] for s in seqlets if i < len(s["seq"])] #Collect the amino acid at position i from all seqlets that are long enough to have that position
        if column:
            #key = column.count : compare max value based on column.count('Residue') 
            motif += max(set(column), key=column.count) #Find the most common amino acid in that column and add it to the motif

    return motif

def cluster_seqlets(all_seqlets, k=3, alpha=0.90, similarity_threshold=0.6):
    """
    Wrapper method to perform TF-MoDISco-style IG-aware clustering (modular version)
    k - k-mer size for sequence similarity
    alpha - weight for combining k-mer and IG similarity
    similarity_threshold - threshold for clustering (0.6 means 60% similarity or higher will
    be merged into the same cluster)
    
    alpha: 0.90 means 90% weight to k-mer similarity and 10% to IG similarity.
    IG scores are noisy; sequence is more stable biologically. So Ig similarity is given less weight to prevent overfitting to noise
    and improve cluster quality.
    
    """

    # 1. flatten
    flattened_seqlets = flatten_seqlets(all_seqlets)

    if len(flattened_seqlets) == 0:
        return []

    seqs = [s["seq"] for s in flattened_seqlets]
    scores = [s["score"] for s in flattened_seqlets]

    # 2. similarities
    kmer_sim = compute_kmer_similarity(seqs, k=k)
    ig_sim = compute_ig_similarity(scores)

    sim = combine_similarity(kmer_sim, ig_sim, alpha=alpha)

    # 3. clustering
    dist = to_distance(sim)
    labels = cluster_distance_matrix(dist, similarity_threshold) #cluster IDs for each sample.

    # 4. build clusters
    clusters = build_clusters(labels, flattened_seqlets) # eg: {cid_0(int): [seqlet1, seqlet2], cid_1: [seqlet3, seqlet4], ...}

    # 5. output
    output = []

    for motif_id, seqlets in clusters.items():
        output.append({
            "motif_id": motif_id,
            "consensus": consensus_motif(seqlets),
            "seqlets": seqlets
        })

    return output

def build_pfm(seqlets, alphabet=AA_ALPHABET):
    """
    Peak-centered Position Frequency Matrix (PFM)
    FIX: avoids left/right bias using center alignment + peak correction
    """

    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

    max_len = max(len(s["seq"]) for s in seqlets)
    center = max_len // 2

    pfm = np.zeros((max_len, len(alphabet)), dtype=np.float32)

    for s in seqlets:
        seq = s["seq"]

        # peak-aware alignment (BEST PRACTICE)
        if "peak_pos" in s:
            offset = center - (s["peak_pos"] - s["start"])
        else:
            offset = center - len(seq) // 2

        for i, aa in enumerate(seq):
            pos = i + offset

            if 0 <= pos < max_len and aa in aa_to_idx:
                pfm[pos, aa_to_idx[aa]] += 1

    return pfm

def build_pwm(seqlets, alphabet=AA_ALPHABET, pseudocount=0.01):
    """
    Converts PFM → PWM with proper pseudocount smoothing.
    """

    pfm = build_pfm(seqlets, alphabet)

    # add pseudocount BEFORE normalization (IMPORTANT FIX)
    pfm = pfm + pseudocount

    pwm = pfm / np.sum(pfm, axis=1, keepdims=True)

    return pwm

def compute_information_content(pwm):
    """
    Shannon information content per position (bits)
    """

    pwm = np.clip(pwm, 1e-8, 1.0)

    entropy = -np.sum(pwm * np.log2(pwm), axis=1)

    max_entropy = np.log2(pwm.shape[1])

    ic = max_entropy - entropy

    return ic

def plot_motif_logo(pwm, title="", save_path=None):

    L, A = pwm.shape

    # 🔥 FIX 1: compact figure for short motifs
    fig_width = max(2, L * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 2.5))

    ic = compute_information_content(pwm)

    for i in range(L):

        column = pwm[i]
        sorted_idx = np.argsort(column)

        y_offset = 0.0

        for idx in sorted_idx:

            prob = column[idx]
            if prob < 0.01:
                continue

            aa = AA_ALPHABET[idx]
            height = prob * ic[i]

            #  FIX 2: capped font size
            fontsize = min(18, 6 + height * 20)

            ax.text(
                i,
                y_offset,
                aa,
                fontsize=fontsize,
                ha="center",
                va="bottom",
                fontweight="bold"
            )

            y_offset += height

    # 🔥 FIX 3: tighter axis control
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(L))
    ax.set_xticklabels(range(1, L + 1), fontsize=8)

    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(0, max(1.0, np.max(ic) + 0.2))

    ax.set_xlabel("Position", fontsize=9)
    ax.set_ylabel("Information", fontsize=9)
    ax.set_yticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def filter_clusters(clusters, min_consensus=0.4, min_seqlets=2):
    """
    Remove noisy / unstable motifs before logo generation.
    """

    filtered = []

    for c in clusters:

        if len(c["seqlets"]) < min_seqlets:
            continue

        strength = compute_consensus_strength(c["seqlets"])

        if strength < min_consensus:
            continue

        filtered.append(c)

    return filtered
    
    
def normalize_clusters(cluster_output):
    """
    Ensures plotting-compatible format:

    [
        {
            "motif_id": int,
            "seqlets": [{"seq": str, "score": float}]
        }
    ]
    """

    # CASE 1: already correct format
    if isinstance(cluster_output, list) and len(cluster_output) > 0:
        first = cluster_output[0]

        if isinstance(first, dict) and "seqlets" in first:
            return cluster_output

    # CASE 2: flattened seqlet rows (your actual case)
    from collections import defaultdict

    grouped = defaultdict(list)

    for r in cluster_output:
        grouped[r["motif_id"]].append({
            "seq": r["seq"],
            "score": r.get("score", 0.0),
            "start": r.get("start", -1),
            "end": r.get("end", -1),
        })

    return [
        {
            "motif_id": k,
            "seqlets": v
        }
        for k, v in grouped.items()
    ]
    
def save_clusters_to_csv(cluster_output, save_path):
    """
    Save clustered seqlets to CSV (flattened format).

    Args:
        cluster_output: output from cluster_seqlets()
        save_path: file path ending in .csv
    """

    rows = []

    for cluster in cluster_output:
        motif_id = cluster["motif_id"]
        consensus = cluster["consensus"]

        for s in cluster["seqlets"]:
            rows.append({
                "motif_id": motif_id,
                "consensus": consensus,
                "peptide": s.get("peptide", ""),
                "seqlet": s["seq"],
                "start": s.get("start", -1),
                "end": s.get("end", -1),
                "score": s.get("score", 0.0),
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"✓ Saved Clusters {len(df)} seqlets to {save_path}")
    
def compute_consensus_strength(seqlets):
    """
    Measures how conserved the motif is.

    Returns:
        value between 0 and 1
    """

    if len(seqlets) == 0:
        return 0.0

    sequences = [s["seq"] for s in seqlets]

    max_len = max(len(s) for s in sequences)

    # pad sequences
    padded = [
        seq.ljust(max_len, "-")
        for seq in sequences
    ]

    conservation_scores = []

    for pos in range(max_len):

        residues = [seq[pos] for seq in padded]

        counts = {}

        for aa in residues:
            counts[aa] = counts.get(aa, 0) + 1

        max_freq = max(counts.values()) / len(residues)

        conservation_scores.append(max_freq)

    return np.mean(conservation_scores)

def generate_top_motif_logos(
    clusters,
    ranked_motifs,
    top_k=10,
    save_dir=""
):

    os.makedirs(save_dir, exist_ok=True)

    # map motif_id → cluster
    cluster_map = {c["motif_id"]: c for c in clusters}

    for motif in ranked_motifs[:top_k]:

        motif_id = motif["motif_id"]

        if motif_id not in cluster_map:
            continue

        cluster = cluster_map[motif_id]
        seqlets = cluster["seqlets"]

        if len(seqlets) < 2:
            continue

        # build PWM
        pwm = build_pwm(seqlets)

        title = (
            f"Motif {motif_id} | "
            f"Score: {motif['score']:.3f} | "
            f"n={len(seqlets)} | "
            f"{motif.get('consensus','')}"
        )

        save_path = os.path.join(save_dir, f"motif_{motif_id}.png")

        plot_motif_logo(
            pwm,
            title=title,
            save_path=save_path
        )

    print(f"✓ Saved motif logos to: {save_dir}")

# ---------------------------------------------------
# Motif importance score
# ---------------------------------------------------
def compute_motif_importance(cluster):
    """
    motif_score =
        mean(seqlet_score)
        * log(number_of_seqlets + 1)
        * consensus_strength
    """

    seqlets = cluster["seqlets"]

    if len(seqlets) == 0:
        return 0.0

    scores = [
        abs(s.get("score", 0.0))
        for s in seqlets
    ]

    mean_score = np.mean(scores)
    
    global_max = max(
        abs(s.get("score",0))
        for c in clusters
        for s in c["seqlets"]
    )
    
    mean_score /= (global_max + 1e-8)

    n_seqlets = len(seqlets)

    consensus_strength = compute_consensus_strength(seqlets)

    '''motif_score = (
        mean_score
        * np.log(n_seqlets + 1)
        * consensus_strength
    )'''
    
    motif_score = (
        mean_score
        * np.sqrt(n_seqlets)
        * consensus_strength
    )

    return motif_score
# ---------------------------------------------------
# Ranked motif summary plot
# ---------------------------------------------------
def rank_motifs(clusters, top_n=None):
    """
    Compute and rank motif importance.

    Returns:
        [
            {
                "motif_id": int,
                "score": float,
                "consensus": str,
                "n_seqlets": int #Number of seqlets in the cluster (motif support)
            }
        ]
    """

    motif_scores = []

    for cluster in clusters:

        motif_scores.append({
            "motif_id": cluster["motif_id"],
            "score": compute_motif_importance(cluster),
            "consensus": cluster.get("consensus", ""),
            "n_seqlets": len(cluster["seqlets"])
        })

    motif_scores = sorted(
        motif_scores,
        key=lambda x: x["score"],
        reverse=True
    )

    if top_n is not None:
        motif_scores = motif_scores[:top_n]

    return motif_scores


def save_ranked_motifs(ranked_motifs,
                       save_path="ig_results/ranked_motifs.csv"):
    """
    Save ranked motifs to CSV.

    Args:
        ranked_motifs:
            output from rank_motifs()

        save_path:
            CSV file path
    """

    if len(ranked_motifs) == 0:
        print("No ranked motifs to save.")
        return

    df = pd.DataFrame(ranked_motifs)

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        f"✓ Saved {len(df)} ranked motifs to {save_path}"
    )

    return df

def plot_ranked_motifs(
    ranked_motifs,
    save_path="ig_results/ranked_motifs.png"
):
    """
    Plot motif importance ranking (single-column journal optimized).
    """

    motif_ids = [
        f'M{m["motif_id"]}'
        for m in ranked_motifs
    ]

    scores = [
        m["score"]
        for m in ranked_motifs
    ]

    labels = [
        f'{motif_id}\n{m["consensus"]}'
        for motif_id, m in zip(motif_ids, ranked_motifs)
    ]

    n = len(scores)

    # -----------------------------
    # Single-column journal size
    # Typical width: 3.3–3.6 inches
    # -----------------------------
    fig_width = 3.6
    fig_height = max(2.8, min(0.35 * n + 1.2, 4.5))

    plt.figure(figsize=(fig_width, fig_height))

    bars = plt.bar(
        range(n),
        scores,
        width=0.65
    )

    # -----------------------------
    # Axis styling (journal compact)
    # -----------------------------
    plt.xticks(
        range(n),
        labels,
        rotation=45,
        ha="right",
        fontsize=8
    )

    plt.yticks(fontsize=8)

    plt.ylabel("Motif Importance Score", fontsize=9)
    plt.xlabel("Motif", fontsize=9)
    plt.title("Ranked Motif Summary", fontsize=10, pad=6)

    # -----------------------------
    # Value annotations (smaller + cleaner)
    # -----------------------------
    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0
        )

    plt.tight_layout(pad=0.6)

    plt.savefig(
        save_path,
        dpi=600,  # higher DPI for publication
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

    print(f"✓ Saved ranked motif plot to {save_path}")

###Dendogram 
def pwm_to_vector(pwm, max_len):
    """
    Convert PWM to fixed-length vector.

    Pads shorter PWMs with zeros.
    """

    L, A = pwm.shape

    padded = np.zeros((max_len, A), dtype=np.float32)

    padded[:L] = pwm

    return padded.flatten()

def compute_pwm_similarity(clusters):
    """
    Compute motif similarity using PWMs.

    Returns
    -------
    similarity_matrix
    motif_ids
    """

    pwms = []

    max_len = 0

    for cluster in clusters:

        pwm = build_pwm(cluster["seqlets"])

        pwms.append(pwm)

        max_len = max(max_len, pwm.shape[0])

    vectors = [
        pwm_to_vector(pwm, max_len)
        for pwm in pwms
    ]

    vectors = np.array(vectors)

    similarity = cosine_similarity(vectors)

    motif_ids = [
        c["motif_id"]
        for c in clusters
    ]

    return similarity, motif_ids

def build_consensus_matrix(clusters):
    """
    Convert consensus motifs into a fixed-length matrix.

    Returns
    -------
    matrix : ndarray (n_motifs, max_len)
    motif_ids : list
    """

    motifs = [
        c["consensus"]
        for c in clusters
    ]

    max_len = max(len(m) for m in motifs)

    matrix = []

    for motif in motifs:

        row = list(motif.ljust(max_len, "-"))

        matrix.append(row)

    return np.array(matrix), max_len

def get_dendrogram_order(clusters):
    """
    Returns leaf ordering from PWM dendrogram.
    """

    from scipy.cluster.hierarchy import linkage
    from scipy.cluster.hierarchy import dendrogram
    from scipy.spatial.distance import squareform

    similarity, motif_ids = compute_pwm_similarity(clusters)

    distance = 1 - similarity

    np.fill_diagonal(distance, 0)

    condensed = squareform(distance)

    linkage_matrix = linkage(
        condensed,
        method="average",
        optimal_ordering=True
    )

    dendro = dendrogram(
        linkage_matrix,
        no_plot=True
    )

    return linkage_matrix, dendro["leaves"]

def reorder_consensus_matrix(
    consensus_matrix,
    clusters,
    leaf_order
):
    """
    Reorder rows according to dendrogram leaves.
    """

    reordered_matrix = consensus_matrix[leaf_order]

    reordered_clusters = [
        clusters[i]
        for i in leaf_order
    ]

    return reordered_matrix, reordered_clusters

def get_residue_color(aa):
    """
    AMP-oriented residue classes.
    """

    POSITIVE = {"K", "R", "H"}
    NEGATIVE = {"D", "E"}
    HYDROPHOBIC = {"W", "F", "I", "L", "V", "M", "A"}
    POLAR = {"S", "T", "N", "Q", "Y"}
    SPECIAL = {"G", "P", "C"}

    if aa in POSITIVE:
        return "#4C72B0"  # blue

    if aa in NEGATIVE:
        return "#C44E52"  # red

    if aa in HYDROPHOBIC:
        return "#DD8452"  # orange

    if aa in POLAR:
        return "#55A868"  # green

    if aa in SPECIAL:
        return "#8172B2"  # purple

    return "#DDDDDD"



def plot_consensus_heatmap(
    ax,
    consensus_matrix
):
    from matplotlib.patches import Rectangle
    n_rows, n_cols = consensus_matrix.shape

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)

    ax.invert_yaxis()

    for r in range(n_rows):

        for c in range(n_cols):

            aa = consensus_matrix[r, c]

            color = get_residue_color(aa)

            ax.add_patch(
                Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1
                )
            )

            ax.text(
                c + 0.5,
                r + 0.5,
                aa,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black"
            )

    ax.set_xticks(np.arange(n_cols) + 0.5)

    ax.set_xticklabels(
        range(1, n_cols + 1),
        fontsize=9
    )

    ax.set_yticks([])

    ax.set_title(
        "Consensus Residues",
        fontsize=12
    )

    for spine in ax.spines.values(): #top left right bottom borders lines
        spine.set_visible(False)

def add_residue_legend(ax):
    from matplotlib.patches import Patch

    handles = [

        Patch(
            facecolor="#4C72B0",
            label="Positive (K,R,H)"
        ),

        Patch(
            facecolor="#DD8452",
            label="Hydrophobic"
        ),

        Patch(
            facecolor="#55A868",
            label="Polar"
        ),

        Patch(
            facecolor="#C44E52",
            label="Negative"
        ),

        Patch(
            facecolor="#8172B2",
            label="Special"
        )
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=9
    )

def plot_dendrogram_with_consensus_heatmap(
    clusters,
    score_map,
    save_path="ig_results/dendrogram_consensus_heatmap.png"
):

    linkage_matrix, leaf_order = get_dendrogram_order(clusters)

    consensus_matrix, max_len = build_consensus_matrix(clusters)

    consensus_matrix, ordered_clusters = reorder_consensus_matrix(
        consensus_matrix,
        clusters,
        leaf_order
    )

    labels = [
        f"{c['consensus']} (n={len(c['seqlets'])})"
        for c in ordered_clusters
    ]

    n_motifs = len(clusters)

    # -----------------------------
    # Two-column journal sizing
    # Typical width: 7 inches total
    # -----------------------------
    fig_width = 7.0

    # Keep height compact (avoid overly tall figures)
    fig_height = max(3.5, min(0.35 * n_motifs, 6.5))

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[2.2, 1.0],  # slightly more compact dendrogram
        wspace=0.08
    )

    # ---------------------
    # Dendrogram
    # ---------------------
    ax_tree = fig.add_subplot(gs[0])

    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="right",
        leaf_font_size=7,  # smaller for journal
        color_threshold=np.mean(linkage_matrix[:, 2]),
        ax=ax_tree
    )

    ax_tree.set_title(
        "PWM Similarity Dendrogram",
        fontsize=11,
        pad=6
    )

    ax_tree.set_xlabel(
        "PWM Distance",
        fontsize=10
    )

    ax_tree.tick_params(axis='y', labelsize=7)

    # ---------------------
    # Consensus heatmap
    # ---------------------
    ax_heat = fig.add_subplot(gs[1])

    plot_consensus_heatmap(ax_heat, consensus_matrix)

    add_residue_legend(ax_heat)

    ax_heat.set_title(
        "Consensus",
        fontsize=11,
        pad=6
    )

    plt.tight_layout(pad=0.4)

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()

def plot_pwm_dendrogram(
    clusters,
    score_map,
    save_path="ig_results/pwm_dendrogram.png"
):

    

    similarity, motif_ids = compute_pwm_similarity(clusters)

    distance = 1 - similarity

    np.fill_diagonal(distance, 0)

    condensed = squareform(distance)

    linkage_matrix = linkage(
        condensed,
        method="average",
        optimal_ordering=True
    )

    labels = []

    for c in clusters:

        labels.append(
            f"{c['consensus']} "
            f"(n={len(c['seqlets'])})"
        )

    n_motifs = len(clusters)

    fig_height = max(8, n_motifs * 0.4)

    plt.figure(figsize=(12, fig_height))

    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="right",
        leaf_font_size=8,
        color_threshold=np.mean(linkage_matrix[:,2])
    )

    plt.xlabel("PWM Distance")
    plt.title("Hierarchical Clustering of Discovered Motifs")

    cutoff = 0.45

    plt.axvline(
        cutoff,
        linestyle="--",
        linewidth=1
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

def filter_amp_motifs(clusters):
    filtered = []

    for c in clusters:
        motif = c["consensus"]

        # ---- hard filters ----
        if is_low_complexity_motif(motif):
            continue

        net_charge = motif_net_charge(motif)
        hydro = motif_hydrophobicity(motif)
        hmoment = motif_hydrophobic_moment(motif)
        amph = motif_amphipathicity(motif)

        amp_score = amp_validity_score(
            motif,
            net_charge,
            hydro,
            hmoment,
            amph
        )

        # ---- AMP acceptance rule ----
        if (
            amp_score > 1.5 and          # main biological threshold
            net_charge >= 1 and         # AMPs are usually cationic
            len(motif) >= 3
        ):
            c["amp_score"] = amp_score
            filtered.append(c)

    return filtered

def load_clusters_from_csv(csv_path):

    import pandas as pd

    df = pd.read_csv(csv_path)

    clusters = []

    for motif_id, group in df.groupby("motif_id"):

        seqlets = []

        for _, row in group.iterrows():
            seqlets.append({
                "seq": row["seqlet"],   # FIX: treat seqlet as seq
                "peptide": row["peptide"],
                "start": row["start"],
                "end": row["end"],
                "score": row["score"]
            })

        clusters.append({
            "motif_id": motif_id,
            "consensus": group["consensus"].iloc[0],
            "seqlets": seqlets
        })

    return clusters

def load_ranked_motifs(csv_path):
    """
    Load ranked motifs saved from rank_motifs().
    Returns: list of dicts sorted by score (pipeline-compatible).
    """

    df = pd.read_csv(csv_path)

    # Ensure correct types
    df["motif_id"] = df["motif_id"].astype(int)
    df["score"] = df["score"].astype(float)
    df["n_seqlets"] = df["n_seqlets"].astype(int)

    # Sort by score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Convert to list-of-dicts (IMPORTANT FIX)
    ranked_motifs = df.to_dict(orient="records")

    return ranked_motifs

if __name__ == "__main__":
    base_dir = "result/motif_discovery"
    dataset_path = data_util.DATASET_PATH_ECOLI_PROTT5
    ig_result_npy_file = "ig_results.npy"
    peak_percentile = 85 #50
    expand_percentile = 65 #55

    merge_gap = 1 #0

    alpha = 0.90
    similarity_threshold = 0.70
    
    # results = compute_ig(model_name=MODEL_NAME, model_index=MODEL_INDEX, dataset_path=dataset_path)
    # save_ig_results(results, save_dir='result', file_name=ig_result_npy_file)

    '''loaded_results = load_ig_results(save_dir='result', file_name=ig_result_npy_file)

    

    extract_seqlets_adaptive_from_results = extract_seqlets_adaptive(
        loaded_results,
        peak_percentile=peak_percentile,
        expand_percentile=expand_percentile,
        min_len=3,
        max_len=25,
        merge_gap=merge_gap, #1
    )

    save_seqlets_to_csv(extract_seqlets_adaptive_from_results, save_path=f"{base_dir}/seqlets.csv")
    
    
    clusters = cluster_seqlets(extract_seqlets_adaptive_from_results, k=3, alpha=alpha, similarity_threshold=similarity_threshold)
    
    save_clusters_to_csv(clusters, f"{base_dir}/motifs.csv")'''
   
    #Load clusters from CSV (if already computed)
    clusters = load_clusters_from_csv(f"{base_dir}/motifs.csv")
    clusters = filter_clusters(clusters)
    
    # NEW STEP: AMP biological filtering
    amp_clusters = filter_amp_motifs(clusters)
    
    # ranked_motifs = rank_motifs(amp_clusters, top_n=12)
    ranked_motifs = load_ranked_motifs(f"{base_dir}/ranked_motifs_A0.9_S0.7.csv")

    score_map = {
        m["motif_id"]: m["score"]
        for m in ranked_motifs
    }

    plot_dendrogram_with_consensus_heatmap(
        amp_clusters,
        score_map,
        save_path=f"{base_dir}/dendrogram_amp_only_A{alpha}_S{similarity_threshold}_NNNNN.png"
    )


    # save_ranked_motifs(
    #     ranked_motifs,
    #     save_path=f"{base_dir}/ranked_motifs_A{alpha}_S{similarity_threshold}.csv"
    # )

    plot_ranked_motifs(
        ranked_motifs,
        save_path=f"{base_dir}/ranked_motifs_A{alpha}_S{similarity_threshold}_NNNNNN.png"
    )

    # save_motif_analysis(
    #     csv_input_path=f"{base_dir}/ranked_motifs_A{alpha}_S{similarity_threshold}.csv",
    #     output_csv_path=f"{base_dir}/analyzed_motifs_A{alpha}_S{similarity_threshold}.csv"
    # )