#!/usr/bin/env python3
"""
Dual-branch GNN + Transformer for raw MIC regression
Features:
- ProtT5 embeddings
- GNN on contact map
- Physico-chemical + Blosum features
- Transformer encoder for sequence branch
- Attention-based fusion of branches
- Predict raw MIC directly
"""
import argparse
import os
from pathlib import Path
import random
from dataclasses import asdict
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, kendalltau

import data_util, data_visualization
from experimental_config import ExperimentConfig
from Model_Dual import GraphAttentionNetwork,  TransformerEncoderReadout, CrossAttentionFusion
from sequence_cnn import SequenceCNN, DilatedConvAttention
from validation_pearson import ValPearsonCallback
# --------------------------- GPU Setup ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# --------------------------- Hyperparameters ---------------------------
cfg = ExperimentConfig()
EMB_DIM = 1024 #For ProtT5 and ProtBert Model 
# EMB_DIM = 1280 #For ESM2 Model

PHYSIO_DIM = cfg.physio_feature_dim + cfg.blosum_feature_dim + cfg.sinusoidal_feature_dim  #32+20+32+=84 #using only physio physio (32+ ncbias:9(not used)) + Blosum features(20) + Sinusoidal PE(32)

from physiochem_feature_extractor import PHYSIO_LABELS
# --------------------------- Reproducibility ---------------------------
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_global_seed(cfg.seed)

# --------------------------- Dataset ---------------------------
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

# --------------------------- Model ---------------------------
class ImprovedDualBranchGNN_AttentionFusion(keras.Model):
    """Dual-branch GNN + Transformer for raw MIC regression"""
    def __init__(self, cfg: ExperimentConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # Sequence branch (data from  node-features ie. ProtT5 embeddings)
        self.seq_proj = layers.Dense(cfg.seq_dense1, activation=cfg.seq_proj_A)
        self.seq_transformer = TransformerEncoderReadout(
            num_heads=cfg.transformer_heads,
            embed_dim=cfg.seq_dense1,
            dense_dim=cfg.transformer_ff_dim,
            batch_size=cfg.batch_size
        )
        self.seq_out = layers.Dense(cfg.seq_dense2, activation=cfg.seq_out_A)
        self.seq_bottleneck = layers.Dense(cfg.seq_bottleneck_dim, activation=cfg.seq_bottleneck_A)

        # **Residual Dense
        # self.seq_skip_dense = layers.Dense(128)  # performance dropped 
        
        # Graph branch
        '''self.gnn = GraphAttentionNetwork(
            atom_dim=EMB_DIM, hidden_units=cfg.gnn_hidden,
            num_heads=cfg.gnn_heads, num_layers=cfg.gnn_layers)'''
        
        # self.gnn_proj = layers.Dense(cfg.gnn_dense, activation=cfg.gnn_proj_A)
        # self.gnn_bottleneck = layers.Dense(cfg.gnn_bottleneck_dim, activation=cfg.gnn_bottleneck_A) 
        
        # Physio branch
        self.physio_proj = layers.Dense(cfg.physio_proj, activation=cfg.physio_proj_A) 
        # self.blosum_proj = layers.Dense(cfg.physio_proj, activation=cfg.physio_proj_A)#New 
        # self.sinus_proj  = layers.Dense(cfg.physio_proj, activation=cfg.physio_proj_A)#New
       
        # self.pos_proj    = layers.Dense(cfg.pos_proj, activation=cfg.physio_proj_A)#New

        self.physio_bottleneck = layers.Dense(cfg.physio_bottleneck_dim, activation=cfg.physio_bottleneck_A) 

        #Cross attention fusion layer
        self.cross_attn_fusion = CrossAttentionFusion(dim=cfg.seq_bottleneck_dim, num_heads=4, dropout=0.1)

        #seq cnn branch 
        #self.seq_cnn = SequenceCNN()
        self.seq_cnn = DilatedConvAttention(hidden_dim=128, num_heads=4, dropout=0.1)
        self.seq_cnn_norm = layers.LayerNormalization() 
        self.seq_cnn_dropout = layers.Dropout(cfg.fuse_dropout)

       
        # Fusion
        self.attn_dense = layers.Dense(1)
        self.fuse_norm = layers.LayerNormalization()
        self.fuse_dropout = layers.Dropout(cfg.fuse_dropout)
        
        # Output: raw MIC
        self.out = layers.Dense(1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "cfg": asdict(self.cfg)
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        cfg_dict = config.pop("cfg")
        # Convert dict back to dataclass
        cfg = ExperimentConfig(**cfg_dict)
        return cls(cfg, **config)
    
    def modality_wise_projection(self, physio):
        '''
        # Split physio modalities
        # ----------------------------
        physio_base = physio[:, 0:32] #32
        blosum      = physio[:, 32:52] #20
        sinusoid    = physio[:, 52:84] #32
        position    = physio[:, 84:144] #60

        # Project each modality with 32 units
        # physio_p = self.physio_proj(physio_base)    
        # blosum_p = self.blosum_proj(blosum)          
        # sinus_p  = self.sinus_proj(sinusoid)         
        # pos_p    = self.pos_proj(position)          

        # Fuse all global features
        # ----------------------------
        physio_feat = tf.concat([physio_p, blosum_p, sinus_p, pos_p], axis=-1)   # (B, 128) | concat along last dim
        
        
        physio_feat = self.physio_bottleneck(physio_feat)

        return physio_feat'''
        return None
    
    def modality_wise_mini_projection(self, physio):
        '''# Split physio modalities
        # ----------------------------
        physio_base = physio[:, 0:32] #32
        blosum      = physio[:, 32:52] #20
        sinusoid    = physio[:, 52:84] #32
        position    = physio[:, 84:144] #60

        # reduce 60 features into 12 features
        pos_p    = self.pos_proj(position)          

        # Fuse all global features
        # ----------------------------
        physio_feat = tf.concat([physio_base, blosum, sinusoid, pos_p], axis=-1)   # (B, 96)

        physio_feat = self.physio_proj(physio_feat) # Convert it to 32 dim 
        physio_feat = self.physio_bottleneck(physio_feat) #128 dim

        return physio_feat'''
        return None


    def call(self, inputs, training=False):
        nodes = inputs['atom_features'] # (total_nodes, EMB_DIM): ProtT5 embeddings (residue label emb)
        # edges = inputs['pair_indices'] #binary contact map edges
        # weights = inputs['edge_weights'] #binary edge weights (distance-based)
        prot_ids = inputs['molecule_indicator'] #indicator index for sequence/protein id in batch
        physio = inputs['physio_features']
        seq_ids = inputs['seq_ids']

        # Sequence branch
        mean_pool = tf.math.unsorted_segment_mean(nodes, prot_ids, tf.reduce_max(prot_ids) + 1) #It computes one graph-level embedding per protein by averaging all node embeddings that belong to the same protein.
        seq = self.seq_proj(mean_pool)
        seq = self.seq_transformer(seq, training=training)
        seq = self.seq_out(seq)
        seq_feat = self.seq_bottleneck(seq)
        
        # Physio branch
        physio_feat = self.physio_proj(physio) #Physio + Blosum + sinu + pos-aware features 
        physio_feat = self.physio_bottleneck(physio_feat)

        #cnn feature 
        seq_cnn_feat = self.seq_cnn(seq_ids) #128 dim
        seq_cnn_feat =  self.seq_cnn_norm(seq_cnn_feat) #stabilizes the scale of activations across the features
        seq_cnn_feat = self.seq_cnn_dropout(seq_cnn_feat, training=training)

        # CNN Modulation
        seq_feat = seq_feat * (1 + cfg.cnn_gating_threshold * tf.tanh(seq_cnn_feat)) #gating trick(used in alphafold) for cnn+protT5

        # Fusion based output testing(Ablation Test Cross Attention Fusion)
        '''fused = self.get_cross_attention_weighted_feature_fused(training=training, seq_feat=seq_feat, gnn_feat=None, physio_feat=physio_feat) #gnn feature removed (Perf. dropped)
        return self.out(fused)'''

        #Attention weighted fusion 
        '''fused = self.get_attention_weighted_feature_fused(training=training, seq_feat=seq_feat, gnn_feat=None, physio_feat=physio_feat, seq_cnn_feat=seq_cnn_feat) 
        return self.out(fused)'''

        # Output
        return self.out(seq_feat + physio_feat)

    def get_attention_weighted_feature_fused(self, training, seq_feat, gnn_feat, physio_feat, seq_cnn_feat):
        # concat_feats = tf.stack([seq_feat, gnn_feat, physio_feat, seq_cnn_feat], axis=1) #cnn based
        # concat_feats = tf.stack([seq_feat, gnn_feat, physio_feat], axis=1)
        concat_feats = tf.stack([seq_feat,  physio_feat], axis=1)
        # concat_feats = tf.stack([seq_feat, physio_feat], axis=1) #gnn feautre removed (Perf. dropped)
        attn_scores = tf.nn.softmax(self.attn_dense(concat_feats), axis=1)
        fused = tf.reduce_sum(concat_feats * attn_scores, axis=1) #Weighted sum based on attention scores
        fused = self.fuse_norm(fused)
        fused = self.fuse_dropout(fused, training=training)
        return fused
    
    def get_cross_attention_weighted_feature_fused(self, training, seq_feat, gnn_feat, physio_feat):
        fused, attn_scores = self.cross_attn_fusion(
            seq_feat,
            gnn_feat,
            physio_feat,
            training=training

        )
        fused = self.fuse_norm(fused)
        fused = self.fuse_dropout(fused, training=training)

        return fused

# --------------------------- Metrics ---------------------------
def compute_metrics(model, dataset, model_path, model_index=0, visualize=False, write_results=False): 
    y_true, y_pred = [], []
    for inputs, labels in dataset:
        preds = model(inputs, training=False)
        y_true.extend(labels.numpy().flatten())
        y_pred.extend(preds.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "R2": r2_score(y_true, y_pred),
        "Pearson": pearsonr(y_true, y_pred)[0],
        "Kendall": kendalltau(y_true, y_pred)[0]
    }

    if write_results:
        write_path = f"{model_path}/result_{model_index}.txt"
        with open(write_path, "w") as f:
            f.write(f"{metrics}\n")

        print(f"Results written to {write_path}")
    
    if visualize:
        save_path = f"{model_path}/correlation_{model_index}.png"
        data_visualization.plot_correlation(y_true, y_pred, save_path=save_path)
    return metrics

def get_model_path(model_name):
    model_path = Path(f"./model/{model_name}")
    model_path.mkdir(parents=True, exist_ok=True)

    #Save configuration Parameters
    cfg_save_path = model_path / f"config.json"
    return model_path, cfg_save_path
# --------------------------- Training ---------------------------
def train_model(train_f, val_f, test_f, model_name, mdl_index):
    train_ds = make_dataset(train_f, shuffle=True)
    val_ds = make_dataset(val_f)
    test_ds = make_dataset(test_f)

    model = ImprovedDualBranchGNN_AttentionFusion(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss=tf.keras.losses.MeanSquaredError(), #tf.keras.losses.Huber(delta=1.0)
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    
    

    model_path, config_path = get_model_path(model_name)
   
    cfg.save_config(config_path) #Save configuration Parameters
    
    val_pearson_history = []

    callbacks = [
        EarlyStopping(monitor="val_rmse", mode="min", patience=cfg.patience, restore_best_weights=True),
        ModelCheckpoint(str(model_path / f"model_{mdl_index}"), save_best_only=True, monitor='val_rmse', mode='min'),
        ReduceLROnPlateau(monitor="val_rmse", factor=0.5, patience=50, min_lr=1e-6),
        ValPearsonCallback(val_ds, val_pearson_history)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=2)

    print(f"*******Model Summary****** {model.summary()}")

    history.history["val_Pearson"] = val_pearson_history
    data_visualization.plot_train_val_with_pearson(
        history,
        save_path=model_path / f"loss_curve_fold{mdl_index}.png"
    )


    metrics = compute_metrics(model, test_ds, model_path, model_index=mdl_index)
    print(f"[Fold {mdl_index}] Test metrics: {metrics}")
    return model, metrics

def load_model_and_evaluate_test(
    model_name, datasets_index=[0], dataset_path=''):
    """
    Load a saved model from disk and evaluate on test_ds
    using the existing compute_metrics method.
    """
    metrics = []
    datasets = data_util.load_datasets(datasets_index=datasets_index, dataset_path=dataset_path)
    
    for model_index in datasets_index:
        model_dir = Path("model") / model_name / f"model_{model_index}"

        # _, _, test_f = datasets[model_index]
        _, _, test_f = datasets[datasets_index.index(model_index)]
        test_ds = make_dataset(test_f)

        # Load model with required custom objects
        model = tf.keras.models.load_model(
            model_dir,
            custom_objects={
                "ImprovedDualBranchGNN_AttentionFusion": ImprovedDualBranchGNN_AttentionFusion,
                "GraphAttentionNetwork": GraphAttentionNetwork,
                "TransformerEncoderReadout": TransformerEncoderReadout,
                "DilatedConvAttention": DilatedConvAttention,
            },
            compile=False
        )
        print(f"*******Loaded Model Summary****** {model.summary()}")

        print(f"Evaluating dataset {dataset_index} with model_{dataset_index}")


        # Reuse existing metric computation
        metric = compute_metrics(
            model=model,
            dataset=test_ds,
            model_path= f"model/{model_name}",
            model_index=model_index
        )
        metrics.append(metric)

    print("All metrics:", metrics)

    return metrics


# --------------------------- Execute ---------------------------
def execute(model_name, datasets_index=[0], save_file='metrics_results.csv', dataset_path=''):
    datasets = data_util.load_datasets(datasets_index=datasets_index, dataset_path=dataset_path)
    all_metrics = []
    for i, (train_f, val_f, test_f) in enumerate(datasets):
        _, metrics = train_model(train_f, val_f, test_f, model_name, i)
        all_metrics.append(metrics)
    
    model_path, config_path = get_model_path(model_name)
    data_util.save_results_table(all_metrics, filename=str(model_path / save_file))

    return all_metrics

def load_model(model_name, model_index):
    """
    Load a saved model from disk and evaluate on test_ds
    using the existing compute_metrics method.
    """
    model_dir = Path("./model") / model_name / f"model_{model_index}"
    # Load model with required custom objects
    model = tf.keras.models.load_model(
        model_dir,
        custom_objects={
            "ImprovedDualBranchGNN_AttentionFusion": ImprovedDualBranchGNN_AttentionFusion,
            "GraphAttentionNetwork": GraphAttentionNetwork,
            "TransformerEncoderReadout": TransformerEncoderReadout,
            "DilatedConvAttention": DilatedConvAttention,
        },
        compile=False
    )
    print(f"*******Loaded Model Summary****** {model.summary()}")
    return model

def load_input_for_integrated_gradients(datasets_index=[0]):
    datasets = data_util.load_datasets(datasets_index=datasets_index)
    for i, (train_f, val_f, test_f) in enumerate(datasets):
        test_ds = make_dataset(test_f)
        for inputs, labels in test_ds:
            return inputs, labels
        
def integrated_gradients(model, inputs, m_steps=32):
    """
    Compute IG for atom_features and physio_features only
    """
    atom = tf.cast(inputs["atom_features"], tf.float32)
    phys = tf.cast(inputs["physio_features"], tf.float32)

    atom_base = tf.zeros_like(atom)
    phys_base = tf.zeros_like(phys)

    total_grad_atom = tf.zeros_like(atom)
    total_grad_phys = tf.zeros_like(phys)

    for alpha in tf.linspace(0.0, 1.0, m_steps):
        atom_interp = atom_base + alpha * (atom - atom_base)
        phys_interp = phys_base + alpha * (phys - phys_base)

        with tf.GradientTape() as tape:
            tape.watch([atom_interp, phys_interp])

            preds = model({
                **inputs,
                "atom_features": atom_interp,
                "physio_features": phys_interp
            }, training=False)

        grads = tape.gradient(preds, [atom_interp, phys_interp])
        total_grad_atom += grads[0]
        total_grad_phys += grads[1]

    ig_atom = (atom - atom_base) * total_grad_atom / m_steps #Input change * average sensitivity along the path
    ig_phys = (phys - phys_base) * total_grad_phys / m_steps

    return ig_atom, ig_phys
       
def compute_mean_integrated_gradients(model, dataset, m_steps=32):
    total_atom = None
    total_phys = None
    count = 0

    for inputs, _ in dataset:
        ig_atom, ig_phys = integrated_gradients(model, inputs, m_steps)

        # Reduce per sample | take mean absolute IG per feature across batch (otherwise negative and positive cancel out)
        ig_atom = tf.reduce_mean(tf.abs(ig_atom), axis=0)   # (1024,)
        ig_phys = tf.reduce_mean(tf.abs(ig_phys), axis=0)   # (84,)

        if total_atom is None:
            total_atom = ig_atom
            total_phys = ig_phys
        else:
            total_atom += ig_atom
            total_phys += ig_phys

        count += 1

    mean_ig_atom = (total_atom / count).numpy()
    mean_ig_phys = (total_phys / count).numpy()

    return mean_ig_atom, mean_ig_phys

def compute_sequence_descriptor_submodality_scores(mean_ig_phys, physio_labels):
    """
    Compute mean integrated gradient (IG) scores for sub-modalities
    within the sequence descriptor features: BLOSUM, Physicochemical, and Positional Encoding.

    Parameters
    ----------
    mean_ig_phys : np.ndarray
        Array of IG scores for all sequence descriptor features.
    physio_labels : list of str
        List of labels corresponding to each sequence descriptor feature.

    Returns
    -------
    submodality_scores : dict
        Dictionary with keys 'BLOSUM', 'Physicochemical', 'PositionalEncoding'
        and values as mean IG scores for each sub-modality.
    """
    # Identify indices for each sub-modality
    blosum_idx = [i for i, lbl in enumerate(physio_labels) if lbl.startswith("BLOSUM")]
    posenc_idx = [i for i, lbl in enumerate(physio_labels) if lbl.startswith("PosEnc")]
    
    # Treat all other labels as physicochemical
    phys_idx = [i for i in range(len(physio_labels)) if i not in blosum_idx + posenc_idx]

    # Compute mean IG for each sub-modality
    blosum_score = np.mean(mean_ig_phys[blosum_idx]) if blosum_idx else 0.0
    posenc_score = np.mean(mean_ig_phys[posenc_idx]) if posenc_idx else 0.0
    phys_score   = np.mean(mean_ig_phys[phys_idx]) if phys_idx else 0.0

    # Return as dictionary
    submodality_scores = {
        "BLOSUM": blosum_score,
        "Physico": phys_score,
        "PositionalEncoding": posenc_score
    }
    return submodality_scores


def compute_cnn_gating_contribution(model, dataset):
    """
    Compute the effect of CNN-based modulation on predictions.
    Returns the difference in predictions with CNN gate ON vs OFF.
    
    Args:
        model: trained ImprovedDualBranchGNN_AttentionFusion
        dataset: tf.data.Dataset
    Returns:
        cnn_diff: np.array of differences per peptide
        preds_with_cnn: np.array predictions with CNN
        preds_without_cnn: np.array predictions without CNN
    """
    preds_with_cnn = []
    preds_without_cnn = []

    for inputs, _ in dataset:
        # --- With CNN gate (normal) ---
        pred_on = model(inputs, training=False).numpy().flatten()
        preds_with_cnn.extend(pred_on)

        # --- Without CNN gate ---
        # Temporarily disable CNN gate by setting gating threshold to 0
        original_threshold = cfg.cnn_gating_threshold
        cfg.cnn_gating_threshold = 0.0
        print(f"Temporarily setting CNN gating threshold to {cfg.cnn_gating_threshold} for ablation original was: {original_threshold}")
        pred_off = model(inputs, training=False).numpy().flatten()
        preds_without_cnn.extend(pred_off)
        # Restore original threshold
        cfg.cnn_gating_threshold = original_threshold
        print(f"Restored CNN gating threshold to {cfg.cnn_gating_threshold}")

    preds_with_cnn = np.array(preds_with_cnn)
    preds_without_cnn = np.array(preds_without_cnn)
    cnn_diff = preds_with_cnn - preds_without_cnn  # Positive → CNN increased prediction

    return cnn_diff, preds_with_cnn, preds_without_cnn


def plot_combined_contributions(model, dataset, physio_labels, save_path="./visualization/IG/combined_contributions_colored.png"):
    """
    Plot CNN gating effect, modality importance, and top physio features
    in a single figure with 3 subplots, with color-coded top physio features.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -------------------- CNN gating contribution --------------------
    cnn_diff, preds_with_cnn, preds_without_cnn = compute_cnn_gating_contribution(model, dataset)
    cnn_mean = np.mean(cnn_diff)
    cnn_std = np.std(cnn_diff)

    # -------------------- Modality importance --------------------
    mean_ig_atom, mean_ig_phys = compute_mean_integrated_gradients(model, dataset)
    modality_scores = [np.mean(mean_ig_atom), np.mean(mean_ig_phys)]

    #Compute sub-modality scores Blosum , Physicochemical, Positional Encoding
    submod_scores = compute_sequence_descriptor_submodality_scores(mean_ig_phys, physio_labels)
    print("Submodality scores:", submod_scores)

    # -------------------- Top physio features --------------------
    top_k = 10
    top_idx = np.argsort(mean_ig_phys)[-top_k:]
    top_physio_labels = [physio_labels[i] for i in top_idx]
    top_physio_scores = mean_ig_phys[top_idx]

    # Assign colors based on category
    top_colors = []
    for lbl in top_physio_labels:
        if lbl.startswith("BLOSUM"):
            top_colors.append("#1f77b4")  # Blue
        elif lbl.startswith("PosEnc"):
            top_colors.append("#2ca02c")  # Green
        else:
            top_colors.append("#d62728")  # Red for physio

    # -------------------- Plotting --------------------
    fig, axs = plt.subplots(1, 4, figsize=(20,5), constrained_layout=True) #18,5 3

    # 1. CNN gating effect
    axs[0].hist(cnn_diff, bins=30, color="#2ca02c", alpha=0.7)
    axs[0].axvline(cnn_mean, color='red', linestyle='--', label=f"Mean = {cnn_mean:.3f}")
    axs[0].set_xlabel("Prediction Diff (CNN ON - OFF)")
    axs[0].set_ylabel("Number of Peptides")
    axs[0].set_title("CNN Gate Contribution (A)")
    axs[0].legend()

    # 2. Modality importance
    axs[1].bar(["SEQ Embedding Features", "SEQ Descriptor Features"],
               modality_scores,
               color=["#1f77b4", "#d62728"])
    axs[1].set_ylabel("Mean |Integrated Gradient|")
    axs[1].set_title("Modality Importance (B)")

    # 3. Sub-modality importance
    axs[2].bar(list(submod_scores.keys()), list(submod_scores.values()),
               color=["#1f77b4", "#d62728", "#2ca02c"])
    axs[2].set_ylabel("Mean |Integrated Gradient|")
    axs[2].set_title("Sub-modality Importance (C)")

    # 4. Top physio features (color-coded)
    axs[3].barh(top_physio_labels, top_physio_scores, color=top_colors)
    axs[3].set_xlabel("Mean |Integrated Gradient|")
    axs[3].set_title(f"Top {top_k} Sequence Descriptor Features (D)")
    axs[3].invert_yaxis()  # highest at top

    

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    # -------------------- Print stats --------------------
    print("CNN Gate Mean Contribution:", cnn_mean, "Std:", cnn_std)
    print("Modality Importance - ProtT5:", modality_scores[0], " Physio:", modality_scores[1])
    print("Sequence Descriptor Sub-modality Scores:")
    for key, val in submod_scores.items():
        print(f"  {key}: {val:.4f}")
    print("Top Physio Features IG (colored):")
    for label, score in zip(top_physio_labels[::-1], top_physio_scores[::-1]):
        print(f"  {label}: {score:.4f}")


def compute_feature_contributions(model_name, model_index, save_path_base="./visualization/IG"):
    model = load_model(model_name, model_index=model_index)
    _, _, test_f = data_util.load_datasets([model_index])[0]
    test_ds = make_dataset(test_f)

    

    # Compute Integrated Gradients and plot modality importance and top physio features    
    physio_labels = (
    PHYSIO_LABELS +
    [f"BLOSUM_{aa}" for aa in list("ACDEFGHIKLMNPQRSTVWY")] +
    [f"PosEnc_{i+1}" for i in range(32)]
    )

    plot_combined_contributions(model, test_ds, physio_labels, save_path=os.path.join(save_path_base, "combined_contributions.png"))



if __name__ == "__main__":
    #Choose Dataset Path
    # dataset_path = data_util.DATASET_PATH_P_AERUGINOSA_PROTT5
    dataset_path = data_util.DATASET_PATH_ECOLI_PROTT5
    
    parser = argparse.ArgumentParser(description="Train, Predict or Plot Feature Contribution.")
    parser.add_argument(
        "mode",
        choices=["train", "predict", "plot"],
        help="Mode to run: train, predict or visualize feature contributions."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ModProt",
        help="Model name (default: ModProt)"
    )
    parser.add_argument(
        "--datasets",
        type=int,
        nargs="+",
        default=[0],
        help="List of dataset indices (default: 0)"
    )
    args = parser.parse_args()

    
    if args.mode == "train":
        model_name = args.model
        datasets_index = args.datasets
        metrics = execute(model_name=model_name, datasets_index=datasets_index, dataset_path=dataset_path)
    elif args.mode == "predict":
        model_name = args.model
        datasets_index = args.datasets
        metrics = load_model_and_evaluate_test(model_name=model_name, datasets_index=datasets_index, dataset_path=dataset_path)
    elif args.mode == "plot":
        model_name = args.model
        datasets_index = args.datasets
        for model_index in datasets_index:
            compute_feature_contributions(
                model_name=model_name,
                model_index=model_index,
                save_path_base=f"./visualization/IG/foldnew_{model_index}"
            )
    else:
        raise ValueError("Invalid mode. Choose from: train, predict, plot")


    
    
    
