###The defination of model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from experimental_config import ExperimentConfig

cfg = ExperimentConfig()

@tf.keras.utils.register_keras_serializable()
class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(feature_dim, self.units),
            # initializer=self.kernel_initializer,
            initializer = keras.initializers.GlorotUniform(seed=42),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        self.kernel_attention = self.add_weight(
            name="kernel_attention",
            shape=(self.units * 2, 1),
            initializer=keras.initializers.GlorotUniform(seed=43),
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        inputs: tuple (node_states, edges, edge_weights)
        node_states: [num_nodes, feature_dim]
        edges: [num_edges, 2]
        edge_weights: [num_edges] (float)
        """
        node_states, edges, edge_weights = inputs
        num_nodes = tf.shape(node_states)[0]
        node_states_transformed = tf.matmul(node_states, self.kernel)
        

        # Handle empty edge case with tf.cond
        def process_edges():
            # Gather node features for each edge
            node_states_expanded = tf.gather(node_states_transformed, edges)
            node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))

            # Compute attention scores | This is attention energy computation, not a feature transformation. so leaky relu is fine
            attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))
            attention_scores = tf.squeeze(attention_scores, -1)

            #  ADDed DROPOUT avoid textbook attention over-confidence
            # attention_scores = tf.nn.dropout(attention_scores, rate=0.1)#did not helped so commented

            # Incorporate edge weights
            attention_scores = attention_scores * edge_weights

            # Clip and exponentiate for stability
            attention_scores = tf.exp(tf.clip_by_value(attention_scores, -2, 2))

            # Sum attention scores per source node
            attention_scores_sum = tf.math.unsorted_segment_sum(
                attention_scores,
                edges[:, 0],
                num_segments=num_nodes,
            )
            
            # Gather sum for each edge (instead of repeat + bincount)
            attention_scores_sum_gathered = tf.gather(attention_scores_sum, edges[:, 0])

            # Normalize attention scores 
            attention_scores_norm = attention_scores / (attention_scores_sum_gathered + 1e-8)

            # Compute Degree-Aware attention normalization | normal attention normalization is better 
            # attention_scores_norm = degree_aware_attn_norm(attention_scores, attention_scores_sum_gathered, edges, num_nodes)

            # Aggregate neighbor features
            node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
            out = tf.math.unsorted_segment_sum(
                node_states_neighbors * attention_scores_norm[:, tf.newaxis],
                edges[:, 0],
                num_segments=num_nodes,
            )
            return out

        
        
        def no_edges():
            return node_states_transformed
        
        # Check if edges exist
        has_edges = tf.greater(tf.shape(edges)[0], 0)
        return tf.cond(has_edges, process_edges, no_edges)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

@tf.keras.utils.register_keras_serializable()
class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.merge_type = merge_type

        self.attention_layers = None 
    
    def build(self, input_shape):
        # Create attention layers here to ensure proper tracking
        self.attention_layers = [
            GraphAttention(self.units) for _ in range(self.num_heads)
        ]
        super().build(input_shape)

    def call(self, inputs):
        atom_features, edges, edge_weights = inputs  # add edge_weights

        outputs = [
            att([atom_features, edges, edge_weights]) for att in self.attention_layers
        ]

        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

        return  tf.nn.relu(outputs) #|tf.nn.gelu(outputs) #gelu performed badly here


    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_heads": self.num_heads,
            "merge_type": self.merge_type,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs

        # Convert to RaggedTensor grouped by molecule_indicator
        ragged_features = tf.RaggedTensor.from_value_rowids(
            atom_features, molecule_indicator, nrows=self.batch_size
        )

        # Pad each protein to max length in the batch
        padded = ragged_features.to_tensor(default_value=0.0)

        return padded  # shape: [batch_size, max_num_nodes, feature_dim]

    def get_config(self):
        config = super().get_config()
        config.update({"batch_size": self.batch_size})
        return config



@tf.keras.utils.register_keras_serializable()
class TransformerEncoderReadout(layers.Layer):
    """
    Protein-level Transformer encoder.
    Input:  [batch_size, feature_dim]
    Output: [batch_size, feature_dim]
    """
    def __init__(
        self,
        num_heads=8,
        embed_dim=256,
        dense_dim=512,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.dropout = dropout

        self.self_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(dense_dim, activation=cfg.ffn_A, kernel_initializer=keras.initializers.GlorotUniform(seed=101)),
            layers.Dense(embed_dim, kernel_initializer=keras.initializers.GlorotUniform(seed=102)),
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.drop1 = layers.Dropout(dropout,seed=120)
        self.drop2 = layers.Dropout(dropout, seed=121)

    def call(self, x, training=False):
        """
        x: [batch, embed_dim]
        """
        # Add fake sequence length = 1
        x = x[:, tf.newaxis, :]  # [B, 1, D]

        # Self-attention
        attn_out = self.self_attn(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))

        # Feed-forward
        ffn_out = self.ffn(x, training=training)
        x = self.norm2(x + self.drop2(ffn_out, training=training))

        # Remove fake sequence dimension
        return tf.squeeze(x, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "dropout": self.dropout,
        })
        return config


        
def GraphAttentionNetwork(atom_dim, hidden_units, num_heads, num_layers):
    
    # Input layers
    node_features = layers.Input((atom_dim,), dtype="float32", name="atom_features")
    pair_indices = layers.Input((2,), dtype="int32", name="pair_indices")
    edge_weights = layers.Input((), dtype="float32", name="edge_weights")  # NEW
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    
    # Preprocess features | gelu  performed bad here
    x = layers.Dense(hidden_units * num_heads, activation=cfg.x_A, kernel_initializer=keras.initializers.GlorotUniform(seed=111))(node_features)
    
    # Multi-head graph attention layers
    for _ in range(num_layers):
        residual = x
        x_att = MultiHeadGraphAttention(hidden_units, num_heads)([x, pair_indices, edge_weights])
        x = x_att + residual   #  No LayerNorm here
    
    # Transformer encoder readout
    x = TransformerEncoderReadout(embed_dim=hidden_units * num_heads)(x)

    
    x = layers.LayerNormalization()(x)   # New | Graph-level normalization

    # Output layer (linear for regression)
    outputs = layers.Dense(cfg.gnn_base_output_dim, kernel_initializer=keras.initializers.GlorotUniform(seed=113))(x)  # No activation for regression
    
    # Build model
    model = keras.models.Model(
        inputs=[node_features, pair_indices, edge_weights, molecule_indicator],
        outputs=outputs
    )

    
    return model

class CrossAttentionFusion(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Transformer-style cross-attention
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=dropout
        )

        # Layer norms for stability
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

        # Projection after attention
        self.proj = tf.keras.layers.Dense(dim)

        # Learnable modality scoring
        self.score = tf.keras.layers.Dense(1)

    def call(self, seq_feat, gnn_feat, physio_feat, training=False):
        """
        Inputs:
            seq_feat    : [B, D]
            gnn_feat    : [B, D]
            physio_feat : [B, D]
        Returns:
            fused       : [B, D]
            attn_scores : [B, 3, 1]
        """

        # Convert each modality into a token
        tokens = tf.stack([seq_feat, gnn_feat, physio_feat], axis=1)   # [B, 3, D]

        # Cross-attention across modalities
        attn = self.mha(
            query=tokens,
            key=tokens,
            value=tokens,
            training=training
        )                                                             # [B, 3, D]

        # Residual + normalization
        tokens = self.norm1(tokens + attn)

        # Projection
        tokens = self.norm2(self.proj(tokens))

        # Learn modality importance
        attn_scores = tf.nn.softmax(self.score(tokens), axis=1)       # [B, 3, 1]

        # Attention-weighted fusion
        fused = tf.reduce_sum(tokens * attn_scores, axis=1)          # [B, D]

        return fused, attn_scores
