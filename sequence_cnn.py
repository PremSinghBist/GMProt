import tensorflow as tf
from keras import layers
from experimental_config import ExperimentConfig
cfg = ExperimentConfig()

class SequenceCNN(layers.Layer):
    def __init__(self):
        super().__init__()
        #mask_zero:token with index 0 is treated as padding and will be ignored 
        self.embed = layers.Embedding(cfg.cnn_input_dim, cfg.cnn_emb_output_dim, mask_zero=True) # (B, L, emb_output_dim=24)

        self.conv3  = layers.Conv1D(cfg.cnn_filters, 3, padding="same", activation="relu") #(B,L,cnn_filters=64)
        self.conv5  = layers.Conv1D(cfg.cnn_filters, 5, padding="same", activation="relu")
        self.conv7  = layers.Conv1D(cfg.cnn_filters, 7, padding="same", activation="relu") 
        self.conv11 = layers.Conv1D(cfg.cnn_filters, 11, padding="same", activation="relu")

        self.pool = layers.GlobalMaxPooling1D() #(B, L, cnn_filters=64) ->(B, 64) | collapses the sequence length L dimension:
        self.proj = layers.Dense(128, activation="relu")

    def call(self, seq_ids):
        x = self.embed(seq_ids)  # (B, L, emb_dim)

        # Create mask: 1 for real residues, 0 for padding | True for real(1), False(0) for fake
        #cast: convert boolean to float
        mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)  # (B, L) eg: [1, 1, 0,0 ,0]
        mask = tf.expand_dims(mask, -1)  # (B, L, 1)
        x = x * mask #zeros out emb at padded pos 

        # To prevent padded positions from affecting GlobalMaxPooling because zeros alos count for max
        # set padded positions to very negative value after conv
        def masked_pool(conv_out):
            # conv_out: (B, L, filters)
            large_negative = -1e9
            #Mask(1: real, 0: fake) |  1.0 - mask filips -> 1 =fake now, real pos=0 |  
            #so MaxPool will never pick a padded value, because real values are much larger than -1e9
            #conv_out at real positions stays the same | padded becomes extremly neg. 
            conv_out_masked = conv_out + (1.0 - mask) * large_negative
            return self.pool(conv_out_masked) #apply max pooling

        c3  = masked_pool(self.conv3(x))
        c5  = masked_pool(self.conv5(x))
        c7  = masked_pool(self.conv7(x))
        c11 = masked_pool(self.conv11(x))

        x = tf.concat([c3, c5, c7, c11], axis=-1)  # (B, 64*4=256)
        return self.proj(x)  # (B, 128)
class DilatedConvAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        # -------- Dilated Conv Block --------
        self.conv1 = tf.keras.layers.Conv1D(hidden_dim, 3, padding='same', dilation_rate=1, activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(hidden_dim, 3, padding='same', dilation_rate=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(hidden_dim, 3, padding='same', dilation_rate=4, activation='relu')

        self.norm = tf.keras.layers.LayerNormalization()

        # -------- Attention --------
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads
        )

        # self.attn_dense = tf.keras.layers.Dense(1)  # Reduced Performance | for attention pooling weights

        self.dropout = tf.keras.layers.Dropout(dropout)

        # -------- Projection --------
        self.proj = tf.keras.layers.Dense(hidden_dim)

    def call(self, x, training=False):
        """
        x: (B, L) or (B, L, C)
        """

        # If input is token IDs → embed first
        if len(x.shape) == 2:
            x = tf.one_hot(x, depth=25)  # assuming vocab size = 25 (amino acids)

        # -------- Dilated Convolutions --------
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # Multi-scale fusion
        x = x1 + x2 + x3 #element-wise sum to fuse multi-scale features
        # x = tf.concat([x1, x2, x3], axis=-1) #also try concat fusion (perf. dropped) | concatenation along feature dimension, preserves all multi-scale features but increases dim by 3x
        x = self.norm(x)

        # -------- Self Attention --------
        attn_out = self.attn(x, x)
        attn_out = self.dropout(attn_out, training=training)

        # Residual connection
        x = x + attn_out

        # --------| Max pooling, Preserve max feature  --------
        x = tf.reduce_max(x, axis=1)   # (B, hidden_dim) 

        #-------- Attention Pooling instead of max Pooling --------
        '''scores = self.attn_dense(x)          # shape: (batch_size, seq_len, 1)
        weights = tf.nn.softmax(scores, axis=1)
        x = tf.reduce_sum(weights * x, axis=1)  '''

        # -------- Final projection --------
        x = self.proj(x)

        return x

