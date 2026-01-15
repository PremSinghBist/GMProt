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


