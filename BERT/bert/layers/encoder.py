import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout
from bert.layers.encoder_layer import EncoderLayer

class Encoder(tf.keras.layers.Layer):
    def __init__(self,n,  h, vocab_size, max_length, n_segments, d_model, d_ff, dropout_rate, eps):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.encoder_layers = [EncoderLayer(h, d_model, d_ff, dropout_rate, eps) for _ in range(n)]
        self.tok_embed = Embedding(vocab_size, output_dim=d_model)
        self.pos_embed = Embedding(max_length, output_dim=d_model)
        self.seg_embed = Embedding(n_segments, output_dim=d_model)
        self.dropout = Dropout(dropout_rate)

    def call(self, input_ids, segment_ids, is_train, mask):
        batch_size, len_q = input_ids.shape
        pos = tf.range(len_q, dtype=tf.int64)
        pos_expand_dim = tf.expand_dims(pos, axis=0)
        pos_broadcast = tf.broadcast_to(pos_expand_dim, [batch_size, len_q])
        encoder_out = self.tok_embed(input_ids) + self.pos_embed(pos_broadcast) + self.seg_embed(segment_ids)

        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer(encoder_out, is_train, mask)
        
        return encoder_out

