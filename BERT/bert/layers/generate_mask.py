from library import *

def generate_padding_mask(seq_q, seq_k):
        # TODO: Update document
        batch_size, len_q = seq_q.shape
        batch_size, len_k = seq_k.shape
        # eq(zero) is PAD token
        result = tf.cast(seq_k == 0, dtype=tf.bool)
        result_expand_dim = tf.expand_dims(result, axis=1) # batch_size x 1 x len_k
        return tf.broadcast_to(result_expand_dim, [batch_size,len_q, len_k]) # batch_size x len_q x len_k