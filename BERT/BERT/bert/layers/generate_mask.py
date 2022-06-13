from library import *

def generate_padding_mask(seq_q, seq_k):
        """
            Splitting item to heads
            Parameters
            ----------
            seq_q: tensor
                shape: (batch_size, max_length)
            seq_k: tensor
                shape: (batch_size, max_length)
            Returns
            ----------
            result_broadcast: tensor
                shape: (batch_size, max_length, max_length)
        """
        
        # TODO: Update document
        batch_size, len_q = seq_q.shape
        batch_size, len_k = seq_k.shape
        # eq(zero) is PAD token
        result = tf.cast(seq_k == 0, dtype=tf.bool)
        result_expand_dim = tf.expand_dims(result, axis=1) # batch_size x 1 x max_length
        
        result_broadcast = tf.broadcast_to(result_expand_dim, [batch_size,len_q, len_k]) # batch_size x max_length x max_length
        
        return result_broadcast