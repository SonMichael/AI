import tensorflow as tf
from tensorflow.keras.layers import Dense


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, h ):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        # Num of heads
        self.h = h
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.wo = Dense(d_model)

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
            Calculate Attention score
            Parameters

            ----------
            d_heads = d_model/n_heads
            q: tensor
                query
                shape: (batch_size, n_heads, max_length, d_heads)
            k: tensor
                key
                shape: (batch_size, n_heads, max_length, d_heads)
            v: tensor
                value
                shape: (batch_size, n_heads, max_length, d_heads)
            mask: tensor
                shape:(batch_size, n_heads, max_length, max_length)

            Returns
            ----------
            attention_weights: tensor 
                Attention Scores between Query and Key
                shape: (batch_size, n_heads, max_length, max_length)
            out: tensor
                Attention Weights on Value
                shape: (batch_size, n_heads, max_length, d_heads)
        """
        dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)#64
        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk) #(batch_size, n_heads, max_length, max_length)
        if len(mask) > 0:
            mask = tf.cast(mask, dtype=tf.float32)
            attention_scores += (mask * -1e30)
        attention_weights =  tf.nn.softmax(attention_scores, axis=-1)#(batch_size, n_heads, max_length, max_length)
        out = tf.matmul(attention_weights, v) #(batch_size, n_heads, max_length, d_heads)
        return out, attention_weights

    def splitting_head(self, x):
        """
            Splitting item to heads
            Parameters
            ----------
            x: tensor
                query/key/value
                shape: (batch_size, max_length, d_model)
            Returns
            ----------
            d_heads = d_model/n_heads
            xs: tensor
                splitted heads
                shape: (batch_size, n_heads, max_length, d_heads)

        """
        
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1] 
        d_model = tf.shape(x)[2] 

        # assert d_model % self.h == 0
        
        hd_v = d_model // self.h
        
        x = tf.reshape(x, (batch_size, length, self.h, hd_v))#(batch_size, max_length, n_heads, d_heads)
        
        xs = tf.transpose(x, [0, 2, 1, 3])#(batch_size, n_heads, max_length, d_heads)
        
        return xs
        


    def call(self, q, k, v, mask):
        """
            Splitting item to heads
            Parameters
            ----------
            x: tensor
                shape: (batch_size, max_length, d_model)
            q: tensor
                shape: (batch_size, max_length, d_model)
            k: tensor
                shape: (batch_size, max_length, d_model)
            mask: tensor
                shape: (batch_size, max_length, max_length)
            Returns
            ----------
            d_heads = d_model/n_heads
            final: tensor
                shape: (batch_size, max_length, d_model)
            attention_weights: tensor
                shape: (batch_size, n_heads, max_length, max_length)

        """

        batch_size = tf.shape(q)[0]
        vocab_size = tf.shape(q)[1]
        qw = self.wq(q) # (batch_size, max_length, d_model)
        kw = self.wk(k) # (batch_size, max_length, d_model)
        vw = self.wv(v) # (batch_size, max_length, d_model)
        
        # Splitting Head

        heads_qw = self.splitting_head(qw) # (batch_size, n_heads, max_length, d_heads)
        heads_kw = self.splitting_head(kw) # (batch_size, n_heads, max_length, d_heads)
        heads_vw = self.splitting_head(vw) # (batch_size, n_heads, max_length, d_heads)

        # Do Attention
        mask_expand = tf.expand_dims(mask, axis = 1)# (batch_size, 1, max_length, max_length)
        
        mask_broadcast = tf.broadcast_to(mask_expand, [batch_size, self.h, vocab_size, vocab_size])#(batch_size, n_heads, max_length, max_length)
        
        #out: (batch_size, n_heads, max_length, d_heads), attention_weights: (batch_size, n_heads, max_length, max_length)
        out, attention_weights = self.scaled_dot_product_attention(heads_qw, heads_kw, heads_vw, mask_broadcast)

        # Transpose out back

        out = tf.transpose(out, [0, 2, 1, 3])# (batch_size, max_length, n_heads, d_heads)
        

        out = tf.reshape(out, (batch_size, tf.shape(qw)[1], self.d_model)) # (batch_size, max_length, d_model)
        

        final = self.wo(out) # (batch_size, max_length, d_model)
        

        return final, attention_weights