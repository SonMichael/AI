from library import *
from bert.layers.encoder import Encoder
from tensorflow.keras.layers import Dense, LayerNormalization
from bert.layers.generate_mask import generate_padding_mask

class Bert(tf.keras.models.Model):
    def __init__(self,n, h, vocab_size, max_length, n_segments, d_model, d_ff, dropout_rate, eps):
        super(Bert, self).__init__()
        self.encoder = Encoder(n, h, vocab_size, max_length, n_segments, d_model, d_ff, dropout_rate, eps)
        self.next_sentence_classifier = Dense(2)
        #decoder MLM
        self.mlm_dense = Dense(d_model, activation = 'relu')
        self.mlm_layernorm1 = LayerNormalization(epsilon=eps)
        self.mlm_classifier = Dense(max_length)
        


    def call(self, input_ids, segment_ids, is_train, masked_pos):
        """
            Parameters
            ----------
            input_ids: tensor,
                shape: (batch_size, vocab_size)
            segment_ids: tensor,
                shape: (batch_size, max_length)
            is_train: bool,
            masked_pos: tensor,
                shape: (batch_size, max_pred)
            Returns
            ----------
            logits_mlm: tensor
                shape: (batch_size, max_pred, vocab_size)
            logits_nsp: tensor
                shape: (batch_size, 2)
        """
        #NSP
        enc_self_attn_mask = generate_padding_mask(input_ids,input_ids)# (batch_size, max_length, max_length)
        encoder_out = self.encoder(input_ids, segment_ids, is_train, enc_self_attn_mask) # (batch_size, max_length, d_model)
        logits_nsp = self.next_sentence_classifier(encoder_out[:, 0]) #(batch_size, 2)

        #decoder MLM
        tok_embed_weight = self.encoder.tok_embed.weights[0] # (vocab_size ,d_model)
        self.mlm_classifier._weights = [tok_embed_weight]
        batch_size, max_pred = masked_pos.shape
        d_model = tok_embed_weight.shape[-1]
        masked_pos_broad_cast = tf.broadcast_to(masked_pos[:, :, None], [batch_size, max_pred, d_model]) #(batch_size, max_pred, d_model)
        h_masked = tf.gather(encoder_out,masked_pos_broad_cast , axis=1, batch_dims=1) #(batch_size, max_pred, d_model, d_model)
        h_masked_reshape = h_masked[:,:, -1]#(batch_size, max_pred, d_model)
        h_masked_norm = self.mlm_layernorm1(self.mlm_dense(h_masked_reshape))#(batch_size, max_pred, d_model)
        logits_mlm = self.mlm_classifier(h_masked_norm)#(batch_size, max_pred, vocab_size)
        return logits_mlm, logits_nsp
