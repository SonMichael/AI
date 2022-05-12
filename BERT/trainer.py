from bert.layers.generate_mask import generate_padding_mask
import tensorflow as tf

class Trainer:
	def __init__(self, model, optimizer, epochs, checkpoint_folder):
		self.model = model
		self.optimizer = optimizer
		self.epochs = epochs
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')
		self.checkpoint = tf.train.Checkpoint(model = self.model, optimizer = self.optimizer)
		self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_folder, max_to_keep=3)

	def loss_function(self, real, pred, loss_type):
		if loss_type == "CategoricalCrossentropy":
			cross_entropy = tf.keras.losses.CategoricalCrossentropy()
			loss = cross_entropy(real, pred)
			return tf.reduce_mean(loss)
		bce = tf.keras.losses.BinaryCrossentropy()
		loss = bce(real, pred)
		return loss

	def train_step(self, input_ids, segment_ids, masked_tokens, masked_pos, are_next):
		"""
            Parameters
            ----------
			are_next: tensor,
                shape: (batch_size)
            masked_tokens: tensor,
                shape: (batch_size, max_pred)
            
        """
		# TODO: Update document
		are_next_expand = tf.expand_dims(are_next, axis = 1 )
		
		with tf.GradientTape() as tape:
			#logits_mlm = (batch_size, max_pred, vocab_size), logits_nsp = (batch_size, 2)
			logits_mlm, logits_nsp = self.model(input_ids, segment_ids,True, masked_pos) 
			batch_size, max_pred, vocab_size =  logits_mlm.shape
			masked_tokens_one_hot = tf.one_hot(masked_tokens, vocab_size, axis=-1)
			# logits_mlm_tranpose = tf.transpose(logits_mlm, [0,2, 1]) #(batch_size, max_pred, vocab_size)
			mask_loss = self.loss_function(masked_tokens_one_hot, logits_mlm, "CategoricalCrossentropy")
			seg_loss = self.loss_function(are_next_expand, logits_nsp, "BinaryCrossentropy")
			d_loss = mask_loss + seg_loss

		# Compute gradients
		grads = tape.gradient(d_loss, self.model.trainable_variables)

		# Update weights
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

		# Compute metrics
		self.train_loss.update_state(d_loss)
		# self.train_accuracy.update_state(self.cal_acc(masked_tokens, logits_mlm))

		return {"loss": self.train_loss.result()}

	def fit(self, data):
		print('=============Training Progress================')
		print('----------------Begin--------------------')
		# Loading checkpoint
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
			print('Restored checkpoint manager !')
		input_ids, segment_ids, masked_tokens, masked_positions, are_next = data
		for epoch in range(self.epochs):
			self.train_loss.reset_states()
			# self.train_accuracy.reset_states()
			self.train_step(input_ids, segment_ids, masked_tokens, masked_positions, are_next)
			print(f'Epoch {epoch } Loss {self.train_loss.result():.3f}')
			if (epoch + 1) % 5 == 0:
				saved_path = self.checkpoint_manager.save()
				print('Checkpoint was saved at {}'.format(saved_path))
		print('----------------Done--------------------')

	def predict(self, train_dataset, text, number_dict):
		print('=============Inference Progress================')
		print('----------------Begin--------------------')
		# Loading checkpoint
		print("latest_checkpoint", self.checkpoint_manager.latest_checkpoint)
		if self.checkpoint_manager.latest_checkpoint:
			self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint).expect_partial()
			print('Restored checkpoint manager !')
		#masked_tokens = (1, max_pred), are_next = (1), input_ids = (1, vocab_size), segment_ids = (1, max_length), masked_pos = (1, max_pred)
		input_ids, segment_ids, masked_tokens, masked_positions, are_next = train_dataset
		print(text)
		print([[number_dict[w.numpy()]] for w in input_ids[0] if number_dict[w.numpy()] != '[PAD]'])
		#logits_mlm = (1, max_pred, vocab_size), logits_nsp = (1, 2)
		logits_mlm, logits_nsp = self.model(input_ids, segment_ids, False, masked_positions)
		print('masked tokens list : ',[pos.numpy() for pos in masked_tokens[0] if pos.numpy() != 0])
		logits_lm_max = tf.reduce_max(logits_mlm[0], axis =1)
		print('predict masked tokens list : ',[pos.numpy() for pos in logits_lm_max if pos != 0])

		logits_nsp_max = tf.reduce_max(logits_nsp, axis =1)
		print('isNext : ', True if are_next else False)
		print('predict isNext : ', True if logits_nsp_max  else False )
