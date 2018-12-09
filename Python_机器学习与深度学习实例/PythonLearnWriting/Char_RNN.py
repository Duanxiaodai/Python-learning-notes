# Char_RNN
# 作者：Charles
# 公众号：Charles的皮卡丘
import config
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq as seq2seq
import numpy as np
import time
import os
import sys


# 数据读取和生成训练用的词向量
class DataGen():
	def __init__(self, filepath, args):
		self.seq_length = args.seq_length
		self.batch_size = args.batch_size
		with open(filepath, encoding='utf-8') as f:
			self.text = f.read()
			self.text = self.text.replace('\n', '')
		self.total_len = len(self.text)
		self.words = list(set(self.text))
		self.words.sort()
		self.vocabulary_size = len(self.words)
		self.char2idx_dict = {w: i for i, w in enumerate(self.words)}
		self.idx2char_dict = {i: w for i, w in enumerate(self.words)}
		self._pointer = 0
		self.save_metadata(args.metadata)
	def char2idx(self, c):
		idx = self.char2idx_dict[c]
		return idx
		# if idx:
		# 	return idx
		# else:
		# 	self.vocabulary_size += 1
		# 	self.char2idx_dict[c] = self.vocabulary_size
		# 	return self.vocabulary_size
	def idx2char(self, idx):
		return self.idx2char_dict[idx]
	def save_metadata(self, filename):
		with open(filename, 'w', encoding='utf-8') as f:
			f.write('idx\tchar\n')
			for i in range(self.vocabulary_size):
				c = self.idx2char(i)
				f.write('{}\t{}\n'.format(i, c))
	def next_batch(self):
		x_batches = []
		y_batches = []
		for i in range(self.batch_size):
			if self._pointer + self.seq_length + 1 >= self.total_len:
				self._pointer = 0
			batch_x = self.text[self._pointer: self._pointer+self.seq_length]
			batch_y = self.text[self._pointer+1: self._pointer+self.seq_length+1]
			self._pointer += self.seq_length
			batch_x = [self.char2idx(c) for c in batch_x]
			batch_y = [self.char2idx(c) for c in batch_y]
			x_batches.append(batch_x)
			y_batches.append(batch_y)
		return x_batches, y_batches


# rnn
class Model():
	def __init__(self, args, text, test=False):
		if test:
			args.batch_size = 1
			args.seq_length = 1
		with tf.name_scope('inputs'):
			self.input_text = tf.placeholder(
				tf.int32, [args.batch_size, args.seq_length])
			self.target_text = tf.placeholder(
				tf.int32, [args.batch_size, args.seq_length])
		with tf.name_scope('model'):
			# LSTM单元 state_size为隐藏层的大小
			self.cell = rnn.BasicLSTMCell(args.state_size)
			self.cells = rnn.MultiRNNCell([self.cell] * args.num_layers)
			self.initial_state = self.cells.zero_state(
				args.batch_size, tf.float32)
			with tf.variable_scope('rnnlm'):
				w = tf.get_variable('softmax_w', [args.state_size, text.vocabulary_size])
				b = tf.get_variable('softmax_b', [text.vocabulary_size])
				with tf.device('/cpu:0'):
					embedding = tf.get_variable(
						'embedding', [text.vocabulary_size, args.state_size])
					inputs = tf.nn.embedding_lookup(embedding, self.input_text)
			outputs, last_state = tf.nn.dynamic_rnn(
				self.cells, inputs, initial_state=self.initial_state)
		with tf.name_scope('loss'):
			output = tf.reshape(outputs, [-1, args.state_size])
			self.logits = tf.matmul(output, w) + b
			self.probs = tf.nn.softmax(self.logits)
			self.last_state = last_state
			targets = tf.reshape(self.target_text, [-1])
			loss = seq2seq.sequence_loss_by_example([self.logits],
													[targets],
													[tf.ones_like(targets, dtype=tf.float32)])
			self.loss_avg = tf.reduce_sum(loss) / args.batch_size
			tf.summary.scalar('loss', self.loss_avg)
		with tf.name_scope('optimize'):
			self.lr = tf.placeholder(tf.float32, [])
			tf.summary.scalar('learning_rate', self.lr)
			optimizer = tf.train.AdamOptimizer(self.lr)
			tvars = tf.trainable_variables()
			grads = tf.gradients(self.loss_avg, tvars)
			for g in grads:
				tf.summary.histogram(g.name, g)
			grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
			self.train_op = optimizer.apply_gradients(zip(grads, tvars))
			self.merged_op = tf.summary.merge_all()


# 训练
def train(text, model, args):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(args.log_dir, sess.graph)
		pconfig = projector.ProjectorConfig()
		embed = pconfig.embeddings.add()
		embed.tensor_name = 'rnnlm/embedding:0'
		embed.metadata_path = args.metadata
		projector.visualize_embeddings(writer, pconfig)
		max_iter = args.n_epoch * \
					(text.total_len // args.seq_length) // args.batch_size
		for i in range(max_iter):
			learning_rate = args.learning_rate * \
							(args.decay_rate ** (i // args.decay_steps))
			x_batch, y_batch = text.next_batch()
			feed_dict = {
				model.input_text: x_batch,
				model.target_text: y_batch,
				model.lr: learning_rate
			}
			train_loss, summary, _, _ = sess.run([model.loss_avg,
												  model.merged_op,
												  model.last_state,
												  model.train_op],
												  feed_dict)
			if i % 10 == 0:
				writer.add_summary(summary, global_step=i)
				print('[Step]:{}/{}, [Train_loss]:{:4f}'.format(i, max_iter, train_loss))
			if i % 2000 == 0 or (i + 1) == max_iter:
				saver.save(sess, os.path.join(args.log_dir, 'model.ckpt'), global_step=i)


# 生成文本
# prime用于预热
def GenText(text, model, args, prime):
	saver = tf.train.Saver()
	with tf.Session() as sess:
		ckpt = tf.train.latest_checkpoint(args.log_dir)
		saver.restore(sess, ckpt)
		state = sess.run(model.cells.zero_state(1, tf.float32))
		for word in prime[:-1]:
			x = np.zeros((1, 1))
			x[0, 0] = text.char2idx(word)
			feed_dict = {model.input_text: x, model.initial_state: state}
			state = sess.run(model.last_state, feed_dict)
		word = prime[-1]
		contents = prime
		for i in range(args.gen_num):	
			x = np.zeros([1, 1])
			x[0, 0] = text.char2idx(word)
			feed_dict = {model.input_text: x, model.initial_state: state}
			probs, state = sess.run([model.probs, model.last_state], feed_dict)
			prob = probs[0]
			word = text.idx2char(np.argmax(prob))
			time.sleep(0.5)
			# sys.stdout.flush()
			# print(word, end='')
			contents += word
		return contents




if __name__ == '__main__':
	print('[INFO]:Char_RNN Example...')
	print('[Author]:Charles')
	print('[Usage]:python3 Char_RNN.py \nEnter 1 to train, enter 0 to gen...')
	choice = input('[Choice]:Train<1> or Gen<0>:')
	text = DataGen('./JayLyrics.txt', config.HParam())
	prime = u'是曾与你躲过雨的屋檐'
	if choice == '1':
		model = Model(config.HParam(), text, False)
		train(text, model, config.HParam())
	elif choice == '0':
		model = Model(config.HParam(), text, True)
		contents = GenText(text, model, config.HParam(), prime)
		with open('./results.txt', 'w') as f:
			f.write(contents)
	else:
		print('[Error]:Enter cannot recognized...')