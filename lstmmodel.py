import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn
from tqdm import tqdm
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python import pywrap_tensorflow

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '/cpu:0'


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.len_sentence = 70
        self.num_epochs = 3  # 在一个num_epochs中，所有训练集数据使用一次
        self.n_classes = 53
        self.cnn_size = 230
        self.num_layers = 1
        self.pos_size = 5  # 位置嵌入维度
        self.pos_num = 123
        self.word_embedding = 50  # 单词嵌入维度
        self.keep_prob = 0.5
        self.batch_size = 128  # 每个批次的大小
        self.n_steps = 20
        self.lr = 0.001
        self.training_iters = 100000
        self.display_step = 10
        # 网络参数
        self.n_input = 28
        # 隐藏层大小
        self.n_hidden = 128


def attention(inputs):
    # Trainable parameters
    # hidden_size = inputs.shape[2].value
    hidden_size = 128
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas


class CNN():

    def __init__(self, word_embeddings, setting):

        self.vocab_size = setting.vocab_size
        self.len_sentence = len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.n_classes = n_classes = setting.n_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr
        self.batch_size = setting.batch_size
        self.training_iters = setting.training_iters

        self.display_step = setting.display_step
        # 网络参数
        self.n_input = setting.n_input
        self.n_steps = setting.n_steps
        # 隐藏层大小
        self.n_hidden = setting.n_hidden
        self.n_classes = setting.n_classes

        initializer = tf.keras.initializers.glorot_normal

        # 使用这些参数获取现有变量或创建一个新变量。
        # 定义tf中用到的变量
        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
        # relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.cnn_size])

        # placeholder和feed_dict绑定，使用placeholder是要在运行时再给tf一个输入的值
        # 使用feed_dict在Session.run()时提供输入值。
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.len_sentence], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, self.len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, self.len_sentence], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        # 在word_embedding中找到input_word
        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)
        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)

        self.inputsOfLSTM = tf.concat(axis=2, values=[self.input_word_ebd, self.input_pos1_ebd, self.input_pos2_ebd])

        self.inputs = tf.reshape(self.inputsOfLSTM, [-1, self.len_sentence, self.word_embedding + self.pos_size * 2, 1])
        self.conv = layers.conv2d(inputs=self.inputs, num_outputs=self.cnn_size, kernel_size=[3, 60], stride=[1, 60],
                                  padding='SAME')
        self.max_pool = layers.max_pool2d(self.conv, kernel_size=[70, 1], stride=[1, 1])
        self.sentence = tf.reshape(self.max_pool, [-1, self.cnn_size])  # sentence.shape:  (?, 230)

        # self.inputsOfLSTM = tf.nn.dropout(self.inputsOfLSTM, 0.5)
        # 双向LSTM + attention
        with tf.variable_scope("bi-lstm"):
            _fw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, initializer=initializer())
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(_fw_cell, self.keep_prob)
            _bw_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, initializer=initializer())
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(_bw_cell, self.keep_prob)

            self.value, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                            inputs=self.inputsOfLSTM,
                                                            # inputsOfLSTM.shape:  (?, 70, 60)
                                                            dtype=tf.float32)

            self.value = tf.add(self.value[0], self.value[1])
        # 添加attention
        with tf.variable_scope('attention'):
            self.attn, self.alphas = attention(self.value)

        print("attn ", self.attn)
        print("value ", self.value)
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attn, self.keep_prob)

        with tf.variable_scope('output'):
            self.logits = tf.layers.dense(self.h_drop, self.n_classes, kernel_initializer=initializer())

        # LSTM
        # lstmCell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        #
        # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
        #
        # self.value, _ = tf.nn.dynamic_rnn(lstmCell, self.inputsOfLSTM, dtype=tf.float32)
        # # weght
        # self.weight = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_classes]), name='weight')
        # #
        # bias = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name='bias')
        # #
        # print("value.shape: ", self.value.shape)
        # #
        # print("weight.shape: ", self.weight.shape)
        #
        # value = tf.transpose(self.value, [1, 0, 2])
        #
        # self.last = tf.gather(value, int(value.get_shape()[0]) - 1)
        #
        # self.prediction = (tf.matmul(self.last, self.weight) + bias)
        #
        # self.drop = layers.dropout(self.prediction, keep_prob=self.keep_prob)
        #
        # # 1 3.59 0.656  2 3.47 0.632  3 3.37 0.67
        # self.logits = layers.fully_connected(inputs=self.h_drop, num_outputs=self.n_classes, activation_fn=tf.nn.softmax)
        # 奖励
        # r = tf.nn.softmax(self.logits)
        # prediction = tf.argmax(r, axis=1)
        print("logits ", self.logits)
        prediction = tf.nn.softmax(self.logits)

        self.reward = tf.log(tf.reduce_sum(self.input_y * prediction, axis=1))
        # 交叉损失
        # self.cross_loss = -tf.reduce_mean(tf.reduce_sum(self.input_y * tf.log(self.logits), axis=1))
        # self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
        #                                                       weights_list=tf.trainable_variables())
        # self.final_loss = self.cross_loss + self.l2_loss
        with tf.variable_scope("loss"):
            self.cross_loss = -tf.reduce_mean(tf.reduce_sum(self.input_y * tf.log(prediction), axis=1))
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                                  weights_list=tf.trainable_variables())
            self.final_loss = self.cross_loss + self.l2_loss
            # self.final_loss = tf.reduce_mean(losses) + 0.0001 * self.l2
        # self.final_loss = tf.reduce_mean(losses) + self.l2_loss

        self.pred = tf.argmax(prediction, axis=1)
        self.pred_prob = tf.reduce_max(prediction, axis=1)

        # self.y_label = tf.argmax(self.input_y, axis=1)
        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.input_y, axis=1)), 'float'))

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.final_loss)

        # 返回所有用' trainable=True '创建的变量。
        self.tvars = tf.trainable_variables()

        # manual update parameters 将tvars中的value值转换为index_holder
        # 先将tvars中的数据，转换为一个placeholder，内容为 index_holder，并添加到tvars_holders中
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        # 枚举tvars将value赋值为tvars_holders的值，添加到update_tvar_holder中
        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


def train(path_train_word, path_train_pos1, path_train_pos2, path_train_y, save_path):
    print('reading wordembedding')
    # 加载词向量嵌入
    wordembedding = np.load('./data/vec.npy', allow_pickle=True)

    print('reading training data')

    cnn_train_word = np.load(path_train_word, allow_pickle=True)
    cnn_train_pos1 = np.load(path_train_pos1, allow_pickle=True)
    cnn_train_pos2 = np.load(path_train_pos2, allow_pickle=True)
    cnn_train_y = np.load(path_train_y, allow_pickle=True)

    settings = Settings()
    settings.vocab_size = len(wordembedding)
    # settings.n_classes = len(cnn_train_y[0])
    settings.n_steps = len(cnn_train_word) // settings.batch_size
    # model = CNN(word_embeddings=wordembedding, setting=settings)

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for epoch in range(1, settings.num_epochs + 1):

                # 进度条
                bar = tqdm(range(settings.n_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000'.format(epoch))

                for _ in bar:
                    # 在cnn_train_y中随机选择batch_size个唯一随机元素。
                    sample_list = random.sample(range(len(cnn_train_y)), settings.batch_size)

                    batch_train_word = [cnn_train_word[x] for x in sample_list]
                    batch_train_pos1 = [cnn_train_pos1[x] for x in sample_list]
                    batch_train_pos2 = [cnn_train_pos2[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]

                    # 将训练数据添加到feed_dict中
                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.input_pos1] = batch_train_pos1
                    feed_dict[model.input_pos2] = batch_train_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob

                    # 训练数据。

                    _, loss, accuracy = sess.run(
                        [model.train_op, model.final_loss, model.accuracy],
                        feed_dict=feed_dict)

                    bar.set_description('epoch {} loss={:.6f} accuracy={:.6f}'.format(epoch, loss, accuracy))
                    # break
                # 训练完保存sess
                saver.save(sess, save_path=save_path)
                # break


class interaction():

    def __init__(self, sess, save_path='model/model.ckpt3'):

        self.settings = Settings()
        wordembedding = np.load('./data/vec.npy', allow_pickle=True)

        self.sess = sess
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.model = CNN(word_embeddings=wordembedding, setting=self.settings)
        sess.run(tf.global_variables_initializer())
        reader = pywrap_tensorflow.NewCheckpointReader(save_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # Print tensor name and values
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, save_path)

        self.train_word = np.load('./data/train_word.npy', allow_pickle=True)
        self.train_pos1 = np.load('./data/train_pos1.npy', allow_pickle=True)
        self.train_pos2 = np.load('./data/train_pos2.npy', allow_pickle=True)
        self.y_train = np.load('data/train_y.npy', allow_pickle=True)

        # 测试数据
        self.testall_word = np.load('./data/testall_word.npy', allow_pickle=True)
        self.testall_pos1 = np.load('./data/testall_pos1.npy', allow_pickle=True)
        self.testall_pos2 = np.load('./data/testall_pos2.npy', allow_pickle=True)

    # 计算奖励
    def reward(self, batch_test_word, batch_test_pos1, batch_test_pos2, batch_test_y):

        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.input_y] = batch_test_y
        feed_dict[self.model.keep_prob] = 1
        outputs = (self.sess.run(self.model.reward, feed_dict=feed_dict))
        return (outputs)

    # 计算句子的向量嵌入
    def sentence_ebd(self, batch_test_word, batch_test_pos1, batch_test_pos2, batch_test_y):
        feed_dict = {
            self.model.input_word: batch_test_word,
            self.model.input_pos1: batch_test_pos1,
            self.model.input_pos2: batch_test_pos2,
            self.model.input_y: batch_test_y,
            self.model.keep_prob: 1
        }
        outputs = self.sess.run(self.model.sentence, feed_dict=feed_dict)
        return (outputs)

    # 计算准确率
    def test(self, batch_test_word, batch_test_pos1, batch_test_pos2):
        feed_dict = {
            self.model.input_word: batch_test_word,
            self.model.input_pos1: batch_test_pos1,
            self.model.input_pos2: batch_test_pos2,
            self.model.keep_prob: 1
        }
        relation, prob = self.sess.run([self.model.pred, self.model.pred_prob], feed_dict=feed_dict)

        return (relation, prob)

    def update_cnn(self, update_word, update_pos1, update_pos2, update_y, updaterate):

        num_steps = len(update_word) // self.settings.batch_size
        # init = tf.global_variables_initializer()
        with self.sess.as_default():

            tvars_old = self.sess.run(self.model.tvars)

            for i in tqdm(range(num_steps)):
                batch_word = update_word[i * self.settings.batch_size:(i + 1) * self.settings.batch_size]
                batch_pos1 = update_pos1[i * self.settings.batch_size:(i + 1) * self.settings.batch_size]
                batch_pos2 = update_pos2[i * self.settings.batch_size:(i + 1) * self.settings.batch_size]
                batch_y = update_y[i * self.settings.batch_size:(i + 1) * self.settings.batch_size]

                feed_dict = {}
                feed_dict[self.model.input_word] = batch_word
                feed_dict[self.model.input_pos1] = batch_pos1
                feed_dict[self.model.input_pos2] = batch_pos2
                feed_dict[self.model.input_y] = batch_y
                feed_dict[self.model.keep_prob] = self.settings.keep_prob
                # _, loss, accuracy = sess.run([self.model.train_op,self.model.final_loss, self.model.accuracy], feed_dict=feed_dict)
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

            # get tvars_new
            tvars_new = self.sess.run(self.model.tvars)

            # update old variables of the target network
            tvars_update = self.sess.run(self.model.tvars)
            for index, var in enumerate(tvars_update):
                tvars_update[index] = updaterate * tvars_new[index] + (1 - updaterate) * tvars_old[index]

            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)

    # 计算每个关系的准确率
    def produce_ac(self):

        testall_word = self.testall_word
        testall_pos1 = self.testall_pos1
        testall_pos2 = self.testall_pos2
        dict_ac = {}
        len_batch = len(testall_word)

        with self.sess.as_default():
            for batch in tqdm(range(len_batch)):
                batch_word = testall_word[batch]
                batch_pos1 = testall_pos1[batch]
                batch_pos2 = testall_pos2[batch]

                (tmp_relation, tmp_prob) = self.test(batch_word, batch_pos1, batch_pos2)
                tmp_prob = list(tmp_prob)
                tmp_relation = list(tmp_relation)
                dict_ac.setdefault(tmp_relation[0], []).append(tmp_prob[0])

            for k, v in dict_ac.items():
                dict_ac[k] = np.mean(np.array(v))

            return dict_ac

    def produce_average_reward(self):

        train_word = self.train_word
        train_pos1 = self.train_pos1
        train_pos2 = self.train_pos2
        y_train = self.y_train
        all_reward_list = []
        len_batch = len(train_word)

        with self.sess.as_default():
            for batch in tqdm(range(len_batch)):
                batch_word = train_word[batch]
                batch_pos1 = train_pos1[batch]
                batch_pos2 = train_pos2[batch]
                # batch_y = train_y[batch]
                batch_y = [y_train[batch] for x in range(len(batch_word))]

                tmp_reward = self.reward(batch_word, batch_pos1, batch_pos2, batch_y)

                all_reward_list += list(tmp_reward)

            all_reward_list = np.array(all_reward_list)
            average_reward = np.mean(all_reward_list)
            average_reward = np.array(average_reward)

            return average_reward

    def produce_new_embedding(self):

        # produce reward sentence_ebd  average_reward
        train_word = self.train_word
        train_pos1 = self.train_pos1
        train_pos2 = self.train_pos2
        y_train = self.y_train
        all_sentence_ebd = []
        all_reward = []
        all_reward_list = []
        len_batch = len(train_word)

        with self.sess.as_default():
            for batch in tqdm(range(len_batch)):
                batch_word = train_word[batch]
                batch_pos1 = train_pos1[batch]
                batch_pos2 = train_pos2[batch]
                # batch_y = train_y[batch]
                batch_y = [y_train[batch] for x in range(len(batch_word))]

                tmp_sentence_ebd = self.sentence_ebd(batch_word, batch_pos1, batch_pos2, batch_y)
                tmp_reward = self.reward(batch_word, batch_pos1, batch_pos2, batch_y)

                all_sentence_ebd.append(tmp_sentence_ebd)
                all_reward.append(tmp_reward)
                all_reward_list += list(tmp_reward)

            all_reward_list = np.array(all_reward_list)
            average_reward = np.mean(all_reward_list)
            average_reward = np.array(average_reward)

            all_sentence_ebd = np.array(all_sentence_ebd)
            all_reward = np.array(all_reward)

            return average_reward, all_sentence_ebd, all_reward

    def save_cnnmodel(self, save_path):
        with self.sess.as_default():
            self.saver.save(self.sess, save_path=save_path)

    def tvars(self):
        with self.sess.as_default():
            tvars = self.sess.run(self.model.tvars)
            return tvars

    def update_tvars(self, tvars_update):
        with self.sess.as_default():
            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)


# produce reward sentence_ebd  average_reward
def produce_rldata(save_path):
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # start = time.time()
            interact = interaction(sess, save_path)
            average_reward, all_sentence_ebd, all_reward = interact.produce_new_embedding()
            np.save('data/average_reward.npy', average_reward)
            np.save('data/all_sentence_ebd.npy', all_sentence_ebd)
            np.save('data/all_reward.npy', all_reward)
            print(average_reward)

            dict_ac = interact.produce_ac()
            print(dict_ac)


if __name__ == '__main__':
    # train model
    print('train model')
    train('cnndata/cnn_train_word.npy', 'cnndata/cnn_train_pos1.npy', 'cnndata/cnn_train_pos2.npy','cnndata/cnn_train_y.npy', 'model/bi_lstm_attn_model/bi_lstm_attn_model.ckpt')

    # train('data/train_word.npy', 'data/train_pos1.npy', 'data/train_pos2.npy',
    #       'data/train_y.npy', 'model/origin_cnn_model.ckpt')

    # produce reward sentence_ebd  average_reward for rlmodel
    print('produce reward sentence_ebd  average_reward for rlmodel')
    produce_rldata(save_path='model/bi_lstm_attn_model/bi_lstm_attn_model.ckpt')
