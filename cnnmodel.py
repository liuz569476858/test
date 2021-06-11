import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn
from tqdm import tqdm



class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.len_sentence = 70
        self.num_epochs = 3  # 在一个num_epochs中，所有训练集数据使用一次
        self.num_classes = 53
        self.cnn_size = 230
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        self.word_embedding = 50
        self.keep_prob = 0.5
        self.batch_size = 300  # 每个批次的大小
        self.num_steps = 10000
        self.lr = 0.001


class CNN():

    def __init__(self, word_embeddings, setting):

        self.vocab_size = setting.vocab_size
        self.len_sentence = len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes = setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr


        # 使用这些参数获取现有变量或创建一个新变量。
        # 定义tf中用到的变量
        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
        # relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.cnn_size])

        # placeholder和feed_dict绑定，使用placeholder是要在运行时再给tf一个输入的值
        # 使用feed_dict在Session.run()时提供输入值。
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)

        # 在word_embedding中找到input_word
        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)
        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)

        print("input_word_ebd: ", self.input_word_ebd)
        print("input_pos1_ebd: ", self.input_pos1_ebd)
        print("input_pos2_ebd: ", self.input_pos2_ebd)

        # 将values的列表沿维度“axis”连接起来。
        self.inputs = tf.concat(axis=2, values=[self.input_word_ebd, self.input_pos1_ebd, self.input_pos2_ebd])

        print("inputs: ", self.inputs)

        self.inputs = tf.reshape(self.inputs, [-1, self.len_sentence, self.word_embedding + self.pos_size * 2, 1])

        print("inputs: ", self.inputs)

        '''
        卷积层
        input：张量，必须是 half、float32、float64 三种类型之一。
        kernel_size: 一个整数，或者包含了两个整数的元组/队列，表示卷积窗的高和宽。
        strides：整数列表。长度是 4 的一维向量。输入的每一维度的滑动窗口步幅。必须与指定格式维度的顺序相同。
        padding：可选字符串为 SAME、VALID。要使用的填充算法的类型。卷积方式
        '''
        self.conv = layers.conv2d(inputs=self.inputs, num_outputs=self.cnn_size, kernel_size=[3, 60], stride=[1, 60],
                             padding='SAME')
        print("conv: ", self.conv)
        '''
        最大池化
        kernel_size：长度 >=4 的整数列表。输入张量的每个维度的窗口大小。
        strides：长度 >=4 的整数列表。输入张量的每个维度的滑动窗口的步幅。
        '''
        self.max_pool = layers.max_pool2d(self.conv, kernel_size=[70, 1], stride=[1, 1])

        print("max_pool: ",self.max_pool)
        # 全连接层
        # 将最大池化后的数据转换结构[[0~cnn_size],[0~cnn_size]]
        self.sentence = tf.reshape(self.max_pool, [-1, self.cnn_size])

        print("sentence: ", self.sentence)
        # 计算sentence的双曲正切，一般和dropout连用
        self.tanh = tf.nn.tanh(self.sentence)
        # tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
        self.drop = layers.dropout(self.tanh, keep_prob=self.keep_prob)


        # 添加一个完全连接的层。返回运算结果
        self.outputs = layers.fully_connected(inputs=self.drop, num_outputs=self.num_classes, activation_fn=tf.nn.softmax)

        print("outputs: ", self.outputs)
        print("input_y: ", self.input_y)

        '''
        self.y_index =  tf.argmax(self.input_y,1,output_type=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.outputs)[0]) * tf.shape(self.outputs)[1] + self.y_index
        self.responsible_outputs = - tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.outputs, [-1]),self.indexes)))
        '''
        # loss 损失函数
        # self.cross_loss = -tf.reduce_mean( tf.log(tf.reduce_sum( self.input_y  * self.outputs ,axis=1)))


        # 交叉损失
        self.cross_loss = -tf.reduce_mean(tf.reduce_sum(self.input_y * tf.log(self.outputs), axis=1))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.outputs, labels=self.input_y
        ))
        # 奖励
        self.reward = tf.log(tf.reduce_sum(self.input_y * self.outputs, axis=1))

        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())

        self.final_loss = self.cross_loss + self.l2_loss

        # accuracy
        # arg_max 返回一维张量中最大的值所在的位置
        self.pred = tf.argmax(self.outputs, axis=1)
        self.pred_prob = tf.reduce_max(self.outputs, axis=1)

        self.y_label = tf.argmax(self.input_y, axis=1)
        # 先比较pred和y_label，结果存放在一个布尔型列表中
        # 将结果转换为float类型
        # 计算所有float类型结果的平均值
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.y_label), 'float'))

        # minimize loss
        # 使用Adam算法最小化损失
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

    cnn_train_word = np.load(path_train_word)
    cnn_train_pos1 = np.load(path_train_pos1)
    cnn_train_pos2 = np.load(path_train_pos2)
    cnn_train_y = np.load(path_train_y)

    settings = Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(cnn_train_word) // settings.batch_size

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            # 实现权重的初始化
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            # 运行tf并初始化所有的变量
            sess.run(tf.global_variables_initializer())
            # 构造函数用于保存和恢复变量，也可以用于保存model
            # saver = tf.train.Saver()
            # saver.restore(sess,save_path=save_path)
            for epoch in range(1, settings.num_epochs + 1):

                # 进度条
                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000'.format(epoch))

                for _ in bar:
                    # 在cnn_train_y中随机选择batch_size个唯一随机元素。
                    sample_list = random.sample(range(len(cnn_train_y)), settings.batch_size)
                    # 同理
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
                    _, loss, cross_loss, cost, accuracy = sess.run([model.train_op, model.final_loss, model.cross_loss, model.cost, model.accuracy],
                                                 feed_dict=feed_dict)

                    conv = sess.run(model.conv, feed_dict=feed_dict)
                    max_pool = sess.run(model.max_pool, feed_dict=feed_dict)
                    sentence = sess.run(model.sentence, feed_dict=feed_dict)
                    tanh = sess.run(model.tanh, feed_dict=feed_dict)
                    drop = sess.run(model.drop, feed_dict=feed_dict)
                    outputs = sess.run(model.outputs, feed_dict=feed_dict)
                    pred = sess.run(model.pred, feed_dict=feed_dict)
                    y_label = sess.run(model.y_label, feed_dict=feed_dict)

                    # print("conv: ", conv[0])
                    # print("max_pool: ", max_pool[0])
                    print("sentence[0]: ", sentence[0])
                    print("sentence: ", sentence)
                    # print("tanh: ", tanh[0])
                    # print("drop: ", drop[0])
                    # print("outputs: ", outputs)
                    # print("pred: ", pred)
                    # print("y_label: ", y_label)


                    bar.set_description('epoch {} loss={:.6f} accuracy={:.6f}'.format(epoch, loss, accuracy))
                    # print('epoch {} cross_loss={:.6f} cost={:.6f}'.format(epoch, cross_loss, cost))
                    # break
                # 训练完保存sess
                # saver.save(sess, save_path=save_path)
                # break


class interaction():

    def __init__(self, sess, save_path='model/model.ckpt3'):

        self.settings = Settings()
        wordembedding = np.load('./data/vec.npy', allow_pickle=True)

        self.sess = sess
        with tf.variable_scope("model"):
            self.model = CNN(word_embeddings=wordembedding, setting=self.settings)

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
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.input_y] = batch_test_y
        feed_dict[self.model.keep_prob] = 1
        outputs = self.sess.run(self.model.sentence, feed_dict=feed_dict)
        return (outputs)

    # 计算准确率
    def test(self, batch_test_word, batch_test_pos1, batch_test_pos2):
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.keep_prob] = 1
        relation, prob = self.sess.run([self.model.pred, self.model.pred_prob], feed_dict=feed_dict)

        return (relation, prob)

    def update_cnn(self, update_word, update_pos1, update_pos2, update_y, updaterate):

        num_steps = len(update_word) // self.settings.batch_size

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
        dict_ac={}
        len_batch = len(testall_word)

        with self.sess.as_default():
            for batch in tqdm(range(len_batch)):
                batch_word = testall_word[batch]
                batch_pos1 = testall_pos1[batch]
                batch_pos2 = testall_pos2[batch]

                (tmp_relation, tmp_prob) = self.test(batch_word, batch_pos1, batch_pos2)
                tmp_prob=list(tmp_prob)
                tmp_relation=list(tmp_relation)
                dict_ac.setdefault(tmp_relation[0],[]).append(tmp_prob[0])

            for k,v in dict_ac.items():
                dict_ac[k]=np.mean(np.array(v))

            return dict_ac

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

            dict_ac = interact.produce_ac()

            np.save('data/average_reward.npy', average_reward)
            np.save('data/all_sentence_ebd.npy', all_sentence_ebd)
            np.save('data/all_reward.npy', all_reward)

            print(average_reward)
            print(dict_ac)


if __name__ == '__main__':
    # train model
    print('train model')
    train('cnndata/cnn_train_word.npy', 'cnndata/cnn_train_pos1.npy', 'cnndata/cnn_train_pos2.npy',
          'cnndata/cnn_train_y.npy', 'model/origin_cnn_model.ckpt')

    # produce reward sentence_ebd  average_reward for rlmodel
    print('produce reward sentence_ebd  average_reward for rlmodel')
    # produce_rldata(save_path='model/origin_cnn_model.ckpt')





