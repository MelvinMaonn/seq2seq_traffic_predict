import os
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import \
    InputLayer, Conv1d, MaxPool1d, \
    RNNLayer, DropoutLayer, DenseLayer, \
    LambdaLayer, ReshapeLayer, ConcatLayer, \
    Conv2d, MaxPool2d, FlattenLayer, DynamicRNNLayer, \
    DeConv2d, BatchNormLayer, ElementwiseLayer, \
    SubpixelConv2d, Seq2Seq, ExpandDimsLayer, TileLayer

import src.config as config
import lib.metrics as metrics

class Spacial_Model():

    def __init__(
            self,
            model_name,
            start_learning_rate,
            decay_rate,
            decay_steps
    ):
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.model_name = model_name

        self.__create_placeholders__()
        self.__create_model__()
        self.__create_loss__()
        self.__create_training_op__()

    def __create_placeholders__(self):
        self.x_root = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.in_seq_length, config.road_num],
            name='input_x_root'
        )
        self.x_neighbour = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.in_seq_length, config.num_neighbour],
            name='input_x_neighbour'
        )
        self.decode_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, config.road_num], # start_id at beginning
            name="decode_root_seqs"
        )
        self.decode_seqs_test = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, 1, 1], # start_id at beginning
            name="decode_root_seqs_test"
        )
        self.target_seqs = tf.placeholder(
            dtype=tf.float32,
            shape=[config.batch_size, config.out_seq_length + 1, config.road_num], # end_id at end
            name="target_root_seqs"
        )
        self.global_step = tf.placeholder(
            dtype=tf.int32,
            shape=[],
            name="global_step"
        )

    def __create_model__(self):
        self.train_net = self.__get_network__(
            self.x_root,
            self.x_neighbour,
            self.decode_seqs,
            is_train=True,
            reuse=False,
        )
        self.test_net = self.__get_network__(
            self.x_root,
            self.x_neighbour,
            self.decode_seqs_test,
            is_train=False,
            reuse=True,
        )
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __get_mape__(self, out, target):
        return tf.reduce_mean(tf.reduce_mean(tf.abs(out - target) / target, [1, 2]))

    def __create_loss__(self):
        self.mae_copy = tl.cost.absolute_difference_error(
            tf.slice(self.x_root, [0, config.in_seq_length - config.out_seq_length, 0], [config.batch_size, config.out_seq_length, config.road_num]),
            tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, config.road_num]),
            is_mean=True
        )
        # train loss
        self.nmse_train_loss = tl.cost.normalized_mean_square_error(self.train_net.outputs, self.target_seqs)

        outputs_noend = tf.slice(self.train_net.outputs, [0, 0, 0], [config.batch_size, config.out_seq_length, config.road_num])
        target_seqs_noend = tf.slice(self.target_seqs, [0, 0, 0], [config.batch_size, config.out_seq_length, config.road_num])
        self.nmse_train_noend = tl.cost.normalized_mean_square_error(outputs_noend, target_seqs_noend)

        preds = tf.reshape(outputs_noend, [-1])
        labels = tf.reshape(target_seqs_noend, [-1])
        self.rmse_train_noend = metrics.masked_rmse_tf(preds, labels, 0)
        # self.mae_train_noend = tf.reduce_mean(tf.abs(tf.reshape(self.train_net.outputs, [-1]) - tf.reshape(self.target_seqs, [-1])))
        self.mae_train_noend = metrics.masked_mae_tf(tf.reshape(self.train_net.outputs, [-1]), tf.reshape(self.target_seqs, [-1]), 0)
        self.mape_train_noend = metrics.masked_mape_tf(preds, labels, 0)
        # test loss
        '''
        self.nmse_test_loss = tl.cost.normalized_mean_square_error(self.test_net.outputs, self.target_seqs)
        self.nmse_test_noend = tl.cost.normalized_mean_square_error(outputs_noend, target_seqs_noend)
        self.rmse_test_noend = metrics.masked_rmse_tf(preds, labels, 0)
        self.mae_test_noend = metrics.masked_mae_tf(tf.reshape(self.train_net.outputs, [-1]), tf.reshape(self.target_seqs, [-1]), 0)
        self.mape_test_noend = metrics.masked_mape_tf(preds, labels, 0)
        '''
        self.nmse_test_loss = self.nmse_train_loss
        self.nmse_test_noend = self.nmse_train_noend
        self.rmse_test_noend = self.rmse_train_noend
        self.mae_test_noend = self.mae_train_noend
        self.mape_test_noend = self.mape_train_noend
        # adaptive train loss
        self.train_loss = self.nmse_train_loss
        self.test_loss = self.nmse_test_loss

    def __create_training_op__(self):
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.start_learning_rate,
            global_step=self.global_step,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
            name="learning_rate"
        )
        all_vars = tl.layers.get_variables_with_name(self.model_name)
        self.optim = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.train_loss, var_list=all_vars)

    def __get_network__(self, encode_seq, neighbour_seq, decode_seq, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            inputs_x_root = InputLayer(encode_seq, name='in_root')
            inputs_x_nbor = InputLayer(neighbour_seq, name="in_neighbour")

            # encoding neighbour graph information
            n = ReshapeLayer(inputs_x_nbor, (config.batch_size * config.in_seq_length, config.num_neighbour), "reshape1")
            n.outputs = tf.expand_dims(n.outputs, axis=-1)
            n = Conv1d(n, 4, 4, 1, act=tf.identity, padding='SAME', W_init=w_init, name='conv1')
            n = BatchNormLayer(n, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='bn1')
            n = MaxPool1d(n, 2, 2, padding='valid', name='maxpool1')
            n = FlattenLayer(n, name="flatten1")
            n = ReshapeLayer(n, (config.batch_size, config.in_seq_length, -1), name="reshape1_back")

            net_encode = ConcatLayer([inputs_x_root, n], concat_dim=-1, name="encode")
            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=1,
                return_seq_2d=True,
                name='seq2seq'
            )
            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            net_out = DenseLayer(net_rnn, n_units=1, act=tf.identity, name='dense2')
            if is_train:
                net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, config.road_num), name="reshape_out")
            else:
                net_out = ReshapeLayer(net_out, (config.batch_size, 1, config.road_num), name="reshape_out")

            self.net_rnn = net_rnn

            return net_out

class Seq2Seq_Model(Spacial_Model):

    def __create_model__(self):
        self.train_net = self.__get_network__(
            self.x_root,
            self.decode_seqs,
            is_train=True,
            reuse=False,
        )
        '''
        self.test_net = self.__get_network__(
            self.x_root,
            self.decode_seqs_test,
            is_train=False,
            reuse=True,
        )
        '''
        self.train_net.print_params(False)
        self.train_net.print_layers()

    def __get_network__(self, encode_seq, decode_seq, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        g_init = tf.random_normal_initializer(1., 0.02)

        with tf.variable_scope(self.model_name, reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            net_encode = InputLayer(encode_seq, name='in_root')

            net_decode = InputLayer(decode_seq, name="decode")

            net_rnn = Seq2Seq(
                net_encode, net_decode,
                cell_fn=tf.contrib.rnn.BasicLSTMCell,
                n_hidden=config.dim_hidden,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length=tl.layers.retrieve_seq_length_op(net_encode.outputs),
                decode_sequence_length=tl.layers.retrieve_seq_length_op(net_decode.outputs),
                initial_state_encode=None,
                # dropout=(0.8 if is_train else None),
                dropout=None,
                n_layer=2,
                return_seq_2d=True,
                name='seq2seq'
            )
            # net_out = DenseLayer(net_rnn, n_units=64, act=tf.identity, name='dense1')
            net_out = DenseLayer(net_rnn, n_units=config.road_num, act=tf.identity, name='dense2')
            if is_train:
                net_out = ReshapeLayer(net_out, (config.batch_size, config.out_seq_length + 1, config.road_num), name="reshape_out")
            else:
                net_out = ReshapeLayer(net_out, (config.batch_size, 1, config.road_num), name="reshape_out")

            self.net_rnn = net_rnn

            return net_out



if __name__ == "__main__":
    '''
    model = Spacial_Model(
        model_name="spacial_model",
        start_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8,
    )
    '''

    model = Seq2Seq_Model(
        model_name="seq2seq_model",
        start_learning_rate=0.001,
        decay_steps=400,
        decay_rate=0.8,
    )

