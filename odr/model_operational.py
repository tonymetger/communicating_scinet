#   Copyright 2020 communicating_scinet (https://github.com/tonymetger/communicating_scinet)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from math import ceil
import os
import numpy as np
import glob
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import l2_regularizer
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2
import pickle as pickle
from . import io
from .data_handler import Dataset


class OperationalNetwork(object):

    def __init__(self, encoder_num=2, decoder_num=4,
                 input_sizes=[20, 20], latent_sizes=[2, 2], question_sizes=[1, 1, 1, 1], answer_sizes=[1, 1, 1, 1],
                 encoder_num_units=[100, 100], decoder_num_units=[100, 100], name='Unnamed',
                 tot_epochs=0, load_file=None):
        """
        Params:
        =======
        encoder_num: 
            number of encoders
        decoder_num: 
            number of decoders
        input_sizes: 
            [length of input for enc_1, length of input for enc_2, ...]
        latent_size: 
            [number of latent neurons for enc_1, number of latent neurons for enc_2, ...]
        question_size: 
            [question length for dec_1, question length for dec_2, ...]
        answer_size: 
            [answer length for dec_1, answer length for dec_2, ...]
        encoder_num_units, decoder_num_units: 
            Number of neurons in encoder and decoder hidden layers. Everything is fully connected. 
            All encoders and decoders have the same hidden layer sizes.
        name: 
            used for tensorboard
        Note:
            tot_epochs and load_file are used internally for loading and saving, don't pass anything to them manually.
        """

        self.graph = tf.Graph()

        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.input_sizes = input_sizes
        self.latent_sizes = latent_sizes
        self.question_sizes = question_sizes
        self.answer_sizes = answer_sizes
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.name = name
        self.tot_epochs = tot_epochs

        # Set up neural network
        self.graph_setup()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            initialize_uninitialized(self.session)

        # Load saved network
        self.load_file = load_file
        if self.load_file is not None:
            self.load(self.load_file)

    #########################################
    #           Public interface            #
    #########################################

    def train(self, epoch_num, batch_size, learning_rate, training_data, validation_data,
              reg_loss_factor=0., gamma=0.01, pretrain=False, nloc_factor=1.,
              test_step=None, progress_bar=lambda x: x, add_feeds={}):
        """
        Trains the network.

        Params:
        =======
        epoch_num (int): 
            number of training epochs
        batch_size (int), learning_rate (float): 
            self-explanatory
        training_data, validation_data: 
            Dataset objects
        reg_loss_factor:
            for weight regularization
        gamma:
            coefficient in cost function weighing the cost of communicating with decoders
        pretrain:
            if True, selection noise level is fixed and very low
        nloc_factor:
            coefficient in cost function to weigh contribution from interaction experiment (specific to em_glof example, set to 1 otherwise)
        test_step (int, optional): 
            network is tested on validation data after this number of epochs and tensorboard summaries are written
        progress_bar:
            can pass tqdm_notebook here if desired
        add_feeds:
            add custom stuff to feed_dict

        """
        with self.graph.as_default():
            initialize_uninitialized(self.session)
            vd_dict = self.gen_data_dict(validation_data)

            for epoch_iter in progress_bar(list(range(epoch_num))):

                if test_step is not None and self.tot_epochs > 0 and self.tot_epochs % test_step == 0:
                    self.test(validation_data, training_data)

                self.tot_epochs += 1

                for step, data_dict in enumerate(self.gen_batch(training_data, batch_size)):
                    parameter_dict = {self.learning_rate: learning_rate, self.reg_loss_factor: reg_loss_factor,
                                      self.nloc_factor: nloc_factor,
                                      self.gamma: gamma}
                    feed_dict = {**data_dict, **parameter_dict, **add_feeds}

                    if pretrain:
                        self.session.run(self.pretraining_op, feed_dict=feed_dict)
                    else:
                        self.session.run(self.training_op, feed_dict=feed_dict)

    def test(self, data, t_data=None):
        """
        Test accuracy of neural network by comparing mean of output distribution to actual values.

        Params:
        =======
        data: 
            Dataset object
        """
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data)
            summary = self.session.run(self.vd_summaries, feed_dict={self.reg_loss_factor: 0., self.nloc_factor: 1., self.gamma: 1., **data_dict})
            self.summary_writer.add_summary(summary, global_step=self.tot_epochs)

            if t_data is not None:
                data_dict = self.gen_data_dict(t_data)
                summary_td = self.session.run(self.td_summaries, feed_dict={self.reg_loss_factor: 0., self.nloc_factor: 1., self.gamma: 1., **data_dict})
                self.summary_writer.add_summary(summary_td, global_step=self.tot_epochs)

    def run(self, data, layer, additional_params={}):
        """
        Run the network and output return the result.

        Params:
        =======
        data: 
            Dataset object
        layer: 
            specifies the layer that is run
        """
        with self.graph.as_default():
            data_dict = self.gen_data_dict(data)
            return self.session.run(layer, feed_dict={**data_dict, self.reg_loss_factor: 0., **additional_params})

    def save(self, file_name):
        """
        Saves state variables (weights, biases) of neural network

        Params:
        =======
        file_name (str): 
            model is saved in folder tf_save as file_name.ckpt
        """
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, io.tf_save_path + file_name + '.ckpt')
            params = {'encoder_num': self.encoder_num,
                      'decoder_num': self.decoder_num,
                      'input_sizes': self.input_sizes,
                      'latent_sizes': self.latent_sizes,
                      'question_sizes': self.question_sizes,
                      'answer_sizes': self.answer_sizes,
                      'encoder_num_units': self.encoder_num_units,
                      'decoder_num_units': self.decoder_num_units,
                      'tot_epochs': self.tot_epochs,
                      'name': self.name}
            with open(io.tf_save_path + file_name + '.pkl', 'wb') as f:
                pickle.dump(params, f)
            print("Saved network to file " + file_name)

    @classmethod
    def from_saved(cls, file_name, change_params={}):
        """
        Initializes a new network from saved data.

        file_name (str): 
            model is loaded from tf_save/file_name.ckpt
        """
        with open(io.tf_save_path + file_name + '.pkl', 'rb') as f:
            params = pickle.load(f)
        params['load_file'] = file_name
        for p in change_params:
            params[p] = change_params[p]
        print(params)
        return cls(**params)

    #########################################
    #        Private helper functions       #
    #########################################

    def graph_setup(self):
        """
        Set up the computation graph for the neural network based on the parameters set at initialization
        """
        with self.graph.as_default():

            #######################
            # Define placeholders #
            #######################
            self.gamma = tf.placeholder(tf.float32, shape=[], name='gamma')
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.reg_loss_factor = tf.placeholder(tf.float32, shape=[], name='reg_loss_factor')
            self.nloc_factor = tf.placeholder(tf.float32, shape=[], name='nloc_factor')
            self.total_latent_size = np.sum(self.latent_sizes)

            self.inputs = [tf.placeholder(tf.float32, [None, self.input_sizes[k]], name='input{}'.format(k))
                           for k in range(self.encoder_num)]

            self.question_inputs = [
                tf.placeholder(tf.float32, shape=[None, self.question_sizes[i]], name='q_dec{}'.format(i))
                for i in range(self.decoder_num)
            ]

            self.answers = [
                tf.placeholder(tf.float32, shape=[None, self.answer_sizes[i]], name='q_dec{}'.format(i))
                for i in range(self.decoder_num)
            ]

            self.select_noise = [
                tf.placeholder(tf.float32, shape=[None, self.total_latent_size], name='select_noise_{}'.format(i))
                for i in range(self.decoder_num)
            ]

            def fc_layer(in_layer, num_outputs, activation_fn, collection='std'):
                return fully_connected(in_layer, num_outputs, activation_fn,
                                       weights_regularizer=l2_regularizer(1.),
                                       biases_regularizer=l2_regularizer(1.),
                                       variables_collections=[collection])

            ##########################################
            # Set up variables and computation graph #
            ##########################################
            self.individual_latent = []
            for k in range(self.encoder_num):
                with tf.variable_scope('encoder_{}'.format(k)):
                    temp_layer = self.inputs[k]
                    for n in self.encoder_num_units:
                        temp_layer = fc_layer(temp_layer, num_outputs=n, activation_fn=tf.nn.elu)
                    self.individual_latent.append(fc_layer(temp_layer, num_outputs=self.latent_sizes[k], activation_fn=tf.identity))

            with tf.variable_scope('latent_layer'):
                self.full_latent = tf.concat(self.individual_latent, axis=1)
                latent_std = tf.math.sqrt(tf.nn.moments(self.full_latent, axes=[0])[1])
                self.select_logs = []
                self.dec_inputs = []
                for n in range(self.decoder_num):
                    with tf.variable_scope('select_dec{}'.format(n)):
                        selectors = tf.get_variable('sf_log',
                                                    initializer=tf.initializers.constant(-10.),
                                                    shape=self.total_latent_size,
                                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'sel'])
                        self.select_logs.append(selectors)
                        self.dec_inputs.append(self.full_latent + latent_std * tf.exp(selectors) * self.select_noise[n])

            self.outputs = []
            for n in range(self.decoder_num):
                with tf.variable_scope('dec{}'.format(n)):
                    temp_layer = tf.concat([self.dec_inputs[n], self.question_inputs[n]], axis=1, name='dec_in')

                    for q in self.decoder_num_units:
                        temp_layer = fc_layer(temp_layer, num_outputs=q, activation_fn=tf.nn.elu)

                    out = np.pi / 2. * fc_layer(temp_layer, num_outputs=self.answer_sizes[n], activation_fn=tf.identity)

                self.outputs.append(out)

            #####################
            # Cost and training #
            #####################
            with tf.name_scope('cost'):
                sel_cost_list = []
                ans_cost_list = []
                for n in range(self.decoder_num):
                    sel_cost_list.append(tf.reduce_mean(self.select_logs[n]))
                    ans_cost_list.append(tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.answers[n], self.outputs[n]), axis=1)))

                self.cost_select = (-1) * tf.add_n(sel_cost_list)
                loc_cut = int(ceil(self.decoder_num / 2))
                self.cost_loc = tf.add_n([ans_cost_list[i] for i in range(0, loc_cut)], name='cost_local')
                self.cost_nloc = tf.add_n([ans_cost_list[i] for i in range(loc_cut, self.decoder_num)], name='cost_local')
                self.weighted_cost = (self.cost_loc + self.nloc_factor * self.cost_nloc) / (1. + self.nloc_factor)

            with tf.name_scope('reg_loss'):
                self.reg_loss = tf.losses.get_regularization_loss()

            with tf.name_scope('optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

                self.training_op = self.train_op_from_loss(optimizer, self.weighted_cost)
                self.pretraining_op = self.train_op_from_loss(optimizer, self.weighted_cost, collections=['std', 'loc_decoder', 'nloc_decoder'])

            #########################
            # Tensorboard summaries #
            #########################

            chart = []
            for i in range(self.decoder_num):
                chart.append(layout_pb2.Chart(
                    title='Decoder {}'.format(i),
                    multiline=layout_pb2.MultilineChartContent(
                        tag=[r'^sf_log_{}'.format(i)]
                    )
                ))

            layout_summary = summary_lib.custom_scalar_pb(
                layout_pb2.Layout(category=[
                    layout_pb2.Category(
                        title='Select factors',
                        chart=chart)
                ])
            )

            tf.summary.scalar('cost_select', self.cost_select, collections=['vd'])
            tf.summary.scalar('cost', self.weighted_cost, collections=['vd'])
            tf.summary.scalar('cost_td', self.weighted_cost, collections=['td'])
            tf.summary.scalar('cost_loc', self.cost_loc, collections=['vd'])
            tf.summary.scalar('cost_nloc', self.cost_nloc, collections=['vd'])
            tf.summary.scalar('reg_loss', self.reg_loss, collections=['vd'])

            for i in range(self.decoder_num):
                for l in range(self.total_latent_size):
                    tf.summary.scalar('sf_log_{}_{}'.format(i, l), self.select_logs[i][l], collections=['vd'])

            for i in range(len(self.decoder_num_units)):
                weight_id = '' if i == 0 else '_{}'.format(i)
                for j in range(self.decoder_num):
                    tf.summary.histogram('dec{}_weight_{}'.format(j, i),
                                         self.graph.get_tensor_by_name('dec{}/fully_connected{}/weights:0'.format(j, weight_id)),
                                         collections=['vd'])

            for i in range(len(self.encoder_num_units)):
                weight_id = '' if i == 0 else '_{}'.format(i)
                for k in range(self.encoder_num):
                    tf.summary.histogram('enc_weight_{}'.format(i),
                                         self.graph.get_tensor_by_name('encoder_{}/fully_connected{}/weights:0'.format(k, weight_id)),
                                         collections=['vd'])

            self.summary_writer = tf.summary.FileWriter(io.tf_log_path + self.name + '/', graph=self.graph)
            self.summary_writer.add_summary(layout_summary)
            self.summary_writer.flush()
            self.vd_summaries = tf.summary.merge_all(key='vd')
            self.td_summaries = tf.summary.merge_all(key='td')

    def train_op_from_loss(self, optimizer, loss, collections=[tf.GraphKeys.TRAINABLE_VARIABLES]):
        var_list = []
        for col in collections:
            var_list += tf.get_collection(col)
        gvs = optimizer.compute_gradients(loss + self.reg_loss_factor * self.reg_loss + self.gamma * self.cost_select, var_list=var_list)
        capped_gvs = []
        for grad, var in gvs:
            if grad is not None:
                capped_gvs.append((tf.clip_by_value(grad, -10., 10.), var))
            else:
                capped_gvs.append((grad, var))
        return optimizer.apply_gradients(capped_gvs)

    def gen_batch(self, data, batch_size, shuffle=True):
        """
        Generate batches for training the network.

        Params:
        =======
        data: 
            either a dataset object or a list [input_data, questions, answers], 
            where the items in the list are of the same format as they are in the Dataset object
        batch_size (int):
            /
        shuffle (bool): 
            if True, data is shuffled before batches are created
        """
        if type(data) is Dataset:
            data = [np.array(data.input_data),
                    np.array(data.questions),
                    np.array(data.answers)]
        if shuffle:
            p = np.random.permutation(len(data[0]))
            data = [data[i][p] for i in [0, 1, 2]]
        for i in range(int(len(data[0]) / batch_size)):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            batch = [data[j][batch_slice] for j in [0, 1, 2]]
            yield self.gen_data_dict(batch)

    def gen_data_dict(self, data):
        """
        Params:
        =======
        data: 
            either a dataset object or a list [input_data, questions, answers], 
            where the items in the list are of the same format as they are in the Dataset object
        """
        data_dict = {}
        if type(data) is Dataset:
            data = [np.array(data.input_data),
                    np.array(data.questions),
                    np.array(data.answers)]
        for k in range(self.encoder_num):
            data_dict[self.inputs[k]] = data[0][:, k]
            for n in range(self.decoder_num):
                data_dict[self.question_inputs[n]] = data[1][:, n]
                data_dict[self.answers[n]] = data[2][:, n]
                data_dict[self.select_noise[n]] = np.random.normal(size=[len(data[0]), self.total_latent_size])

        return data_dict

    def load(self, file_name):
        """
        Loads network, params as in save
        """
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, io.tf_save_path + file_name + '.ckpt')
            print("Loaded network from file " + file_name)


###########
# Helpers #
###########


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
