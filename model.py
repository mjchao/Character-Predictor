'''
Created on Aug 23, 2016

@author: mjchao
'''
import numpy as np
import tensorflow as tf
import reader
import train_config


class LSTM(object):
    """Represents an LSTM cell graph.

    The input training sample is (sequence, label). The outputs are returned
    by GetOutput.
    """

    def __init__(self, num_hidden_units, forget_bias, batch_size,
                 chars_per_sample, dictionary_size, sequence):
        self._weights = tf.Variable(tf.random_normal([num_hidden_units,
                                                      dictionary_size]))
        self._biases = tf.Variable(tf.random_normal([dictionary_size]))
        self._lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units,
                                                       forget_bias)
        self._outputs, _ = tf.nn.rnn(self._lstm_cell, sequence,
                                     dtype=tf.float32)

    def GetOutput(self):
        """Returns the output of this LSTM cell graph on the inputs.

        Returns:
            (Tensor) This LSTM cell's prediction on the inputs.
        """
        return tf.matmul(self._outputs[-1], self._weights) + self._biases


class CharacterPredictorModel(object):

    def __init__(self, config):
        self._config = config
        self._reader = reader.TrainingDataReader(config.char_to_id_dictionary,
                                                 config.data_file,
                                                 config.chars_per_sample,
                                                 config.batch_size)
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._BuildGraph()

        self._session = tf.Session(graph=self._graph)

    def _BuildGraph(self):
        self._inputs = tf.placeholder(tf.int32, [self._config.batch_size,
                                                 self._config.chars_per_sample])
        self._labels = tf.placeholder(tf.int32, [self._config.batch_size])
        self._lstm_cell = LSTM(self._config.num_hidden_units,
                               self._config.forget_bias,
                               self._config.batch_size,
                               self._config.char_to_id_dictionary.Size(),
                               self._inputs)
        self._pred = self._lstm_cell.GetOutput()
        self._cost = tf.reduce_mean(tf.gather(self._pred, self._labels))
        self._correct_preds = tf.equal(tf.argmax(self._pred, 1), self._labels)
        self._accuracy = tf.reduce_sum(tf.cast(self._correct_preds, tf.int32))

        if (self._config.optimizer == train_config.Optimizers.SGD):
            self._train_op = tf.train.GradientDescentOptimizer(
                self._config.learning_rate).minimize(self._cost)
        elif (self._config.optimizer == train_config.Optimizers.ADAM):
            self._train_op = tf.train.AdamOptimizer(
                self._config.learning_rate).minimize(self._cost)


    def Train(self):
        with self._session.as_default():
            tf.initialize_all_variables().run()
            for i in range(self._config.train_iters):
                sequences, labels = self._reader.GetBatch()
                cost, acc = self._session.run([self._cost, self._accuracy],{
                    self._inputs: sequences, self._labels: labels})
                if i % self._config.checkpoint_frequency == 0:
                    print "Checkpoint i: Cost = %.6f, Accuracy = %.2f" %(cost,
                                                                         acc)
