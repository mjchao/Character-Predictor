'''
Created on Aug 23, 2016

@author: mjchao
'''
import numpy as np
import sys
import tensorflow as tf
import reader
import train_config


class LSTM(object):
    """Represents an LSTM cell graph.

    The input training sample is (sequence, label). The outputs are returned
    by GetOutput.
    """

    def __init__(self, num_hidden_units, forget_bias, dictionary_size,
                 sequence):
        self._weights = tf.Variable(tf.random_normal([num_hidden_units,
                                                      dictionary_size]))
        self._biases = tf.Variable(tf.random_normal([dictionary_size]))
        self._lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units,
                                                       forget_bias)
        self._outputs, _ = tf.nn.rnn(
            self._lstm_cell, [tf.cast(sequence, tf.float32)], dtype=tf.float32)

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
        self._inputs = tf.placeholder(tf.float32, [self._config.batch_size,
                                                 self._config.chars_per_sample])
        self._labels = tf.placeholder(tf.float32,
            [self._config.batch_size,
             self._config.char_to_id_dictionary.Size()])
        self._lstm_cell = LSTM(self._config.num_hidden_units,
                               self._config.forget_bias,
                               self._config.char_to_id_dictionary.Size(),
                               self._inputs)
        self._pred = self._lstm_cell.GetOutput()
        self._cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self._pred, self._labels))
        self._correct_preds = tf.equal(tf.argmax(self._pred, 1),
                                       tf.argmax(self._labels, 1))
        self._num_correct = tf.reduce_sum(tf.cast(self._correct_preds,
                                                  tf.int32))
        self._accuracy = tf.cast(
            self._num_correct, tf.float32) / self._config.batch_size

        if self._config.optimizer == train_config.Optimizers.SGD:
            self._train_op = tf.train.GradientDescentOptimizer(
                self._config.learning_rate).minimize(self._cost)
        elif self._config.optimizer == train_config.Optimizers.ADAM:
            self._train_op = tf.train.AdamOptimizer(
                self._config.learning_rate).minimize(self._cost)

        self._init = tf.initialize_all_variables()

    def _ConvertToOneHot(self, labels):
        one_hot = np.zeros([len(labels),
                            self._config.char_to_id_dictionary.Size()],
                           dtype=np.float32)
        for i, label in enumerate(labels):
            one_hot[i][label] = 1
        return one_hot

    def Train(self):
        with self._session.as_default():
            self._init.run()

            total_cost = total_acc = 0.0
            total_iters = 0
            for i in range(self._config.train_iters):
                sequences, labels = self._reader.GetBatch()
                cost, acc = self._session.run([self._cost, self._accuracy], {
                    self._inputs: sequences,
                    self._labels: self._ConvertToOneHot(labels)})

                total_cost += cost
                total_acc += acc
                total_iters += 1
                avg_cost = total_cost / total_iters
                avg_acc = total_acc / total_iters
                sys.stdout.write("\r Iteration %d: Average cost = %.6f, "
                                 "Average accuracy = %.4f" %(i,
                                                             avg_cost, avg_acc))
                sys.stdout.flush()
                if i % self._config.checkpoint_frequency == 0:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
