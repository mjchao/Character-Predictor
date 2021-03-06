'''
Created on Aug 23, 2016

@author: mjchao
'''
import sys
import numpy as np
import tensorflow as tf
import reader
import train_config


class LSTM(object):
    """Represents an LSTM cell graph.

    The input training sample is (sequence, label). The outputs are returned
    by GetOutput.
    """

    def __init__(self, num_hidden_units, batch_size, forget_bias,
                 dictionary_size, sequence):
        # Components for training
        self._weights = tf.Variable(tf.random_normal([num_hidden_units,
                                                      dictionary_size]))
        self._biases = tf.Variable(tf.random_normal([dictionary_size]))
        self._lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_units,
                                                       forget_bias)
        self._train_saved_state = tf.Variable(tf.zeros([batch_size, dictionary_size]))
        self._train_saved_output = tf.Variable(tf.zeros([1, batch_size,
                                                         num_hidden_units]))
        self._train_outputs, _ = tf.nn.rnn(
            self._lstm_cell, [tf.cast(sequence, tf.float32)],
            self._train_saved_state, dtype=tf.float32)

        # Components for generating new sequences
        self._write_saved_state = tf.Variable(tf.zeros([1, dictionary_size]))
        self._write_saved_output = tf.Variable(tf.zeros([1, 1,
                                                         num_hidden_units]))
        with tf.variable_scope("write"):
            self._write_output, self._write_state = tf.nn.rnn(
                self._lstm_cell, [tf.cast(sequence, tf.float32)],
                self._write_saved_state, dtype=tf.float32)

    def GetTrainOutput(self):
        """Returns the output of this LSTM cell graph on the training inputs.

        Returns:
            (Tensor) This LSTM cell's prediction on the inputs.
        """
        with tf.control_dependencies(
            [self._train_saved_output.assign(self._train_outputs),
             self._train_saved_state.assign(self._train_saved_state)]):
            return tf.matmul(
                self._train_outputs[-1], self._weights) + self._biases

    def GetWriteOutput(self):
        """Returns the output of this LSTM cell graph for user inputs.

        Returns:
            (Tensor) A character that this LSTM writes to continue the user
                input.
        """
        with tf.control_dependencies(
            [self._write_saved_state.assign(self._write_state),
             self._write_saved_output.assign(self._write_output)]):
            return tf.matmul(self._write_output[-1],
                             self._weights) + self._biases


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
        self._inputs = tf.placeholder(tf.float32,
                                      [None, self._config.chars_per_sample])
        self._labels = tf.placeholder(
            tf.float32, [self._config.batch_size,
                         self._config.char_to_id_dictionary.Size()])
        self._lstm_cell = LSTM(self._config.num_hidden_units,
                               self._config.batch_size,
                               self._config.forget_bias,
                               self._config.char_to_id_dictionary.Size(),
                               self._inputs)
        self._pred = self._lstm_cell.GetTrainOutput()
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

        self._write_pred = self._lstm_cell.GetWriteOutput()
        self._write_id = tf.argmax(self._write_pred, 1)[0]
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
            for i in range(self._config.train_iters):
                sequences, labels = self._reader.GetBatch()
                cost, acc, _ = self._session.run(
                    [self._cost, self._accuracy, self._train_op],
                    {self._inputs: sequences,
                     self._labels: self._ConvertToOneHot(labels)})
                sys.stdout.write("\r Iteration %d: Cost = %.6f, "
                                 "Accuracy = %.4f" %(i, cost, acc))
                sys.stdout.flush()
                if i % self._config.checkpoint_frequency == 0:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

    def _Predict(self, string):
        sequence = reader.ConvertStringToSequence(
            self._config.char_to_id_dictionary, string)
        return self._session.run(self._write_id, {self._inputs: [sequence]})

    def ContinueWriting(self, start_sequence, num_chars):
        result = start_sequence
        curr_sequence = start_sequence
        for _ in range(num_chars):
            next_id = self._Predict(curr_sequence)
            next_char = self._config.char_to_id_dictionary.GetChar(next_id)
            curr_sequence = (curr_sequence[1:] + next_char)
            result += next_char
        return result

