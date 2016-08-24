'''
Created on Aug 23, 2016

@author: mjchao
'''
import dictionary


class Optimizers(object):
    """Enum for various optimizers that can be used in training.
    """
    SGD = range(1)


class TrainConfig(object):
    """Parameters for the training algorithm.
    """

    def __init__(self, data_file="data/corpus1.txt",
                 char_to_id_dictionary=dictionary.CharToIdDictionary(),
                 learning_rate=0.001, optimizer=Optimizers.SGD,
                 train_iters=100000, chars_per_sample=16, batch_size=128,
                 checkpoint_frequency=1000):
        self.data_file = data_file
        self.char_to_id_dictionary = char_to_id_dictionary
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_iters = train_iters
        self.chars_per_sample = chars_per_sample
        self.batch_size = batch_size
        self.checkpoint_frequency = checkpoint_frequency


def DefaultConfig():
    """Creates a default training configuration:

        Default configurations:
            data file: data/corpus1.txt
            learning rate: 0.001
            optimizer: stochastic gradient descent
            train iterations: 100000
            characters per sample: 16
            batch size: 128
            checkpoint frequency: once per 1000 iterations

    Returns:
        config: (TrainConfig) The default training configuration.
    """
    return TrainConfig()
