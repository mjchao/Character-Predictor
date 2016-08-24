'''
Created on Aug 23, 2016

@author: mjchao
'''
import tensorflow as tf
import reader


class CharacterPredictorModel(object):

    def __init__(self, config):
        self._config = config
        self._reader = reader.TrainingDataReader(config.char_to_id_dictionary,
                                                 config.data_file,
                                                 config.chars_per_sample,
                                                 config.batch_size)
        self._BuildGraph()

    def _BuildGraph(self):
        pass

    def Train(self):
        pass