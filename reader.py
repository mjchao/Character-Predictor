'''
Created on Aug 23, 2016

@author: mjchao
'''
import numpy as np


class TrainingDataReader(object):

    def __init__(self, dictionary, filename, batch_size=128):
        self._dictionary = dictionary
        self._filename = filename
        self._batch_size = batch_size

        # Variables to keep track of where we are in the file
        self._reader = open(self._filename)
        self._epoch = 0

    def GetBatch(self):
        rtn = np.zeros((self._batch_size,), dtype=np.int32)
        fill_from_idx = 0
        chars_to_read = self._batch_size
        while chars_to_read > 0:
            chars_for_filling = list(self._reader.read(chars_to_read))
            chars_read = len(chars_for_filling)
            chars_to_read -= chars_read
            rtn[fill_from_idx:fill_from_idx + chars_read] = [
                self._dictionary.GetId(c) for c in chars_for_filling]
            fill_from_idx += chars_read

            # If the file didn't have enough to fill the batch, we need
            # to start a new epoch.
            if chars_to_read > 0:
                self._epoch += 1
                self._reader = open(self._filename)

        return rtn
