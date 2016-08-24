'''
Created on Aug 23, 2016

@author: mjchao
'''
import numpy as np


class TrainingDataReader(object):
    """Reads training data.
    """

    def __init__(self, dictionary, filename, sample_size=16, batch_size=128):
        """Creates a TrainingDataReader to read from a text file.

        Args:
            dictionary: The dictionary to use to convert characters to IDs. It
                just has to supply a GetId() function that takes a char.
            filename: (string) The file from which to read training data.
            sample_size: (int) The number of characters per training sample.
            batch_size: (int) The number of samples per batch of gradient
                descent.
        """
        self._dictionary = dictionary
        self._filename = filename
        self._sample_size = sample_size
        self._batch_size = batch_size

        # Variables to keep track of where we are in the file
        self._reader = open(self._filename)
        self._epoch = 0

    def GetSample(self):
        """Gets the next sample.
        
        Returns:
            sample: (numpy array) An array of size (sample_size,) that is the
                sequence of characters in the training sample.
        """
        rtn = np.zeros((self._sample_size,), dtype=np.int32)
        fill_from_idx = 0
        chars_to_read = self._sample_size
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

    def GetBatch(self):
        """Gets the next batch of training examples.
        
        Returns:
            batch: (numpy array) A numpy array of size (batch_size, sample_size)
                that is the next batch of training data.
        """
        return np.array([self.GetSample() for _ in range(self._batch_size)])

    def GetEpoch(self):
        """Gets the current epoch.

        Returns:
            epoch: (int) The number of passes made through the entire corpus.
        """
        return self._epoch
