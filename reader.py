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

        self._reader = open(self._filename)
        self._epoch = 0

        # The buffer is an array-based queue that serves as a sliding window
        # over the training data file. Every time we get a sample, we slide the
        # window forward by 1 character.
        self._buffer = []
        self._buffer_start_idx = 0
        self._label = ""
        self._resetting_epoch = False
        self._ResetEpoch()


    def ConsumeChars(self, num_chars):
        """Consumes characters from the training data file.

        Args:
            num_chars: (int) The number of characters to consume

        Returns:
            (list) A list of length num_chars that are the characters consumed.
            Or None if the buffer was reset due to changing epoch.
        """
        chars_to_read = num_chars
        consumed_chars = []
        fill_from_idx = 0
        while chars_to_read > 0:
            chars_for_filling = list(self._reader.read(chars_to_read))
            chars_read = len(chars_for_filling)

            # Prevent infinite loop on empty files
            if chars_read == 0 and chars_to_read == self._sample_size:
                raise RuntimeError("Input file is empty.")

            chars_to_read -= chars_read
            consumed_chars += chars_for_filling
            fill_from_idx += chars_read

            # If the file didn't have enough to fill the batch, we need
            # to start a new epoch.
            if chars_to_read > 0:
                self._epoch += 1
                self._ResetEpoch()
                return None

        return consumed_chars

    def _ResetEpoch(self):
        if self._resetting_epoch:
            raise RuntimeError("File is too small to fill a batch.")
        self._resetting_epoch = True
        self._reader = open(self._filename)
        new_chars = self.ConsumeChars(self._sample_size + 1)
        self._buffer = [self._dictionary.GetId(c) for c in
                        new_chars[:-1]]
        self._buffer_start_idx = 0
        self._label = self._dictionary.GetId(new_chars[-1])
        self._resetting_epoch = False

    def GetSample(self):
        """Gets the next sample.

        Returns:
            sequence: (numpy array) An array of size (sample_size,) that is the
                sequence of characters in the training sample.
            label: (int) The character that should come next in the sequence.
        """
        sequence = np.array(self._buffer[self._buffer_start_idx:] +
                            self._buffer[:self._buffer_start_idx])
        label = self._label

        # Move the sliding window buffer forward by 1 character.
        self._buffer[self._buffer_start_idx] = label
        self._buffer_start_idx += 1
        self._buffer_start_idx %= self._sample_size


        next_label = self.ConsumeChars(1)
        # If we reset due to changing epoch, we don't have to do anything.
        # The ResetEpoch() method will take care of everything.
        if next_label is not None:
            self._label = self._dictionary.GetId(next_label[0])
        return sequence, label

    def GetBatch(self):
        """Gets the next batch of training examples.

        Returns:
            sequences: (numpy array) A numpy array of shape
                (batch_size, sample_size) that is the next batch of training
                data sequences.
            labels: (numpy array) A numpy array of shape (batch_size) that are
                the labels for the next batch of training data sequences.
        """
        sequences = []
        labels = []
        for _ in range(self._batch_size):
            sequence, label = self.GetSample()
            sequences.append(sequence)
            labels.append(label)
        return sequences, labels

    def GetEpoch(self):
        """Gets the current epoch.

        Returns:
            epoch: (int) The number of passes made through the entire corpus.
        """
        return self._epoch
