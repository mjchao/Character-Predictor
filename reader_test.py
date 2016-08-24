'''
Created on Aug 23, 2016

@author: mjchao
'''
import unittest
import os
import numpy as np
import dictionary
import reader


class ReaderTest(unittest.TestCase):

    _TEST_DATA_PATH = os.path.join("test_data", "test_corpus_tiny.txt")

    def _CreateTestDictionary(self):
        return dictionary.CharToIdDictionary()

    def _CreateTestReader(self, char_to_id_dict, filename, sample_size,
                          batch_size):
        return reader.TrainingDataReader(char_to_id_dict, filename, sample_size,
                                         batch_size)

    def _AssertCorrectBatches(self, char_to_id_dict,
                              found, expected_sequences, expected_labels):
        sequence_ids = np.array([np.array([
            char_to_id_dict.GetId(c) for c in seq]) for seq in expected_sequences])
        label_ids = np.array([char_to_id_dict.GetId(c) for c in
                              expected_labels])
        for (i, sample) in enumerate(found):
            np.testing.assert_equal(sample[0], sequence_ids[i])
            np.testing.assert_equal(sample[1], label_ids[i])

    def testGetSampleSize1(self):
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 1, 1)
        found = [test_reader.GetSample() for _ in range(3)]
        expected_sequences = [" ", "a", " "]
        expected_labels = ["a", " ", "b"]
        self._AssertCorrectBatches(char_to_id_dict, found, expected_sequences,
                                   expected_labels)

    def testGetSampleSize3(self):
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 3, 1)
        found = [test_reader.GetSample() for _ in range(4)]
        expected_sequences = [" a ", "a b", " b ", "b c"]
        expected_labels = ["b", " ", "c", " "]
        self._AssertCorrectBatches(char_to_id_dict, found, expected_sequences,
                                   expected_labels)

    def testGetSampleSize7(self):
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 7, 1)
        found = [test_reader.GetSample() for _ in range(7)]
        expected_sequences = [" a b c ", "a b c 1", " b c 1 ", " a b c ",
                              "a b c 1", " b c 1 ", " a b c "]
        expected_labels = ["1", " ", "2", "1", " ", "2", "1"]
        self._AssertCorrectBatches(char_to_id_dict, found, expected_sequences,
                                   expected_labels)

    def testGetSampleSize9(self):
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 9, 1)
        found = [test_reader.GetSample() for _ in range(3)]
        expected_sequences = [" a b c 1 ", " a b c 1 ", " a b c 1 "]
        expected_labels = ["2", "2", "2"]
        self._AssertCorrectBatches(char_to_id_dict, found, expected_sequences,
                                   expected_labels)

    def testGetSampleSize10(self):
        """Test sample size larger than characters in 1 epoch.
        """
        char_to_id_dict = self._CreateTestDictionary()
        with self.assertRaises(RuntimeError):
            self._CreateTestReader(char_to_id_dict, ReaderTest._TEST_DATA_PATH,
                                   10, 1)

    def testGetBatch(self):
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 7, 7)
        sequences, labels = test_reader.GetBatch()
        found = zip(sequences, labels)
        expected_sequences = [" a b c ", "a b c 1", " b c 1 ", " a b c ",
                              "a b c 1", " b c 1 ", " a b c "]
        expected_labels = ["1", " ", "2", "1", " ", "2", "1"]
        self._AssertCorrectBatches(char_to_id_dict, found, expected_sequences,
                                   expected_labels)

if __name__ == "__main__":
    unittest.main()
