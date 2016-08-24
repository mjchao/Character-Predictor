'''
Created on Aug 23, 2016

@author: mjchao
'''
import unittest

import numpy as np
import dictionary
import reader


class ReaderTest(unittest.TestCase):

    def _CreateTestDictionary(self):
        return dictionary.CharToIdDictionary()

    def _CreateTestReader(self, char_to_id_dict, filename, batch_size):
        return reader.TrainingDataReader(char_to_id_dict, filename, batch_size)

    def _BuildExpectedBatches(self, char_to_id_dict, batch_strings):
        return [np.array([char_to_id_dict.GetId(c) for c in batch],
                         dtype=np.int32) for batch in batch_strings]

    def testGetBatchSize1(self):
        """Test smallest batch size possible.
        """
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(
            char_to_id_dict, "test_corpus_tiny.txt", 1)
        expected = self._BuildExpectedBatches(
            char_to_id_dict, [" ", "a", " ", "b", " ", "c", " ", "1", " ", "2"])
        found = [test_reader.GetBatch() for _ in range(10)]
        np.testing.assert_equal(found, expected)

    def testGetBatchSize3(self):
        """Test medium batch size.
        """
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(
            char_to_id_dict, "test_corpus_tiny.txt", 3)
        expected_batches = [" a ", "b c", " 1 ", "2 a", " b "]
        expected = [np.array([char_to_id_dict.GetId(c) for c in batch],
                             dtype=np.int32) for batch in expected_batches]
        found = [test_reader.GetBatch() for _ in range(5)]
        np.testing.assert_equal(found, expected)

    def testGetBatchSize15(self):
        """Test batch sizes larger than the entire corpus.
        """
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             "test_corpus_tiny.txt", 15)
        expected_batches = [" a b c 1 2 a b ", "c 1 2 a b c 1 2",
                            " a b c 1 2 a b "]
        expected = [np.array([char_to_id_dict.GetId(c) for c in batch],
                             dtype=np.int32) for batch in expected_batches]
        found = [test_reader.GetBatch() for _ in range(3)]
        np.testing.assert_equal(found, expected)


if __name__ == "__main__":
    unittest.main()
