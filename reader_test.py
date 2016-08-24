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

    def _BuildExpectedBatches(self, char_to_id_dict, batch_strings):
        return [np.array([char_to_id_dict.GetId(c) for c in batch],
                         dtype=np.int32) for batch in batch_strings]

    def testGetSampleSize1(self):
        """Test smallest batch size possible.
        """
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(
            char_to_id_dict, ReaderTest._TEST_DATA_PATH, 1, 1)
        expected = self._BuildExpectedBatches(
            char_to_id_dict, [" ", "a", " ", "b", " ", "c", " ", "1", " ", "2"])
        found = [test_reader.GetSample() for _ in range(10)]
        np.testing.assert_equal(found, expected)
        self.assertEqual(test_reader.GetEpoch(), 0)

    def testGetSampleSize3(self):
        """Test medium batch size.
        """
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(
            char_to_id_dict, ReaderTest._TEST_DATA_PATH, 3, 1)
        expected_batches = [" a ", "b c", " 1 ", "2 a", " b "]
        expected = self._BuildExpectedBatches(char_to_id_dict, expected_batches)
        found = [test_reader.GetSample() for _ in range(5)]
        np.testing.assert_equal(found, expected)
        self.assertEqual(test_reader.GetEpoch(), 1)

    def testGetSampleSize15(self):
        """Test batch sizes larger than the entire corpus.
        """
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 15, 1)
        expected_batches = [" a b c 1 2 a b ", "c 1 2 a b c 1 2",
                            " a b c 1 2 a b "]
        expected = self._BuildExpectedBatches(char_to_id_dict, expected_batches)
        found = [test_reader.GetSample() for _ in range(3)]
        np.testing.assert_equal(found, expected)
        self.assertEqual(test_reader.GetEpoch(), 4)

    def testGetBatch(self):
        char_to_id_dict = self._CreateTestDictionary()
        test_reader = self._CreateTestReader(char_to_id_dict,
                                             ReaderTest._TEST_DATA_PATH, 15, 3)
        expected_batches = [" a b c 1 2 a b ", "c 1 2 a b c 1 2",
                            " a b c 1 2 a b "]
        expected = np.array(self._BuildExpectedBatches(char_to_id_dict,
                                                       expected_batches))
        found = test_reader.GetBatch()
        np.testing.assert_equal(found, expected)
        self.assertEqual(test_reader.GetEpoch(), 4)


if __name__ == "__main__":
    unittest.main()
