__author__ = "Hernan Contigiani"
__version__ = '1'

import unittest
from unittest import TestCase

import numpy as np

from clase_1.indexing import Indexer


class IndexingTestCase(TestCase):

    def test_ind2idx(self):
        ids = np.array([15, 12, 14, 10, 1, 2, 1], dtype=np.int64)
        indexer = Indexer(ids)
        expected_id2idx = np.array([-1, 4, 5, -1, -1, -1, -1, -1, -1, -1, 3, -1, 1, -1, 2, 0], dtype=np.int64)
        np.testing.assert_equal(indexer.id2idx, expected_id2idx)

    def test_get_user_idx(self):
        ids = np.array([15, 12, 14, 10, 1, 2, 1], dtype=np.int64)
        indexer = Indexer(ids)

        id = 15
        expected_idx = 0
        np.testing.assert_equal(indexer.get_user_idx(id), expected_idx)

        id = 12
        expected_idx = 1
        np.testing.assert_equal(indexer.get_user_idx(id), expected_idx)

        id = 3
        expected_idx = -1
        np.testing.assert_equal(indexer.get_user_idx(id), expected_idx)

    def test_get_user_id(self):
        ids = np.array([15, 12, 14, 10, 1, 2, 1], dtype=np.int64)
        indexer = Indexer(ids)

        idx = 0
        expected_id = 15
        np.testing.assert_equal(indexer.get_user_id(idx), expected_id)

        idx = 4
        expected_id = 1
        np.testing.assert_equal(indexer.get_user_id(idx), expected_id)


if __name__ == "__main__":
    unittest.main()
