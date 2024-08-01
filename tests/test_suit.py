# test_zipnn.py

import unittest
from test_one_model import test_compression_decompression_float


class TestSuite(unittest.TestCase):

    def test_compression_decompression_float(self):
        test_compression_decompression_float(self)


if __name__ == "__main__":
    unittest.main()
