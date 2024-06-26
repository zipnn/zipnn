# test_zipnn.py

import unittest
from test_one_model import test_compression_decompression_one_model_method
from test_one_model import test_compression_decompression_one_model_byte_file
from test_one_model import test_compression_decompression_one_model_lossy

# from test_two_models import test_compression_decompression_two_models


class TestSuite(unittest.TestCase):

    def test_compression_decompression_one_model_method(self):
        test_compression_decompression_one_model_method(self)

    def test_compression_decompression_one_model_byte_file(self):
        test_compression_decompression_one_model_byte_file(self)

    def test_compression_decompression_one_model_one_model_lossy(self):
        test_compression_decompression_one_model_lossy(self)


if __name__ == "__main__":
    unittest.main()
