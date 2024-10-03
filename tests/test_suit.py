# test_zipnn.py

import unittest
from test_one_model import test_compression_decompression_float
from simple_tests import test_byte_torch_streaming

class TestSuite(unittest.TestCase):

    def test_compression_decompression_float(self):
        test_compression_decompression_float(self)

    def test_byte_torch_streaming(self):
        test_byte_torch_streaming()
    


if __name__ == "__main__":
    unittest.main()
