# test_zipnn.py

import unittest
from test_one_model import test_compression_decompression_float
from simple_stress_tests import test_byte_torch_streaming
from tests_auto_mode import test_auto_mode

class TestSuite(unittest.TestCase):

    def test_compression_decompression_float(self):
        test_compression_decompression_float(self)

    def test_byte_torch_streaming(self):
        test_byte_torch_streaming()
 
    def test_auto_mode(self):
        test_auto_mode(self)
    


if __name__ == "__main__":
    unittest.main()
