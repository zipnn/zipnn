import unittest
from zipnn import ZipNN
import random
import torch
import zstd
import pickle
import time
import sys
import subprocess
import os


def build_vars_dict():
    vars_dict = {
            "method" : "zstd",
            "input_format" : "byte",
            "bytearray_dtype" : "float32",
            "threads" : 1,
            "bg" : 0
    }
    return vars_dict


def update_vars_dict(vars_dict, **kwargs):
    for key, value in kwargs.items():
        if key in vars_dict:
            vars_dict[key] = value
        else:
            print(f"Warning: {key} is not in the dictionary and will be ignored.")
    return vars_dict


def test_zipnn(self, original_bin, original_tensor, vars_dict):
    zipnn = ZipNN(**vars_dict)
    # Act: Compress and then decompress
    start_time = time.time()
    if vars_dict["input_format"] == "byte":
        compressed_zipnn_byte = zipnn.compress(original_bin)
    elif vars_dict["input_format"] == "torch":
        compressed_zipnn_byte = zipnn.compress(original_tensor)
    else:
        sys.exit(f"Unsupported input_format")
    compress_time = time.time() - start_time
    start_time = time.time()
    decompressed_zipnn = zipnn.decompress(compressed_zipnn_byte)
    decompress_time = time.time() - start_time

    # Asserts
    if vars_dict["input_format"] == "byte":
        input_byte = original_bin
    elif vars_dict["input_format"] == "torch":
        input_byte = original_tensor.numpy().tobytes()
    else:
        sys.exit(f"Unsupported decompressed_ret_type")

    if vars_dict["input_format"] == "byte":
        decompressed_zipnn_byte = decompressed_zipnn
    elif vars_dict["input_format"] == "torch":
        decompressed_zipnn_byte = decompressed_zipnn.numpy().tobytes()
    else:
        sys.exit(f"Unsupported decompressed_ret_type")

    if vars_dict["input_format"] == "torch":
        self.assertEqual(original_tensor.shape, decompressed_zipnn.shape)
        self.assertEqual(original_tensor.dtype, decompressed_zipnn.dtype)

    compress_ratio = len(compressed_zipnn_byte) / len(original_bin)
    var_str = f"compress_ratio {compress_ratio:.2f} compression_time = {compress_time} decompression_time {decompress_time} original_len {len(original_bin)}"
    for var, value in zipnn.__dict__.items():
        var_str += f" {var}: {value} "
    print(var_str)



def run_few_config(
    self,
    original_bin,
    original_tensor,
    vars_dict,
    method_list,
    input_format_list,
    bytearray_dtype_list,
    bg_list,
    threads_list,

):
    # one model different method "zstd","lz4","snappy"
    print("Check different methods  zstd,lz4,snappy with Byte Gorup 4")
    for method in method_list:
        for input_format in input_format_list:
            for bytearray_dtype in bytearray_dtype_list:
                for bg in bg_list:
                    for threads in threads_list:
                            vars_dict = update_vars_dict(
                                vars_dict,
                                method=method,
                                input_format=input_format,
                                bytearray_dtype = bytearray_dtype,
                                bg = bg,
                                threads = threads

                            )
                            if (input_format == "byte"):
                                print(f"{method} {input_format} /bytearray_dtype={bytearray_dtype}/bg={bg}/threads={threads}")
                            else:    
                                print(f"{method} {input_format} /bg={bg}/threads={threads}")

                            test_zipnn(self, original_bin, original_tensor, vars_dict)


def build_tensors_and_vars():
    # Arrange: Original data to compress (a byte array)

    random.seed(42)
    vars_dict = build_vars_dict()

    # Generate random floats between low_rand and high_rand

    # From -0.02 to 0.02
    size = 1024*1024
    original_tensor = torch.rand(size) * 0.04 - 0.02

    # Convert the tensor to a numpy array
    np_array = original_tensor.numpy()

    # Convert the numpy array to bytes
    original_bin = np_array.tobytes()
    print (f"original length in bytes {len(original_bin)}")
    return vars_dict, original_tensor, original_bin


def test_compression_decompression_one_model_method(self):
    # one model different method "zstd","lz4","snappy" with and without byte grouping
    vars_dict, original_tensor, original_bin = build_tensors_and_vars()

    print("Check different methods  zstd,lz4,snappy with Byte Gorup 4 and vanilla method (Byte Group = 1)")
    run_few_config(
        self,
        original_bin,
        original_tensor,
        vars_dict,
        method_list=["zstd", "lz4", "snappy"],
        input_format_list=["byte", "torch"],
        bytearray_dtype_list = ["float32"],
        bg_list=[0, 2, 4],
        threads_list = [1, 8]
    )


# if __name__ == '__main__':
#    unittest.main()
