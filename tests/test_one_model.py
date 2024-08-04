import random
import sys
import time
import torch
from zipnn import ZipNN


def build_vars_dict():
    vars_dict = {
        "method": "zstd",
        "input_format": "byte",
        "bytearray_dtype": "float32",
        "threads": 1,
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
    original_bin_saved = bytearray(original_bin)
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
        decompressed_zipnn_byte = decompressed_zipnn

    if vars_dict["input_format"] == "torch":
        self.assertEqual(original_tensor.shape, decompressed_zipnn.shape)
        self.assertEqual(original_tensor.dtype, decompressed_zipnn.dtype)
    else:
        self.assertEqual(original_bin_saved[1], decompressed_zipnn_byte[1])

    compress_ratio = len(compressed_zipnn_byte) / len(original_bin)
    var_str = f"compress_ratio {compress_ratio:.2f} compression_time = {compress_time} decompression_time {decompress_time} original_len {len(original_bin)}"
    for var, value in vars_dict.items():
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
    threads_list,
):
    # one model different method "zstd","lz4","snappy"
    for method in method_list:
        for input_format in input_format_list:
            for bytearray_dtype in bytearray_dtype_list:
                for threads in threads_list:
                    vars_dict = update_vars_dict(
                        vars_dict, method=method, input_format=input_format, bytearray_dtype=bytearray_dtype, threads=threads
                    )
                    if input_format == "byte":
                        print(f"{method} {input_format} /bytearray_dtype={bytearray_dtype}/threads={threads}")
                    else:
                        print(f"{method} {input_format} / threads={threads}")

                    test_zipnn(self, original_bin, original_tensor, vars_dict)


def build_tensors_and_vars(dtype):
    # Arrange: Original data to compress (a byte array)

    random.seed(42)
    vars_dict = build_vars_dict()

    # Generate random floats between low_rand and high_rand

    if dtype == torch.float32:
        bytearray_dtype = "float32"
    elif dtype == torch.bfloat16:
        bytearray_dtype = "bfloat16"
    elif dtype == torch.float16:
        bytearray_dtype = "float16"

    element_size = torch.tensor([], dtype=dtype).element_size()
    #    num_elements = 1024*1024*1024 // element_size
    num_elements = 1024 * 1024 // element_size

    element_size = torch.tensor([], dtype=dtype).element_size()

    # Create a tensor of these many elements of type float32
    # Initialize the tensor with random numbers from a uniform distribution between -1 and 1
    original_tensor = torch.rand(num_elements, dtype=dtype) * 2 - 1

    # Convert the tensor to a numpy array
    if dtype == torch.bfloat16:
        tensor_uint16 = original_tensor.view(torch.uint16)
        np_array = tensor_uint16.numpy()
    else:
        np_array = original_tensor.numpy()

    # Convert the numpy array to bytes
    original_bin = np_array.tobytes()
    print(f"original length in bytes {len(original_bin)}")
    return vars_dict, original_tensor, original_bin, bytearray_dtype


def test_compression_decompression_float(self):
    # one model different method "zstd","lz4","snappy" with and without byte grouping

    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        vars_dict, original_tensor, original_bin, bytearray_dtype = build_tensors_and_vars(dtype)

        print("Check different standart option with different dtypes")
        run_few_config(
            self,
            original_bin,
            original_tensor,
            vars_dict,
            method_list=["zstd"],
            input_format_list=["byte", "torch"],
            bytearray_dtype_list=[bytearray_dtype],
            threads_list=[1],
        )


# if __name__ == '__main__':
#    unittest.main()
