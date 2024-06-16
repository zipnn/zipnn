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
        "method": "zstd",
        "delta_compressed_type": None,  # None no delta compression / for delta compression use "byte" or "file" - "torch" TBD
        "bg_partitions": 4,  # how namy Byte Goruping partitions -> Currently supported 4 (ideal for int32/ float32) and 1 (no Byte Grouping)
        "bg_compression_threshold": 0.99,  # Compression threshold of Byte Grouping
        "reorder_signbit": False,  # Move the sign bit to the lsb
        "lossy_compressed_type": None,
        "lossy_compressed_factor": 27,
        "is_streaming": False,  # Only to "file" at the moment
        "streaming_chunk_kb": 64 * 1024,  # if is_straming is True, the chunk size is MB
        "input_type": "byte",  # byte, torch, file (in case of file the data is the file name)
        "input_file": "byte",  # byte, torch, file in case of file, the is none
        "compressed_ret_type": "byte",  # byte, file
        "compressed_file": None,  # path to the compress file, if compress_type is "file"
        "decompressed_ret_type": "byte",  # byte, torch, file
        "decompressed_file": None,  # The file of the decompression
        "zstd_level": 3,
        "zstd_threads": 8,
        "lz4_compression_level": 0,
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
    if vars_dict["input_type"] == "byte":
        compressed_zipnn = zipnn.compress(original_bin)
    elif vars_dict["input_type"] == "file":
        compressed_zipnn = zipnn.compress(vars_dict["input_file"])
    elif vars_dict["input_type"] == "torch":
        compressed_zipnn = zipnn.compress(original_tensor)
    else:
        sys.exit(f"Unsupported input_type")
    compress_time = time.time() - start_time
    start_time = time.time()
    if vars_dict["compressed_ret_type"] == "file":
        decompressed_zipnn = zipnn.decompress(vars_dict["compressed_file"])
    else:
        decompressed_zipnn = zipnn.decompress(compressed_zipnn)
    decompress_time = time.time() - start_time

    # Asserts
    if vars_dict["input_type"] == "byte":
        input_byte = original_bin
    elif vars_dict["input_type"] == "file":
        with open(vars_dict["input_file"], "rb") as in_file_handler:
            input_byte = in_file_handler.read()
    elif vars_dict["input_type"] == "torch":
        input_byte = original_tensor.numpy().tobytes()
    else:
        sys.exit(f"Unsupported decompressed_ret_type")

    if vars_dict["decompressed_ret_type"] == "byte":
        decompressed_zipnn_byte = decompressed_zipnn
    elif vars_dict["decompressed_ret_type"] == "file":
        with open(vars_dict["decompressed_file"], "rb") as in_file_handler:
            decompressed_zipnn_byte = in_file_handler.read()
    elif vars_dict["decompressed_ret_type"] == "torch":
        decompressed_zipnn_byte = decompressed_zipnn.numpy().tobytes()
    else:
        sys.exit(f"Unsupported decompressed_ret_type")

    if vars_dict["lossy_compressed_type"] is None:
        self.assertEqual(input_byte, decompressed_zipnn_byte)

    if vars_dict["input_type"] == "torch" == vars_dict["decompressed_ret_type"] == "torch":
        self.assertEqual(original_tensor.shape, decompressed_zipnn.shape)
        self.assertEqual(original_tensor.dtype, decompressed_zipnn.dtype)
        if vars_dict["lossy_compressed_type"] is None:
            self.assertEqual(torch.all(torch.eq(original_tensor, decompressed_zipnn)), torch.tensor(True))
        else:
            differences = torch.abs(original_tensor - decompressed_zipnn)
            any_less_than_X = torch.all(differences > 2 ** (-1 * (vars_dict["lossy_compressed_factor"]))).any()
            if any_less_than_X != torch.tensor(False):
                sys.exit(
                    f"The lossy compression has a problem the differences between original_tensor and decompressed_zipnn are too large"
                )

    if vars_dict["compressed_ret_type"] == "byte":
        compressed_zipnn_byte = compressed_zipnn
    elif vars_dict["compressed_ret_type"] == "file":
        with open(vars_dict["compressed_file"], "rb") as in_file_handler:
            compressed_zipnn_byte = in_file_handler.read()

    if vars_dict["input_type"] == "file" and vars_dict["decompressed_ret_type"] == "file":
        self.assertEqual(compare_files(vars_dict["input_file"], vars_dict["decompressed_file"]), True)

    compress_ratio = len(compressed_zipnn_byte) / len(original_bin)
    var_str = f"compress_ratio {compress_ratio:.2f} compression_time = {compress_time} decompression_time {decompress_time} original_len {len(original_bin)}"
    for var, value in zipnn.__dict__.items():
        var_str += f" {var}: {value} "
    print(var_str)

    # delete files
    delete_file_if_exist(vars_dict["compressed_file"])
    delete_file_if_exist(vars_dict["decompressed_file"])


def delete_file_if_exist(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def write_file(file_path, data):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def compare_files(file1, file2):
    result = subprocess.run(["diff", file1, file2], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print("Files are identical.")
        return True
    elif result.returncode == 1:
        print("Files are different.")
        return False
    else:
        print("Diff command failed:", result.stderr)
        return None


def run_few_methods_types_lossy(
    self,
    original_bin,
    original_tensor,
    vars_dict,
    method_list: list,
    input_type_list: list,
    compressed_ret_type_list: list,
    decompressed_ret_type_list: list,
    input_file: str,
    compressed_file: str,
    decompressed_file: str,
    bg_partitions_list: list,
    lossy_type_list: list,
    lossy_factor_list: list,
):
    # one model different method "zstd","lz4","snappy"
    print("Check different methods  zstd,lz4,snappy with Byte Gorup 4")
    for bg_partitions in bg_partitions_list:
        for method in method_list:
            for input_type in input_type_list:
                for compressed_ret_type in compressed_ret_type_list:
                    for decompressed_ret_type in decompressed_ret_type_list:
                        if lossy_type_list is not None:
                            for lossy_type in lossy_type_list:
                                for lossy_factor in lossy_factor_list:
                                    print(
                                        f"bg {vars_dict['bg_partitions']} {method} {input_type}/{compressed_ret_type}/{decompressed_ret_type} lossy {lossy_factor} ",
                                        end="",
                                    )
                                    vars_dict = update_vars_dict(
                                        vars_dict,
                                        method=method,
                                        input_type=input_type,
                                        compressed_ret_type=compressed_ret_type,
                                        decompressed_ret_type=decompressed_ret_type,
                                        input_file=input_file,
                                        compressed_file=compressed_file,
                                        decompressed_file=decompressed_file,
                                        lossy_compressed_type=lossy_type,
                                        lossy_compressed_factor=lossy_factor,
                                        bg_partitions=bg_partitions,
                                    )
                                    test_zipnn(self, original_bin, original_tensor, vars_dict)
                        else:
                            vars_dict = update_vars_dict(
                                vars_dict,
                                method=method,
                                input_type=input_type,
                                compressed_ret_type=compressed_ret_type,
                                decompressed_ret_type=decompressed_ret_type,
                                input_file=input_file,
                                compressed_file=compressed_file,
                                decompressed_file=decompressed_file,
                                lossy_compressed_type=None,
                                lossy_compressed_factor=None,
                                bg_partitions=bg_partitions,
                            )
                            print(
                                f"bg {vars_dict['bg_partitions']} {method} {input_type}/{compressed_ret_type}/{decompressed_ret_type} lossless ",
                                end="",
                            )
                            test_zipnn(self, original_bin, original_tensor, vars_dict)


def build_tensors_and_vars():
    # Arrange: Original data to compress (a byte array)

    input_file = "tests/in.pkl"
    compressed_file = "tests/comp.zipnn"
    decompressed_file = "tests/out.pkl"

    random.seed(42)
    vars_dict = build_vars_dict()

    # Generate random floats between low_rand and high_rand

    # From -0.02 to 0.02
    size = 1024*1024
    original_tensor = torch.rand(size) * 0.04 - 0.02
    write_file(input_file, original_tensor)

    # Convert the tensor to a numpy array
    np_array = original_tensor.numpy()

    # Convert the numpy array to bytes
    original_bin = np_array.tobytes()
    print (f"original length in bytes {len(original_bin)}")
    return vars_dict, original_tensor, original_bin, input_file, compressed_file, decompressed_file


def test_compression_decompression_one_model_method(self):
    # one model different method "zstd","lz4","snappy" with and without byte grouping
    vars_dict, original_tensor, original_bin, input_file, compressed_file, decompressed_file = build_tensors_and_vars()

    print("Check different methods  zstd,lz4,snappy with Byte Gorup 4 and vanilla method (Byte Group = 1)")
    run_few_methods_types_lossy(
        self,
        original_bin,
        original_tensor,
        vars_dict,
        method_list=["zstd", "lz4", "snappy"],
        input_type_list=["byte"],
        compressed_ret_type_list=["byte"],
        decompressed_ret_type_list=["byte"],
        input_file=input_file,
        compressed_file=compressed_file,
        decompressed_file=decompressed_file,
        bg_partitions_list=[4, 1],
        lossy_type_list=None,
        lossy_factor_list=None,
    )


def test_compression_decompression_one_model_byte_file(self):
    # one model "byte"/"file" with "ZSTD"
    vars_dict, original_tensor, original_bin, input_file, compressed_file, decompressed_file = build_tensors_and_vars()

    print('\nCheck different input/compress/decompress types "byte"/"file" ')
    run_few_methods_types_lossy(
        self,
        original_bin,
        original_tensor,
        vars_dict,
        method_list=["zstd"],
        input_type_list=["byte", "file"],
        compressed_ret_type_list=["byte", "file"],
        decompressed_ret_type_list=["byte", "file"],
        input_file=input_file,
        compressed_file=compressed_file,
        decompressed_file=decompressed_file,
        bg_partitions_list=[4],
        lossy_type_list=None,
        lossy_factor_list=None,
    )


def test_compression_decompression_one_model_lossy(self):
    # one model "lossy  with "ZSTD"
    vars_dict, original_tensor, original_bin, input_file, compressed_file, decompressed_file = build_tensors_and_vars()

    print("\nCheck lossy")
    run_few_methods_types_lossy(
        self,
        original_bin,
        original_tensor,
        vars_dict,
        method_list=["zstd", "lz4", "snappy"],
        input_type_list=["torch"],
        compressed_ret_type_list=["byte"],
        decompressed_ret_type_list=["byte", "torch"],
        input_file=input_file,
        compressed_file=compressed_file,
        decompressed_file=decompressed_file,
        bg_partitions_list=[4],
        lossy_type_list=["integer"],
        lossy_factor_list=[16, 23, 27, 31],
    )


# if __name__ == '__main__':
#    unittest.main()
