#!/usr/bin/env python3
"""
This script compresses a .safetensors file.
"""
from argparse import ArgumentParser
from subprocess import check_call
from sys import executable


def check_and_install_zipnn():
    """
    Checks if zipnn is installed, and install it otherwise.
    """
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        check_call(
            [
                executable,
                "-m",
                "pip",
                "install",
                "zipnn",
                "--upgrade",
            ]
        )
        import zipnn


def compress_safetensors_file(filename):
    """
    Compress a safetensors file.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    from zipnn import ZipNN
    from zipnn.util_header import EnumFormat
    from zipnn.util_torch import zipnn_is_floating_point
    from zipnn.util_safetensors import (
        build_compressed_tensor_info,
        set_compressed_tensors_metadata,
        COMPRESSED_DTYPE, COMPRESSION_METHOD
    )
    import torch

    assert filename.endswith(".safetensors")

    tensors = {}
    compressed_tensor_infos = {}
    with safe_open(filename, "pt", "cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            if not zipnn_is_floating_point(EnumFormat.TORCH.value, tensor, tensor.dtype):
                tensors[name] = tensor
                continue

            compressed_tensor_info = build_compressed_tensor_info(tensor)

            znn = ZipNN(
                input_format="torch",
                bytearray_dtype=tensor.dtype,
                method=COMPRESSION_METHOD)
            compressed_buf = znn.compress(tensor)

            uncompressed_size = tensor.element_size() * tensor.nelement()
            compressed_size = len(compressed_buf)
            if compressed_size >= uncompressed_size:
                tensors[name] = tensor
                continue

            compressed_tensor = torch.frombuffer(compressed_buf, dtype=COMPRESSED_DTYPE)
            tensors[name] = compressed_tensor
            compressed_tensor_infos[name] = compressed_tensor_info

        metadata = f.metadata()

    set_compressed_tensors_metadata(compressed_tensor_infos, metadata)
    save_file(tensors, filename[: -(len(".safetensors"))] + ".znn.safetensors", metadata)


if __name__ == "__main__":
    parser = ArgumentParser(description="Enter a file path to compress.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Specify the path to the file to compress.",
    )

    args = parser.parse_args()
    check_and_install_zipnn()
    compress_safetensors_file(args.input_file)
