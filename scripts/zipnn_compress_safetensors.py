#!/usr/bin/env python3
"""
This script compresses a .safetensors file.
"""
from argparse import ArgumentParser
from subprocess import check_call
from sys import executable
import argparse
import time
import multiprocessing
import os
import subprocess



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


def compress_safetensors_file(filename,delete=False,force=False,hf_cache=False,method=None,threads=None):
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

    load_time_sum=0
    comp_time_sum=0
    comp_len=0
    og_len=0

    tensors = {}
    compressed_tensor_infos = {}

    compressed_path=filename[: -(len(".safetensors"))] + ".znn.safetensors"
    if not force and os.path.exists(compressed_path):
        user_input = (
            input(f"{compressed_path} already exists; overwrite (y/n)? ").strip().lower()
        )

        if user_input not in ("yes", "y"):
            print(f"Skipping {filename}...")
            return
    print(f"Compressing {filename}...")

    time_start=time.time()
    with safe_open(filename, "pt", "cpu") as f:
        load_time_sum+=time.time()-time_start
        
        for name in f.keys():
            time_start=time.time()
            tensor = f.get_tensor(name)
            load_time_sum+=time.time()-time_start
            
            if not zipnn_is_floating_point(EnumFormat.TORCH.value, tensor, tensor.dtype):
                tensors[name] = tensor
                continue

            compressed_tensor_info = build_compressed_tensor_info(tensor)
            znn = ZipNN(
                input_format="torch",
                bytearray_dtype=tensor.dtype,
                method = method if method is not None else COMPRESSION_METHOD,
                threads=threads)

            uncompressed_size = tensor.element_size() * tensor.nelement()
            tensor_save = tensor.clone()
            time_start=time.time()
            compressed_buf = znn.compress(tensor)
            comp_time_sum+=time.time()-time_start
            uncompressed_buf = znn.decompress(compressed_buf)
            compressed_size = len(compressed_buf)       
            og_len+=uncompressed_size
            print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(tensor_save.to(torch.uint8), uncompressed_buf.to(torch.uint8)))
            


            if compressed_size >= uncompressed_size:
                tensors[name] = tensor
                comp_len+=uncompressed_size
                continue
            comp_len+=compressed_size
            print (compressed_size/uncompressed_size)

            compressed_tensor = torch.frombuffer(compressed_buf, dtype=COMPRESSED_DTYPE)
            tensors[name] = compressed_tensor
            compressed_tensor_infos[name] = compressed_tensor_info

        metadata = f.metadata()

    set_compressed_tensors_metadata(compressed_tensor_infos, metadata)
    time_start=time.time()
    save_file(tensors, filename[: -(len(".safetensors"))] + ".znn.safetensors", metadata)
    write_time=time.time()-time_start

    
    if delete and not hf_cache:
        print(f"Deleting {filename}...")
        os.remove(filename)

    if hf_cache:
        # If the file is in the Hugging Face cache, fix the symlinks
        print(f"{YELLOW}Reorganizing Hugging Face cache...{RESET}")
        try:
            snapshot_path = os.path.dirname(filename)
            blob_name = os.path.join(snapshot_path, os.readlink(filename))
            os.rename(compressed_path, blob_name)
            os.symlink(blob_name, compressed_path)
                
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            raise Exception(f"Error reorganizing Hugging Face cache: {e}")

    print(f"Compressed {filename} to {compressed_path} using {znn.threads} threads")
    print(f"sum of load times: {load_time_sum}s")
    print(f"sum of comp times: {comp_time_sum}s")
    print(f"comp file written in {write_time}s, ratio is {comp_len/og_len}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Enter a file path to compress.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Specify the path to the file to compress.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of a single file instead of compression",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="A flag that forces overwriting when compressing.",
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="A flag that indicates if the file is in the Hugging Face cache.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["HUFFMAN", "ZSTD", "FSE", "AUTO"],
        default="HUFFMAN",
        help="Specify the method to use. Default is HUFFMAN.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="The amount of threads to be used.",
    )
    args = parser.parse_args()
    optional_kwargs = {}
    if args.delete:
        optional_kwargs["delete"] = args.delete
    if args.force:
        optional_kwargs["force"] = args.force
    if args.hf_cache:
        optional_kwargs["hf_cache"] = args.hf_cache
    if args.method:
        optional_kwargs["method"] = args.method
    if args.threads:
        optional_kwargs["threads"] = args.threads#
    check_and_install_zipnn()
    compress_safetensors_file(args.input_file,**optional_kwargs)
