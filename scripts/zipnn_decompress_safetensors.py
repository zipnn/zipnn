#!/usr/bin/env python3
"""
This script decompresses a .safetensors file.
"""
from argparse import ArgumentParser
from subprocess import check_call
from sys import executable
import time
import os
import multiprocessing


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


def decompress_safetensors_file(filename, delete=False,force=False,hf_cache=False,threads=None):
    """
    Decompress a safetensors file.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    from zipnn import ZipNN
    from zipnn.util_header import EnumFormat
    from zipnn.util_torch import zipnn_is_floating_point,ZipNNDtypeEnum
    from zipnn.util_safetensors import (
        get_compressed_tensors_metadata,
        COMPRESSED_DTYPE, COMPRESSION_METHOD
    )
    import torch

    assert filename.endswith("znn.safetensors")

    load_time_sum=0
    decomp_time_sum=0
    comp_len=0
    decomp_len=0
    
    tensors = {}
    compressed_tensor_infos = {}

    decompressed_path=filename[: -(len(".znn.safetensors"))] + ".safetensors"
    if not force and os.path.exists(decompressed_path):
        user_input = (
            input(f"{decompressed_path} already exists; overwrite (y/n)? ").strip().lower()
        )

        if user_input not in ("yes", "y"):
            print(f"Skipping {filename}...")
            return
    print(f"Decompressing {filename}...")
    
    time_start=time.time()
    with safe_open(filename, "pt", "cpu") as f:
        load_time_sum+=time.time()-time_start
        
        L=f.metadata()
        D=get_compressed_tensors_metadata(L)
        znn = ZipNN(
                input_format="torch",
                bytearray_dtype=COMPRESSED_DTYPE,
                method=COMPRESSION_METHOD,
                threads=threads)
        for name in f.keys():
            time_start=time.time()
            tensor = f.get_tensor(name)
            #tmp=D.get(name, {}).get("dtype")
            #print(f"Tensor '{name}' dtype: {tensor.dtype},pre:{COMPRESSED_DTYPE}, from dixt: {tmp}")
            #znn = ZipNN(
            #    input_format="torch",
            #    bytearray_dtype=D.get(name, {}).get("dtype") or COMPRESSED_DTYPE,
            #    method=COMPRESSION_METHOD,
            #    threads=threads)
            load_time_sum+=time.time()-time_start
            
            #L=f.metadata()
            #D=get_compressed_tensors_metadata(L)
            if name not in D.keys(): 
                tensors[name]=tensor
                continue
            
            comp_len+=tensor.element_size() * tensor.nelement()
            time_start=time.time()
            decompressed_buf = znn.decompress(tensor.contiguous().numpy())
            decomp_time_sum+=time.time()-time_start
            decomp_len+=decompressed_buf.element_size() * decompressed_buf.nelement()

            tensors[name] = decompressed_buf

        metadata = f.metadata()
        if metadata:
            metadata.pop("znn_compressed_vectors", None)

    time_start=time.time()
    save_file(tensors, decompressed_path, metadata)
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
            os.rename(decompressed_path, blob_name)
            os.symlink(blob_name, decompressed_path)
                
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            raise Exception(f"Error reorganizing Hugging Face cache: {e}")
    
    print(f"Decompressed {filename} to {decompressed_path} using {znn.threads} threads")
    print(f"sum of load times: {load_time_sum}s")
    print(f"sum of decomp times: {decomp_time_sum}s")
    print(f"decomp file written in {write_time}s, ratio is {comp_len/decomp_len}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Enter a file path to decompress.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Specify the path to the file to decompress.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of the single compressed file instead after decompression",
    )
    parser.add_argument(
        "--force", action="store_true", help="A flag that forces overwriting when decompressing."
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="A flag that indicates if the file is in the Hugging Face cache.",
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
    if args.threads:
        optional_kwargs["threads"] = args.threads#
    
    check_and_install_zipnn()
    decompress_safetensors_file(args.input_file,**optional_kwargs)
