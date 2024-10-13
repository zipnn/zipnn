import argparse
from zipnn import ZipNN
import torch
import time
import zstandard as zstd
import numpy as np
import os
import requests

def download_file(url, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    print("Start downloading file")
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("End downloading file")

def main(file_path, method, start_byte, end_byte, bytearray_dtype):
    if not os.path.exists(file_path):
        url = 'https://huggingface.co/ibm-granite/granite-3b-code-base/resolve/main/model-00002-of-00002.safetensors?download=true'
        download_file(url, file_path)

    with open(file_path, 'rb') as file:
        file_bytes = file.read()

    original_bytes = file_bytes[start_byte:end_byte]
    original_bytes_saved = bytearray(original_bytes)
    print(f"Length of original bytes: {len(original_bytes)/1024/1024/1024:.2f} GB")
    
    is_torch_numpy_byte = 0  # byte
    threads = 1

    # Map bytearray_dtype to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[bytearray_dtype]

    zipnn = ZipNN(input_format="byte", threads=threads, bytearray_dtype=bytearray_dtype, method=method, is_streaming=True)

    # Compress the byte string
    start_time = time.time()
    compressed_data = zipnn.compress(original_bytes)
    compression_time = time.time() - start_time
    print(f"Compressed data ratio: {len(compressed_data)/len(original_bytes):.4f}, Time: {compression_time:.2f} seconds")

    # Decompress the byte string
    start_time = time.time()
    decompressed_data = zipnn.decompress(compressed_data)
    decompression_time = time.time() - start_time
    print(f"Decompression time: {decompression_time:.2f} seconds")

    # Verify the result
    if (method != "TRUNCATE"):
        print(f"Are the original and decompressed byte strings the same? {original_bytes_saved == decompressed_data}")
    if (method == "TRUNCATE"):
        print(f"Are all the decompressed bytes zero, with the same size as the original data? {bytes(len(original_bytes_saved)) == decompressed_data}")

if __name__ == "__main__":

    # example: 
    # python3 simple_example_granite.py data/granite-3b-code-base.2.bin
    parser = argparse.ArgumentParser(description="Compress and decompress data using ZipNN\n example: ```\n python simple_example_granite.py data/granite-3b-code-base.2.bin --method HUFFMAN --bytearray_dtype bfloat16 --start_byte 100000000 --end_byte 1173741824\n```")
    parser.add_argument("file_path", type=str, help="Path to the input file")
    parser.add_argument("--method", type=str, default="HUFFMAN", choices=["AUTO", "HUFFMAN", "ZSTD", "TRUNCATE", "FSE"], help="Compression method")
    parser.add_argument("--start_byte", type=int, default=100000000, help="Start byte for compression")
    parser.add_argument("--end_byte", type=int, default=1173741824, help="End byte for compression")
    parser.add_argument("--bytearray_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Data type for bytearray")
    
    args = parser.parse_args()
    
    main(args.file_path, args.method, args.start_byte, args.end_byte, args.bytearray_dtype)
