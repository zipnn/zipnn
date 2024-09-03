from zipnn import ZipNN
import torch
import os
import copy





def test_byte_torch_streaming():
    # Instance of the ZipNN class
    zpn_torch = ZipNN(input_format='torch')
    zpn_bytes = ZipNN()

    KB=1024
    MB=1024*KB
    # Define the tensor sizes in kilobytes and megabytes
    sizes_kb = [255, 256, 257, 511, 512, 513, 1024]  # sizes in KB
    sizes_mb = [0.99, 1, 1.01, 1.99, 2, 2.1]    # sizes in MB
    chunk_sizes=[int(0.5*MB),int(0.9*MB),int(1.1*MB-1),int(4.9*MB),int(5.1*MB-1),int(11*MB)]
    chunk_sizes=[int(2**19),int(2**20),int(2**21),int(2**23),int(2**24)]

    # Function to create a tensor of a specific size
    def create_tensor(size_in_kb):
        num_elements = size_in_kb * 1024  
        return torch.rand(num_elements, dtype=torch.bfloat16) * 2 - 1

    def create_random_bytes(size_in_kb):
        size_in_bytes = size_in_kb * 1024  
        return os.urandom(size_in_bytes)

    # Test each size in kilobytes
    for size in sizes_kb:
        print(f"\nTesting size: {size}KB")
        original_tensor = create_tensor(size)
        compressed_data = zpn_torch.compress(original_tensor)
        decompressed_data = zpn_torch.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(original_tensor, decompressed_data))

        original_bytes=create_random_bytes(size)
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_bytes.compress(original_bytes)
        decompressed_data = zpn_bytes.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [BYTES]? ", copy_bytes == decompressed_data)
        

    # Test each size in megabytes
    for size in sizes_mb:
        print(f"\nTesting size: {size}MB")
        original_tensor = create_tensor(int(size * 1024))  
        compressed_data = zpn_torch.compress(original_tensor)
        decompressed_data = zpn_torch.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(original_tensor, decompressed_data))

        original_bytes=create_random_bytes(int(size*1024))
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_bytes.compress(original_bytes)
        decompressed_data = zpn_bytes.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [BYTES]? ", copy_bytes == decompressed_data)

    # Test each size in chunk sizes
    for size in chunk_sizes:
        print(f"\nTesting chunk size: {size} Bytes")
        original_bytes = create_random_bytes(int(10*1024))
        copy_bytes=bytearray(original_bytes)
        zpn_streaming = ZipNN(is_streaming=True,streaming_chunk_kb=size)
        compressed_data = zpn_streaming.compress(original_bytes)
        decompressed_data = zpn_streaming.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [STREAMING BYTES]? ",copy_bytes == decompressed_data)
