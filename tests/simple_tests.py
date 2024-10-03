from zipnn import ZipNN
import torch
import os
import copy
import numpy as np





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

    # Delta(byte)
    zpn_delta = ZipNN(delta_compressed_type="byte")
    for size in [int(1024*10)]:
        a=create_random_bytes(int(size))
        b=create_random_bytes(int(size))
        c=create_random_bytes(int(size))
        original_bytes=a+b
        second_data=a+c
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_delta.compress(original_bytes,delta_second_data=second_data)
        decompressed_data = zpn_delta.decompress(compressed_data,delta_second_data=second_data)
        print("Are the original and decompressed byte strings the same [DELTA BUFFER]? ", copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.")

    # Streaming delta(byte)
    for size in [int(1024*8)]:
        zpn_streaming_delta = ZipNN(delta_compressed_type="byte",is_streaming=True)
        a=create_random_bytes(int(size))
        b=create_random_bytes(int(size))
        c=create_random_bytes(int(size))
        original_bytes=a+b
        second_data=a+c
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_streaming_delta.compress(original_bytes,delta_second_data=second_data)
        decompressed_data = zpn_streaming_delta.decompress(compressed_data,delta_second_data=second_data)
        print("Are the original and decompressed byte strings the same [STREAMING DELTA BUFFER]? ", copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.")

    # Delta from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'temp_file.bin')
    for size in [int(1024*8)]:
        zpn_streaming_delta = ZipNN(delta_compressed_type="file")
        a=create_random_bytes(int(size))
        b=create_random_bytes(int(size))
        c=create_random_bytes(int(size))
        original_bytes=a+b
        second_data=a+c
        # Write second_data to file
        with open(file_path, 'wb') as f:
            f.write(second_data)
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_streaming_delta.compress(original_bytes,delta_second_data=file_path)
        decompressed_data = zpn_streaming_delta.decompress(compressed_data,delta_second_data=file_path)
        print("Are the original and decompressed byte strings the same [DELTA FILE]? ", copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.")

    # Streaming delta from file
    for size in [int(1024*8)]:
        zpn_streaming_delta = ZipNN(delta_compressed_type="file",is_streaming=True)
        a=create_random_bytes(int(size))
        b=create_random_bytes(int(size))
        c=create_random_bytes(int(size))
        original_bytes=a+b
        second_data=a+c
        # Write second_data to file
        with open(file_path, 'wb') as f:
            f.write(second_data)
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_streaming_delta.compress(original_bytes,delta_second_data=file_path)
        decompressed_data = zpn_streaming_delta.decompress(compressed_data,delta_second_data=file_path)
        print("Are the original and decompressed byte strings the same [DELTA FILE STREAMING]? ", copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.") 

    # Float32
    for size in [int(1024*8)]:  # Size in number of float32 elements
        zpn_streaming_delta = ZipNN(bytearray_dtype="float32")
        original_bytes = np.random.rand(size).astype(np.float32)
        copy_bytes = bytearray(original_bytes)
        compressed_data = zpn_streaming_delta.compress(original_bytes)
        decompressed_data = zpn_streaming_delta.decompress(compressed_data)
        decompressed_array = np.frombuffer(decompressed_data, dtype=np.float32)
        print("Are the original and decompressed float32 arrays the same [FLOAT32]? ", np.array_equal(np.frombuffer(copy_bytes, dtype=np.float32), decompressed_array))
        
        if not np.array_equal(np.frombuffer(copy_bytes, dtype=np.float32), decompressed_array):
            raise ValueError("Error - original and decompressed float32 arrays are NOT equal.")

    # Streaming float32
    for size in [int(1024*8)]:  # Size in number of float32 elements
        zpn_streaming_delta = ZipNN(bytearray_dtype="float32",is_streaming=True)
        original_bytes = np.random.rand(size).astype(np.float32)
        copy_bytes = bytearray(original_bytes)
        compressed_data = zpn_streaming_delta.compress(original_bytes)
        decompressed_data = zpn_streaming_delta.decompress(compressed_data)
        decompressed_array = np.frombuffer(decompressed_data, dtype=np.float32)
        print("Are the original and decompressed float32 arrays the same [STREAMING FLOAT32]? ", np.array_equal(np.frombuffer(copy_bytes, dtype=np.float32), decompressed_array))
        
        if not np.array_equal(np.frombuffer(copy_bytes, dtype=np.float32), decompressed_array):
            raise ValueError("Error - original and decompressed float32 arrays are NOT equal.")

    # Streaming delta float32
    for size in [int(1024*8)]:  # Size in number of float32 elements
        zpn_streaming_delta = ZipNN(bytearray_dtype="float32", is_streaming=True,delta_compressed_type="byte")
        
        # Create random float32 arrays
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        c = np.random.rand(size).astype(np.float32)
        
        # Convert arrays to byte format for compression
        original_bytes = np.concatenate([a, b]).tobytes()
        second_data = np.concatenate([a, c]).tobytes()
        copy_bytes = bytearray(original_bytes)
        
        # Compress and decompress
        compressed_data = zpn_streaming_delta.compress(original_bytes, delta_second_data=second_data)
        decompressed_data = zpn_streaming_delta.decompress(compressed_data, delta_second_data=second_data)
        
        # Convert decompressed data back to float32
        decompressed_array = np.frombuffer(decompressed_data, dtype=np.float32)
        
        print("Are the original and decompressed float32 arrays the same [STREAMING DELTA FLOAT32]? ", np.array_equal(np.frombuffer(copy_bytes, dtype=np.float32), decompressed_array))
        
        if not np.array_equal(np.frombuffer(copy_bytes, dtype=np.float32), decompressed_array):
            raise ValueError("Error - original and decompressed float32 arrays are NOT equal.")





    if os.path.exists(file_path):
        os.remove(file_path)
