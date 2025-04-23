from zipnn import ZipNN
import torch
import os
import copy
import numpy as np
import sys
from safetensors.torch import save_file
from safetensors.torch import safe_open


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
        original_tensor_clone=original_tensor.clone()
        compressed_data = zpn_torch.compress(original_tensor)
        decompressed_data = zpn_torch.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(original_tensor_clone, decompressed_data))
        if not torch.equal(original_tensor_clone, decompressed_data):
            raise ValueError("Error - original file and decompressed file are NOT equal.")

        original_bytes=create_random_bytes(size)
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_bytes.compress(original_bytes)
        decompressed_data = zpn_bytes.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [BYTES]? ", copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.")
        

    # Test each size in megabytes
    for size in sizes_mb:
        print(f"\nTesting size: {size}MB")
        original_tensor = create_tensor(int(size * 1024))
        original_tensor_clone=original_tensor.clone()
        compressed_data = zpn_torch.compress(original_tensor)
        decompressed_data = zpn_torch.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(original_tensor_clone, decompressed_data))
        if not torch.equal(original_tensor_clone, decompressed_data):
            raise ValueError("Error - original file and decompressed file are NOT equal.")

        original_bytes=create_random_bytes(int(size*1024))
        copy_bytes=bytearray(original_bytes)
        compressed_data = zpn_bytes.compress(original_bytes)
        decompressed_data = zpn_bytes.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [BYTES]? ", copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.")
        

    # Test each size in chunk sizes
    for size in chunk_sizes:
        print(f"\nTesting chunk size: {size} Bytes")
        original_bytes = create_random_bytes(int(10*1024))
        copy_bytes=bytearray(original_bytes)
        zpn_streaming = ZipNN(is_streaming=True,streaming_chunk=size)
        compressed_data = zpn_streaming.compress(original_bytes)
        decompressed_data = zpn_streaming.decompress(compressed_data)
        print("Are the original and decompressed byte strings the same [STREAMING BYTES]? ",copy_bytes == decompressed_data)
        if not copy_bytes == decompressed_data:
            raise ValueError("Error - original file and decompressed file are NOT equal.")

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
        original_bytes = np.random.rand(size).astype(np.float32).tobytes()
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
        original_bytes = np.random.rand(size).astype(np.float32).tobytes()
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
    
    # Safetensors
    # Add the scripts directory to the path
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
    sys.path.append(script_dir)

    # Import the functions from the scripts
    from zipnn_compress_safetensors import compress_safetensors_file
    from zipnn_decompress_safetensors import decompress_safetensors_file

    # Create a tensor with half repetitive and half random data
    size = (100, 100)
    half_size = size[0] // 2
    repetitive_part = torch.full((half_size, size[1]), 42.0)  # Fill half with constant value
    random_part = torch.randn((half_size, size[1]))  # Random values for the other half
    tensor = torch.cat((repetitive_part, random_part), dim=0)

    # Test different floating point types
    for dtype in [torch.float16,torch.bfloat16,torch.float8_e4m3fn]:  # Removed unsupported types
        tensor = tensor.to(dtype)  # Convert tensor to the specified dtype
        
        # Save tensor to a temporary .safetensors file
        tensor_file = f"temp_tensor_{dtype}.safetensors"
        tensor_name="tensor"
        metadata = {
            f"{tensor_name}_dtype": str(dtype),  # Store dtype as a string
            f"{tensor_name}_shape": str(list(tensor.shape))  # Store shape as a string
        }
        save_file({tensor_name: tensor}, tensor_file, metadata=metadata)  # Save using safetensors
        
        # Define compressed and decompressed file paths
        compressed_file = f"temp_tensor_{dtype}.znn.safetensors"
        decompressed_file = f"temp_tensor_{dtype}.safetensors"
        
        try:
            # Compress the .safetensors file
            compress_safetensors_file(tensor_file, force=True)
            
            # Decompress the file
            decompress_safetensors_file(compressed_file, force=True)
            
            # Load the decompressed tensor
            with safe_open(decompressed_file, framework="pt", device="cpu") as f:
                decompressed_tensor = f.get_tensor("tensor")  # Get tensor by its key
            
            # Check if decompressed tensor is close to original
            if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                tensor_uint8 = tensor.view(torch.uint8)
                decompressed_uint8 = decompressed_tensor.view(torch.uint8)
                assert torch.equal(tensor_uint8, decompressed_uint8), f"Decompressed tensor does not match original for dtype {dtype}!"
            else:
                assert torch.allclose(tensor, decompressed_tensor), f"Decompressed tensor does not match original for dtype {dtype}!"

            print(f"Test passed for dtype {dtype}: zipnn_safetensors works correctly!")
        
        finally:
            # Clean up files
            for file in [tensor_file, compressed_file, decompressed_file]:
                if os.path.exists(file):
                    os.remove(file)



    
    if os.path.exists(file_path):
        os.remove(file_path)
