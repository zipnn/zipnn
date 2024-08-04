from zipnn import ZipNN

# instance class:
zpn = ZipNN(input_format='torch')

# Create a 1MB tensor with random numbers from a uniform distribution between -1 and 1 The dtype is bfloat
import torch
print ("Create a 1MB tensor with random numbers")
original_tensor = torch.rand(10124*1024, dtype=torch.bfloat16) * 2 - 1

# Compression:
print ("compressed_data")
compressed_data = zpn.compress(original_tensor)

# Decompression:
print ("decompressed_data")
decompressed_data = zpn.decompress(compressed_data)

# Check for correctness:
print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(original_tensor, decompressed_data))
