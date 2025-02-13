from zipnn import ZipNN

# instance class:
zpn = ZipNN(input_format='torch')

#Generate a 1MB tensor with the first half containing 0s and the second half containing 1s and the dtype is bfloat.
import torch
print ("Generate a 1MB tensor with the first half containing 0s and the second half containing 1s.")
tensor_size = 1024 * 1024  
half_size = tensor_size // 2

original_tensor = torch.cat([torch.zeros(half_size, dtype=torch.bfloat16), torch.ones(half_size, dtype=torch.bfloat16)])
original_tensor_clone=original_tensor.clone()
# Compression:
print (f"Compressing tensor of size {len(original_tensor)} bytes, using {zpn.threads} threads.")
compressed_data = zpn.compress(original_tensor)
remaining_size=len(compressed_data)/len(original_tensor)*100
print(f"Compressed the tensor using {zpn.threads} threads. Remaining size is {remaining_size:.2f}% of original.")

# Decompression:
print ("Decompressing the compressed tensor.")
decompressed_data = zpn.decompress(compressed_data)
# Check for correctness:
print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(original_tensor_clone, decompressed_data))
