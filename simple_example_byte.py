from zipnn import ZipNN

# instance class:
zpn = ZipNN()

# Generate a 1MB byte string with the first half containing 0s and the second half containing 1s.
print("Generate a 1MB byte string with the first half containing 0s and the second half containing 1s.")
tensor_size = 1024 * 1024  
half_size = tensor_size // 2

# Create the byte string
original_data = bytes([0] * half_size + [1] * half_size)
original_data_copy = bytes([0] * half_size + [1] * half_size)

# Compression:
print(f"Compressing byte data of size {len(original_data)} bytes, using {zpn.threads} threads.")
compressed_data = zpn.compress(original_data)
remaining_size = len(compressed_data) / len(original_data) * 100
print(f"Compressed the byte data using {zpn.threads} threads. Remaining size is {remaining_size:.2f}% of original.")

# Decompression:
print("Decompressing the compressed byte data.")
decompressed_data = zpn.decompress(compressed_data)
# Check for correctness:
print("Are the original and decompressed byte strings the same? [BYTES]", original_data_copy == decompressed_data)
