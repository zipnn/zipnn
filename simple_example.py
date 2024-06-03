from zipnn import ZipNN

example_string = b"Example string for compression"

# Initializing the ZipNN class with the default configuration
# for Byte->Byte compression and Byte->Byte decompression
zipnn = ZipNN(method='zstd')
    
# Compress the byte string
compressed_data = zipnn.compress(example_string)
    
# Decompress the byte string back
decompressed_data = zipnn.decompress(compressed_data)

# Verify the result
print("Are the original and decompressed byte strings the same? ", example_string == decompressed_data)
