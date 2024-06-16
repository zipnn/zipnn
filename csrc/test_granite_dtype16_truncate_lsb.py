import numpy as np
import split_dtype  # This is your C extension module
import time
from zipnn import ZipNN
import torch

file_path = "granite-3b-code-base.2.bin" 
with open(file_path, 'rb') as file:
    file_bytes = file.read()

zstd = ZipNN(bg_partitions = 1, zstd_threads=1)

original_bytes = bytearray(file_bytes[100000000:1173741824])
#original_bytes = bytearray(file_bytes[100:200])
start_time = time.time()
original_bytes[0::2] = bytearray([0] * (len(original_bytes) // 2))
print ("zero msb ", time.time() - start_time)
original_bytes_saved = bytearray(original_bytes)
print ("len of original bytes ", len(original_bytes)/1024/1024/1024, " GB")

# Function to print the first 8 bytes in bit representation
def print_bit_representation(data, label):
    print(label + ": " + ' '.join(f'{b:08b}' for b in data[:8]))

# Print bit representation of the original data
print_bit_representation(original_bytes, "Original Data")

start_time = time.time()
orig = zstd.compress(original_bytes)
print (f"% of orig {len(orig)/len(original_bytes)}, {time.time() - start_time}")

start_time = time.time()
revert = zstd.decompress(orig)
print (f"% of revert {revert == original_bytes}, {time.time() - start_time}")
zstd = ZipNN(bg_partitions = 1, zstd_threads=1)

# Measure and perform the split
start_time = time.time()
# bit_mode = 1 [bit_ordering],
# byte_mode = 1 [Truncate LSByte]
# is_review = 0 [no review]
# thread = 1 [one thread]
buf1, buf2 = split_dtype.split_dtype16(original_bytes, 0, 8, 0 ,1) # bit_mode = 1 [bit_ordering], 
split_time = time.time() - start_time

start_time = time.time()
comp1 = zstd.compress(buf1)
print (f"% of comp1 {len(comp1)/len(original_bytes)*2}, {time.time() - start_time}")
if(len(comp1) > (len(original_bytes)/2)):
   len_comp1 = len(original_bytes)/2
else:
   len_comp1 = len(comp1)


start_time = time.time()
revert = zstd.decompress(comp1)
print (f"% of revert comp1 {time.time() - start_time}")
zstd = ZipNN(bg_partitions = 1, zstd_threads=1)



print(f"Split Time: {split_time:.3f} seconds")

# Print bit representations of the splits
print_bit_representation(buf1, "Split Buf1")

# Measure and perform the combine
# bit_mode = 0 [bit_ordering],
# byte_mode = 1 [Truncate LSByte]
# thread = 1 [one thread]
start_time = time.time()
empty_bytearray = bytearray()
combined_bytes = split_dtype.combine_dtype16(buf1, empty_bytearray, 0, 8, 1)
combine_time = time.time() - start_time
print_bit_representation(original_bytes_saved, "Original Data")
print_bit_representation(combined_bytes, "Combine")
print(f"Combine Time: {combine_time:.3f} seconds")

if (original_bytes_saved == combined_bytes):
    print("Test Passed: Original and reverted data are identical.")
else:
    print (original_bytes_saved[3])
    print (combined_bytes[3])
    print("Test Failed: Data mismatch.")

