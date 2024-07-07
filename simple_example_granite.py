from zipnn import ZipNN
import torch
import time
import zstandard as zstd
import numpy as np


file_path = "csrc/granite-3b-code-base.2.bin" 
with open(file_path, 'rb') as file:
    file_bytes = file.read()

original_bytes = file_bytes[100000000:1173741824]
original_bytes = file_bytes[100000000:200000000]
#original_bytes = file_bytes[:]
#original_bytes = file_bytes[100:200]
original_bytes_saved = bytearray(original_bytes)
print ("len of original bytes ", len(original_bytes)/1024/1024/1024, " GB")

is_torch_numpy_byte = 0 # torch 2 / numpy 1/ byte = 0
#is_torch_numpy_byte = 1 # torch 2 / numpy 1/ byte = 0
#is_torch_numpy_byte = 2 # torch 2 / numpy 1/ byte = 0

#dtype = torch.float32
dtype = torch.bfloat16
#dtype = torch.float16
threads = 1

if (dtype == torch.float32):
    bytearray_dtype = "float32"
elif (dtype == torch.bfloat16):
    bytearray_dtype = "bfloat16"
elif (dtype == torch.float16):
    bytearray_dtype = "float16"

element_size = torch.tensor([], dtype=dtype).element_size()
num_elements = 1024*1024*1024 // element_size 
#num_elements = 1024 // element_size 

tensor_bytes = original_bytes
    
if (is_torch_numpy_byte == 2): # Tensor
    zipnn = ZipNN(method='zstd', input_format="torch", threads = threads)
elif (is_torch_numpy_byte == 1): # Numpy   
    zipnn = ZipNN(method='zstd', input_format="numpy", threads = threads)
elif (is_torch_numpy_byte == 0): # Byte 
    zipnn = ZipNN(method='zstd', input_format="byte", threads = threads, bytearray_dtype = bytearray_dtype)
else: 
    raise ValueError("Unsupported input_format")


# Compress the byte string
start_time = time.time()
if (is_torch_numpy_byte == 2): # Tensor
    compressed_data = zipnn.compress(tensor)
elif (is_torch_numpy_byte == 1): # Numpy   
    compressed_data = zipnn.compress(tensor_numpy)
elif (is_torch_numpy_byte == 0): # Byte 
    compressed_data = zipnn.compress(tensor_bytes)

print ("compressed_data remain ", len(compressed_data)/len(tensor_bytes), " time ", time.time() - start_time)
z = zstd.ZstdCompressor(level=3, threads=threads)
start_time = time.time()
c = z.compress(original_bytes_saved)
print ("zstd remain ", len(c)/len(tensor_bytes), " time " , time.time() - start_time)
start_time = time.time()
zd = zstd.ZstdDecompressor()
d = zd.decompress(c)
print ("decompress zstd ", time.time() - start_time)


#Decompress the byte string back
start_time = time.time()
decompressed_data = zipnn.decompress(compressed_data)
print ("decompress zipnn data ", time.time() - start_time)
exit(0)

# Verify the result
if (is_torch_numpy_byte == 2): # Tensor
    print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(tensor, decompressed_data))
elif (is_torch_numpy_byte == 1): # Numpy   
    print("Are the original and decompressed byte strings the same [NUMPY]? ", np.array_equal(tensor.numpy(), decompressed_data))
#print("Are the original and decompressed byte strings the same? ", tensor_bytes == decompressed_data)
elif (is_torch_numpy_byte == 0): # Byte 
    print("Are the original and decompressed byte strings the same [BYTE]? ", original_bytes_saved == decompressed_data)