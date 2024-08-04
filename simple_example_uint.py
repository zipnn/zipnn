from zipnn import ZipNN
import torch
import time
import zstandard as zstd
import numpy as np

#is_torch_numpy_byte = 0 # torch 2 / numpy 1/ byte = 0
is_torch_numpy_byte = 1 # torch 2 / numpy 1/ byte = 0
#is_torch_numpy_byte = 2 # torch 2 / numpy 1/ byte = 0

#dtype = torch.uint32
threads = 8

max_val = 2**24 + 2**30
max_val = 2**16 + 2**20
#max_val = 2**8 + 2**15
#max_val = 56


num_elements = 1024*1024*1024 // 4
#num_elements = 1024 // 4 



array = np.random.randint(0, max_val, size=num_elements, dtype=np.uint32)
start_time = time.time()
array_bytes = array.tobytes()
array_bytes_copy = bytearray(array_bytes)
print ("numpy to bytes ", time.time()-start_time)

'''
zipnn_zstd = ZipNN(method='zstd', input_format="byte", bg=1, threads = threads)
start_time = time.time()
czz = zipnn_zstd.compress(array_bytes)
print ("zipnn zstd compress ", time.time()-start_time)
start_time = time.time()
dzz = zipnn_zstd.decompress(czz)
print ("zipnn zstd decompress ", time.time()-start_time)
print("Are the original and decompressed byte strings the same [BYTE]? ", array_bytes_copy == dzz)
'''

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
    compressed_data = zipnn.compress()
elif (is_torch_numpy_byte == 1): # Numpy   
    compressed_data = zipnn.compress(array)
elif (is_torch_numpy_byte == 0): # Byte 
    compressed_data = zipnn.compress()

print ("compressed_data remain ", len(compressed_data)/len(array_bytes), " time ", time.time() - start_time)

z = zstd.ZstdCompressor(level=3, threads=threads)
start_time = time.time()
c = z.compress(array_bytes)
print ("zstd remain ", len(c)/len(array_bytes), " time " , time.time() - start_time)
start_time = time.time()
zd = zstd.ZstdDecompressor()
d = zd.decompress(c)
print ("decompress zstd ", time.time() - start_time)

#Decompress the byte string back
start_time = time.time()
decompressed_data = zipnn.decompress(compressed_data)

print ("decompress zipnn data ", time.time() - start_time)

# Verify the result
if (is_torch_numpy_byte == 2): # Tensor
    print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(tensor, decompressed_data))
elif (is_torch_numpy_byte == 1): # Numpy   
    print("Are the original and decompressed byte strings the same [NUMPY]? ", np.array_equal(array, decompressed_data))
#print("Are the original and decompressed byte strings the same? ", array_bytes == decompressed_data)
elif (is_torch_numpy_byte == 0): # Byte 
    print("Are the original and decompressed byte strings the same [BYTE]? ", array_bytes_copy == decompressed_data)
