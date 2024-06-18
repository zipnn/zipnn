from zipnn import ZipNN
import torch
import time
import zstandard as zstd
import numpy as np

is_torch_numpy_byte = 0 # torch 2 / numpy 1/ byte = 0
is_torch_numpy_byte = 1 # torch 2 / numpy 1/ byte = 0
is_torch_numpy_byte = 2 # torch 2 / numpy 1/ byte = 0

dtype = torch.float32
#dtype = torch.bfloat16
#dtype = torch.float16
threads = 8

if (dtype == torch.float32):
    bytearray_dtype = "float32"
elif (dtype == torch.bfloat16):
    bytearray_dtype = "bfloat16"
elif (dtype == torch.float16):
    bytearray_dtype = "float16"

element_size = torch.tensor([], dtype=dtype).element_size()
num_elements = 1024*1024*1024 // element_size 
#num_elements = 1024 // element_size 

# Create a tensor of these many elements of type float32
# Initialize the tensor with random numbers from a uniform distribution between -1 and 1
tensor = torch.rand(num_elements, dtype=dtype) * 2 - 1

start_time = time.time()
if (dtype == torch.bfloat16):
    tensor_uint16 = tensor.view(torch.uint16)
    tensor_bytes = tensor_uint16.numpy().tobytes() 
    tensor_bytes_copy = bytearray(tensor_bytes)

else:
    tensor_bytes = tensor.numpy().tobytes()  
    tensor_bytes_copy = bytearray(tensor_bytes)
    tensor_numpy = tensor.numpy()
print ("transfer data to bytes is ", time.time() - start_time)

example_string = b"Example string for compression"

# Initializing the ZipNN class with the default configuration
# for Byte->Byte compression and Byte->Byte decompression

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
c = z.compress(tensor_bytes)
print ("zstd remain ", len(c)/len(tensor_bytes), " time " , time.time() - start_time)
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
    print("Are the original and decompressed byte strings the same [NUMPY]? ", np.array_equal(tensor.numpy(), decompressed_data))
#print("Are the original and decompressed byte strings the same? ", tensor_bytes == decompressed_data)
elif (is_torch_numpy_byte == 0): # Byte 
    print("Are the original and decompressed byte strings the same [BYTE]? ", tensor_bytes_copy == decompressed_data)
