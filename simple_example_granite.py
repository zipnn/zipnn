from zipnn import ZipNN
import torch
import time
import zstandard as zstd
import numpy as np
import os
import requests


file_path = "data/granite-3b-code-base.2.bin"
url = 'https://huggingface.co/ibm-granite/granite-3b-code-base/resolve/main/model-00002-of-00002.safetensors?download=true'

#file_path = "data/granite-8b-instruct.2.bin"
#url = 'https://huggingface.co/ibm-granite/granite-8b-code-instruct/resolve/main/model-00002-of-00004.safetensors?download=true'

#file_path = "data/llama3.bin" 

#file_path = "data/mistral.bin" 
# Need authentication -> download from the browser or use Huggingface Token

#file_path = "data/Arcee-Nova.70B.2.bin" 
#url = 'https://huggingface.co/arcee-ai/Arcee-Nova/resolve/main/model-00002-of-00031.safetensors?download=true'

#file_path = "data/Arcee-Nova-Alpha-GGUF.fp16.2.bin" 
#url = 'https://huggingface.co/arcee-ai/Arcee-Nova-GGUF/resolve/main/Arcee-Nova-Alpha-GGUF.fp16-00002-of-00008.gguf?download=true'

#file_path = "data/jamba-v0.1.2.bin"
#url = 'https://huggingface.co/ai21labs/Jamba-v0.1/resolve/main/model-00002-of-00021.safetensors?download=true'

#file_path = "data/llama3-1.bf16.405B.bin"
# Need authentication -> download from the browser or use Huggingface Token
#url = 'https://huggingface.co/meta-llama/Meta-Llama-3.1-405B/resolve/main/model-00002-of-00191.safetensors?download=true'

#file_path = "data/llama3-1.8B.instruct.3.bin" 
# Need authentication -> download from the browser or use Huggingface Token
#url = 'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/resolve/main/model-00003-of-00004.safetensors?download=true'

#file_path = "data/ast-finetuned-audioset-10-10-0.4593.fp32.bin"
#url = "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/resolve/main/model.safetensors?download=true" 

#file_path = "data/xlm-roberta-large.fp32.bin"
#url = "https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/model.safetensors?download=true"

#file_path = "data/Llama-3-8B-Instruct.bin"
#url = "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/model-00003-of-00004.safetensors?download=true"


def download_file(url, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    
    response.raise_for_status()
   
    print("start downloading file")
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print ("end downloading file")        


if not os.path.exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    download_file(url, file_path)

with open(file_path, 'rb') as file:
    file_bytes = file.read()

original_bytes = file_bytes[100000000:1173741824]
#original_bytes = file_bytes[100000000:100524288]
#original_bytes = file_bytes[0:500000]
#original_bytes = file_bytes[:]
#original_bytes = file_bytes[0:20000]
original_bytes_saved = bytearray(original_bytes)
print ("len of original bytes ", len(original_bytes)/1024/1024/1024, " GB")
print ("len of original bytes ", len(original_bytes)/1024, " GB")

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
    zipnn = ZipNN(input_format="torch", threads = threads)
elif (is_torch_numpy_byte == 1): # Numpy   
    zipnn = ZipNN(input_format="numpy", threads = threads)
elif (is_torch_numpy_byte == 0): # Byte 
    zipnn = ZipNN(input_format="byte", threads = threads, bytearray_dtype = bytearray_dtype)
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

# Verify the result
if (is_torch_numpy_byte == 2): # Tensor
    print("Are the original and decompressed byte strings the same [TORCH]? ", torch.equal(tensor, decompressed_data))
elif (is_torch_numpy_byte == 1): # Numpy   
    print("Are the original and decompressed byte strings the same [NUMPY]? ", np.array_equal(tensor.numpy(), decompressed_data))
#print("Are the original and decompressed byte strings the same? ", tensor_bytes == decompressed_data)
elif (is_torch_numpy_byte == 0): # Byte 
    print("Are the original and decompressed byte strings the same [BYTE]? ", original_bytes_saved == decompressed_data)
