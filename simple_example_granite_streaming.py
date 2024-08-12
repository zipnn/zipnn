from zipnn import ZipNN
import torch
import time
import zstandard as zstd
import numpy as np
import os
import requests

start=time.time()
###################################
####    Downloading Granite    ####
###################################


file_path = "data/granite-3b-code-base.2.bin"
url = 'https://huggingface.co/ibm-granite/granite-3b-code-base/resolve/main/model-00002-of-00002.safetensors?download=true'

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


##########################
####    Init ZipNN    ####
##########################


bytearray_dtype = "bfloat16"
threads = 1
zipnn = ZipNN()
input_path = file_path
output_path = "data/granite-3b-code-base.2.bin.zpn"
output_decomp_path="data/streamed_granite-3b-code-base.2.bin"
CHUNK_SIZE=1048576 #1MB


#####################################
####    Streaming Compression    ####
#####################################

start_time = time.time()
with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
    while chunk := infile.read(CHUNK_SIZE):
        compressed_chunk = zipnn.compress(chunk)
        if compressed_chunk:
            outfile.write(compressed_chunk)
print ("compress zipnn data ", time.time() - start_time)


#######################################
####    Streaming Decompression    ####
#######################################

start_time = time.time()
with open(output_path, 'rb') as infile, open(output_decomp_path, 'wb') as outfile:
    d_data=b''
    while header:= infile.read(20):
        mv_header=memoryview(header)
        mid_chunk_len=int.from_bytes(mv_header[16:20], byteorder="little")-20
        ##mid_chunk_len=int.from_bytes(header[16:20], byteorder="little")-20
        #chunk=header+infile.read(mid_chunk_len)
        chunk_data = infile.read(mid_chunk_len)
        #decompressed_chunk = zipnn.decompress(chunk)
        decompressed_chunk = zipnn.decompress(header + chunk_data)
        if decompressed_chunk:
            d_data+=decompressed_chunk
            outfile.write(d_data)
            d_data=b''
print ("decompress zipnn data ", time.time() - start_time)

##########################
####    Comparison    ####
##########################


with open(input_path, 'rb') as file1, open(output_decomp_path, 'rb') as file2:
    while True:
        chunk1 = file1.read(CHUNK_SIZE)
        chunk2 = file2.read(CHUNK_SIZE)
        if chunk1 != chunk2:
            print(input_path+","+output_decomp_path+" are not equal!")
            break

        if not chunk1:
            print(input_path+", "+output_decomp_path+" are equal!")
            break


finish=time.time()
print("Overall time: " + str(finish - start))

