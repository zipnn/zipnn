import os
import sys
import shutil
import filecmp
import subprocess


# Paths and additional imports
current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
sys.path.append(module_dir)
os.makedirs(data_dir, exist_ok=True)
import zipnn_compress
import zipnn_decompress

# Downloading files, one for bfloat16 and one for float32.
url_16 = "https://huggingface.co/ibm-granite/granite-20b-functioncalling/resolve/main/model-00009-of-00009.safetensors"
url_32 = "https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1/resolve/main/model.safetensors"

file_16 = os.path.join(data_dir, "model-00009-of-00009.safetensors")
file_32 = os.path.join(data_dir, "model.safetensors")

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        subprocess.run(['wget', '-O', dest_path, url], check=True)
        print(f"Downloaded {dest_path}")
    else:
        print(f"File {dest_path} already exists, skipping download.")

download_file(url_16, file_16)
download_file(url_32, file_32)

# Create copies of the files for comparison later
file_16_copy = shutil.copy2(file_16, os.path.join(data_dir, "model-00009-of-00009_copy.safetensors"))
file_32_copy = shutil.copy2(file_32, os.path.join(data_dir, "model_copy.safetensors"))

# bfloat16 compression and decompression
print("\n bfloat16: Compression starts.")
zipnn_compress.compress_file(file_16, streaming_chunk_size=2097152)
print("Compression ended, decompression starts.")
zipnn_decompress.decompress_file(file_16 + ".znn")
print("Decompression ended.")
print("Are the files equal? " + str(filecmp.cmp(file_16, file_16_copy, shallow=False)))

# float32 compression and decompression
print("\n float32: Compression starts.")
zipnn_compress.compress_file(file_32, dtype='float32')
print("Compression ended, decompression starts.")
zipnn_decompress.decompress_file(file_32 + ".znn")
print("Decompression ended.")
print("Are the files equal? " + str(filecmp.cmp(file_32, file_32_copy, shallow=False)))

