import os
from zipnn import ZipNN
import sys
import filecmp
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
sys.path.append(module_dir)

import zipnn_compress
import zipnn_decompress

# bfloat16 compression and decompression
file_16 = "file16.bin"
file_16_copy=shutil.copy2(file_16, "file_16_copy.bin")

# Can change the streaming compression chunk_size 
zipnn_compress.compress_file(file_16,CHUNK_SIZE=2097152)
zipnn_decompress.decompress_file(file_16+".zpn")
print("Are the files equal? "+str(filecmp.cmp(file_16, file_16_copy, shallow=False)))


# float32 compression and decompression
file_32 = "file32.bin"
file_32_copy=shutil.copy2(file_32, "file_32_copy.bin")
 
zipnn_compress.compress_file(file_32,dtype='float32')
zipnn_decompress.decompress_file(file_32+".zpn")
print("Are the files equal? "+str(filecmp.cmp(file_32, file_32_copy, shallow=False)))
