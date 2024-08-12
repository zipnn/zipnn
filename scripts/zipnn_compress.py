import os
from zipnn import ZipNN
import sys
import argparse

def compress_file(input_file,dtype="",CHUNK_SIZE=1048576):
    # Define the output file name by adding the '.zpn' suffix
    output_file = input_file + '.zpn'

    # Init ZipNN
    #CHUNK_SIZE=1048576 #1MB
    if dtype:
        zipnn = ZipNN(bytearray_dtype='float32')
    else:
        zipnn = ZipNN()

    # Compress
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        while chunk := infile.read(CHUNK_SIZE):
            compressed_chunk = zipnn.compress(chunk)
            if compressed_chunk:
                outfile.write(compressed_chunk)
        print(f"Compressed {input_file} to {output_file}")    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress.py <file>")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Enter a file to compress, (optional) dtype, (optional) chunksize.")
    parser.add_argument('input_file', type=str, help='The file to compress')
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression')
    parser.add_argument('--chunk_size', type=float, help='An optional chunk size in MB. Default is 1MB')
    args = parser.parse_args()
    optional_args = []
    if args.float32:
        optional_args.append(32)
    if args.chunk_size is not None:
        optional_args.append(int(1048576*args.chunk_size))
    
    compress_file(args.input_file, *optional_args)

