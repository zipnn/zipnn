import os
from zipnn import ZipNN
import sys
import argparse

def parse_streaming_chunk_size(streaming_chunk_size):
    if str(streaming_chunk_size).isdigit():
        final = int(streaming_chunk_size)  # Convert the digit string to an integer
    else:
        # Extract the numeric part of the string
        size_value = int(streaming_chunk_size[:-2])
        size_unit = streaming_chunk_size[-2].lower()

        if size_unit == 'k':
            final = 1024 * size_value
        elif size_unit == 'm':
            final = 1024 * 1024 * size_value
        elif size_unit == 'g':
            final = 1024 * 1024 * 1024 * size_value
        else:
            raise ValueError(f"Invalid size unit: {size_unit}. Use 'k', 'm', or 'g'.")
        
    return final

def compress_file(input_file,dtype="",streaming_chunk_size=1048576):
    # Define the output file name by adding the '.zpn' suffix
    output_file = input_file + '.zpn'

    # Handle streaming chunk size
    streaming_chunk_size=parse_streaming_chunk_size(streaming_chunk_size)
    
    # Init ZipNN
    #streaming_chunk_size=1048576 #1MB
    if dtype:
        zipnn = ZipNN(bytearray_dtype='float32')
    else:
        zipnn = ZipNN()

    
    # Compress
    file_size_before=0
    file_size_after=0
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        while chunk := infile.read(streaming_chunk_size):
            file_size_before+=len(chunk)
            compressed_chunk = zipnn.compress(chunk)
            if compressed_chunk:
                file_size_after+=len(compressed_chunk)
                outfile.write(compressed_chunk)
        print(f"Compressed {input_file} to {output_file}")
        print ("Original size: "+str(file_size_before)+", size after compression: "+str(file_size_after)+", Remaining size is "+str(file_size_after/file_size_before*100)+"% of original")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress.py <file>")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Enter a file to compress, (optional) dtype, (optional) streaming chunk size.")
    parser.add_argument('input_file', type=str, help='The file to compress')
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression')
    parser.add_argument('--streaming_chunk_size', type=str, help='An optional streaming chunk size. The format is int (for size in Bytes) or int+KB/MB/GB. Default is 1MB')
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs['dtype'] = 32
    if args.streaming_chunk_size is not None:
        optional_kwargs['streaming_chunk_size'] = args.streaming_chunk_size
        
    compress_file(args.input_file, **optional_kwargs)

