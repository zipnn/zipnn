import os
from zipnn import ZipNN
import sys
import argparse

def decompress_file(input_file,dtype=""):
    # Verify the file suffix
    if not input_file.endswith('.zpn'):
        raise ValueError("Input file does not have the '.zpn' suffix")

    # Define the output file name by removing the '.zpn' suffix
    output_file = input_file[:-4]

    # Init ZipNN
    if dtype:
        zipnn = ZipNN(bytearray_dtype='float32')
    else:
        zipnn = ZipNN()
    
    # Decompress
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        d_data=b''
        while header:= infile.read(20):
            mv_header=memoryview(header)
            mid_chunk_len=int.from_bytes(mv_header[16:20], byteorder="little")-20
            chunk_data = infile.read(mid_chunk_len)
            decompressed_chunk = zipnn.decompress(header + chunk_data)
            if decompressed_chunk:
                d_data+=decompressed_chunk
                outfile.write(d_data)
                d_data=b''
        print(f"Decompressed {input_file} to {output_file}")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decompress.py <file.zpn>")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Enter a file to decompress, (optional) dtype.")
    parser.add_argument('input_file', type=str, help='The file to decompress')
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression')
    args = parser.parse_args()
    optional_args = []
    if args.float32:
        optional_args.append(32)
        
    input_file = sys.argv[1]
    decompress_file(args.input_file, *optional_args)

