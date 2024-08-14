import os
import subprocess
import sys
import argparse


MB = 1024*1024
GB = 1024*1024*1024

def check_and_install_zipnn():
    global zipnn
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zipnn", "--upgrade"])
        import zipnn

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
    
    # Init ZipNN
    #streaming_chunk_size=1048576 #1MB
    if dtype:
        zpn = zipnn.ZipNN(bytearray_dtype='float32')
    else:
        zpn = zipnn.ZipNN()

    
    # Compress
    file_size_before=0
    file_size_after=0
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        while chunk := infile.read(streaming_chunk_size):
            file_size_before+=len(chunk)
            compressed_chunk = zpn.compress(chunk)
            if compressed_chunk:
                file_size_after+=len(compressed_chunk)
                outfile.write(compressed_chunk)
        print(f"Compressed {input_file} to {output_file}")
        print (f'Original size:  {file_size_before/GB:.02f}GB size after compression: {file_size_after/GB:.02f}GB, Remaining size is {file_size_after/file_size_before*100:.02f}% of original')


def compress_files_with_suffix(suffix,dtype="",streaming_chunk_size=1048576,path="."):

    # Handle streaming chunk size
    streaming_chunk_size=parse_streaming_chunk_size(streaming_chunk_size)

    # List all files in the current directory
    files_found = False
    for file_name in os.listdir(path):
        # Check if the file has the specified suffix
        if file_name.endswith(suffix):
            files_found = True
            print(f"Compressing {file_name}...")
            if (path!="."):
                file_name=path+"/"+file_name
            compress_file(file_name,dtype=dtype,streaming_chunk_size=streaming_chunk_size)

    if not files_found:
        print(f"No files with the suffix '{suffix}' found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_files.py <suffix>")
        print("Example: python compress_files.py 'safetensors'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Enter a suffix to compress, (optional) dtype, (optional) streaming chunk size, (optional) path to files.")
    parser.add_argument('suffix', type=str, help='Specify the file suffix to compress all files with that suffix. If a single file name is provided, only that file will be compressed.')
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression')
    parser.add_argument('--streaming_chunk_size', type=str, help='An optional streaming chunk size. The format is int (for size in Bytes) or int+KB/MB/GB. Default is 1MB')
    parser.add_argument('--path', type=str, help='Path to files to compress')
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs['dtype'] = 32
    if args.streaming_chunk_size is not None:
        optional_kwargs['streaming_chunk_size'] = args.streaming_chunk_size
    if args.path is not None:
        optional_kwargs['path'] = args.path
    
    check_and_install_zipnn()
    compress_files_with_suffix(args.suffix, **optional_kwargs)

