import os
import subprocess
import sys
import argparse
from zipnn import ZipNN

def check_and_install_zipnn():
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zipnn"])
        import zipnn

def decompress_zpn_files(dtype="",path=".",input_file=None,delete=False):
    import zipnn

    if input_file:
        if (path!="."):
            input_file=path+"/"+input_file
        if os.path.exists(input_file):
            if delete:
                print(f"Deleting {input_file}...")
                os.remove(input_file)
            else:
                print(f"Decompressing {input_file}...")
                decompress_file(input_file,dtype=dtype)
        else:
            print(f"Error: The file {input_file} does not exist.")
        exit(0)
    
    # List all files in the current directory
    for file_name in os.listdir(path):
        # Check if the file has a .zpn suffix
        if file_name.endswith('.zpn'):
            if (path!="."):
                file_name=path+"/"+file_name
            if delete:
                print(f"Deleting {file_name}...")
                os.remove(file_name)
            else:
                print(f"Decompressing {file_name}...")
                decompress_file(file_name,dtype=dtype)

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
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(description="Compresses all .zpn files. (optional) dtype.")
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression.')
    parser.add_argument('--path', type=str, help='Path to folder of files to decompress.')
    parser.add_argument('--input_file', type=str, help='Name of file if only a single file needs decompression.')
    parser.add_argument('--delete', action='store_true', help='A flag that triggers deletion of a single compressed file instead of decompression')
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs['dtype'] = 32
    if args.path is not None:
        optional_kwargs['path'] = args.path
    if args.input_file is not None:
        optional_kwargs['input_file'] = args.input_file
    if args.delete:
        optional_kwargs['delete'] = args.delete
    
    decompress_zpn_files(**optional_kwargs)

