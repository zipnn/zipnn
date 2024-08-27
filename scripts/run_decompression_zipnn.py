import os
import subprocess
import sys
import argparse
from zipnn import ZipNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import zipnn

def check_and_install_zipnn():
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zipnn"])
        import zipnn

def decompress_zpn_files(dtype="",path=".",file=None,delete=False,force=False):
    import zipnn
    input_file=file
    if input_file:
        if (path!="."):
            input_file=path+"/"+input_file
        if os.path.exists(input_file):
            if delete:
                print(f"Deleting {input_file}...")
                os.remove(input_file)
            else:
                decompressed_path=input_file[:-4]
                if (not force and os.path.exists(decompressed_path)):
                    user_input = input(f"{decompressed_path} already exists; overwrite (y/n)? ").strip().lower()
                    if user_input!="y" and user_input!="yes":
                        print(f"Skipping {input_file}...")
                        return
                print(f"Decompressing {input_file}...")
                decompress_file(input_file,dtype=dtype)
        else:
            print(f"Error: The file {input_file} does not exist.")
        #exit(0)
        return
    
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
                decompressed_path=file_name[:-4]
                if (not force and os.path.exists(decompressed_path)):
                    user_input = input(f"{decompressed_path} already exists; overwrite (y/n)? ").strip().lower()
                    if user_input!="y" and user_input!="yes":
                        print(f"Skipping {file_name}...")
                        continue
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
<<<<<<< HEAD
        zpn = zipnn.ZipNN(is_streaming=True,streaming_chunk_kb=int(1024*1024),bytearray_dtype='float32')
=======
        zpn = zipnn.ZipNN(bytearray_dtype='float32')
>>>>>>> origin/main
    else:
        zpn = zipnn.ZipNN(is_streaming=True,streaming_chunk_kb=int(1024*1024))
    
    # Decompress
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        d_data=b''
<<<<<<< HEAD
        chunk= infile.read()
        d_data += zpn.decompress(chunk)
        outfile.write(d_data)
=======
        while header:= infile.read(20):
            mv_header=memoryview(header)
            mid_chunk_len=int.from_bytes(mv_header[16:20], byteorder="little")-20
            chunk_data = infile.read(mid_chunk_len)
            decompressed_chunk = zpn.decompress(header + chunk_data)
            if decompressed_chunk:
                d_data+=decompressed_chunk
                outfile.write(d_data)
                d_data=b''
>>>>>>> origin/main
        print(f"Decompressed {input_file} to {output_file}")


if __name__ == "__main__":
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(description="Compresses all .zpn files. (optional) dtype.")
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression.')
    parser.add_argument('--path', type=str, help='Path to folder of files to decompress.')
    parser.add_argument('--file', type=str, help='Name of file if only a single file needs decompression.')
    parser.add_argument('--delete', action='store_true', help='A flag that triggers deletion of a single compressed file instead of decompression')
    parser.add_argument('--force', action='store_true', help='A flag that forces overwriting when decompressing.')
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs['dtype'] = 32
    if args.path is not None:
        optional_kwargs['path'] = args.path
    if args.file is not None:
        optional_kwargs['file'] = args.file
    if args.delete:
        optional_kwargs['delete'] = args.delete
    if args.force:
        optional_kwargs['force'] = args.force
    
    decompress_zpn_files(**optional_kwargs)

