import os
import subprocess
import sys
import argparse

def check_and_install_zipnn():
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zipnn"])
        import zipnn

def compress_files_with_suffix(suffix,dtype="",CHUNK_SIZE=1048576):
    import zipnn
    from zipnn_compress import compress_file

    # List all files in the current directory
    files_found = False
    for file_name in os.listdir('.'):
        # Check if the file has the specified suffix
        if file_name.endswith(suffix):
            files_found = True
            print(f"Compressing {file_name}...")
            compress_file(file_name,dtype=dtype,CHUNK_SIZE=CHUNK_SIZE)

    if not files_found:
        print(f"No files with the suffix '{suffix}' found.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_files.py <suffix>")
        print("Example: python compress_files.py 'safetensors'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Enter a suffix to compress, (optional) dtype, (optional) chunksize.")
    parser.add_argument('suffix', type=str, help='The file to compress')
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression')
    parser.add_argument('--chunk_size', type=float, help='An optional chunk size in MB. Default is 1MB')
    args = parser.parse_args()
    optional_args = []
    if args.float32:
        optional_args.append(32)
    if args.chunk_size is not None:
        optional_args.append(int(1048576*args.chunk_size))
    
    check_and_install_zipnn()
    compress_files_with_suffix(args.suffix, *optional_args)

