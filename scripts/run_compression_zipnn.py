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

def compress_files_with_suffix(suffix,dtype="",streaming_chunk_size=1048576,path="."):
    import zipnn
    from zipnn_compress import compress_file

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

    parser = argparse.ArgumentParser(description="Enter a suffix to compress, (optional) dtype, (optional) streaming chunk size.")
    parser.add_argument('suffix', type=str, help='The file to compress')
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

