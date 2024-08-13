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

def decompress_zpn_files(dtype="",path="."):
    import zipnn
    from zipnn_decompress import decompress_file
    
    # List all files in the current directory
    for file_name in os.listdir(path):
        # Check if the file has a .zpn suffix
        if file_name.endswith('.zpn'):
            print(f"Decompressing {file_name}...")
            if (path!="."):
                file_name=path+"/"+file_name
            decompress_file(file_name,dtype=dtype)

if __name__ == "__main__":
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(description="Compresses all .zpn files. (optional) dtype.")
    parser.add_argument('--float32', action='store_true', help='A flag that triggers float32 compression')
    parser.add_argument('--path', type=str, help='Path to files to decompress')
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs['dtype'] = 32
    if args.path is not None:
        optional_kwargs['path'] = args.path
    
    decompress_zpn_files(**optional_kwargs)

