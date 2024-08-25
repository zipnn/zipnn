import os
import subprocess
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import zipnn

KB = 1024
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
            final = KB * size_value
        elif size_unit == 'm':
            final = MB * size_value
        elif size_unit == 'g':
            final = GB * size_value
        else:
            raise ValueError(f"Invalid size unit: {size_unit}. Use 'k', 'm', or 'g'.")
        
    return final

def compress_file(input_file,dtype="",streaming_chunk_size=1048576):
    # Define the output file name by adding the '.zpn' suffix
    output_file = input_file + '.zpn'
    
    # Init ZipNN
    #streaming_chunk_size=1048576 #1MB
    if dtype:
        zpn = zipnn.ZipNN(bytearray_dtype='float32',is_streaming=True,streaming_chunk_kb=int(1024*1024))
    else:
        zpn = zipnn.ZipNN(is_streaming=True,streaming_chunk_kb=int(1024*1024))

    # Compress
    file_size_before=0
    file_size_after=0
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        chunk = infile.read()
        file_size_before+=len(chunk)
        compressed_chunk = zpn.compress(chunk)
        if compressed_chunk:
            file_size_after+=len(compressed_chunk)
            outfile.write(compressed_chunk)
    print(f"Compressed {input_file} to {output_file}")
    print (f'Original size:  {file_size_before/GB:.02f}GB size after compression: {file_size_after/GB:.02f}GB, Remaining size is {file_size_after/file_size_before*100:.02f}% of original')


def compress_files_with_suffix(suffix,dtype="",streaming_chunk_size=1048576,path=".",delete=False,r=False,force=False):

    # Handle streaming chunk size`
    streaming_chunk_size=parse_streaming_chunk_size(streaming_chunk_size)
    directories_to_search = os.walk(path) if r else [(path, [], os.listdir(path))]
    files_found=False
    for root, _, files in directories_to_search:
        for file_name in files:
            if file_name.endswith(suffix):
                files_found = True
                full_path = os.path.join(root, file_name)
                
                if delete:
                    print(f"Deleting {full_path}...")
                    os.remove(full_path)
                else:
                    compressed_path=full_path+".zpn"
                    if(not force and os.path.exists(compressed_path)):
                        user_input = input(f"{compressed_path} already exists; overwrite (y/n)? ").strip().lower()
                        if user_input!="y" and user_input!="yes":
                            print(f"Skipping {full_path}...")
                            continue
                    print(f"Compressing {full_path}...")
                    compress_file(full_path, dtype=dtype, streaming_chunk_size=streaming_chunk_size)

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
    parser.add_argument('--delete', action='store_true', help='A flag that triggers deletion of a single file instead of compression')
    parser.add_argument('-r', action='store_true', help='A flag that triggers recursive search on all subdirectories')
    parser.add_argument('--recursive', action='store_true', help='A flag that triggers recursive search on all subdirectories')
    parser.add_argument('--force', action='store_true', help='A flag that forces overwriting when compressing.')
    args = parser.parse_args()
    optional_kwargs = {}
    if args.float32:
        optional_kwargs['dtype'] = 32
    if args.streaming_chunk_size is not None:
        optional_kwargs['streaming_chunk_size'] = args.streaming_chunk_size
    if args.path is not None:
        optional_kwargs['path'] = args.path
    if args.delete:
        optional_kwargs['delete'] = args.delete
    if args.r or args.recursive:
        optional_kwargs['r'] = args.r
    if args.force:
        optional_kwargs['force'] = args.force
    
    check_and_install_zipnn()
    compress_files_with_suffix(args.suffix, **optional_kwargs)

