import os
import subprocess
import sys
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

def check_and_install_zipnn():
    try:
        import zipnn
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "zipnn",
                "--upgrade",
            ]
        )
        import zipnn


def parse_streaming_chunk_size(
    streaming_chunk_size,
):
    if str(streaming_chunk_size).isdigit():
        final = int(streaming_chunk_size)
    else:
        size_value = int(streaming_chunk_size[:-2])
        size_unit = streaming_chunk_size[-2].lower()

        if size_unit == "k":
            final = KB * size_value
        elif size_unit == "m":
            final = MB * size_value
        elif size_unit == "g":
            final = GB * size_value
        else:
            raise ValueError(f"Invalid size unit: {size_unit}. Use 'k', 'm', or 'g'.")

    return final


def compress_file(
    input_file,
    dtype="bfloat16",
    streaming_chunk_size=1048576,
    delete=False,
    force=False,
    hf_cache=False,
    method="HUFFMAN",
    verification=False,#
    test=False,#
    is_streaming=False
):
    import zipnn

    streaming_chunk_size = parse_streaming_chunk_size(streaming_chunk_size)
    full_path = input_file
    if not os.path.exists(full_path):
        print(f"{RED}File not found{RESET}")
        return

    compressed_path = full_path + ".znn"
    if not test and not force and os.path.exists(compressed_path):
        user_input = (
            input(f"{compressed_path} already exists; overwrite (y/n)? ").strip().lower()
        )
        if user_input not in ("yes", "y"):
            print(f"Skipping {full_path}...")
            return
    print(f"Compressing {full_path}...")
    #
    output_file = input_file + ".znn"
    zpn = zipnn.ZipNN(
            bytearray_dtype=dtype,
            is_streaming=is_streaming,
            streaming_chunk_kb=streaming_chunk_size,
            method=method
        )
    file_size_before = 0
    file_size_after = 0
    start_time = time.time()
    if not test:
        with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
            chunk = infile.read()
            file_size_before += len(chunk)
            compressed_chunk = zpn.compress(chunk)
            if compressed_chunk:
                file_size_after += len(compressed_chunk)
                outfile.write(compressed_chunk)
    else:
        test_buffer=bytearray()
        with open(input_file, "rb") as infile:
            chunk = infile.read()
            file_size_before += len(chunk)
            compressed_chunk = zpn.compress(chunk)
            if compressed_chunk:
                file_size_after += len(compressed_chunk)
                test_buffer+=compressed_chunk
    #
    if verification:
        if test:
            with open(input_file, "rb") as f:
                file_data2 = f.read()
            assert (zpn.decompress(test_buffer)==file_data2), "Decompressed file should be equal to original file."
        else:
            with open(input_file, "rb") as infile, open(output_file, "rb") as outfile:
                file_data1=infile.read()
                file_data2=outfile.read()
            decompressed_data=zpn.decompress(file_data2)
            assert (file_data1==decompressed_data), "Decompressed file should be equal to original file."
        print("Verification successful.")
    #
    end_time = time.time() - start_time
    print(f"Compressed {input_file} to {output_file}")
    print(
        f"{GREEN}Original size:  {file_size_before/GB:.02f}GB size after compression: {file_size_after/GB:.02f}GB, Remaining size is {file_size_after/file_size_before*100:.02f}% of original, time: {end_time:.02f}{RESET}"
    )

    if delete and not hf_cache:
        print(f"Deleting {full_path}...")
        os.remove(full_path)

    if hf_cache:
        # If the file is in the Hugging Face cache, fix the symlinks
        print(f"{YELLOW}Reorganizing Hugging Face cache...{RESET}")
        try:
            snapshot_path = os.path.dirname(input_file)
            blob_name = os.path.join(snapshot_path, os.readlink(input_file))
            os.rename(output_file, blob_name)
            os.symlink(blob_name, output_file)
            if os.path.exists(input_file):
                os.remove(input_file)
        except Exception as e:
            raise Exception(f"Error reorganizing Hugging Face cache: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter a file path to compress.")
    parser.add_argument(
        "input_file",
        type=str,
        help="Specify the path to the file to compress.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Specify the data type. Default is bfloat16.",
    )
    parser.add_argument(
        "--streaming_chunk_size",
        type=str,
        help="An optional streaming chunk size. The format is int (for size in Bytes) or int+KB/MB/GB. Default is 1MB",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of a single file instead of compression",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="A flag that forces overwriting when compressing.",
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="A flag that indicates if the file is in the Hugging Face cache.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["HUFFMAN", "ZSTD", "FSE", "AUTO"],
        default="HUFFMAN",
        help="Specify the method to use. Default is HUFFMAN.",
    )
    parser.add_argument(#
        "--verification",
        action="store_true",
        help="A flag that verifies that a compression can be decompressed correctly.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="A flag to not write the compression to a file.",
    )#
    parser.add_argument(
        "--is_streaming",
        action="store_true",
        help="A flag to compress using streaming.",
    )#
    args = parser.parse_args()
    optional_kwargs = {}
    if args.dtype:
        optional_kwargs["dtype"] = args.dtype
    if args.streaming_chunk_size is not None:
        optional_kwargs["streaming_chunk_size"] = args.streaming_chunk_size
    if args.delete:
        optional_kwargs["delete"] = args.delete
    if args.force:
        optional_kwargs["force"] = args.force
    if args.hf_cache:
        optional_kwargs["hf_cache"] = args.hf_cache
    if args.method:
        optional_kwargs["method"] = args.method
    if args.verification:#
        optional_kwargs["verification"] = args.verification
    if args.test:
        optional_kwargs["test"] = args.test#
    if args.is_streaming:
        optional_kwargs["is_streaming"] = args.is_streaming#

    check_and_install_zipnn()
    compress_file(args.input_file, **optional_kwargs)
