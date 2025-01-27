import os
import sys
import argparse
import time
import multiprocessing
from util import check_and_install_zipnn, RESET, GREEN, GB, YELLOW

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def decompress_file(input_file, delete=False, force=False, hf_cache=False,threads=multiprocessing.cpu_count()):
    import zipnn

    if not input_file.endswith(".znn"):
        raise ValueError("Input file does not have the '.znn' suffix")

    if os.path.exists(input_file):
        decompressed_path = input_file[:-4]
        if not force and os.path.exists(decompressed_path):

            user_input = (
                input(f"{decompressed_path} already exists; overwrite (y/n)? ").strip().lower()
            )

            if user_input not in ("yes", "y"):
                print(f"Skipping {input_file}...")
                return
        print(f"Decompressing {input_file}...")

        output_file = input_file[:-4]
        zpn = zipnn.ZipNN(is_streaming=True,threads=threads)

        file_size_before = 0
        file_size_after = 0
        start_time = time.time()
        with open(input_file, "rb") as infile, open(output_file, "wb") as outfile:
            d_data = b""
            chunk = infile.read()
            file_size_before = len(chunk)
            d_data += zpn.decompress(chunk)
            file_size_after = len(d_data)
            outfile.write(d_data)
            print(f"Decompressed {input_file} to {output_file} using {threads} threads")
        end_time = time.time() - start_time

        print(
            f"{GREEN}Back to original size: {file_size_after/GB:.02f}GB size before decompression: {file_size_before/GB:.02f}GB, time: {end_time:.02f}{RESET}"
            )


        if delete and not hf_cache:
            print(f"Deleting {input_file}...")
            os.remove(input_file)

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

    else:
        print(f"Error: The file {input_file} does not exist.")


if __name__ == "__main__":
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(description="Enter a file path to decompress.")
    parser.add_argument("input_file", type=str, help="Specify the path to the file to decompress.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of a single compressed file instead of decompression",
    )
    parser.add_argument(
        "--force", action="store_true", help="A flag that forces overwriting when decompressing."
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="A flag that indicates if the file is in the Hugging Face cache.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The amount of threads to be used.",
    )
    args = parser.parse_args()
    optional_kwargs = {}
    if args.delete:
        optional_kwargs["delete"] = args.delete
    if args.force:
        optional_kwargs["force"] = args.force
    if args.hf_cache:
        optional_kwargs["hf_cache"] = args.hf_cache
    if args.threads:
        optional_kwargs["threads"] = args.threads#

    decompress_file(args.input_file, **optional_kwargs)
