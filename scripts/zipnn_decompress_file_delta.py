import os
import subprocess
import sys
import argparse
import multiprocessing

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
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zipnn"])
        import zipnn


def decompress_file(input_file, delta_file, delete=False, force=False, hf_cache=False,threads=multiprocessing.cpu_count()):
    import zipnn

    if not input_file.endswith(".znn"):
        raise ValueError("Input file does not have the '.znn' suffix")

    if os.path.exists(input_file) and os.path.exists(delta_file):
        if delete and not hf_cache:
            raise ValueError(f"{RED}Delete not supported yet for delta decompression.{RESET}")
            print(f"Deleting {input_file}...")
            os.remove(input_file)
        else:
            decompressed_path = input_file[:-4]
            if not force and os.path.exists(decompressed_path):

                user_input = input(f"{decompressed_path} already exists; overwrite (y/n)? ").strip().lower()

                if user_input not in ("yes", "y"):
                    print(f"Skipping {input_file}...")
                    return
            print(f"Decompressing {input_file}...")

            output_file = input_file.split("_delta_")[0] + ".bin"
            zpn = zipnn.ZipNN(is_streaming=True, delta_compressed_type="file",threads=threads)  ##

            #
            with open(input_file, "rb") as f:
                file_data = f.read()
            decompressed_data = zpn.decompress(file_data, delta_second_data=delta_file)
            with open(output_file, "wb") as f_out:
                f_out.write(decompressed_data)
            #
            print(f"Decompressed {input_file} to {output_file} using {threads} threads")
            file_size_before = len(file_data)
            file_size_after = len(decompressed_data)
            print(
                f"{GREEN}Original size:  {file_size_before/GB:.05f}GB size after decompression: {file_size_after/GB:.05f}GB, Remaining size is {file_size_after/file_size_before*100:.02f}% of original"
            )
            #
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
        print(f"Error: File doesn't exist.")


if __name__ == "__main__":
    check_and_install_zipnn()

    parser = argparse.ArgumentParser(description="Enter a file path to decompress.")
    parser.add_argument("input_file", type=str, help="Specify the path to the file to decompress.")
    parser.add_argument(
        "delta_file",
        type=str,
        help="Specify the path to the delta file.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of a single compressed file instead of decompression",
    )
    parser.add_argument("--force", action="store_true", help="A flag that forces overwriting when decompressing.")
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
        
    decompress_file(args.input_file, args.delta_file, **optional_kwargs)
