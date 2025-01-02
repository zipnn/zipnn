import os
import subprocess
import sys
import argparse
from pathlib import Path
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from zipnn_compress_file import compress_file
import multiprocessing
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), ".."
        )
    )
)


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
        size_value = int(
            streaming_chunk_size[:-2]
        )
        size_unit = streaming_chunk_size[
            -2
        ].lower()

        if size_unit == "k":
            final = KB * size_value
        elif size_unit == "m":
            final = MB * size_value
        elif size_unit == "g":
            final = GB * size_value
        else:
            raise ValueError(
                f"Invalid size unit: {size_unit}. Use 'k', 'm', or 'g'."
            )

    return final

def replace_in_file(file_path, old: str, new: str) -> None:
    """Given a file_path, replace all occurrences of `old` with `new` inpalce."""

    with open(file_path, 'r') as file:
        file_data = file.read()

    file_data = file_data.replace(old, new)

    with open(file_path, 'w') as file:
        file.write(file_data)

def compress_files_with_suffix(
    suffix,
    dtype="bfloat16",
    streaming_chunk_size=1048576,
    path=".",
    delete=False,
    r=False,
    force=False,
    max_processes=1,
    hf_cache=False,
    model="",
    branch="main",
    method="HUFFMAN",
    verification=False,#
    test=False,#
    is_streaming=False,
    threads=multiprocessing.cpu_count()
):
    import zipnn

    overwrite_first=True
    file_list = []
    streaming_chunk_size = (
        parse_streaming_chunk_size(
            streaming_chunk_size
        )
    )
    if model:
        if not hf_cache:
            raise ValueError(
                "Must specify --hf_cache when using --model"
            )
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            raise ImportError(
                "huggingface_hub not found. Please pip install huggingface_hub."
            )  
        cache = scan_cache_dir()
        repo = next((repo for repo in cache.repos if repo.repo_id == model), None)

        if repo is not None:
            print(f"Found repo {model} in cache")
            
            # Get the latest revision path
            hash = ''
            try:
                with open(os.path.join(repo.repo_path, 'refs', branch), "r") as ref:
                    hash = ref.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"Branch {branch} not found in repo {model}")
            
            path = os.path.join(repo.repo_path, 'snapshots', hash)

    directories_to_search = (
        os.walk(path)
        if r
        else [(path, [], os.listdir(path))]
    )
    files_found = False
    for root, _, files in directories_to_search:
        for file_name in files:
            if file_name.endswith(suffix):
                compressed_path = (
                    file_name + ".znn"
                )
                if not test and not force and os.path.exists(
                    os.path.join(root, compressed_path)
                ):
                    #
                    if overwrite_first:
                        overwrite_first=False
                        user_input = (
                            input(
                                f"Compressed files already exists; Would you like to overwrite them all (y/n)? "
                            )
                            .strip()
                            .lower()
                        )
                        if user_input not in (
                            "y",
                            "yes",
                        ):
                            print(
                                f"No forced overwriting."
                            )
                        else:
                            print(
                                f"Overwriting all compressed files."
                            )
                            force=True
                    #
                    if not force and not test:
                        user_input = (
                            input(
                                f"{compressed_path} already exists; overwrite (y/n)? "
                            )
                            .strip()
                            .lower()
                        )
                        if user_input not in (
                            "y",
                            "yes",
                        ):
                            print(
                                f"Skipping {file_name}..."
                            )
                            continue
                files_found = True
                full_path = os.path.join(
                    root, file_name
                )
                file_list.append(full_path)

    if file_list and hf_cache:
        try:
            from transformers.utils import (
                SAFE_WEIGHTS_INDEX_NAME,
                WEIGHTS_INDEX_NAME
            )
        except ImportError:
            raise ImportError(
                "Transformers not found. Please pip install transformers."
            )
        
        if os.path.exists(os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)):
            print(f"{YELLOW}Fixing Hugging Face model json...{RESET}")
            blob_name = os.path.join(path, os.readlink(os.path.join(path, SAFE_WEIGHTS_INDEX_NAME)))
            replace_in_file(
                    file_path=blob_name,
                    old=f"{suffix}",
                    new=f"{suffix}.znn"
                )
        elif os.path.exists(os.path.join(path, WEIGHTS_INDEX_NAME)):
            print(f"{YELLOW}Fixing Hugging Face model json...{RESET}")
            blob_name = os.path.join(path, os.readlink(os.path.join(path, WEIGHTS_INDEX_NAME)))
            replace_in_file(
                    file_path=blob_name,
                    old=f"{suffix}",
                    new=f"{suffix}.znn"
                )

    with ProcessPoolExecutor(
        max_workers=max_processes
    ) as executor:
        future_to_file = {
            executor.submit(
                compress_file,
                file,
                dtype,
                streaming_chunk_size,
                delete,
                True,
                hf_cache,
                method,
                verification,
                test,
                is_streaming,
                threads
            ): file
            for file in file_list[:max_processes]
        }
        file_list = file_list[max_processes:]
        while future_to_file:
            for future in as_completed(
                future_to_file
            ):
                file = future_to_file.pop(future)

                try:
                    future.result()
                except Exception as exc:
                    print(
                        f"{RED}File {file} generated an exception: {exc}{RESET}"
                    )

                if file_list:
                    next_file = file_list.pop(0)
                    future_to_file[
                        executor.submit(
                            compress_file,
                            next_file,
                            dtype,
                            streaming_chunk_size,
                            delete,
                            True,
                            hf_cache,
                            method,
                            verification,
                            test,
                            is_streaming,
                            threads
                        )
                    ] = next_file

    if not files_found:
        print(
            f"{RED}No files with the suffix '{suffix}' found.{RESET}"
        )

    print(f"{GREEN}All files compressed{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enter a suffix to compress, (optional) dtype, (optional) streaming chunk size, (optional) path to files."
    )
    parser.add_argument(
        "suffix",
        type=str,
        help="Specify the file suffix to compress all files with that suffix. If a single file name is provided, only that file will be compressed.",
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
        "--path",
        type=str,
        help="Path to files to compress",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="A flag that triggers deletion of a single file instead of compression",
    )
    parser.add_argument(
        "-r",
        action="store_true",
        help="A flag that triggers recursive search on all subdirectories",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="A flag that triggers recursive search on all subdirectories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="A flag that forces overwriting when compressing.",
    )
    parser.add_argument(
        "--max_processes",
        type=int,
        help="The amount of maximum processes.",
    )
    parser.add_argument(
        "--hf_cache",
        action="store_true",
        help="A flag that indicates if the file is in the Hugging Face cache. Must either specify --model or --path to the model's snapshot cache.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Only when using --hf_cache, specify the model name or path. E.g. 'ibm-granite/granite-7b-instruct'",
    )
    parser.add_argument(
        "--model_branch",
        type=str,
        default="main",
        help="Only when using --model, specify the model branch. Default is 'main'",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["HUFFMAN", "ZSTD", "FSE", "AUTO"],
        default="AUTO",
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
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="The amount of threads to be used.",
    )
    args = parser.parse_args()
    optional_kwargs = {}
    if args.dtype:
        optional_kwargs["dtype"] = args.dtype
    if args.streaming_chunk_size is not None:
        optional_kwargs[
            "streaming_chunk_size"
        ] = args.streaming_chunk_size
    if args.path is not None:
        optional_kwargs["path"] = args.path
    if args.delete:
        optional_kwargs["delete"] = args.delete
    if args.r or args.recursive:
        optional_kwargs["r"] = args.r
    if args.force:
        optional_kwargs["force"] = args.force
    if args.max_processes:
        optional_kwargs["max_processes"] = (
            args.max_processes
        )
    if args.hf_cache:
        optional_kwargs["hf_cache"] = args.hf_cache
    if args.model:
        optional_kwargs["model"] = args.model
    if args.model_branch:
        optional_kwargs[
            "branch"
        ] = args.model_branch
    if args.method:
        optional_kwargs["method"] = args.method
    if args.verification:#
        optional_kwargs["verification"] = args.verification
    if args.test:
        optional_kwargs["test"] = args.test#
    if args.is_streaming:
        optional_kwargs["is_streaming"] = args.is_streaming#
    if args.threads:
        optional_kwargs["threads"] = args.threads#
    check_and_install_zipnn()
    compress_files_with_suffix(
        args.suffix, **optional_kwargs
    )
